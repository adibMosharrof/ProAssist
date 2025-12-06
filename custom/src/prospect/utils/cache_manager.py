"""KV Cache Manager for VLM streaming inference"""

import logging
from typing import Optional, Any, Tuple

from prospect.context_strategies import BaseContextStrategy


logger = logging.getLogger(__name__)


class KVCacheManager:
    """
    Manages KV cache state for streaming inference.
    
    Handles:
    - Cache state tracking
    - Initial cache storage (for drop_middle strategy)
    - Cache length queries
    
    Note: With Approach 2, cache compression happens automatically inside
    the model's prepare_inputs_for_generation(), so this class mainly
    tracks state.
    """
    
    def __init__(self, context_strategy: Optional[BaseContextStrategy] = None):
        """
        Initialize cache manager
        
        Args:
            context_strategy: Context strategy for overflow handling
        """
        self.context_strategy = context_strategy
        self.past_key_values: Optional[Any] = None
        self.last_msg_tokens: Optional[Any] = None
        self.initial_kv_cache: Optional[Any] = None
        
    def reset(self):
        """Reset all cache state (call at start of new video)"""
        self.past_key_values = None
        self.last_msg_tokens = None
        self.initial_kv_cache = None
        logger.debug("Cache state reset")
    
    def update_cache(self, new_cache: Any):
        """
        Update cache after generation
        
        Args:
            new_cache: New KV cache from model.generate()
        """
        self.past_key_values = new_cache
        
        # Store initial cache for drop_middle strategy (only first time)
        if self.initial_kv_cache is None and new_cache is not None:
            self.initial_kv_cache = new_cache
            if self.context_strategy and hasattr(self.context_strategy, 'set_initial_cache'):
                self.context_strategy.set_initial_cache(new_cache)
                logger.debug(f"Stored initial KV cache: {self.get_cache_length()} tokens")
    
    def get_cache(self) -> Optional[Any]:
        """Get current KV cache"""
        return self.past_key_values
    
    def get_cache_length(self) -> int:
        """
        Get current KV cache sequence length
        
        Returns:
            Number of tokens in cache, or 0 if cache is empty
        """
        if self.past_key_values is None:
            return 0
        # past_key_values: tuple of (keys, values) per layer
        # Shape: [batch, num_heads, seq_len, head_dim]
        return self.past_key_values[0][0].shape[2]
    
    def has_cache(self) -> bool:
        """Check if cache exists"""
        return self.past_key_values is not None
    
    def get_cache_stats(self) -> dict:
        """
        Get cache statistics for logging
        
        Returns:
            Dict with cache stats (length, has_initial, strategy_name)
        """
        return {
            'cache_length': self.get_cache_length(),
            'has_initial_cache': self.initial_kv_cache is not None,
            'strategy': self.context_strategy.name if self.context_strategy else 'none',
        }
