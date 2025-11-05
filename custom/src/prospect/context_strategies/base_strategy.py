"""Base context strategy class and enum"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import logging
import torch


logger = logging.getLogger(__name__)


class ContextStrategy(Enum):
    """Available context overflow handling strategies"""
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE_AND_DROP = "summarize_and_drop"
    SUMMARIZE_WITH_DST = "summarize_with_dst"


class BaseContextStrategy(ABC):
    """
    Base class for context overflow handling strategies.
    
    Each strategy decides what to do when the KV cache exceeds the maximum
    sequence length during streaming inference.
    """
    
    def __init__(
        self,
        max_seq_len: int,
        reserved_seq_len: int = 128,
        **kwargs
    ):
        """
        Initialize context strategy
        
        Args:
            max_seq_len: Maximum sequence length before overflow
            reserved_seq_len: Reserved tokens for new input
            **kwargs: Strategy-specific parameters
        """
        self.max_seq_len = max_seq_len
        self.reserved_seq_len = reserved_seq_len
        self.ctxlen_to_reduce = max_seq_len - reserved_seq_len
        
    @abstractmethod
    def should_reduce_cache(self, current_seq_len: int) -> bool:
        """
        Check if cache should be reduced
        
        Args:
            current_seq_len: Current KV cache sequence length
            
        Returns:
            True if cache should be reduced
        """
        pass
    
    @abstractmethod
    def handle_overflow(
        self,
        past_key_values: Any,
        last_msg: Any,
        **context
    ) -> Tuple[Any, Any]:
        """
        Handle KV cache overflow
        
        Args:
            past_key_values: Current KV cache
            last_msg: Last message/token to preserve
            **context: Additional context (model, processor, current frame, etc.)
            
        Returns:
            Tuple of (new_past_key_values, new_last_msg)
        """
        pass
    
    def compress_cache(
        self,
        past_key_values: Any,
        attention_mask: Optional[torch.Tensor] = None,
        **context
    ) -> Tuple[Any, Optional[torch.Tensor]]:
        """
        Compress KV cache for model's prepare_inputs_for_generation.
        
        This method is called by the model during generation to automatically
        compress the cache when it exceeds limits. It wraps handle_overflow()
        and also updates the attention mask to match the new cache size.
        
        Args:
            past_key_values: Current KV cache
            attention_mask: Current attention mask (optional)
            **context: Additional context
            
        Returns:
            Tuple of (compressed_kv_cache, updated_attention_mask)
            Note: compressed_kv_cache can be None if strategy clears cache
        """
        # Use existing handle_overflow logic
        new_kv_cache, _ = self.handle_overflow(past_key_values, None, **context)
        
        # Update attention mask to match new cache size
        new_attention_mask = self._update_attention_mask(
            attention_mask, past_key_values, new_kv_cache
        )
        
        return new_kv_cache, new_attention_mask
    
    def _update_attention_mask(
        self,
        attention_mask: Optional[torch.Tensor],
        old_kv_cache: Any,
        new_kv_cache: Any
    ) -> Optional[torch.Tensor]:
        """
        Update attention mask to match compressed cache size.
        
        Args:
            attention_mask: Original attention mask
            old_kv_cache: Original KV cache (before compression)
            new_kv_cache: New KV cache (after compression)
            
        Returns:
            Updated attention mask or None
        """
        if attention_mask is None or new_kv_cache is None:
            return attention_mask
        
        old_seq_len = old_kv_cache[0][0].shape[2]
        new_seq_len = new_kv_cache[0][0].shape[2]
        
        if old_seq_len == new_seq_len:
            # No compression happened
            return attention_mask
        
        # Default: keep first tokens up to new length
        # Subclasses can override for custom behavior (e.g., drop_middle)
        return attention_mask[:, :new_seq_len]
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging"""
        pass
