"""Drop middle context strategy - keep first and last"""

import logging
from typing import Any, Tuple, Optional
import torch

from prospect.context_strategies import BaseContextStrategy


logger = logging.getLogger(__name__)


def trim_kv_cache(
    past_key_values: Tuple,
    start_idx: int,
    end_idx: int
) -> Tuple:
    """
    Trim KV cache to keep only tokens from start_idx to end_idx
    
    Args:
        past_key_values: Tuple of (keys, values) for each layer
        start_idx: Start index (inclusive)
        end_idx: End index (exclusive)
        
    Returns:
        Trimmed KV cache
    """
    trimmed = []
    for keys, values in past_key_values:
        trimmed.append((
            keys[:, :, start_idx:end_idx, :],
            values[:, :, start_idx:end_idx, :]
        ))
    return tuple(trimmed)


class DropMiddleStrategy(BaseContextStrategy):
    """
    Drop middle portion of KV cache, keep initial and recent context.
    
    Strategy:
    - Keep initial context (first few frames + dialogue)
    - Keep recent context (last N tokens)
    - Drop everything in the middle
    
    This preserves:
    - Task setup and initial instructions (from beginning)
    - Recent activity (from end)
    
    Memory: Bounded by init_len + last_keep_len
    Quality: Better than drop_all, loses middle context
    """
    
    def __init__(
        self,
        max_seq_len: int,
        reserved_seq_len: int = 128,
        last_keep_len: int = 512,
        **kwargs
    ):
        """
        Initialize drop middle strategy
        
        Args:
            max_seq_len: Maximum sequence length
            reserved_seq_len: Reserved tokens for new input
            last_keep_len: Number of recent tokens to keep
        """
        super().__init__(max_seq_len, reserved_seq_len, **kwargs)
        self.last_keep_len = last_keep_len
        self.init_kv_cache: Optional[Tuple] = None
        
    def set_initial_cache(self, past_key_values: Any):
        """
        Store initial KV cache (called after first user input)
        
        Args:
            past_key_values: KV cache from first turn
        """
        self.init_kv_cache = past_key_values
        logger.debug(f"Stored initial KV cache: {past_key_values[0][0].shape[2]} tokens")
    
    def should_reduce_cache(self, current_seq_len: int) -> bool:
        """Check if we've exceeded the threshold"""
        return current_seq_len >= self.ctxlen_to_reduce
    
    def handle_overflow(
        self,
        past_key_values: Any,
        last_msg: Any,
        **context
    ) -> Tuple[Any, Any]:
        """
        Keep initial + recent context, drop middle
        
        Args:
            past_key_values: Current KV cache
            last_msg: Last message to preserve
            **context: Unused for this strategy
            
        Returns:
            (concatenated_kv_cache, last_msg)
        """
        if self.init_kv_cache is None:
            logger.warning("No initial cache stored, falling back to drop all")
            return None, last_msg
        
        curr_seq_len = past_key_values[0][0].shape[2]
        init_kv_cache_len = self.init_kv_cache[0][0].shape[2]
        
        # Check if we even need to drop
        if curr_seq_len < self.last_keep_len:
            logger.debug("Current length < last_keep_len, no drop needed")
            return past_key_values, last_msg
        
        # Calculate where to start keeping recent tokens
        start = curr_seq_len - self.last_keep_len
        if start < init_kv_cache_len:
            start = init_kv_cache_len
        
        # Trim to get only recent tokens
        last_kv_cache = trim_kv_cache(past_key_values, start, curr_seq_len)
        
        # Concatenate initial + recent
        new_kv_cache = []
        for (init_keys, init_values), (last_keys, last_values) in zip(
            self.init_kv_cache, last_kv_cache
        ):
            new_kv_cache.append((
                torch.cat([init_keys, last_keys], dim=2),
                torch.cat([init_values, last_values], dim=2)
            ))
        
        new_kv_cache = tuple(new_kv_cache)
        new_len = new_kv_cache[0][0].shape[2]
        
        logger.debug(
            f"DROP_MIDDLE: {curr_seq_len} → {new_len} tokens "
            f"(init: {init_kv_cache_len}, recent: {self.last_keep_len})"
        )
        
        return new_kv_cache, last_msg
    
    @property
    def name(self) -> str:
        return "drop_middle"
