"""Drop all context strategy - simplest approach"""

import logging
from typing import Any, Tuple

from prospect.context_strategies import BaseContextStrategy


logger = logging.getLogger(__name__)


class DropAllStrategy(BaseContextStrategy):
    """
    Drop all KV cache when overflow occurs.
    
    This is the simplest strategy:
    - When context exceeds limit, drop ALL frames and dialogue from KV cache
    - Keep only the last message/token
    - Start fresh with minimal context
    
    Memory: Resets to ~0 tokens
    Quality: May lose important context
    """
    
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
        Drop all KV cache, keep only last message
        
        Args:
            past_key_values: Current KV cache (will be dropped)
            last_msg: Last message to preserve
            **context: Unused for this strategy
            
        Returns:
            (None, last_msg) - KV cache is completely cleared
        """
        logger.debug("DROP_ALL strategy: Clearing all KV cache")
        return None, last_msg
    
    @property
    def name(self) -> str:
        return "drop_all"
