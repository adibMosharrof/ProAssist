"""Context handling strategies for managing KV cache overflow"""

from enum import Enum
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any
import torch


class ContextStrategy(Enum):
    """Available context overflow handling strategies"""
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE_AND_DROP = "summarize_and_drop"


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
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for logging"""
        pass
