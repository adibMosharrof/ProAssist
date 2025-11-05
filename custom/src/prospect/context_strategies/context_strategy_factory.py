"""Factory for creating context handling strategies"""

import logging
from typing import Dict, Any

from prospect.context_strategies import ContextStrategy, BaseContextStrategy
from prospect.context_strategies.drop_all import DropAllStrategy
from prospect.context_strategies.drop_middle import DropMiddleStrategy
from prospect.context_strategies.summarize_and_drop import SummarizeAndDropStrategy
from prospect.context_strategies.summarize_with_dst import SummarizeWithDSTStrategy


logger = logging.getLogger(__name__)


class ContextStrategyFactory:
    """Factory for creating context overflow handling strategies"""
    
    @staticmethod
    def create_strategy(
        strategy_type: str,
        max_seq_len: int,
        reserved_seq_len: int = 128,
        **kwargs
    ) -> BaseContextStrategy:
        """
        Create a context handling strategy
        
        Args:
            strategy_type: Type of strategy (drop_all, drop_middle, summarize_and_drop)
            max_seq_len: Maximum sequence length before overflow
            reserved_seq_len: Reserved tokens for new input
            **kwargs: Strategy-specific parameters
            
        Returns:
            Instantiated context strategy
            
        Raises:
            ValueError: If strategy_type is unknown
        """
        try:
            strategy_enum = ContextStrategy(strategy_type)
        except ValueError:
            raise ValueError(
                f"Unknown context strategy: {strategy_type}. "
                f"Available: {[s.value for s in ContextStrategy]}"
            )
        
        logger.info(f"Creating context strategy: {strategy_type}")
        
        if strategy_enum == ContextStrategy.DROP_ALL:
            return DropAllStrategy(
                max_seq_len=max_seq_len,
                reserved_seq_len=reserved_seq_len,
                **kwargs
            )
        
        elif strategy_enum == ContextStrategy.DROP_MIDDLE:
            last_keep_len = kwargs.pop('last_keep_len', 512)
            return DropMiddleStrategy(
                max_seq_len=max_seq_len,
                reserved_seq_len=reserved_seq_len,
                last_keep_len=last_keep_len,
                **kwargs
            )
        
        elif strategy_enum == ContextStrategy.SUMMARIZE_AND_DROP:
            summary_max_length = kwargs.pop('summary_max_length', 512)
            summary_prompt = kwargs.pop('summary_prompt', "Summarize the task progress so far.")
            initial_sys_prompt = kwargs.pop('initial_sys_prompt', None)
            task_knowledge = kwargs.pop('task_knowledge', None)
            return SummarizeAndDropStrategy(
                max_seq_len=max_seq_len,
                reserved_seq_len=reserved_seq_len,
                summary_max_length=summary_max_length,
                summary_prompt=summary_prompt,
                initial_sys_prompt=initial_sys_prompt,
                task_knowledge=task_knowledge,
                **kwargs
            )
        
        elif strategy_enum == ContextStrategy.SUMMARIZE_WITH_DST:
            summary_max_length = kwargs.pop('summary_max_length', 512)
            dst_file = kwargs.pop('dst_file', None)
            initial_sys_prompt = kwargs.pop('initial_sys_prompt', None)
            return SummarizeWithDSTStrategy(
                max_seq_len=max_seq_len,
                reserved_seq_len=reserved_seq_len,
                summary_max_length=summary_max_length,
                dst_file=dst_file,
                initial_sys_prompt=initial_sys_prompt,
                **kwargs
            )
        
        else:
            raise ValueError(f"Strategy {strategy_type} not implemented")
