"""Simple chat formatter for VLM models"""

import logging
from typing import List, Dict, Any


logger = logging.getLogger(__name__)


class ChatFormatter:
    """
    Simple chat formatter that wraps tokenizer's chat template functionality.
    
    This provides a consistent interface for formatting chat messages,
    used primarily by context strategies like summarize_and_drop.
    """
    
    def __init__(self, tokenizer: Any):
        """
        Initialize chat formatter
        
        Args:
            tokenizer: HuggingFace tokenizer with apply_chat_template support
        """
        self.tokenizer = tokenizer
        
    def apply_chat_template(
        self,
        messages: List[Dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True
    ) -> str:
        """
        Apply chat template to format messages
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
                      Example: [{"role": "system", "content": "You are helpful"}]
            tokenize: If True, return token IDs instead of string
            add_generation_prompt: If True, add prompt for generation
            
        Returns:
            Formatted chat string or token IDs
            
        Raises:
            RuntimeError: If chat template application fails
        """
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=tokenize,
                add_generation_prompt=add_generation_prompt
            )
            return formatted
        except Exception as e:
            logger.error(f"Chat template failed: {e}")
            raise RuntimeError(f"Failed to apply chat template: {e}")
    
    def format_system_message(self, content: str) -> str:
        """
        Format a system message
        
        Args:
            content: System message content
            
        Returns:
            Formatted system message string
        """
        return self.apply_chat_template(
            [{"role": "system", "content": content}],
            tokenize=False,
            add_generation_prompt=False
        )
    
    def format_user_message(self, content: str) -> str:
        """
        Format a user message
        
        Args:
            content: User message content
            
        Returns:
            Formatted user message string
        """
        return self.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=False,
            add_generation_prompt=True
        )
