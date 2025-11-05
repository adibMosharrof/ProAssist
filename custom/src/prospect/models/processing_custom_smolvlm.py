"""Custom processor for SmolVLM2 with streaming support"""

import logging
import torch
from transformers import AutoProcessor
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


class CustomSmolVLMProcessor:
    """
    Processor for CustomSmolVLM with streaming support.
    
    Handles:
    - Frame-by-frame input preparation
    - Chat template formatting
    - Token sequence management
    - Text cleanup
    
    This is similar to ProAssist's StreamProcessor but adapted for SmolVLM2.
    """
    
    def __init__(self, base_processor: AutoProcessor):
        """
        Initialize custom processor
        
        Args:
            base_processor: Standard HuggingFace AutoProcessor for SmolVLM2
        """
        self.base_processor = base_processor
        self.tokenizer = base_processor.tokenizer
        self.image_processor = base_processor.image_processor
    
    def __call__(self, *args, **kwargs):
        """Delegate to base processor for standard calls"""
        return self.base_processor(*args, **kwargs)
    
    def decode(self, *args, **kwargs):
        """Delegate decode to tokenizer"""
        return self.tokenizer.decode(*args, **kwargs)
    
    def batch_decode(self, *args, **kwargs):
        """Delegate batch_decode to tokenizer"""
        return self.tokenizer.batch_decode(*args, **kwargs)
    
    def get_input_sequence(
        self,
        num_images: int,
        messages: List[Dict[str, str]],
        first_turn: bool = False
    ) -> Tuple[torch.LongTensor, str]:
        """
        Prepare input sequence for streaming inference.
        
        Similar to ProAssist's StreamProcessor.get_input_sequence()
        
        Args:
            num_images: Number of images in this turn
            messages: List of message dicts with 'role' and 'content' keys
            first_turn: Whether this is the first turn (applies full template)
            
        Returns:
            (input_ids, input_str) - Tokenized IDs and formatted string
        """
        # Format text messages
        if messages:
            if first_turn:
                # Apply full chat template for first turn
                input_str_txt = self._apply_chat_template(messages)
            else:
                # Append messages for subsequent turns
                input_str_txt = ""
                for msg in messages:
                    input_str_txt += self._add_message(msg)
        else:
            input_str_txt = ""
        
        # Add image tokens
        input_str_img = self._add_img_tokens(num_images)
        input_str = input_str_txt + input_str_img
        
        # Tokenize
        input_ids = self.tokenizer(
            input_str,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"]
        
        return input_ids, input_str
    
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """
        Apply chat template (SmolVLM2 format)
        
        Args:
            messages: List of message dicts
            
        Returns:
            Formatted chat string
        """
        # SmolVLM2 uses standard chat template
        try:
            formatted = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            return formatted
        except Exception as e:
            logger.warning(f"Chat template failed: {e}, using simple format")
            # Fallback to simple format
            result = ""
            for msg in messages:
                result += self._add_message(msg)
            return result
    
    def _add_message(self, message: Dict) -> str:
        """
        Add a single message to the conversation
        
        Args:
            message: Dict with 'role' and 'content'
            
        Returns:
            Formatted message string
        """
        role = message.get("role", "user")
        content = message.get("content", "")
        
        # Simple format that works with most models
        return f"{role.capitalize()}: {content}\n"
    
    def _add_img_tokens(self, num_images: int) -> str:
        """
        Add image placeholder tokens
        
        Args:
            num_images: Number of images to add
            
        Returns:
            String with image tokens
        """
        # SmolVLM2 uses <image> token
        return "<image>" * num_images
    
    def add_last_assistant_message(
        self,
        model_inputs: Dict[str, torch.Tensor],
        last_msg: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepend last assistant message to current input.
        
        This is used to maintain dialogue context in KV cache.
        Similar to ProAssist's StreamProcessor.add_last_assistant_message()
        
        Args:
            model_inputs: Current frame's model inputs
            last_msg: Last message tokens to prepend
            
        Returns:
            Updated model_inputs with last message prepended
        """
        if last_msg is None:
            return model_inputs
        
        input_ids = model_inputs["input_ids"]
        
        # Ensure last_msg is a tensor
        if isinstance(last_msg, str):
            last_msg = self.tokenizer(
                last_msg,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]
        
        # Concatenate: [last_msg] + [current_input]
        input_ids = torch.cat([last_msg.to(input_ids.device), input_ids], dim=-1)
        model_inputs["input_ids"] = input_ids
        
        return model_inputs
    
    def cleanup_text(self, text: str) -> Tuple[str, Optional[str]]:
        """
        Clean up generated text.
        
        Similar to ProAssist's StreamProcessor.cleanup_text()
        
        Args:
            text: Raw generated text
            
        Returns:
            (cleaned_text, role) - Cleaned text and extracted role (if any)
        """
        # Remove special tokens
        text = text.strip()
        
        # Remove image tokens
        text = text.replace("<image>", "")
        text = text.replace("<|image|>", "")
        
        # Remove common artifacts
        text = text.replace("<|endoftext|>", "")
        text = text.replace("<|end|>", "")
        
        # Extract role if present
        if ":" in text:
            parts = text.split(":", 1)
            role_candidate = parts[0].strip().lower()
            if role_candidate in ["assistant", "user", "system"]:
                return parts[1].strip(), role_candidate
        
        # Handle empty responses
        if text == ".":
            text = ""
        
        return text, None
