"""Summarize and drop strategy - generate text summary of context"""

import logging
from typing import Any, Tuple, Optional, Callable

import torch

from prospect.context_strategies import BaseContextStrategy


logger = logging.getLogger(__name__)


class SummarizeAndDropStrategy(BaseContextStrategy):
    """
    Generate text summary of context, then drop all KV cache.
    
    Strategy:
    - When overflow occurs, ask model to summarize progress
    - Model "sees" all frames + dialogue via KV cache
    - Model generates text summary (e.g., "completed steps 1-3, working on step 4")
    - Drop ALL KV cache (frames + dialogue)
    - Keep only the text summary
    - Continue with summary as context
    
    This is ProAssist's main strategy for long videos.
    
    Memory: Resets to ~100-500 tokens (summary only)
    Quality: Best - semantic information preserved in text
    """
    
    def __init__(
        self,
        max_seq_len: int,
        reserved_seq_len: int = 128,
        summary_max_length: int = 512,
        summary_prompt: str = "Summarize the task progress so far.",
        initial_sys_prompt: Optional[str] = None,
        task_knowledge: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize summarize and drop strategy
        
        Args:
            max_seq_len: Maximum sequence length
            reserved_seq_len: Reserved tokens for new input
            summary_max_length: Max tokens for generated summary
            summary_prompt: Prompt for summarization
            initial_sys_prompt: Initial system prompt to prepend to summary
            task_knowledge: Task knowledge to append to summary
        """
        super().__init__(max_seq_len, reserved_seq_len, **kwargs)
        self.summary_max_length = summary_max_length
        self.summary_prompt = summary_prompt
        self.initial_sys_prompt = initial_sys_prompt
        self.task_knowledge = task_knowledge
        
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
        Generate summary and drop all KV cache
        
        Args:
            past_key_values: Current KV cache (contains all history)
            last_msg: Last message (will be replaced by summary)
            **context: Must contain:
                - model: VLM model for generation
                - processor: Processor for tokenization
                - current_frame: Current frame being processed
                - num_frames: Number of frames in current input
                - chat_formatter: Chat template formatter
                
        Returns:
            (None, summary_message) - KV cache cleared, summary as new context
        """
        # Extract required context
        model = context.get('model')
        processor = context.get('processor')
        current_frame = context.get('current_frame')
        num_frames = context.get('num_frames', 1)
        chat_formatter = context.get('chat_formatter')
        
        if not all([model, processor, current_frame, chat_formatter]):
            logger.error("Missing required context for summarization, falling back to drop all")
            return None, last_msg
        
        try:
            # Generate summary using current frame + all past context
            summary = self._generate_summary(
                model=model,
                processor=processor,
                current_frame=current_frame,
                num_frames=num_frames,
                past_key_values=past_key_values
            )
            
            logger.info(f"Generated summary: {summary}")
            
            # Optionally prepend initial system prompt
            if self.initial_sys_prompt:
                summary = f"{self.initial_sys_prompt} {summary}"
            
            # Optionally append task knowledge
            if self.task_knowledge:
                summary = f"{summary} {self.task_knowledge}"
            
            # Format as system message
            summary_msg = chat_formatter.apply_chat_template(
                [{"role": "system", "content": summary}]
            )
            
            logger.debug(
                f"SUMMARIZE_AND_DROP: Generated {len(summary.split())} word summary, "
                f"dropped {past_key_values[0][0].shape[2]} tokens"
            )
            
            # Return None for KV cache (drop everything), summary as new context
            return None, summary_msg
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}, falling back to drop all")
            return None, last_msg
    
    def _generate_summary(
        self,
        model: Any,
        processor: Any,
        current_frame: Any,
        num_frames: int,
        past_key_values: Any
    ) -> str:
        """
        Generate progress summary using VLM
        
        Args:
            model: VLM model (standard HuggingFace model)
            processor: HuggingFace processor
            current_frame: Current frame's model_inputs dict
            num_frames: Number of frames
            past_key_values: KV cache with all history
            
        Returns:
            Generated summary text
        """
        try:
            # Prepare summarization prompt with current frame
            # Use current frame's image but replace text with summary prompt
            summary_inputs = dict(current_frame)
            
            # Create summary prompt tokens
            summary_text = f"<image>{self.summary_prompt}"
            summary_tokens = processor(
                text=summary_text,
                return_tensors="pt",
                add_special_tokens=False
            )["input_ids"]
            
            # Replace input_ids with summary prompt
            summary_inputs["input_ids"] = summary_tokens
            
            # Move to device
            summary_inputs = {
                k: v.to(model.device) if hasattr(v, 'to') else v
                for k, v in summary_inputs.items()
            }
            
            # Generate summary using accumulated KV cache
            # Model "sees" all previous frames via past_key_values
            with torch.no_grad():
                result = model.generate(
                    **summary_inputs,
                    past_key_values=past_key_values,
                    max_new_tokens=self.summary_max_length,
                    temperature=0.3,
                    do_sample=False,  # Deterministic for summaries
                    use_cache=True,
                    return_dict_in_generate=True,
                )
            
            # Decode summary
            summary_raw = processor.decode(result.sequences[0], skip_special_tokens=True)
            
            # Clean up
            if "Assistant:" in summary_raw:
                summary_raw = summary_raw.split("Assistant:")[-1].strip()
            if self.summary_prompt in summary_raw:
                summary_raw = summary_raw.replace(self.summary_prompt, "").strip()
            
            return summary_raw.strip()
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: simple text summary
            return "Task in progress."
    
    @property
    def name(self) -> str:
        return "summarize_and_drop"
