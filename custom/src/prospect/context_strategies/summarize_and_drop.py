"""Summarize and drop strategy - generate text summary of context"""

import logging
from typing import Any, Tuple, Optional, Callable, Dict

import torch

from prospect.context_strategies.base_strategy import BaseContextStrategy


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
        summary_prompt: Optional[Dict[str, str]] = None,
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
            summary_prompt: Structured prompt for summarization (ProAssist-style)
            initial_sys_prompt: Initial system prompt to prepend to summary
            task_knowledge: Task knowledge to append to summary
        """
        super().__init__(max_seq_len, reserved_seq_len, **kwargs)
        self.summary_max_length = summary_max_length
        
        # Use structured prompt (ProAssist-style) or fallback to simple dict
        if summary_prompt is None:
            self.summary_prompt = {
                "role": "system", 
                "content": "Watch the user's actions and track the task progress. Please summarize the progress."
            }
        else:
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
                - chat_formatter: Chat template formatter
                
        Returns:
            (None, summary_message) - KV cache cleared, summary as new context
        """
        # Extract and validate required context
        required = {'model', 'processor', 'chat_formatter'}
        missing = required - set(context.keys())
        
        if missing:
            raise ValueError(
                f"summarize_and_drop strategy missing required context: {missing}. "
                f"Pass these via compress_cache() kwargs."
            )
        
        model = context['model']
        processor = context['processor']
        chat_formatter = context['chat_formatter']
        trace = context.get('trace', None)  # Get trace if available
        current_timestamp = context.get('current_timestamp', 0.0)
        frame_idx = context.get('frame_idx', 0)
        
        try:
            # Generate summary using current frame + all past context
            summary, summary_prompt = self._generate_summary(
                model=model,
                processor=processor,
                past_key_values=past_key_values
            )
            
            logger.info(f"Generated summary: {summary}")
            
            # Record summary in trace
            if trace is not None and hasattr(trace, 'add_summary'):
                trace.add_summary(
                    timestamp=current_timestamp,
                    frame_idx=frame_idx,
                    summary=summary,
                    prompt=summary_prompt,  # Record the full prompt used
                )
            
            # 3. Format as system message (ProAssist-style)
            # The summary already includes system context, so just format it
            last_msg = chat_formatter.apply_chat_template([
                {"role": "system", "content": summary}
            ])
            
            logger.debug(
                f"SUMMARIZE_AND_DROP: Generated {len(summary.split())} word summary, "
                f"dropped {past_key_values[0][0].shape[2]} tokens"
            )
            
            # 5. Reset cache, keep only summary as context (ProAssist-style)
            return None, last_msg
            
        except Exception as e:
            logger.error(f"Summarization failed: {e}")
            raise RuntimeError(f"Failed to generate summary for summarize_and_drop strategy: {e}")
    
    def _generate_summary(
        self,
        model: Any,
        processor: Any,
        past_key_values: Any
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate progress summary using ProAssist-style approach
        
        Returns:
            Tuple of (summary_text, prompt_used)
        """
        try:
            # 1. Build comprehensive summarization prompt (ProAssist-style)
            # Include system context in the prompt for better summarization
            prompt_parts = []
            
            # Add initial system prompt if available
            if self.initial_sys_prompt:
                prompt_parts.append(self.initial_sys_prompt)
            
            # Add the base summarization instruction
            if isinstance(self.summary_prompt, dict):
                base_content = self.summary_prompt.get('content', 'Please summarize the progress.')
            else:
                base_content = str(self.summary_prompt)
            prompt_parts.append(base_content)
            
            # Add task knowledge if available
            if self.task_knowledge:
                prompt_parts.append(self.task_knowledge)
            
            # Combine all parts
            full_prompt_content = " ".join(prompt_parts)
            
            # Create the structured message
            summarization_message = {
                "role": "system",
                "content": full_prompt_content
            }
            
            # Use processor's get_input_sequence with the comprehensive message
            input_ids, _ = processor.get_input_sequence(
                num_images=0,  # No images for summarization
                messages=[summarization_message],
                first_turn=False
            )
            
            # Prepare model inputs (similar to ProAssist)
            model_inputs = {
                "input_ids": input_ids.to(model.device),
                # Note: In ProAssist, they keep original model_inputs but replace input_ids
                # You may need to adapt this based on your frame/image handling
            }

            # 2. Generate summary using ProAssist's generate_progress_summary approach
            with torch.no_grad():
                # Use joint_embed to get embeddings (ProAssist pattern)
                inputs_embeds = model.joint_embed(**model_inputs)
                
                # Generate summary using fast_greedy_generate (ProAssist pattern)
                output_ids, _ = model.fast_greedy_generate(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    max_length=self.summary_max_length,
                    verbose=False,
                )
            
            # 3. Decode the generated summary
            summary_raw = processor.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            
            # 4. Clean up the summary (remove prompt and assistant prefix)
            summary = self._clean_summary(summary_raw)
            
            return summary.strip(), summarization_message
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback: simple text summary
            fallback_summary = "Task in progress."
            fallback_prompt = self.summary_prompt if isinstance(self.summary_prompt, dict) else {"role": "system", "content": str(self.summary_prompt)}
            return fallback_summary, fallback_prompt
    
    def _clean_summary(self, summary_raw: str) -> str:
        """
        Clean up the generated summary by removing prompts and prefixes
        """
        summary = summary_raw
        
        # Remove assistant prefix if present
        if "Assistant:" in summary:
            summary = summary.split("Assistant:")[-1].strip()
        
        # Remove the original prompt if it appears in the response
        prompt_content = self.summary_prompt.get("content", "")
        if prompt_content in summary:
            summary = summary.replace(prompt_content, "").strip()
        
        return summary
    
    @property
    def name(self) -> str:
        return "summarize_and_drop"
