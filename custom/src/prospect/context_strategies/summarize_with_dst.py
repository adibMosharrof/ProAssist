"""Summarize with DST strategy - use dialogue state tracking to guide summarization"""

import logging
from typing import Any, Tuple, Optional, Dict, List
from pathlib import Path

import pandas as pd
import torch

from prospect.context_strategies.base_strategy import BaseContextStrategy


logger = logging.getLogger(__name__)


class SummarizeWithDSTStrategy(BaseContextStrategy):
    """
    Generate text summary guided by DST (Dialogue State Tracking) state.
    
    Strategy:
    - When overflow occurs, use DST annotations to understand current task state
    - Generate summary that includes: completed steps, current step, next steps
    - Drop ALL KV cache (frames + dialogue)
    - Keep only the DST-guided summary
    - Continue with summary as context
    
    This extends the basic summarize_and_drop with structured task knowledge.
    
    Memory: Resets to ~100-500 tokens (summary only)
    Quality: Expected to be better - uses task structure for better summaries
    """
    
    def __init__(
        self,
        max_seq_len: int,
        reserved_seq_len: int = 128,
        summary_max_length: int = 256,
        dst_file: Optional[str] = None,
        initial_sys_prompt: Optional[str] = None,
        task_knowledge: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize summarize with DST strategy
        
        Args:
            max_seq_len: Maximum sequence length
            reserved_seq_len: Reserved tokens for new input
            summary_max_length: Max tokens for generated summary
            dst_file: Path to DST TSV file with annotations
            initial_sys_prompt: Initial system prompt to prepend to summary
            task_knowledge: Task-specific knowledge to include in prompts
        """
        super().__init__(max_seq_len, reserved_seq_len, **kwargs)
        self.summary_max_length = summary_max_length
        self.initial_sys_prompt = initial_sys_prompt
        self.task_knowledge = task_knowledge
        
        # Load DST annotations
        self.dst_annotations = None
        if dst_file and Path(dst_file).exists():
            try:
                self.dst_annotations = pd.read_csv(dst_file, sep='\t')
                logger.info(f"Loaded DST annotations from {dst_file}: {len(self.dst_annotations)} entries")
            except Exception as e:
                logger.error(f"Failed to load DST file {dst_file}: {e}")
        else:
            logger.warning(f"DST file not provided or doesn't exist: {dst_file}")
        
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
        Generate DST-guided summary and drop all KV cache
        
        Args:
            past_key_values: Current KV cache (contains all history)
            last_msg: Last message (will be replaced by summary)
            **context: Must contain:
                - model: VLM model for generation
                - processor: Processor for tokenization
                - chat_formatter: Chat template formatter
                - current_timestamp: Current video timestamp (for DST lookup)
                
        Returns:
            (None, summary_message) - KV cache cleared, DST-guided summary as new context
        """
        # Extract and validate required context
        required = {'model', 'processor', 'chat_formatter'}
        missing = required - set(context.keys())
        
        if missing:
            raise ValueError(
                f"summarize_with_dst strategy missing required context: {missing}. "
                f"Pass these via compress_cache() kwargs."
            )
        
        model = context['model']
        processor = context['processor']
        chat_formatter = context['chat_formatter']
        current_timestamp = context.get('current_timestamp', 0.0)
        trace = context.get('trace', None)  # Get trace if available
        frame_idx = context.get('frame_idx', 0)
        
        # Get DST state at current timestamp
        dst_state = self._get_dst_state(current_timestamp)
        logger.info(f"DST state at t={current_timestamp:.1f}s: {dst_state}")
        
        # Generate DST-guided summary using ProAssist-style approach
        summary, summary_prompt = self._generate_summary(
            model=model,
            processor=processor,
            past_key_values=past_key_values,
            dst_state=dst_state
        )
        
        logger.info(f"Generated DST-guided summary at t={current_timestamp:.1f}s: {summary}")
        
        # Record summary in trace
        if trace is not None and hasattr(trace, 'add_summary'):
            trace.add_summary(
                timestamp=current_timestamp,
                frame_idx=frame_idx,
                summary=summary,
                prompt=summary_prompt,
                dst_state=dst_state,
            )
        
        # Format as system message (ProAssist-style)
        # The summary already includes system context, so just format it
        last_msg = chat_formatter.apply_chat_template([
            {"role": "system", "content": summary}
        ])
        
        logger.debug(
            f"SUMMARIZE_WITH_DST: Generated {len(summary.split())} word summary "
            f"at {dst_state.get('current_step', 'unknown')}, "
            f"dropped {past_key_values[0][0].shape[2]} tokens"
        )
        
        # Return None for KV cache (drop everything), formatted summary as new context
        return None, last_msg
    
    def _get_dst_state(self, timestamp: float) -> Dict[str, Any]:
        """
        Get DST state at given timestamp
        
        Args:
            timestamp: Current video timestamp in seconds
            
        Returns:
            Dictionary with DST state information:
                - completed_steps: List of completed step names
                - current_step: Current step name
                - current_substep: Current substep name
                - current_action: Current action name
                - next_step: Next step name (if available)
        """
        # Find all entries active at current timestamp
        active = self.dst_annotations[
            (self.dst_annotations['start_ts'] <= timestamp) &
            (self.dst_annotations['end_ts'] >= timestamp)
        ]
        
        # Find completed steps (ended before current timestamp)
        completed = self.dst_annotations[
            (self.dst_annotations['type'] == 'STEP') &
            (self.dst_annotations['end_ts'] < timestamp)
        ]
        
        # Get current step, substep, action
        current_step = active[active['type'] == 'STEP']['name'].iloc[0] if len(active[active['type'] == 'STEP']) > 0 else 'Unknown'
        current_substep = active[active['type'] == 'SUBSTEP']['name'].iloc[0] if len(active[active['type'] == 'SUBSTEP']) > 0 else 'Unknown'
        current_action = active[active['type'] == 'ACTION']['name'].iloc[0] if len(active[active['type'] == 'ACTION']) > 0 else 'Unknown'
        
        # Find next step (starts after current timestamp)
        future_steps = self.dst_annotations[
            (self.dst_annotations['type'] == 'STEP') &
            (self.dst_annotations['start_ts'] > timestamp)
        ].sort_values('start_ts')
        next_step = future_steps['name'].iloc[0] if len(future_steps) > 0 else None
        
        completed_step_names = completed['name'].tolist() if len(completed) > 0 else []
        
        return {
            'completed_steps': completed_step_names,
            'current_step': current_step,
            'current_substep': current_substep,
            'current_action': current_action,
            'next_step': next_step,
            'timestamp': timestamp,
        }
    
    def _create_dst_guided_prompt(self, dst_state: Dict[str, Any]) -> str:
        """
        Create a prompt that guides the model to use DST information for summarization
        
        Focus on making the DST information clear and structured.
        """
        prompt_parts = [
            "Task Progress Summary:",
        ]
        
        # Completed steps
        if dst_state['completed_steps']:
            completed_str = "; ".join(dst_state['completed_steps'])
            prompt_parts.append(f"✓ Completed: {completed_str}")
        
        # Current activity - make this prominent
        prompt_parts.append(f"🔄 Currently: {dst_state['current_step']}")
        if dst_state['current_substep'] != 'Unknown':
            prompt_parts.append(f"   Substep: {dst_state['current_substep']}")
        if dst_state['current_action'] != 'Unknown':
            prompt_parts.append(f"   Action: {dst_state['current_action']}")
        
        # Next step
        if dst_state['next_step']:
            prompt_parts.append(f"➡️ Next: {dst_state['next_step']}")
        
        # Clear instruction
        prompt_parts.append("Based on this task progress and any conversation/visual context, provide a brief summary of the current assembly state.")
        
        full_prompt = " ".join(prompt_parts)
        logger.info(f"DST-guided prompt: {full_prompt}")
        return full_prompt
    
    def _generate_summary(
        self,
        model: Any,
        processor: Any,
        past_key_values: Any,
        dst_state: Dict[str, Any]
    ) -> Tuple[str, Dict[str, str]]:
        """
        Generate progress summary using DST-guided ProAssist-style approach
        
        Returns:
            Tuple of (summary_text, prompt_used)
        """
        try:
            # 1. Build comprehensive DST-guided prompt (ProAssist-style)
            # Include system context and DST information in the prompt for better summarization
            prompt_parts = []
            
            # Add initial system prompt if available
            if self.initial_sys_prompt:
                prompt_parts.append(self.initial_sys_prompt)
            
            # Add task knowledge if available
            if self.task_knowledge:
                prompt_parts.append(self.task_knowledge)
            
            # Add DST-guided summarization instruction
            dst_instruction = self._create_dst_guided_prompt(dst_state)
            prompt_parts.append(dst_instruction)
            
            # Combine all parts
            full_prompt_content = " ".join(prompt_parts)
            
            logger.info(f"Full summarization prompt content: {full_prompt_content}")
            
            # Create the structured message
            summarization_message = {
                "role": "system",
                "content": full_prompt_content
            }
            
            # Use processor's get_input_sequence with the comprehensive DST-guided message
            input_ids, _ = processor.get_input_sequence(
                num_images=0,  # No images for summarization
                messages=[summarization_message],
                first_turn=False
            )
            
            # Prepare model inputs (similar to ProAssist)
            model_inputs = {
                "input_ids": input_ids.to(model.device),
                # Note: In ProAssist, they keep original model_inputs but replace input_ids
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
            
            logger.info(f"Raw generated summary: '{summary_raw}'")
            
            # 4. Clean up the summary (remove prompt and assistant prefix)
            summary = self._clean_summary(summary_raw, full_prompt_content)
            
            return summary.strip(), summarization_message
            
        except Exception as e:
            logger.error(f"DST-guided summary generation failed: {e}")
            # Fallback: simple text summary
            fallback_summary = f"Task progress at {dst_state.get('current_step', 'unknown step')}."
            fallback_prompt = {"role": "system", "content": f"Watch the user's actions and track the task progress. {self._create_dst_guided_prompt(dst_state)}"}
            return fallback_summary, fallback_prompt
    
    def _clean_summary(self, summary_raw: str, prompt_content: str) -> str:
        """
        Clean up the generated summary by removing prompts and prefixes
        
        Args:
            summary_raw: Raw generated summary
            prompt_content: The prompt content that was used
            
        Returns:
            Cleaned summary text
        """
        summary = summary_raw
        
        # Remove assistant prefix if present and take the first complete response
        if "Assistant:" in summary:
            # Split on "Assistant:" and take the first actual response (not the prompt)
            parts = summary.split("Assistant:")
            # Find the first non-empty part after removing whitespace
            for part in parts[1:]:  # Skip the first empty part
                candidate = part.strip()
                if candidate and not candidate.startswith(prompt_content[:50]):  # Avoid prompt fragments
                    summary = candidate
                    break
        
        # If there are still multiple responses, take just the first one
        if "Assistant:" in summary:
            summary = summary.split("Assistant:")[0].strip()
        
        # Only remove the exact prompt if it appears at the beginning
        if summary.startswith(prompt_content):
            summary = summary[len(prompt_content):].strip()
        
        # Remove any leading/trailing punctuation that might remain
        summary = summary.strip(".,;:!? ")
        
        return summary
    
    @property
    def name(self) -> str:
        return "summarize_with_dst"
