"""VLM-based streaming inference runner for PROSPECT"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    SmolVLMForConditionalGeneration,
)

from mmassist.eval.runners.stream_inference import FrameOutput
from prospect.context_strategies.context_strategy_factory import ContextStrategyFactory
from prospect.context_strategies import BaseContextStrategy


logger = logging.getLogger(__name__)


class VLMStreamRunner:
    """
    Custom inference runner using VLM (SmolVLM2) for dialogue generation.

    This runner is compatible with ProAssist's StreamEvaluator framework.
    It generates dialogues at substep transitions using a vision-language model.
    """

    def __init__(
        self,
        model_name: str,
        eval_name: str = "vlm_baseline",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        transition_detection_prompt: str = "",
        dialogue_generation_prompt: str = "",
        fps: int = 2,
        not_talk_threshold: float = 0.5,
        use_gt_substeps: bool = True,
        cache_dir: Optional[str] = None,
        context_strategy_type: str = "none",
        context_strategy_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Initialize VLM stream runner

        Args:
            model_name: HuggingFace model name (e.g., "HuggingFaceTB/SmolVLM2-Instruct")
            eval_name: Name for this evaluation run
            device: Device to run model on
            torch_dtype: torch dtype for model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling
            transition_detection_prompt: Prompt for detecting substeps
            dialogue_generation_prompt: Prompt for generating dialogues
            fps: Frames per second
            not_talk_threshold: Threshold for silence
            use_gt_substeps: Use ground truth substeps (True for baseline)
            cache_dir: HuggingFace cache directory
            context_strategy_type: Type of context overflow handling (none, drop_all, drop_middle, summarize_and_drop)
            context_strategy_config: Additional config for context strategy
        """
        self.model_name = model_name
        self.eval_name = eval_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.transition_detection_prompt = transition_detection_prompt
        self.dialogue_generation_prompt = dialogue_generation_prompt
        self.fps = fps
        self.not_talk_threshold = not_talk_threshold
        self.use_gt_substeps = use_gt_substeps
        
        # Context strategy for KV cache management
        self.context_strategy: Optional[BaseContextStrategy] = None
        if context_strategy_type and context_strategy_type != "none":
            strategy_config = context_strategy_config or {}
            self.context_strategy = ContextStrategyFactory.create_strategy(
                strategy_type=context_strategy_type,
                max_seq_len=kwargs.get('max_seq_len', 4096),
                reserved_seq_len=kwargs.get('reserved_seq_len', 128),
                **strategy_config
            )
            logger.info(f"Context strategy enabled: {self.context_strategy.name}")
        else:
            logger.info("No context strategy (stateless processing)")

        logger.info(f"Loading VLM model: {model_name}")
        logger.info(f"Device: {device}, dtype: {torch_dtype}")

        # Set cache directory if provided
        if cache_dir:
            os.environ["HF_HOME"] = cache_dir
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

        # Load VLM
        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16

        try:
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            # Use SmolVLMForConditionalGeneration directly
            self.model = SmolVLMForConditionalGeneration.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            self.model.eval()
            logger.info("✅ VLM model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise

        # State tracking for transitions
        self.prev_substep = None
        self.current_substep = None
        
        # KV cache state (for context accumulation)
        self.past_key_values = None
        self.last_msg_tokens = None
        self.initial_kv_cache = None  # For drop_middle strategy
        self.use_kv_cache = kwargs.get('use_kv_cache', False)
        
        if self.use_kv_cache:
            logger.info("KV cache accumulation ENABLED")
        else:
            logger.info("KV cache accumulation DISABLED (stateless mode)")

    def run_inference_on_video(
        self, video: Dict[str, Any], output_dir: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Run streaming inference on a video.

        This is the main method called by ProAssist's StreamEvaluator.

        Args:
            video: Dict with keys: video_id, frames, conversation, dst_annotations, fps
            output_dir: Where to save results (optional)

        Returns:
            Dict with predictions in FrameOutput format:
                - predictions: List[FrameOutput]
                - video_id: str
        """
        video_id = video["video_id"]
        frames = video["frames"]
        ground_truth_conv = video.get("conversation", [])
        dst_annotations = video.get("dst_annotations")

        logger.info(f"Running inference on video {video_id}: {len(frames)} frames")
        if self.use_kv_cache:
            logger.info("KV cache accumulation: ENABLED")
        else:
            logger.info("KV cache accumulation: DISABLED (stateless)")

        # Reset state for new video
        self.prev_substep = None
        self.current_substep = None
        self.past_key_values = None
        self.last_msg_tokens = None
        self.initial_kv_cache = None

        outputs = []

        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / self.fps

            # Detect current substep
            curr_substep = self._detect_substep(frame, dst_annotations, timestamp)

            # Check if transition occurred
            is_transition = (
                self.prev_substep is not None
                and curr_substep != self.prev_substep
                and curr_substep is not None
            )

            # Generate dialogue if transition
            if is_transition:
                if self.use_kv_cache:
                    # Generate with KV cache accumulation
                    dialogue = self._generate_dialogue_with_cache(
                        frame, self.prev_substep, curr_substep
                    )
                else:
                    # Generate without KV cache (stateless)
                    dialogue = self._generate_dialogue(
                        frame, self.prev_substep, curr_substep
                    )
                
                logger.debug(
                    f"Frame {frame_idx} ({timestamp:.1f}s): Transition "
                    f"{self.prev_substep} → {curr_substep}: {dialogue}"
                )
            else:
                dialogue = ""  # Silent (no dialogue)

            # Get reference dialogue at this timestamp
            ref_dialogue = self._get_reference_dialogue(ground_truth_conv, timestamp)

            # Create output
            outputs.append(
                FrameOutput(
                    gen=dialogue,
                    ref=ref_dialogue,
                    image=frame,
                    frame_idx_in_stream=frame_idx,
                    timestamp_in_stream=timestamp,
                )
            )

            # Update state
            self.prev_substep = curr_substep
            
            # Log KV cache size if enabled
            if self.use_kv_cache and frame_idx % 50 == 0:
                cache_len = self._get_cache_length()
                logger.debug(f"Frame {frame_idx}: KV cache size = {cache_len} tokens")

        # Count generated dialogues
        num_dialogues = sum(1 for o in outputs if o.gen != "")
        num_refs = sum(1 for o in outputs if o.ref != "")
        logger.info(
            f"Generated {num_dialogues} dialogues "
            f"({num_refs} ground truth dialogues)"
        )

        # StreamEvaluator will save predictions, so we just return them
        return {
            "predictions": outputs,
            "video_id": video_id,
        }

    def _detect_substep(
        self, frame: Image.Image, dst_annotations: Any, timestamp: float
    ) -> Optional[str]:
        """
        Detect current substep from frame.

        For baseline (Day 1): Use ground truth substeps from DST annotations.
        For enhanced (Day 2): Use VLM to predict substep.
        """
        if self.use_gt_substeps and dst_annotations is not None:
            # Use ground truth substeps
            substeps = dst_annotations[
                (dst_annotations["type"].str.upper() == "SUBSTEP")
                & (dst_annotations["start_ts"] <= timestamp)
                & (dst_annotations["end_ts"] >= timestamp)
            ]
            if len(substeps) > 0:
                return substeps.iloc[0]["name"]
            return None

        # Fallback: Ask VLM to predict substep (for Day 2 DST-enhanced)
        prompt = self.transition_detection_prompt

        try:
            inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=50,
                    temperature=0.3,
                    do_sample=False,  # Deterministic for substep detection
                )

            substep = self.processor.decode(outputs[0], skip_special_tokens=True)
            return substep.strip()
        except Exception as e:
            logger.warning(f"VLM substep detection failed: {e}")
            return None

    def _generate_dialogue(
        self, frame: Image.Image, prev_substep: str, curr_substep: str
    ) -> str:
        """Generate dialogue for transition using VLM"""
        prompt = self.dialogue_generation_prompt.format(
            prev_substep=prev_substep, curr_substep=curr_substep
        )

        try:
            inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                )

            # Decode and clean up
            dialogue = self.processor.decode(outputs[0], skip_special_tokens=True)

            # Remove the prompt if it's in the output
            # The model often repeats the prompt, so we need to clean it
            # Split by "Assistant:" and take the last part
            if "Assistant:" in dialogue:
                dialogue = dialogue.split("Assistant:")[-1].strip()

            # Also try to remove the original prompt if it's still there
            if prompt in dialogue:
                dialogue = dialogue.replace(prompt, "").strip()

            return dialogue.strip()

        except Exception as e:
            logger.warning(f"VLM dialogue generation failed: {e}")
            return ""
    
    def _generate_dialogue_with_cache(
        self, frame: Image.Image, prev_substep: str, curr_substep: str
    ) -> str:
        """
        Generate dialogue with KV cache accumulation.
        
        This method maintains past_key_values across generations,
        allowing the model to have context from previous frames.
        """
        prompt = self.dialogue_generation_prompt.format(
            prev_substep=prev_substep, curr_substep=curr_substep
        )

        try:
            # Prepare inputs (frame + prompt)
            inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Generate with KV cache accumulation
                # Note: past_key_values contains history from previous generations
                result = self.model.generate(
                    **inputs,
                    past_key_values=self.past_key_values,
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    do_sample=self.do_sample,
                    use_cache=True,
                    return_dict_in_generate=True,
                )
                
                # Extract output and updated cache
                output_ids = result.sequences
                self.past_key_values = result.past_key_values

            # Decode dialogue
            dialogue = self.processor.decode(output_ids[0], skip_special_tokens=True)

            # Clean up prompt leakage
            if "Assistant:" in dialogue:
                dialogue = dialogue.split("Assistant:")[-1].strip()
            if prompt in dialogue:
                dialogue = dialogue.replace(prompt, "").strip()
            
            dialogue = dialogue.strip()
            
            # Store initial cache for drop_middle strategy
            if self.initial_kv_cache is None and self.past_key_values is not None:
                self.initial_kv_cache = self.past_key_values
                if hasattr(self.context_strategy, 'set_initial_cache'):
                    self.context_strategy.set_initial_cache(self.past_key_values)
                    logger.debug("Stored initial KV cache for drop_middle strategy")

            # Check for overflow and apply context strategy
            if self.context_strategy:
                cache_len = self._get_cache_length()
                if self.context_strategy.should_reduce_cache(cache_len):
                    logger.info(
                        f"KV cache overflow: {cache_len} tokens, "
                        f"applying {self.context_strategy.name} strategy"
                    )
                    self._apply_context_strategy(inputs)

            return dialogue

        except Exception as e:
            logger.warning(f"VLM dialogue generation with cache failed: {e}")
            return ""
    
    def _get_cache_length(self) -> int:
        """Get current KV cache sequence length"""
        if self.past_key_values is None:
            return 0
        # past_key_values: tuple of (keys, values) per layer
        # Shape: [batch, num_heads, seq_len, head_dim]
        return self.past_key_values[0][0].shape[2]
    
    def _apply_context_strategy(self, current_frame_inputs: Dict):
        """Apply context overflow strategy"""
        if not self.context_strategy:
            return
        
        # Prepare context for strategy
        context = {
            'model': self.model,
            'processor': self.processor,
            'current_frame': current_frame_inputs,
            'num_frames': 1,
        }
        
        # For summarize_and_drop, we need a chat formatter
        # Since we're using standard HF processor, we'll create a simple one
        if hasattr(self.context_strategy, '_generate_summary'):
            # Create a simple chat formatter for summarization
            class SimpleChatFormatter:
                @staticmethod
                def apply_chat_template(messages):
                    # Simple template for system messages
                    if messages and messages[0].get("role") == "system":
                        return messages[0]["content"]
                    return ""
            
            context['chat_formatter'] = SimpleChatFormatter()
        
        # Apply strategy
        self.past_key_values, self.last_msg_tokens = \
            self.context_strategy.handle_overflow(
                self.past_key_values,
                self.last_msg_tokens,
                **context
            )
        
        logger.info(
            f"After {self.context_strategy.name}: "
            f"KV cache = {self._get_cache_length()} tokens"
        )

    def _get_reference_dialogue(
        self, conversation: List[Dict], timestamp: float
    ) -> str:
        """Get reference dialogue at given timestamp"""
        for turn in conversation:
            if turn.get("from") == "assistant":
                turn_time = turn.get("timestamp", 0)
                # Exact timestamp match (within 0.5 seconds)
                if abs(turn_time - timestamp) < 0.5:
                    return turn.get("value", "")
        return ""

    def _save_predictions(self, outputs: List[FrameOutput], path: str, video_id: str):
        """Save predictions to JSON"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            "video_id": video_id,
            "model": self.model_name,
            "predictions": [o.to_dict(ignore_keys="image") for o in outputs],
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved predictions to {path}")
