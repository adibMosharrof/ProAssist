"""VLM-based streaming inference runner for PROSPECT"""

import os
import json
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoProcessor
from tqdm.auto import tqdm

from mmassist.eval.runners.stream_inference import FrameOutput
from prospect.context_strategies.context_strategy_factory import ContextStrategyFactory
from prospect.context_strategies import BaseContextStrategy
from prospect.models import (
    SmolVLMWithStrategies,  # Approach 2: Strategy injection
    CustomSmolVLMProcessor,
)
from prospect.utils.cache_manager import KVCacheManager
from prospect.utils.chat_formatter import ChatFormatter
from prospect.timeline_trace.timeline_trace import BaseTrace


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
        trace: Optional[BaseTrace] = None,
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

        # Trace for timeline visualization
        self.trace = trace

        # Context strategy for KV cache management
        self.context_strategy: Optional[BaseContextStrategy] = None
        if context_strategy_type and context_strategy_type != "none":
            strategy_config = context_strategy_config or {}
            self.context_strategy = ContextStrategyFactory.create_strategy(
                strategy_type=context_strategy_type,
                max_seq_len=kwargs.get("max_seq_len", 4096),
                reserved_seq_len=kwargs.get("reserved_seq_len", 128),
                **strategy_config,
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
            # Load base processor first
            base_processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )
            # Wrap with CustomSmolVLMProcessor for streaming support
            self.processor = CustomSmolVLMProcessor(base_processor)

            # Initialize chat formatter
            self.chat_formatter = ChatFormatter(self.processor.tokenizer)
            logger.info("✅ Chat formatter initialized")

            # Use SmolVLMWithStrategies (Approach 1 - ProAssist pattern)
            self.model = SmolVLMWithStrategies.from_pretrained(
                model_name,
                torch_dtype=dtype,
                device_map=device,
                cache_dir=cache_dir,
                trust_remote_code=True,
            )

            self.model.eval()
            logger.info(
                "✅ SmolVLMWithStrategies model loaded successfully (Approach 1 - ProAssist pattern)"
            )
            logger.info(f"   Max sequence length: {kwargs.get('max_seq_len', 4096)}")
            logger.info(
                f"   Context strategy: {self.context_strategy.name if self.context_strategy else 'none'}"
            )
        except Exception as e:
            logger.error(f"Failed to load VLM model: {e}")
            raise

        # State tracking for transitions
        self.prev_substep = None
        self.current_substep = None

        # KV cache manager
        self.cache_manager = KVCacheManager(context_strategy=self.context_strategy)
        self.use_kv_cache = kwargs.get("use_kv_cache", False)

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
        self.cache_manager.reset()

        outputs = []

        total_frames = len(frames)
        with tqdm(
            frames,
            desc=f"Processing {video_id}",
            total=total_frames,
            unit="frame",
            leave=False,
            disable=not logger.isEnabledFor(logging.INFO),
        ) as progress_bar:
            for frame_idx, frame in enumerate(progress_bar):
                timestamp = frame_idx / self.fps
                if frame_idx % 10 == 0:
                    progress_bar.set_postfix_str(
                        f"{frame_idx}/{total_frames} ({timestamp:.1f}s)"
                    )

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
                            frame, self.prev_substep, curr_substep, timestamp, frame_idx
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
                ref_dialogue = self._get_reference_dialogue(
                    ground_truth_conv, timestamp
                )

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
                    cache_len = self.cache_manager.get_cache_length()
                    logger.debug(
                        f"Frame {frame_idx}: KV cache size = {cache_len} tokens"
                    )

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

    def _get_cache_length(self, past_key_values) -> int:
        """Get cache length from past_key_values tuple"""
        if past_key_values is None:
            return 0
        return past_key_values[0][0].shape[2]

    def _save_frame(self, frame: Image.Image, frame_idx: int) -> Optional[str]:
        """
        Save frame image to frames directory if trace has frames_dir configured.

        Returns:
            Relative path to saved frame, or None if not saved
        """
        if self.trace and self.trace.frames_dir:
            try:
                from pathlib import Path

                frames_dir = Path(self.trace.frames_dir)
                frames_dir.mkdir(parents=True, exist_ok=True)

                frame_filename = f"frame_{frame_idx:06d}.jpg"
                frame_path = frames_dir / frame_filename

                # Save frame as JPEG
                frame.save(frame_path, "JPEG", quality=85)

                # Return relative path (just the filename for HTML)
                return frame_filename
            except Exception as e:
                logger.warning(f"Failed to save frame {frame_idx}: {e}")
                return None
        return None

    def _generate_dialogue_with_cache(
        self,
        frame: Image.Image,
        prev_substep: str,
        curr_substep: str,
        timestamp: float = 0.0,
        frame_idx: int = 0,
    ) -> str:
        """
        Generate dialogue with KV cache accumulation using fast_greedy_generate().

        Uses fast_greedy_generate() (ProAssist pattern) to bypass DynamicCache
        and cache_position issues in Transformers 4.36+.

        **Approach 1 (ProAssist Pattern)**: Cache compression happens HERE in the
        runner, BEFORE calling generate(), not inside the model.
        """
        import time

        gen_start = time.time()

        prompt = self.dialogue_generation_prompt.format(
            prev_substep=prev_substep, curr_substep=curr_substep
        )

        try:
            # Prepare inputs (frame + prompt)
            inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Get cache from manager
                past_key_values = (
                    self.cache_manager.get_cache()
                    if self.cache_manager.has_cache()
                    else None
                )

                # Apply compression BEFORE generate() if cache exceeds threshold (ProAssist pattern)
                if past_key_values is not None and self.context_strategy is not None:
                    cache_len = self._get_cache_length(past_key_values)

                    if self.context_strategy.should_reduce_cache(cache_len):
                        logger.info(
                            f"Compressing KV cache before generate: {cache_len} tokens "
                            f"(strategy: {self.context_strategy.name})"
                        )

                        # Compress cache using strategy (pass context for summarize strategies)
                        import time

                        comp_start = time.time()

                        past_key_values, _ = self.context_strategy.compress_cache(
                            past_key_values=past_key_values,
                            attention_mask=None,
                            model=self.model,
                            processor=self.processor,
                            current_frame=frame,
                            chat_formatter=self.chat_formatter,
                            current_timestamp=timestamp,
                            frame_idx=frame_idx,
                            trace=self.trace,  # Pass trace for strategy to record details
                        )

                        comp_time = time.time() - comp_start

                        new_len = 0
                        if past_key_values is not None:
                            new_len = self._get_cache_length(past_key_values)
                            logger.info(
                                f"Cache compressed: {cache_len} → {new_len} tokens"
                            )
                        else:
                            logger.info(f"Cache cleared: {cache_len} → 0 tokens")

                        # Record compression event in trace
                        if self.trace:
                            self.trace.add_compression_event(
                                timestamp=timestamp,
                                frame_idx=frame_idx,
                                tokens_before=cache_len,
                                tokens_after=new_len,
                                strategy_name=self.context_strategy.name,
                                compression_time=comp_time,
                            )

                # Get joint embeddings (text + vision) - ProAssist pattern
                inputs_embeds = self.model.joint_embed(
                    input_ids=inputs["input_ids"],
                    pixel_values=inputs.get("pixel_values"),
                )

                # Generate using fast_greedy_generate (ProAssist pattern)
                output_ids, new_cache = self.model.fast_greedy_generate(
                    inputs_embeds=inputs_embeds,
                    past_key_values=past_key_values,
                    max_length=self.max_new_tokens,
                    verbose=False,
                )

                # Update cache manager with new cache
                self.cache_manager.update_cache(new_cache)

            # Decode dialogue
            dialogue = self.processor.tokenizer.decode(
                output_ids[0], skip_special_tokens=True
            )

            # Clean up prompt leakage
            if "Assistant:" in dialogue:
                dialogue = dialogue.split("Assistant:")[-1].strip()
            if prompt in dialogue:
                dialogue = dialogue.replace(prompt, "").strip()

            # Record generation event in trace
            gen_time = time.time() - gen_start
            if self.trace and dialogue.strip():
                # Save frame and get path
                frame_path = self._save_frame(frame, frame_idx)

                cache_len = (
                    self._get_cache_length(past_key_values)
                    if past_key_values is not None
                    else 0
                )
                self.trace.add_generation_event(
                    timestamp=timestamp,
                    frame_idx=frame_idx,
                    generated_text=dialogue.strip(),
                    generation_time=gen_time,
                    cache_tokens=cache_len,
                    frame_path=frame_path,
                )

            # Note: Cache compression now follows ProAssist pattern - happens in runner
            # BEFORE calling fast_greedy_generate(), not inside the model.
            # Uses fast_greedy_generate() to avoid DynamicCache issues.

            return dialogue.strip()

        except Exception:
            logger.exception("VLM dialogue generation with cache failed")
            raise

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
