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

        # Reset state for new video
        self.prev_substep = None
        self.current_substep = None

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

            # Remove the prompt from the output
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
