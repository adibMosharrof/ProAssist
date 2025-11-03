"""Sanity check runner - returns ground truth dialogues as predictions"""

import logging
from typing import Dict, List, Any, Optional

from mmassist.eval.evaluators.stream_evaluator import FrameOutput


logger = logging.getLogger(__name__)


class SanityCheckRunner:
    """
    Sanity check runner that returns ground truth dialogues as predictions.

    This is a perfect oracle that should achieve near-perfect metrics.
    Used to validate the evaluation pipeline works correctly.

    Key Features:
    - No model loading (instant startup)
    - Returns ground truth dialogues as both gen and ref
    - Timestamp matching with tolerance for FPS conversion
    - Compatible with ProAssist's StreamEvaluator interface
    """

    def __init__(self, fps: float = 2.0, **kwargs):
        """
        Initialize SanityCheckRunner

        Args:
            fps: Frames per second (for frame index calculation)
            **kwargs: Ignored (for compatibility with other runners)
        """
        self.fps = fps
        self.eval_name = "sanity_check"
        logger.info("✅ Initialized SanityCheckRunner (Perfect Oracle)")
        logger.info(f"   FPS: {fps}")

    def run_inference_on_video(
        self, video: Dict[str, Any], output_dir: str = "", **kwargs
    ) -> Dict[str, Any]:
        """
        Return ground truth dialogues as predictions.

        This is a pass-through oracle that uses ground truth as predictions.
        Both gen and ref will be identical, so metrics should be perfect.

        Args:
            video: Dict with keys:
                - video_id: str
                - frames: List[PIL.Image]
                - conversation: List[Dict] with keys: time, content, labels
                - dst_annotations: pd.DataFrame (optional)
                - fps: float

        Returns:
            Dict with:
                - predictions: List[FrameOutput]
                - video_id: str
        """
        video_id = video["video_id"]
        frames = video["frames"]
        ground_truth_conv = video.get("conversation", [])

        logger.info(f"Running sanity check on video: {video_id}")
        logger.info(f"  Total frames: {len(frames)}")
        logger.info(f"  Ground truth dialogues: {len(ground_truth_conv)}")

        # Create mapping of timestamp to dialogue content
        dialogue_map = {}
        for d in ground_truth_conv:
            time = d.get("time", 0)
            content = d.get("content", "")
            if content:  # Only add non-empty dialogues
                if time not in dialogue_map:
                    dialogue_map[time] = []
                dialogue_map[time].append(content)

        # Create FrameOutput for each frame
        outputs = []
        num_dialogues = 0

        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / self.fps

            # Check if there's a dialogue at this timestamp
            # Allow small tolerance (0.5s) for floating point comparison
            dialogue = ""
            ref_dialogue = ""

            for gt_time, dialogues_list in dialogue_map.items():
                if abs(timestamp - gt_time) < 0.5:  # 0.5s tolerance
                    # Use first dialogue at this timestamp
                    dialogue = dialogues_list[0]
                    ref_dialogue = dialogues_list[0]
                    num_dialogues += 1
                    break

            # Create FrameOutput
            outputs.append(
                FrameOutput(
                    gen=dialogue,  # Prediction = ground truth
                    ref=ref_dialogue,  # Reference = ground truth
                    image=frame,
                    frame_idx_in_stream=frame_idx,
                    timestamp_in_stream=timestamp,
                )
            )

        logger.info(f"✅ Returned {num_dialogues} dialogues as predictions (gen==ref)")

        return {
            "predictions": outputs,
            "video_id": video_id,
        }
