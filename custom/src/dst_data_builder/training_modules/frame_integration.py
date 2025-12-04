"""
Frame Integration Module

This module embeds frame information directly into conversation events
rather than using separate frame turns.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig
from pathlib import Path


class FrameIntegration:
    """Embed frame information into conversation events"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.fps = self.training_config.get("fps", 2)

        # Data source configuration
        self.data_source_config = cfg.get("data_source", {})
        self.data_path = self.data_source_config.get("data_path", "data")
        self.frames_subdir = self.data_source_config.get("frames_subdir", "frames")
        self.frame_duration = self.training_config.get("dst_frame_duration", 1)

    def add_frame_metadata(
        self, video_data: Dict[str, Any], dataset_name: str
    ) -> Dict[str, Any]:
        """
        Add frame metadata to conversation events in video data

        Args:
            video_data: Enhanced DST data for a single video
            dataset_name: Name of the dataset

        Returns:
            Updated video data with frame metadata embedded in events
        """
        self.logger.debug(f"Adding frame metadata for dataset: {dataset_name}")

        conversation = video_data.get("conversation", [])
        if not conversation:
            self.logger.warning("No conversation found in video data")
            return video_data

        # Note: frames_file path is NOT added here since DST training doesn't use frames
        # The training dataset will handle frame loading directly from video_uid when needed

        # Apply input style to calculate frame indices for the whole conversation
        # This replaces the per-turn iteration
        updated_conversation = self._apply_input_style(conversation)
        
        # Update conversation in video data
        video_data["conversation"] = updated_conversation

        # Calculate overall frame range
        start_frame_idx, end_frame_idx = self._calculate_overall_frame_range(
            updated_conversation
        )
        video_data["start_frame_idx"] = start_frame_idx
        video_data["end_frame_idx"] = end_frame_idx

        # Validate and clip all turn frames to be within the clip's frame bounds
        updated_conversation = self._validate_and_clip_turn_frames(
            updated_conversation, start_frame_idx, end_frame_idx
        )
        video_data["conversation"] = updated_conversation

        # self.logger.info(
        #     f"Added frame metadata to {len(updated_conversation)} turns, "
        #     f"frame range: {start_frame_idx}-{end_frame_idx}"
        # )

        return video_data

    def _apply_input_style(self, conversation: List[Dict[str, Any]], total_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Apply the configured input style to calculate frame indices for the conversation.
        """
        from custom.src.common.input_styles import get_input_style
        
        # Get input style from config
        # With Hydra defaults, this should be a DictConfig containing the style params
        input_style_cfg = self.cfg.get("input_style")
        
        style_name = "proassist"
        style_config = {
            "fps": self.fps,
            "window_size": self.training_config.get("window_size", 4)
        }

        if input_style_cfg is not None:
            if isinstance(input_style_cfg, (dict, DictConfig)):
                # It's a config object (expected with Hydra defaults)
                style_name = input_style_cfg.get("name", "proassist")
                style_config.update(input_style_cfg)
            else:
                # It's just a string name (legacy/override)
                style_name = str(input_style_cfg)
        
        try:
            style = get_input_style(style_name, style_config)
            return style.calculate_frame_indices(conversation, total_frames)
        except Exception as e:
            self.logger.error(f"Failed to apply input style {style_name}: {e}")
            return conversation

    def _calculate_overall_frame_range(
        self, conversation: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Calculate the overall frame range for entire conversation

        Args:
            conversation: List of conversation turns

        Returns:
            Tuple of (min_start_frame, max_end_frame)
        """
        if not conversation:
            return 0, 0

        start_frames = []
        end_frames = []

        for turn in conversation:
            if "start_frame" in turn and "end_frame" in turn:
                start_frames.append(turn["start_frame"])
                end_frames.append(turn["end_frame"])

        if not start_frames:
            return 0, 0

        return min(start_frames), max(end_frames)

    def _validate_and_clip_turn_frames(
        self,
        conversation: List[Dict[str, Any]],
        clip_start_frame: int,
        clip_end_frame: int,
    ) -> List[Dict[str, Any]]:
        """
        Validate and clip turn frame indices to be within clip bounds.
        
        All turn frames must be within [clip_start_frame, clip_end_frame].
        If frames exceed bounds, they are clipped to fit within the clip.

        Args:
            conversation: List of conversation turns
            clip_start_frame: Start frame index of the clip
            clip_end_frame: End frame index of the clip

        Returns:
            Updated conversation with clipped frame indices
        """
        updated_conversation = []
        
        for turn in conversation:
            updated_turn = turn.copy()
            
            # Only validate turns that have frame information
            if "start_frame" in updated_turn and "end_frame" in updated_turn:
                start_frame = updated_turn["start_frame"]
                end_frame = updated_turn["end_frame"]
                
                # Clip frames to be within clip bounds
                clipped_start = max(start_frame, clip_start_frame)
                clipped_end = min(end_frame, clip_end_frame)
                
                # Log if clipping occurred
                if clipped_start != start_frame or clipped_end != end_frame:
                    self.logger.warning(
                        f"Clipping turn frames from [{start_frame}, {end_frame}] "
                        f"to [{clipped_start}, {clipped_end}] "
                        f"to fit within clip bounds [{clip_start_frame}, {clip_end_frame}]"
                    )
                
                # Ensure valid range after clipping
                if clipped_start > clipped_end:
                    # If clipping resulted in invalid range (start > end), skip this turn's frame info
                    self.logger.warning(
                        f"Turn frames [{start_frame}, {end_frame}] fall completely outside "
                        f"clip bounds [{clip_start_frame}, {clip_end_frame}], skipping frame info"
                    )
                    # Remove frame info from this turn if it's completely outside bounds
                    updated_turn.pop("start_frame", None)
                    updated_turn.pop("end_frame", None)
                else:
                    # Update with clipped values
                    updated_turn["start_frame"] = clipped_start
                    updated_turn["end_frame"] = clipped_end
            
            updated_conversation.append(updated_turn)
        
        return updated_conversation



