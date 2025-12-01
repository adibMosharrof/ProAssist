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

        # Add frame metadata to each conversation turn
        updated_conversation = []
        for turn in conversation:
            updated_turn = self._add_frame_metadata_to_turn(turn)
            updated_conversation.append(updated_turn)

        # Update conversation in video data
        video_data["conversation"] = updated_conversation

        # Calculate overall frame range
        start_frame_idx, end_frame_idx = self._calculate_overall_frame_range(
            updated_conversation
        )
        video_data["start_frame_idx"] = start_frame_idx
        video_data["end_frame_idx"] = end_frame_idx

        # self.logger.info(
        #     f"Added frame metadata to {len(updated_conversation)} turns, "
        #     f"frame range: {start_frame_idx}-{end_frame_idx}"
        # )

        return video_data

    def _add_frame_metadata_to_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add frame metadata to a single conversation turn

        Args:
            turn: Single conversation turn

        Returns:
            Updated turn with frame metadata
        """
        updated_turn = turn.copy()

        # Skip if already has frame metadata
        if "start_frame" in updated_turn and "end_frame" in updated_turn:
            return updated_turn

        # Get timestamp from turn
        timestamp = turn.get("time", 0)

        # Calculate frame range for this turn
        start_frame, end_frame = self._calculate_frame_range_for_timestamp(timestamp)

        # Add frame metadata to turn
        updated_turn["start_frame"] = start_frame
        updated_turn["end_frame"] = end_frame

        # Validate frame alignment
        if not self._validate_frame_alignment(updated_turn):
            self.logger.warning(f"Frame alignment issue for turn at time {timestamp}")

        return updated_turn

    def _validate_frame_alignment(self, turn: Dict[str, Any]) -> bool:
        """
        Validate that frame alignment is reasonable for a single turn

        Args:
            turn: Single conversation turn

        Returns:
            True if frame alignment is reasonable
        """
        # Basic validation - ensure start_frame <= end_frame
        start_frame = turn.get("start_frame", 0)
        end_frame = turn.get("end_frame", 0)

        if start_frame > end_frame:
            return False

        # Ensure reasonable frame count (not too many frames for single turn)
        frame_count = end_frame - start_frame
        if frame_count > self.fps * 5:  # More than 5 seconds of frames for single turn
            self.logger.warning(
                f"Too many frames for single turn: {frame_count}, turn time: {turn['time']}"
            )
            return False

        return True

    def _calculate_frame_range_for_timestamp(self, timestamp: float) -> Tuple[int, int]:
        """
        Calculate frame range for a specific timestamp

        Args:
            timestamp: Time in seconds

        Returns:
            Tuple of (start_frame, end_frame)
        """
        # Calculate center frame
        center_frame = int(timestamp * self.fps)

        # Calculate frame range around the timestamp
        half_duration = self.frame_duration / 2
        start_time = max(0, timestamp - half_duration)
        end_time = timestamp + half_duration

        start_frame = int(start_time * self.fps)
        end_frame = int(end_time * self.fps)

        # Ensure at least one frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        return start_frame, end_frame

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



