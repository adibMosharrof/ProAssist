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

        # Add frames file path
        video_data["frames_file"] = self._get_frames_file_path(video_data, dataset_name)

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

    def calculate_event_frame_ranges(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Tuple[int, int, float]]:
        """
        Calculate frame ranges for all conversation events

        Args:
            conversation: List of conversation turns

        Returns:
            List of (start_frame, end_frame, timestamp) tuples
        """
        frame_ranges = []

        for turn in conversation:
            timestamp = turn.get("time", 0)
            start_frame, end_frame = self._calculate_frame_range_for_timestamp(
                timestamp
            )
            frame_ranges.append((start_frame, end_frame, timestamp))

        return frame_ranges

    def validate_frame_alignment(self, conversation: List[Dict[str, Any]]) -> bool:
        """
        Ensure frame ranges align with conversation temporal flow

        Args:
            conversation: List of conversation turns

        Returns:
            True if frames are properly aligned
        """
        frame_ranges = self.calculate_event_frame_ranges(conversation)

        # Check temporal ordering
        for i in range(1, len(frame_ranges)):
            prev_end = frame_ranges[i - 1][1]
            curr_start = frame_ranges[i][0]

            # Allow some overlap but check for major gaps
            if curr_start > prev_end + self.fps * 2:  # Allow up to 2 seconds gap
                self.logger.warning(
                    f"Large temporal gap between turns: "
                    f"turn {i-1} ends at frame {prev_end}, "
                    f"turn {i} starts at frame {curr_start}"
                )
                return False

        self.logger.debug("Frame alignment validation passed")
        return True

    def _get_frames_file_path(
        self, video_data: Dict[str, Any], dataset_name: str
    ) -> str:
        """
        Generate frames file path for the video

        Args:
            video_data: Video data
            dataset_name: Name of the dataset

        Returns:
            Path to frames file following ProAssist structure: {data_path}/{dataset_name}/{frames_subdir}/{video_uid}.parquet
        """
        video_uid = video_data.get("video_uid", "unknown_video")

        # Construct frames file path following ProAssist structure
        # Format: {data_path}/{dataset_name}/{frames_subdir}/{video_uid}.arrow
        frames_path = Path(
            f"{self.data_path}/{dataset_name}/{self.frames_subdir}/{video_uid}.arrow"
        )

        return str(frames_path)

    def _calculate_overall_frame_range(
        self, conversation: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Calculate overall frame range for the entire conversation

        Args:
            conversation: List of conversation turns

        Returns:
            Tuple of (start_frame, end_frame)
        """
        start_frames = []
        end_frames = []

        for turn in conversation:
            start_frame = turn.get("start_frame", 0)
            end_frame = turn.get("end_frame", 0)
            start_frames.append(start_frame)
            end_frames.append(end_frame)

        if not start_frames:
            return 0, 0

        return min(start_frames), max(end_frames)

    def validate_frame_availability(
        self, video_data: Dict[str, Any], dataset_name: str
    ) -> bool:
        """
        Ensure calculated frames exist within video bounds

        Args:
            video_data: Video data
            dataset_name: Name of the dataset

        Returns:
            True if frames are within bounds
        """
        conversation = video_data.get("conversation", [])
        if not conversation:
            return True

        # Try to get video duration or total frames
        total_frames = self._get_total_frames(video_data)
        if total_frames == 0:
            self.logger.warning("Cannot determine total frames for validation")
            return True

        # Check each turn's frame range
        for turn in conversation:
            start_frame = turn.get("start_frame", 0)
            end_frame = turn.get("end_frame", 0)

            if start_frame < 0 or end_frame > total_frames:
                self.logger.error(
                    f"Frame range {start_frame}-{end_frame} exceeds "
                    f"video bounds (0-{total_frames})"
                )
                return False

        self.logger.debug("Frame availability validation passed")
        return True

    def _get_total_frames(self, video_data: Dict[str, Any]) -> int:
        """
        Get total number of frames in the video

        Args:
            video_data: Video data

        Returns:
            Total frame count, or 0 if unknown
        """
        # Try to get from existing metadata
        if "metadata" in video_data and "total_frames" in video_data["metadata"]:
            return video_data["metadata"]["total_frames"]

        # Try to extract from video_uid or other fields
        video_uid = video_data.get("video_uid", "")

        # Look for frame count patterns in UID
        if "_frames_" in video_uid:
            try:
                parts = video_uid.split("_frames_")
                if len(parts) > 1:
                    return int(parts[1])
            except (ValueError, IndexError):
                pass

        # Default estimation based on conversation duration
        conversation = video_data.get("conversation", [])
        if conversation:
            max_time = max(turn.get("time", 0) for turn in conversation)
            estimated_frames = int(max_time * self.fps)
            return estimated_frames

        return 0

    def get_frame_statistics(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get statistics about frame usage in the conversation

        Args:
            video_data: Video data

        Returns:
            Dictionary with frame statistics
        """
        conversation = video_data.get("conversation", [])
        if not conversation:
            return {}

        frame_ranges = self.calculate_event_frame_ranges(conversation)

        # Calculate statistics
        total_frames_used = sum(
            end_frame - start_frame for start_frame, end_frame, _ in frame_ranges
        )
        unique_frames = len(
            set(
                frame
                for start_frame, end_frame, _ in frame_ranges
                for frame in range(start_frame, end_frame)
            )
        )
        avg_frames_per_turn = (
            total_frames_used / len(conversation) if conversation else 0
        )

        start_frame_idx, end_frame_idx = self._calculate_overall_frame_range(
            conversation
        )
        coverage_percentage = (
            unique_frames / max(1, end_frame_idx - start_frame_idx)
        ) * 100

        return {
            "total_frames_used": total_frames_used,
            "unique_frames": unique_frames,
            "avg_frames_per_turn": avg_frames_per_turn,
            "frame_coverage_percentage": coverage_percentage,
            "start_frame_idx": start_frame_idx,
            "end_frame_idx": end_frame_idx,
            "total_turns": len(conversation),
        }
