"""
Dataset Metadata Generator Module

This module adds ProAssist training dataset metadata to the generated data.
"""

import logging
from typing import Dict, Any, Optional
from omegaconf import DictConfig


class DatasetMetadataGenerator:
    """Generate training dataset metadata following ProAssist format"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.include_quality_metrics = self.training_config.get(
            "include_quality_metrics", True
        )

    def add_training_metadata(
        self,
        video_data: Dict[str, Any],
        dataset_name: str,
        split: str,
        clip_idx: int = 0,
    ) -> Dict[str, Any]:
        """
        Generate dataset metadata for a single video/conversation

        Args:
            video_data: Enhanced DST data for a single video
            dataset_name: Name of the dataset (e.g., 'assembly101')
            split: Data split ('train', 'val', 'test')
            clip_idx: Index of conversation segment (0 for single conversations)

        Returns:
            Updated video_data with training metadata
        """
        self.logger.debug(
            f"Generating metadata for {dataset_name}/{split}, clip_idx: {clip_idx}"
        )

        # Extract basic information
        video_uid = video_data.get("video_uid", "unknown_video")

        # Generate user type and ID (following ProAssist format)
        user_type = self._extract_user_type(video_data)
        user_id = f"{user_type}_{video_data.get('user_id', 0)}"

        # Create metadata following ProAssist structure
        metadata = {
            "dataset": dataset_name,
            "video_uid": video_uid,
            "clip_idx": clip_idx,
            "frames_file": video_data.get("frames_file", ""),
            "max_seq_len": self.training_config.get("max_seq_len", 4096),
            "seq_len": video_data.get("seq_len", 0),
            "num_tokens_per_img": self.training_config.get("num_tokens_per_img", 1),
            "start_frame_idx": video_data.get("start_frame_idx", 0),
            "end_frame_idx": video_data.get("end_frame_idx", 0),
            "conversation": video_data.get("conversation", []),
            "fps": self.training_config.get("fps", 2),
            "metadata": {
                "user_type": user_type,
                "user_id": user_id,
                "task_goal": self._extract_task_goal(video_data),
                "knowledge": video_data.get("knowledge", []),
                "progress": None,  # Not used in training data per user preference
                "add_knowledge": self._should_add_knowledge(),
                "has_summary": False,  # No progress summaries in training
                "summary_only": False,
                "quality": (
                    self._calculate_quality_score(video_data)
                    if self.include_quality_metrics
                    else None
                ),
            },
        }

        # Add quality metrics if enabled
        if self.include_quality_metrics:
            quality_score = self._calculate_quality_score(video_data)
            metadata["quality"] = quality_score
            self.logger.debug(f"Added quality score: {quality_score}")

        # Update video data with metadata
        video_data.update(metadata)

        self.logger.info(
            f"Generated metadata for {video_uid}: user_type={user_type}, clip_idx={clip_idx}"
        )
        return video_data

    def _extract_user_type(self, video_data: Dict[str, Any]) -> str:
        """Extract user interaction type from video data"""
        # Try to get from existing metadata first
        if "metadata" in video_data and "user_type" in video_data["metadata"]:
            return video_data["metadata"]["user_type"]

        # Fallback to default based on content analysis
        conversation = video_data.get("conversation", [])
        if not conversation:
            return "no_talk"  # Default assumption

        # Analyze conversation for user engagement level
        speak_events = [turn for turn in conversation if turn.get("role") == "SPEAK"]
        user_events = [turn for turn in conversation if turn.get("role") == "USER"]

        total_turns = len(conversation)
        user_turns = len(user_events)
        assistant_turns = len(speak_events)

        # Simple heuristic for user type classification
        if user_turns == 0:
            return "no_talk"
        elif user_turns <= total_turns * 0.2:
            return "talk_some"
        else:
            return "talk_more"

    def _extract_task_goal(self, video_data: Dict[str, Any]) -> str:
        """Extract or infer task goal from video data"""
        # Try to get from existing data
        if "metadata" in video_data and "task_goal" in video_data["metadata"]:
            return video_data["metadata"]["task_goal"]

        # Try to get from inferred_goal
        if "inferred_goal" in video_data:
            return video_data["inferred_goal"]

        # Fallback to video UID analysis
        video_uid = video_data.get("video_uid", "")
        if "assembly" in video_uid.lower():
            return "Assembly task"
        elif "cooking" in video_uid.lower():
            return "Cooking task"
        else:
            return "Task completion"

    def _should_add_knowledge(self) -> bool:
        """Determine if knowledge should be added to prompts"""
        return self.training_config.get("add_knowledge", True)

    def _calculate_quality_score(self, video_data: Dict[str, Any]) -> float:
        """Calculate quality score for the conversation (ProAssist format)"""
        # Simple quality assessment based on conversation completeness
        conversation = video_data.get("conversation", [])

        if not conversation:
            return 1.0  # Very low quality for empty conversations

        # Check for required elements
        has_user_turns = any(turn.get("role") == "USER" for turn in conversation)
        has_assistant_turns = any(turn.get("role") == "SPEAK" for turn in conversation)
        has_dst_updates = any(turn.get("role") == "DST_UPDATE" for turn in conversation)

        # Base score
        score = 2.0

        # Add points for quality indicators
        if has_user_turns:
            score += 1.0
        if has_assistant_turns:
            score += 1.0
        if has_dst_updates:
            score += 0.5

        # Length bonus (but cap it)
        length_score = min(len(conversation) / 10.0, 1.0)
        score += length_score

        return min(score, 5.0)  # Cap at 5.0 (ProAssist uses 1-5 scale)

    def validate_data_integrity(self, video_data: Dict[str, Any]) -> bool:
        """Validate that all required metadata fields are present"""
        required_fields = [
            "dataset",
            "video_uid",
            "clip_idx",
            "metadata",
            "conversation",
            "fps",
        ]

        for field in required_fields:
            if field not in video_data:
                self.logger.error(f"Missing required field: {field}")
                return False

        # Validate metadata sub-structure
        metadata = video_data.get("metadata", {})
        required_metadata_fields = [
            "user_type",
            "user_id",
            "task_goal",
            "knowledge",
            "add_knowledge",
            "has_summary",
            "summary_only",
        ]

        for field in required_metadata_fields:
            if field not in metadata:
                self.logger.error(f"Missing required metadata field: {field}")
                return False

        self.logger.debug("Data integrity validation passed")
        return True
