from typing import Any, Dict, Tuple
from dst_data_builder.validators.base_validator import BaseValidator


class TrainingFormatValidator(BaseValidator):
    """Validate ProAssist training format compliance for training samples."""

    def validate(self, training_sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate training sample for ProAssist format compliance.

        Args:
            training_sample: Training data sample to validate

        Returns:
            Tuple of (is_valid: bool, message: str)
        """
        if not isinstance(training_sample, dict):
            return (
                False,
                "Training sample must be a JSON object/dictionary, not "
                + type(training_sample).__name__,
            )

        # Check required fields for ProAssist training format
        required_fields = [
            "video_uid",
            "conversation",
            "dataset",
            "clip_idx",
            "max_seq_len",
            "seq_len",
            "fps",
            "num_tokens_per_img",
        ]
        missing_fields = [
            field for field in required_fields if field not in training_sample
        ]

        if missing_fields:
            return (
                False,
                f"Training sample missing required fields: {missing_fields}. Required fields: {required_fields}",
            )

        # Validate conversation structure
        conversation = training_sample.get("conversation", [])
        if not isinstance(conversation, list):
            return (
                False,
                "Conversation must be an array/list, not "
                + type(conversation).__name__,
            )

        if len(conversation) == 0:
            return (
                False,
                "Conversation cannot be empty. At least one turn required for training.",
            )

        # Validate conversation turns
        for i, turn in enumerate(conversation):
            if not isinstance(turn, dict):
                return (
                    False,
                    f"conversation[{i}] must be an object/dictionary, not {type(turn).__name__}",
                )

            if "role" not in turn:
                return False, f"conversation[{i}] missing required 'role' field"

            if "content" not in turn:
                return False, f"conversation[{i}] missing required 'content' field"

            role = turn.get("role", "")
            if role not in ["system", "user", "assistant", "DST_UPDATE"]:
                return (
                    False,
                    f"conversation[{i}] has invalid role '{role}'. Must be one of: system, user, assistant, DST_UPDATE",
                )

        # Validate numeric fields
        seq_len = training_sample.get("seq_len", 0)
        max_seq_len = training_sample.get("max_seq_len", 4096)

        if not isinstance(seq_len, int) or seq_len < 0:
            return False, f"seq_len must be a non-negative integer, got {seq_len}"

        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            return False, f"max_seq_len must be a positive integer, got {max_seq_len}"

        if seq_len > max_seq_len:
            return False, f"seq_len ({seq_len}) exceeds max_seq_len ({max_seq_len})"

        # Validate other required fields
        fps = training_sample.get("fps")
        if not isinstance(fps, (int, float)) or fps <= 0:
            return False, f"fps must be a positive number, got {fps}"

        num_tokens_per_img = training_sample.get("num_tokens_per_img")
        if not isinstance(num_tokens_per_img, int) or num_tokens_per_img < 0:
            return (
                False,
                f"num_tokens_per_img must be a non-negative integer, got {num_tokens_per_img}",
            )

        # Validate dataset field
        dataset = training_sample.get("dataset")
        if not isinstance(dataset, str) or not dataset.strip():
            return False, "dataset must be a non-empty string"

        # Validate clip_idx
        clip_idx = training_sample.get("clip_idx")
        if not isinstance(clip_idx, int) or clip_idx < 0:
            return False, f"clip_idx must be a non-negative integer, got {clip_idx}"

        # Validate video_uid
        video_uid = training_sample.get("video_uid")
        if not isinstance(video_uid, str) or not video_uid.strip():
            return False, "video_uid must be a non-empty string"

        # Validate conversation flow (optional checks for better quality)
        roles = [turn.get("role", "") for turn in conversation]

        # Check if conversation starts with system prompt
        if roles and roles[0] != "system":
            return (
                False,
                "Training conversation should start with a system prompt role for optimal training",
            )

        return True, ""
