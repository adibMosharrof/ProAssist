"""
Sequence Length Calculator Module

This module calculates token counts for efficient batching and memory management.
"""

import logging
from typing import Dict, Any, List, Optional
from omegaconf import DictConfig

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None


class SequenceLengthCalculator:
    """Calculate sequence lengths for training data optimization"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.max_seq_len = self.training_config.get("max_seq_len", 4096)
        self.num_tokens_per_img = self.training_config.get("num_tokens_per_img", 1)
        self.tokenizer_name = self.training_config.get(
            "tokenizer_name", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )
        self.special_tokens_count = self.training_config.get("special_tokens_count", 10)

        # Initialize tokenizer for token counting
        self.tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer for token counting"""
        if AutoTokenizer is None:
            self.logger.warning(
                "transformers library not available, using approximate token counting"
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            self.logger.info(f"Initialized tokenizer: {self.tokenizer_name}")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer {self.tokenizer_name}: {e}")
            self.logger.info("Using approximate token counting")

    def calculate_text_tokens(self, conversation: List[Dict[str, Any]]) -> int:
        """
        Calculate token count for conversation text content

        Args:
            conversation: List of conversation turns

        Returns:
            Total text token count
        """
        total_tokens = 0

        for turn in conversation:
            # Skip frame turns (these are handled separately)
            if turn.get("role") == "frames":
                continue

            # Count tokens in content
            content = turn.get("content", "")
            if content:
                tokens = self._count_tokens_accurate(content)
                total_tokens += tokens

            # Count tokens in labels if present
            labels = turn.get("labels", "")
            if labels:
                tokens = self._count_tokens_accurate(labels)
                total_tokens += tokens

        # Add role markers and formatting tokens
        role_tokens = len(conversation) * 3  # Rough estimate for role markers
        total_tokens += role_tokens

        return total_tokens

    def calculate_image_tokens(self, conversation: List[Dict[str, Any]]) -> int:
        """
        Calculate token count for all image frames in conversation

        Args:
            conversation: List of conversation turns

        Returns:
            Total image token count
        """
        total_frames = 0

        # Count frames in frame turns (ProAssist format)
        for turn in conversation:
            if turn.get("role") == "frames":
                start_frame = turn.get("start", 0)
                end_frame = turn.get("end", start_frame)
                frame_count = max(0, end_frame - start_frame)
                total_frames += frame_count

        # Count frames embedded in conversation events (DST format)
        for turn in conversation:
            if turn.get("role") in ["SPEAK", "DST_UPDATE"]:
                start_frame = turn.get("start_frame", 0)
                end_frame = turn.get("end_frame", start_frame + 1)
                frame_count = max(0, end_frame - start_frame)
                total_frames += frame_count

        return total_frames * self.num_tokens_per_img

    def calculate_total_sequence_length(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, int]:
        """
        Calculate comprehensive sequence length information

        Args:
            conversation: List of conversation turns

        Returns:
            Dictionary with detailed token counts
        """
        text_tokens = self.calculate_text_tokens(conversation)
        image_tokens = self.calculate_image_tokens(conversation)

        # Add special tokens and formatting overhead
        special_tokens = self.special_tokens_count
        system_overhead = 50  # Estimate for system messages and formatting

        total_tokens = text_tokens + image_tokens + special_tokens + system_overhead

        result = {
            "text_tokens": text_tokens,
            "image_tokens": image_tokens,
            "special_tokens": special_tokens,
            "system_overhead": system_overhead,
            "total_tokens": total_tokens,
            "within_limit": total_tokens <= self.max_seq_len,
        }

        self.logger.debug(
            f"Sequence analysis: text={text_tokens}, image={image_tokens}, "
            f"total={total_tokens}, limit={self.max_seq_len}"
        )

        return result

    def validate_sequence_length(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate sequence length against limits and provide recommendations

        Args:
            conversation: List of conversation turns

        Returns:
            Validation results with recommendations
        """
        token_analysis = self.calculate_total_sequence_length(conversation)

        result = {
            "valid": token_analysis["within_limit"],
            "current_length": token_analysis["total_tokens"],
            "max_length": self.max_seq_len,
            "analysis": token_analysis,
            "recommendations": [],
        }

        if not token_analysis["within_limit"]:
            # Provide recommendations for fixing length issues
            excess_tokens = token_analysis["total_tokens"] - self.max_seq_len

            if token_analysis["image_tokens"] > token_analysis["text_tokens"]:
                result["recommendations"].append(
                    f"Consider reducing frame count (currently {token_analysis['image_tokens']} tokens)"
                )
            else:
                result["recommendations"].append(
                    f"Consider truncating conversation text (currently {token_analysis['text_tokens']} tokens)"
                )

            result["recommendations"].append(
                f"Need to reduce {excess_tokens} tokens to meet limit"
            )

        return result

    def add_sequence_metadata(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add sequence length metadata to video data

        Args:
            video_data: Video data with conversation

        Returns:
            Updated video data with sequence metadata
        """
        conversation = video_data.get("conversation", [])
        if not conversation:
            self.logger.warning("No conversation found in video data")
            return video_data

        # Calculate sequence information
        token_analysis = self.calculate_total_sequence_length(conversation)
        validation = self.validate_sequence_length(conversation)

        # Determine frame range for this conversation segment
        start_frame_idx, end_frame_idx = self._calculate_frame_range(conversation)

        # Update video data with sequence metadata
        video_data.update(
            {
                "seq_len": token_analysis["total_tokens"],
                "start_frame_idx": start_frame_idx,
                "end_frame_idx": end_frame_idx,
                "sequence_analysis": token_analysis,
                "sequence_validation": validation,
            }
        )

        self.logger.debug(
            f"Added sequence metadata: length={token_analysis['total_tokens']}, "
            f"frames={start_frame_idx}-{end_frame_idx}"
        )

        return video_data

    def _count_tokens_accurate(self, text: str) -> int:
        """Count tokens using actual tokenizer if available"""
        if self.tokenizer and text:
            try:
                tokens = self.tokenizer.encode(text, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                self.logger.warning(f"Tokenization failed: {e}, using approximation")

        # Fallback to approximate counting
        return self._count_tokens_approximate(text)

    def _count_tokens_approximate(self, text: str) -> int:
        """Approximate token count (roughly 4 characters per token)"""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _calculate_frame_range(self, conversation: List[Dict[str, Any]]) -> tuple:
        """Calculate overall frame range for conversation"""
        start_frames = []
        end_frames = []

        for turn in conversation:
            if turn.get("role") == "frames":
                start_frames.append(turn.get("start", 0))
                end_frames.append(turn.get("end", 0))
            elif turn.get("role") in ["SPEAK", "DST_UPDATE"]:
                start_frames.append(turn.get("start_frame", 0))
                end_frames.append(turn.get("end_frame", 0))

        if not start_frames:
            return 0, 0

        return min(start_frames), max(end_frames)

    def suggest_optimization(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Suggest optimizations to reduce sequence length if needed

        Args:
            conversation: List of conversation turns

        Returns:
            Optimization suggestions
        """
        validation = self.validate_sequence_length(conversation)

        if validation["valid"]:
            return {
                "needs_optimization": False,
                "message": "Sequence length is within limits",
            }

        suggestions = []

        # Image token optimization
        if (
            validation["analysis"]["image_tokens"]
            > validation["analysis"]["text_tokens"]
        ):
            frames_per_turn = (
                validation["analysis"]["image_tokens"] / self.num_tokens_per_img
            )
            suggestions.append(
                {
                    "type": "reduce_frames",
                    "description": f"Reduce frames per conversation turn",
                    "current_frames_per_turn": frames_per_turn
                    / len([t for t in conversation if t.get("role") == "frames"]),
                    "suggested_action": "Consider using fewer frames per turn or shorter frame ranges",
                }
            )

        # Text truncation suggestion
        if validation["analysis"]["text_tokens"] > self.max_seq_len * 0.7:
            suggestions.append(
                {
                    "type": "truncate_text",
                    "description": "Consider truncating conversation text",
                    "current_text_tokens": validation["analysis"]["text_tokens"],
                    "suggested_action": "Remove less relevant turns or shorten content",
                }
            )

        return {
            "needs_optimization": True,
            "current_length": validation["current_length"],
            "max_length": validation["max_length"],
            "suggestions": suggestions,
        }
