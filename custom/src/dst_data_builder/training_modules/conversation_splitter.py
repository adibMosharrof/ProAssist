"""
Conversation Splitter Module

This module splits long conversations into multiple training samples when they exceed
sequence length limits, preserving context and DST state continuity.
"""

import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig

from transformers import AutoTokenizer
from dst_data_builder.training_modules.dst_enums import count_dst_content_tokens
class ConversationSplitter:
    """Split long conversations for training data creation"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.enable_conversation_splitting = self.training_config.get(
            "enable_conversation_splitting", True
        )
        self.keep_context_length = self.training_config.get(
            "keep_context_length", [5, 20]
        )
        self.max_seq_len = self.training_config.get("max_seq_len", 4096)
        self.num_tokens_per_img = self.training_config.get("num_tokens_per_img", 1)
        # Initialize tokenizer for token counting
        self._tokenizer = None
        self._initialize_tokenizer()

    def _initialize_tokenizer(self):
        """Initialize the tokenizer for token counting"""
        tokenizer_name = self.training_config.get(
            "tokenizer_name", "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        )
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            self.logger.info(f"Initialized tokenizer: {tokenizer_name}")
        except ImportError:
            self.logger.warning("transformers library not available, using approximate token counting")
        except Exception as e:
            self.logger.warning(f"Failed to load tokenizer {tokenizer_name}: {e}")
            self.logger.info("Using approximate token counting for testing")

    def split_conversations(self, video_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split long conversations into multiple training samples

        Args:
            video_data: Enhanced DST data with conversation

        Returns:
            List of video data segments, each representing a training sample
        """
        if not self.enable_conversation_splitting:
            self.logger.debug(
                "Conversation splitting disabled, returning original data"
            )
            return [video_data]

        conversation = video_data.get("conversation", [])
        if not conversation:
            self.logger.warning("No conversation found, returning original data")
            return [video_data]

        # Check if splitting is needed
        if not self._should_split_conversation(conversation):
            self.logger.debug("Conversation within limits, no splitting needed")
            return [video_data]

        self.logger.info("Splitting conversation due to length constraints")

        # Split the conversation
        segments = self._split_conversation_by_length(conversation, video_data)

        self.logger.info(f"Split conversation into {len(segments)} segments")
        return segments

    def _should_split_conversation(self, conversation: List[Dict[str, Any]]) -> bool:
        """
        Determine if a conversation should be split based on length

        Args:
            conversation: List of conversation turns

        Returns:
            True if conversation should be split
        """
        # Estimate conversation length (simplified token counting)
        estimated_tokens = self._estimate_conversation_length(conversation)

        # Use a conservative threshold (80% of max sequence length)
        split_threshold = self.max_seq_len * 0.8

        should_split = estimated_tokens > split_threshold

        self.logger.info(
            f"Split decision: {estimated_tokens} tokens vs threshold {split_threshold} -> {should_split}"
        )

        if should_split:
            self.logger.debug(
                f"Conversation estimated at {estimated_tokens} tokens, "
                f"exceeds threshold of {split_threshold}, will split"
            )

        return should_split

    def _estimate_conversation_length(self, conversation: List[Dict[str, Any]]) -> int:
        """
        Quick estimation of conversation length for split decision

        Args:
            conversation: List of conversation turns

        Returns:
            Estimated token count
        """
        total_tokens = 0

        # Use the same logic as _estimate_turn_length for consistency
        for turn in conversation:
            turn_length = self._estimate_turn_length(turn)
            total_tokens += turn_length

        return total_tokens

    def _split_conversation_by_length(
        self, conversation: List[Dict[str, Any]], original_video_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Split conversation by calculating sequence length as we build it

        Args:
            conversation: List of conversation turns
            original_video_data: Original video data for metadata

        Returns:
            List of conversation segments
        """
        segments = []
        current_segment_turns = []
        current_length_estimate = 0

        for i, turn in enumerate(conversation):
            # Estimate length of adding this turn
            turn_length_estimate = self._estimate_turn_length(turn)

            # Check if adding this turn would exceed limit
            would_exceed = (current_length_estimate + turn_length_estimate) > (
                self.max_seq_len * 0.9
            )

            # If would exceed and we have a substantial segment, create new segment
            if (
                would_exceed and len(current_segment_turns) >= 2
            ):  # Need at least 2 turns
                # Create segment with current conversation
                segment = self._create_conversation_segment(
                    current_segment_turns, original_video_data, len(segments)
                )
                
                # Calculate DST state at this split point for accurate state preservation
                dst_state_at_split = self.compute_dst_state_at_split(
                    conversation, i
                )
                segment["initial_dst_state"] = dst_state_at_split

                # Inject initial DST state into segment metadata
                segment = self.inject_initial_dst_state(segment, conversation)

                segments.append(segment)

                # Start new segment with context overlap
                current_segment_turns = self._create_overlap_context(
                    conversation, i, len(segments)
                )

                # Add DST state info to overlap context for continuity
                if dst_state_at_split:
                    dst_context_time = current_segment_turns[0].get("time", 0) if current_segment_turns else 0
                    # Calculate frame range for DST_CONTEXT turn (use same as first turn or minimal range)
                    fps = self.training_config.get("fps", 2)
                    dst_context_frame = int(dst_context_time * fps)

                    current_segment_turns.insert(0, {
                        "role": "DST_CONTEXT",
                        "content": f"Previous state: {dst_state_at_split}",
                        "time": dst_context_time,
                        "start_frame": dst_context_frame,
                        "end_frame": dst_context_frame + 1  # Single frame for context
                    })

                current_length_estimate = self._estimate_conversation_length(
                    current_segment_turns
                )
            else:
                # Add turn to current segment
                current_segment_turns.append(turn)
                current_length_estimate += turn_length_estimate

        # Add final segment if it has content
        if current_segment_turns:
            segment = self._create_conversation_segment(
                current_segment_turns, original_video_data, len(segments)
            )

            # Inject initial DST state for final segment if it's not the first
            if len(segments) > 0:  # If there were previous segments, this needs initial state
                segment = self.inject_initial_dst_state(segment, conversation)

            segments.append(segment)

        return segments

    def _estimate_turn_length(self, turn: Dict[str, Any]) -> int:
        """
        Estimate token length of a single turn (DST system with embedded frames)

        Args:
            turn: Single conversation turn with embedded frame information

        Returns:
            Estimated token count for this turn

        Raises:
            ValueError: If turn is missing required frame information
        """
        # Check for required frame information - ALL turns should have this in DST system
        if "start_frame" not in turn or "end_frame" not in turn:
            turn_role = turn.get("role", "unknown")
            raise ValueError(
                f"Turn with role '{turn_role}' is missing frame information. "
                f"All conversation turns in DST system should have 'start_frame' and 'end_frame' keys. "
                f"Turn content: {turn.get('content', 'N/A')[:100]}..."
            )

        # Calculate image tokens: num_frames × tokens_per_frame
        start_frame = turn["start_frame"]
        end_frame = turn["end_frame"]
        
        if end_frame <= start_frame:
            raise ValueError(
                f"Invalid frame range: end_frame ({end_frame}) <= start_frame ({start_frame}) "
                f"for turn with role '{turn.get('role', 'unknown')}'"
            )
        
        num_frames = end_frame - start_frame
        
        length_estimate = num_frames * self.num_tokens_per_img

        # Add separator tokens if model uses them
        if self.training_config.get("use_img_sep_token", False):
            length_estimate += max(0, num_frames - 1)

        # Add content tokens for text
        content = turn.get("content", "")
        if content:
            content_tokens = self._count_content_tokens(content, turn)
            length_estimate += content_tokens

        # Add role marker and formatting overhead
        length_estimate += 10

        # Add labels overhead if present
        # if "labels" in turn:
        #     labels = turn["labels"]
        #     if isinstance(labels, list):
        #         length_estimate += len(labels) * 5  # ~5 tokens per label
        #     elif isinstance(labels, str):
        #         length_estimate += len(labels) // 4

        self.logger.debug(
            f"Frame turn: {num_frames} frames × {self.num_tokens_per_img} tokens/frame + "
            f"{len(content)} chars = {length_estimate} total tokens"
        )

        return length_estimate

    def _count_content_tokens(self, content, turn: Dict[str, Any]) -> int:
        """
        Count tokens in conversation content using efficient DST encoding
        
        Args:
            content: Text content or DST content to tokenize
            turn: Turn for context
            
        Returns:
            Token count for content
        """
        if not content:
            return 0
            
        # Handle DST content efficiently using enums
        if turn.get("role") == "DST_UPDATE" or isinstance(content, list):
            return count_dst_content_tokens(content)
        
        # Handle regular string content
        content_str = str(content)
        
        # Use actual tokenizer if available
        if hasattr(self, "_tokenizer") and self._tokenizer:
            try:
                tokens = self._tokenizer.encode(content_str, add_special_tokens=False)
                return len(tokens)
            except Exception as e:
                raise RuntimeError(
                    f"Tokenization failed for content '{content_str[:100]}...': {e}"
                ) from e

        # No tokenizer available - fall back to approximate counting for testing
        # This is acceptable in test environments where transformers might not be available
        return max(len(content_str) // 4, 1)  # Rough approximation: ~4 chars per token

    def _create_conversation_segment(
        self,
        conversation_turns: List[Dict[str, Any]],
        original_video_data: Dict[str, Any],
        segment_index: int,
    ) -> Dict[str, Any]:
        """
        Create a conversation segment from turns

        Args:
            conversation_turns: List of turns for this segment
            original_video_data: Original video data for metadata
            segment_index: Index of this segment

        Returns:
            Video data for this segment
        """
        # Create copy of original video data
        segment_data = original_video_data.copy()

        # Update conversation
        segment_data["conversation"] = conversation_turns

        # Add segment-specific metadata
        segment_data["clip_idx"] = segment_index

        # Calculate frame range for this segment
        frame_range = self._calculate_segment_frame_range(conversation_turns)
        segment_data["start_frame_idx"] = frame_range[0]
        segment_data["end_frame_idx"] = frame_range[1]

        # Add segment statistics
        segment_data["segment_stats"] = {
            "turn_count": len(conversation_turns),
            "estimated_tokens": self._estimate_conversation_length(conversation_turns),
            "frame_range": frame_range,
            "is_split": len(conversation_turns) > 0 and segment_index > 0,
        }

        return segment_data

    def _create_overlap_context(
        self,
        full_conversation: List[Dict[str, Any]],
        split_point: int,
        segment_index: int,
    ) -> List[Dict[str, Any]]:
        """
        Create context overlap for the next segment

        Args:
            full_conversation: Full conversation list
            split_point: Index where split occurs
            segment_index: Index of the new segment

        Returns:
            List of turns for overlap context
        """
        overlap_turns = []

        # Determine overlap duration
        min_overlap, max_overlap = self.keep_context_length
        overlap_duration = random.randint(min_overlap, max_overlap)

        # Calculate how many turns to include for overlap
        overlap_turns_count = 0
        overlap_start_time = 0

        # Work backwards from split point to find overlap turns
        for i in range(split_point - 1, -1, -1):
            turn = full_conversation[i]
            turn_time = turn.get("time", 0)

            if overlap_start_time == 0 or turn_time >= overlap_start_time:
                overlap_start_time = turn_time
                overlap_turns.insert(0, turn)  # Insert at beginning to maintain order
                overlap_turns_count += 1

                # Stop if we've covered enough time
                if len(overlap_turns) > 0:
                    first_overlap_time = overlap_turns[0].get("time", 0)
                    if (turn_time - first_overlap_time) >= overlap_duration:
                        break

        # If no good overlap found, use a few turns before the split
        if not overlap_turns and split_point > 0:
            start_idx = max(0, split_point - 3)
            overlap_turns = full_conversation[start_idx:split_point]

        self.logger.debug(
            f"Created overlap context with {len(overlap_turns)} turns for segment {segment_index}"
        )

        return overlap_turns

    def _calculate_segment_frame_range(
        self, conversation_turns: List[Dict[str, Any]]
    ) -> Tuple[int, int]:
        """
        Calculate frame range for a conversation segment

        Args:
            conversation_turns: List of turns in the segment

        Returns:
            Tuple of (start_frame, end_frame)
        """
        if not conversation_turns:
            return 0, 0

        start_frames = []
        end_frames = []

        for turn in conversation_turns:
            if "start_frame" in turn and "end_frame" in turn:
                start_frames.append(turn["start_frame"])
                end_frames.append(turn["end_frame"])
            elif "time" in turn:
                # Fallback to timestamp-based calculation
                fps = self.training_config.get("fps", 2)
                frame_idx = int(turn["time"] * fps)
                start_frames.append(frame_idx)
                end_frames.append(frame_idx + 1)

        if not start_frames:
            return 0, 0

        return min(start_frames), max(end_frames)

    def calculate_sequence_length(self, conversation: List[Dict[str, Any]]) -> int:
        """
        Calculate sequence length for conversation (for compatibility)

        Args:
            conversation: List of conversation turns

        Returns:
            Estimated sequence length in tokens
        """
        return self._estimate_conversation_length(conversation)


    def generate_clip_indices(self, segments: List[Dict[str, Any]]) -> List[int]:
        """
        Generate incremental clip indices for each segment

        Args:
            segments: List of conversation segments

        Returns:
            List of clip indices
        """
        return [segment.get("clip_idx", i) for i, segment in enumerate(segments)]

    def compute_dst_state_at_split(
        self, full_conversation: List[Dict[str, Any]], split_point: int
    ) -> Dict[str, str]:
        """
        Calculate correct DST state at the start of each clip

        Args:
            full_conversation: Full conversation with DST updates
            split_point: Point where split occurs

        Returns:
            Dictionary mapping step IDs to their states at split point
        """
        # Initialize all steps as not_started
        dst_state = {}

        # Process DST updates up to split point
        for i in range(min(split_point, len(full_conversation))):
            turn = full_conversation[i]

            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            step_id = item.get("id", "")
                            transition = item.get("transition", "")

                            if step_id:
                                if transition == "start":
                                    dst_state[step_id] = "in_progress"
                                elif transition == "complete":
                                    dst_state[step_id] = "completed"

        # Fill in missing steps as not_started
        for step_id in dst_state:
            if dst_state[step_id] not in ["in_progress", "completed"]:
                dst_state[step_id] = "not_started"

        self.logger.debug(f"DST state at split point {split_point}: {dst_state}")
        return dst_state

    def inject_initial_dst_state(
        self,
        segment_data: Dict[str, Any],
        full_conversation: List[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Add initial DST state to clip metadata or system prompt

        Args:
            segment_data: Video data for this segment
            full_conversation: Full conversation (optional, for state calculation)

        Returns:
            Updated segment data with initial DST state
        """
        clip_idx = segment_data.get("clip_idx", 0)

        # For first clip, no initial state needed
        if clip_idx == 0:
            return segment_data

        # Try to get initial state from segment data first (set during splitting)
        initial_dst_state = segment_data.get("initial_dst_state", {})

        # If not available and we have full conversation, calculate it
        if not initial_dst_state and full_conversation:
            # Calculate based on the segment's position in the full conversation
            # We need to find where this segment starts in the full conversation
            segment_conversation = segment_data.get("conversation", [])
            if segment_conversation:
                # Find the starting turn of this segment in the full conversation
                # This is approximate - look for matching turns
                split_point = self._find_segment_start_in_full_conversation(
                    segment_conversation, full_conversation
                )
                if split_point is not None:
                    initial_dst_state = self.compute_dst_state_at_split(
                        full_conversation, split_point
                    )

        # Add initial DST state to metadata
        if "metadata" not in segment_data:
            segment_data["metadata"] = {}

        segment_data["metadata"]["initial_dst_state"] = initial_dst_state
        segment_data["metadata"]["has_initial_dst_state"] = len(initial_dst_state) > 0

        # Optionally add to system prompt if present
        conversation = segment_data.get("conversation", [])
        if conversation and conversation[0].get("role") == "system":
            system_content = conversation[0].get("content", "")
            if "Dialogue Context:" not in system_content and initial_dst_state:
                dst_context = self._format_dst_state_context(initial_dst_state)
                updated_content = (
                    system_content + f"\n\nDialogue Context:\n{dst_context}"
                )
                conversation[0]["content"] = updated_content

        self.logger.debug(
            f"Injected initial DST state for clip {clip_idx}: {initial_dst_state}"
        )
        return segment_data

    def _find_segment_start_in_full_conversation(
        self, segment_conversation: List[Dict[str, Any]], full_conversation: List[Dict[str, Any]]
    ) -> Optional[int]:
        """
        Find where a segment starts in the full conversation

        Args:
            segment_conversation: Conversation turns in this segment
            full_conversation: Full conversation

        Returns:
            Index in full conversation where this segment starts, or None if not found
        """
        if not segment_conversation or not full_conversation:
            return None

        # Look for the first turn of the segment in the full conversation
        first_turn = segment_conversation[0]

        # Skip DST_CONTEXT turns when looking for the match
        if first_turn.get("role") == "DST_CONTEXT":
            if len(segment_conversation) > 1:
                first_turn = segment_conversation[1]
            else:
                return None

        # Find matching turn in full conversation
        for i, turn in enumerate(full_conversation):
            if (turn.get("role") == first_turn.get("role") and
                turn.get("time") == first_turn.get("time") and
                turn.get("content") == first_turn.get("content")):
                return i

        return None

    def _format_dst_state_context(self, dst_state: Dict[str, str]) -> str:
        """
        Format DST state for inclusion in system prompt

        Args:
            dst_state: Dictionary mapping step IDs to states

        Returns:
            Formatted DST context string
        """
        if not dst_state:
            return "No previous dialogue state."

        state_parts = []
        for step_id, state in dst_state.items():
            state_parts.append(f"Step {step_id}: {state}")

        return "Current step states - " + ", ".join(state_parts)

    def validate_split_integrity(
        self, segments: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Validate that conversation splits maintain integrity

        Args:
            segments: List of conversation segments

        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {
                "total_segments": len(segments),
                "total_turns": sum(
                    len(seg.get("conversation", [])) for seg in segments
                ),
            },
        }

        if len(segments) == 1:
            # No splitting occurred
            validation_result["statistics"]["split_occurred"] = False
            return validation_result

        validation_result["statistics"]["split_occurred"] = True

        # Check segment integrity
        for i, segment in enumerate(segments):
            conversation = segment.get("conversation", [])

            # Check that segment has content
            if not conversation:
                validation_result["errors"].append(
                    f"Segment {i} has no conversation turns"
                )
                continue

            # Check clip indices are sequential
            expected_clip_idx = i
            actual_clip_idx = segment.get("clip_idx", -1)
            if actual_clip_idx != expected_clip_idx:
                validation_result["errors"].append(
                    f"Segment {i} has incorrect clip_idx: expected {expected_clip_idx}, got {actual_clip_idx}"
                )

            # Check frame ranges are valid
            start_frame = segment.get("start_frame_idx", 0)
            end_frame = segment.get("end_frame_idx", 0)
            if start_frame >= end_frame:
                validation_result["errors"].append(
                    f"Segment {i} has invalid frame range: {start_frame}-{end_frame}"
                )

        # Check temporal ordering of segments
        for i in range(1, len(segments)):
            prev_end = segments[i - 1].get("end_frame_idx", 0)
            curr_start = segments[i].get("start_frame_idx", 0)

            if curr_start < prev_end:
                validation_result["warnings"].append(
                    f"Segment {i} starts before segment {i-1} ends (temporal overlap)"
                )
            elif curr_start > prev_end + 20:  # Allow some gap
                validation_result["warnings"].append(
                    f"Large gap between segment {i-1} and {i}: {curr_start - prev_end} frames"
                )

        validation_result["valid"] = len(validation_result["errors"]) == 0

        return validation_result
