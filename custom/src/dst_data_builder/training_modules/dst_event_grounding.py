"""
Dialog State Tracking(DST) Event Grounding & Labeling Module

This module embeds frame information and generates initiative/intent labels
for DST_UPDATE and SPEAK events.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig


class DSTEventGrounding:
    """Generate labels and embed frame information for DST events"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = self.cfg.get("training_creation", {})
        self.dst_frame_duration = self.training_config.get("dst_frame_duration", 1)
        self.enable_dst_labels = self.training_config.get("enable_dst_labels", True)

    def add_frames_and_labels(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add frame information and labels to conversation events

        Args:
            video_data: Video data with conversation

        Returns:
            Updated video data with frames and labels
        """
        self.logger.debug("Adding frames and labels to conversation events")

        conversation = video_data.get("conversation", [])
        if not conversation:
            self.logger.warning("No conversation found in video data")
            return video_data

        # Update each conversation turn with frames and labels
        updated_conversation = []
        for turn in conversation:
            updated_turn = self._add_frames_and_labels_to_turn(turn, conversation)
            updated_conversation.append(updated_turn)

        # Update conversation in video data
        video_data["conversation"] = updated_conversation

        # Update statistics
        self._update_grounding_statistics(video_data)

        self.logger.info(
            f"Added frames and labels to {len(updated_conversation)} turns"
        )
        return video_data

    def _add_frames_and_labels_to_turn(self, turn: Dict[str, Any], conversation: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add frame information and labels to a single conversation turn

        Args:
            turn: Single conversation turn
            conversation: Full conversation for context

        Returns:
            Updated turn with frames and labels
        """
        updated_turn = turn.copy()
        role = turn.get("role", "")

        # Add frame information if not already present
        if role in ["system", "user", "assistant", "SPEAK", "DST_UPDATE"] and (
            "start_frame" not in updated_turn or "end_frame" not in updated_turn
        ):
            timestamp = turn.get("time", 0)
            start_frame, end_frame = self._calculate_frame_range_for_timestamp(
                timestamp
            )
            updated_turn["start_frame"] = start_frame
            updated_turn["end_frame"] = end_frame

        # Add current DST state to all conversation turns
        dst_state = self._compute_dst_context_at_turn(turn, conversation)
        if dst_state:
            updated_turn["dst_state"] = dst_state

        # Generate labels based on role
        if self.enable_dst_labels:
            labels = self._generate_event_labels(role, turn)
            if labels:
                updated_turn["labels"] = labels

        return updated_turn

    def _compute_dst_context_at_turn(self, current_turn: Dict[str, Any], conversation: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Compute DST state context at the time of a specific turn

        Args:
            current_turn: The turn for which to compute context
            conversation: Full conversation up to this point

        Returns:
            Dictionary mapping step IDs to their current states
        """
        # Initialize all steps as not_started
        dst_state = {}

        # Find the index of the current turn
        current_turn_index = None
        for i, turn in enumerate(conversation):
            if turn is current_turn:  # Same object reference
                current_turn_index = i
                break

        if current_turn_index is None:
            self.logger.warning("Could not find current turn in conversation")
            return {}

        # Process all DST_UPDATE events up to but not including the current turn
        for i in range(current_turn_index):
            turn = conversation[i]
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
                                elif transition == "pause":
                                    dst_state[step_id] = "paused"
                                elif transition == "resume":
                                    dst_state[step_id] = "in_progress"
                                elif transition == "cancel":
                                    dst_state[step_id] = "cancelled"
                                elif transition == "reset":
                                    dst_state[step_id] = "not_started"
                                elif transition == "fail":
                                    dst_state[step_id] = "failed"

        # Fill in missing steps as not_started
        # We need to know what steps exist - check the dst metadata or infer from transitions
        all_step_ids = set()
        for turn in conversation[:current_turn_index + 1]:  # Include current turn for step discovery
            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and "id" in item:
                            all_step_ids.add(item["id"])

        for step_id in all_step_ids:
            if step_id not in dst_state:
                dst_state[step_id] = "not_started"

        return dst_state

    def _calculate_frame_range_for_timestamp(self, timestamp: float) -> Tuple[int, int]:
        """
        Calculate frame range for a specific timestamp

        Args:
            timestamp: Time in seconds

        Returns:
            Tuple of (start_frame, end_frame)
        """
        fps = self.training_config.get("fps", 2)

        # Calculate center frame
        center_frame = int(timestamp * fps)

        # Calculate frame range around the timestamp
        half_duration = self.dst_frame_duration / 2
        start_time = max(0, timestamp - half_duration)
        end_time = timestamp + half_duration

        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Ensure at least one frame
        if end_frame <= start_frame:
            end_frame = start_frame + 1

        return start_frame, end_frame

    def _generate_event_labels(self, role: str, turn: Dict[str, Any]) -> str:
        """
        Generate appropriate labels for conversation events

        Args:
            role: Role of the turn ('SPEAK', 'DST_UPDATE', etc.)
            turn: The conversation turn

        Returns:
            Labels string
        """
        if role == "SPEAK":
            return self._generate_speak_labels(turn)
        elif role == "DST_UPDATE":
            return self._generate_dst_update_labels(turn)
        elif role == "USER":
            return self._generate_user_labels(turn)
        else:
            return self._generate_generic_labels(role, turn)

    def _generate_speak_labels(self, turn: Dict[str, Any]) -> str:
        """
        Generate labels for SPEAK events (assistant turns)

        Args:
            turn: SPEAK conversation turn

        Returns:
            Labels string for SPEAK events
        """
        # Base label for assistant turns
        labels = ["initiative|instruction"]

        content = turn.get("content", "").lower()

        # Additional modifiers based on content analysis
        if any(
            word in content for word in ["good", "well done", "excellent", "correct"]
        ):
            labels.append("feedback")
        elif any(
            word in content for word in ["information", "explain", "because", "since"]
        ):
            labels.append("info_sharing")
        elif any(
            word in content for word in ["actually", "wait", "correction", "sorry"]
        ):
            labels.append("correction")
        elif any(word in content for word in ["start", "begin", "initiate"]):
            labels.append("initiation")
        elif any(word in content for word in ["complete", "finish", "done"]):
            labels.append("completion")

        return ",".join(labels)

    def _generate_dst_update_labels(self, turn: Dict[str, Any]) -> str:
        """
        Generate labels for DST_UPDATE events

        Args:
            turn: DST_UPDATE conversation turn

        Returns:
            Labels string for DST_UPDATE events
        """
        if not self.enable_dst_labels:
            return ""

        # Base label for DST updates
        labels = ["initiative|dst_update"]

        content = turn.get("content", [])
        if not isinstance(content, list):
            content = [content]

        # Analyze transitions for specific labels
        transitions = [
            item.get("transition", "") for item in content if isinstance(item, dict)
        ]

        if "start" in transitions and "complete" in transitions:
            labels.append("dst_multiple")
        elif "start" in transitions:
            labels.append("dst_start")
        elif "complete" in transitions:
            labels.append("dst_complete")
        else:
            labels.append("dst_state_change")

        # Add step-specific information
        step_ids = [
            item.get("id", "")
            for item in content
            if isinstance(item, dict) and "id" in item
        ]
        if step_ids:
            labels.append(f"steps_{len(set(step_ids))}")

        return ",".join(labels)

    def _generate_user_labels(self, turn: Dict[str, Any]) -> str:
        """
        Generate labels for USER events

        Args:
            turn: USER conversation turn

        Returns:
            Labels string for USER events
        """
        # Base label for user turns
        labels = ["user|input"]

        content = turn.get("content", "").lower()

        # Analyze user input type
        if any(word in content for word in ["help", "assist", "support"]):
            labels.append("help_request")
        elif any(word in content for word in ["what", "how", "why", "where"]):
            labels.append("question")
        elif any(word in content for word in ["do", "perform", "execute"]):
            labels.append("action_request")
        elif any(word in content for word in ["thank", "thanks"]):
            labels.append("gratitude")

        return ",".join(labels)

    def _generate_generic_labels(self, role: str, turn: Dict[str, Any]) -> str:
        """
        Generate labels for other turn types

        Args:
            role: Role of the turn
            turn: The conversation turn

        Returns:
            Labels string for generic events
        """
        return f"{role.lower()}|generic"

    def validate_dst_frame_alignment(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Ensure DST frames align with conversation temporal flow

        Args:
            conversation: List of conversation turns

        Returns:
            Validation results
        """
        validation_result = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "statistics": {},
        }

        if not conversation:
            validation_result["valid"] = False
            validation_result["errors"].append("No conversation found")
            return validation_result

        # Analyze frame alignment
        frame_ranges = []
        dst_events = []

        for i, turn in enumerate(conversation):
            role = turn.get("role", "")
            timestamp = turn.get("time", 0)

            if role in ["SPEAK", "DST_UPDATE"]:
                start_frame = turn.get("start_frame", 0)
                end_frame = turn.get("end_frame", 0)

                frame_ranges.append(
                    {
                        "turn_index": i,
                        "role": role,
                        "timestamp": timestamp,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frame_count": end_frame - start_frame,
                    }
                )

                if role == "DST_UPDATE":
                    dst_events.append(
                        {
                            "turn_index": i,
                            "timestamp": timestamp,
                            "content": turn.get("content", []),
                        }
                    )

        # Check temporal ordering of frames
        for i in range(1, len(frame_ranges)):
            prev_end = frame_ranges[i - 1]["end_frame"]
            curr_start = frame_ranges[i]["start_frame"]

            # Allow some overlap but check for major temporal issues
            if curr_start > prev_end + 10:  # Allow up to 5-second gap (assuming 2fps)
                validation_result["warnings"].append(
                    f"Large temporal gap between turns {i-1} and {i}: "
                    f"turn {i-1} ends at frame {prev_end}, turn {i} starts at frame {curr_start}"
                )

        # Check DST event frame consistency
        fps = self.training_config.get("fps", 2)
        for dst_event in dst_events:
            timestamp = dst_event["timestamp"]
            expected_center_frame = int(timestamp * fps)

            # Find the corresponding frame range
            turn_index = dst_event["turn_index"]
            if turn_index < len(frame_ranges):
                actual_center = (
                    frame_ranges[turn_index]["start_frame"]
                    + frame_ranges[turn_index]["end_frame"]
                ) // 2

                frame_diff = abs(actual_center - expected_center_frame)
                if frame_diff > 2:  # Allow up to 1-second difference
                    validation_result["warnings"].append(
                        f"DST event at timestamp {timestamp}s has large frame alignment difference: "
                        f"expected frame {expected_center_frame}, actual center {actual_center}"
                    )

        # Update validation result
        validation_result["statistics"] = {
            "total_events": len(frame_ranges),
            "dst_events": len(dst_events),
            "avg_frame_count": (
                sum(fr["frame_count"] for fr in frame_ranges) / len(frame_ranges)
                if frame_ranges
                else 0
            ),
        }

        validation_result["valid"] = len(validation_result["errors"]) == 0

        return validation_result

    def _update_grounding_statistics(self, video_data: Dict[str, Any]) -> None:
        """
        Update video data with grounding statistics

        Args:
            video_data: Video data to update
        """
        conversation = video_data.get("conversation", [])

        if not conversation:
            return

        # Count events by type and label
        event_counts = {}
        label_counts = {}

        for turn in conversation:
            role = turn.get("role", "unknown")
            event_counts[role] = event_counts.get(role, 0) + 1

            labels = turn.get("labels", "")
            if labels:
                for label in labels.split(","):
                    label = label.strip()
                    label_counts[label] = label_counts.get(label, 0) + 1

        # Calculate frame statistics
        frame_stats = self._calculate_frame_statistics(conversation)

        # Update metadata
        if "metadata" not in video_data:
            video_data["metadata"] = {}

        video_data["metadata"]["grounding_stats"] = {
            "event_counts": event_counts,
            "label_counts": label_counts,
            "frame_statistics": frame_stats,
            "processed_for_grounding": True,
        }

        self.logger.debug(
            f"Updated grounding stats: {event_counts} events, {len(label_counts)} label types"
        )

    def _calculate_frame_statistics(
        self, conversation: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate frame usage statistics

        Args:
            conversation: List of conversation turns

        Returns:
            Frame statistics dictionary
        """
        frame_ranges = []
        total_frames_used = 0

        for turn in conversation:
            role = turn.get("role", "")
            if role in ["SPEAK", "DST_UPDATE"]:
                start_frame = turn.get("start_frame", 0)
                end_frame = turn.get("end_frame", 0)
                frame_count = max(0, end_frame - start_frame)

                frame_ranges.append(
                    {
                        "role": role,
                        "start_frame": start_frame,
                        "end_frame": end_frame,
                        "frame_count": frame_count,
                    }
                )

                total_frames_used += frame_count

        if not frame_ranges:
            return {"total_frames_used": 0}

        # Calculate statistics
        frame_counts = [fr["frame_count"] for fr in frame_ranges]
        start_frames = [fr["start_frame"] for fr in frame_ranges]
        end_frames = [fr["end_frame"] for fr in frame_ranges]

        return {
            "total_frames_used": total_frames_used,
            "unique_frame_ranges": len(frame_ranges),
            "avg_frames_per_event": sum(frame_counts) / len(frame_counts),
            "min_frame_count": min(frame_counts),
            "max_frame_count": max(frame_counts),
            "overall_start_frame": min(start_frames),
            "overall_end_frame": max(end_frames),
            "frame_coverage": (
                max(end_frames) - min(start_frames)
                if start_frames and end_frames
                else 0
            ),
        }
