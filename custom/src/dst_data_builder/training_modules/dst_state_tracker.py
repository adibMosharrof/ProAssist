"""
DST State Tracker Module

This module maintains accurate DST state across conversation splits for training data integrity.
It tracks DST transitions, computes state at any timestamp, and ensures state consistency.

DST = Dialog State Tracking (not Daylight Saving Time)
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig


class DSTStateTracker:
    """Track and maintain DST state throughout conversations for training data creation"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.validate_transitions = self.training_config.get(
            "validate_transitions", True
        )

        # DST state constants
        self.STATE_VALUES = {"not_started": 0, "in_progress": 1, "completed": 2}

        self.logger.info("DSTStateTrackerModule initialized")

    def track_dst_transitions(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Monitor all DST_UPDATE events in chronological order

        Args:
            video_data: Enhanced DST data with conversation

        Returns:
            Updated video data with transition tracking
        """
        conversation = video_data.get("conversation", [])
        if not conversation:
            return video_data

        # Extract all DST transitions with timestamps
        dst_transitions = self._extract_dst_transitions(conversation)

        # Validate transitions if enabled
        if self.validate_transitions:
            validation_result = self.validate_transition_rules(dst_transitions)
            if not validation_result["valid"]:
                self.logger.warning(
                    f"DST transition validation failed: {validation_result['errors']}"
                )

        # Store transitions for later use
        video_data["_dst_transitions"] = dst_transitions
        video_data["_dst_state_history"] = self._build_state_history(dst_transitions)

        self.logger.debug(f"Tracked {len(dst_transitions)} DST transitions")
        return video_data

    def _extract_dst_transitions(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract all DST transitions from conversation with timestamps

        Args:
            conversation: List of conversation turns

        Returns:
            List of DST transition events with timestamp and transition info
        """
        transitions = []

        for turn in conversation:
            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            step_id = item.get("id", "")
                            transition = item.get("transition", "")
                            turn_time = turn.get("time", 0)

                            if step_id and transition:
                                transitions.append(
                                    {
                                        "timestamp": turn_time,
                                        "step_id": step_id,
                                        "transition": transition,
                                        "turn_index": len(transitions),
                                    }
                                )

        # Sort by timestamp to ensure chronological order
        transitions.sort(key=lambda x: x["timestamp"])

        return transitions

    def _build_state_history(
        self, transitions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Build chronological state history from transitions

        Args:
            transitions: List of DST transition events

        Returns:
            List of state snapshots at each transition point
        """
        state_history = []
        current_state = {}  # All steps start as not_started

        for transition in transitions:
            step_id = transition["step_id"]
            transition_type = transition["transition"]

            # Update state based on transition
            if transition_type == "start":
                current_state[step_id] = "in_progress"
            elif transition_type == "complete":
                current_state[step_id] = "completed"

            # Record state snapshot
            state_snapshot = {
                "timestamp": transition["timestamp"],
                "step_id": step_id,
                "transition": transition_type,
                "state": current_state.copy(),
            }
            state_history.append(state_snapshot)

        return state_history

    def compute_state_at_timestamp(
        self, video_data: Dict[str, Any], timestamp: float
    ) -> Dict[str, str]:
        """
        Calculate DST state at any point in the conversation

        Args:
            video_data: Video data with tracked transitions
            timestamp: Time point to compute state for

        Returns:
            Dictionary mapping step IDs to their states at the given timestamp
        """
        transitions = video_data.get("_dst_transitions", [])
        if not transitions:
            return {}

        # Initialize all steps as not_started
        state = {}

        # Apply transitions up to the given timestamp
        for transition in transitions:
            if transition["timestamp"] <= timestamp:
                step_id = transition["step_id"]
                transition_type = transition["transition"]

                if transition_type == "start":
                    state[step_id] = "in_progress"
                elif transition_type == "complete":
                    state[step_id] = "completed"
            else:
                break  # Transitions are chronologically sorted

        return state

    def compute_state_at_split_point(
        self, conversation: List[Dict[str, Any]], split_point: int
    ) -> Dict[str, str]:
        """
        Calculate DST state at a conversation split point

        Args:
            conversation: Full conversation list
            split_point: Index or turn count where split occurs

        Returns:
            Dictionary mapping step IDs to their states at split point
        """
        # Initialize all steps as not_started
        state = {}

        # Process DST updates up to split point
        processed_turns = 0
        for turn in conversation:
            if processed_turns >= split_point:
                break

            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            step_id = item.get("id", "")
                            transition = item.get("transition", "")

                            if step_id:
                                if transition == "start":
                                    state[step_id] = "in_progress"
                                elif transition == "complete":
                                    state[step_id] = "completed"

            processed_turns += 1

        # Fill in missing steps as not_started (for completeness)
        all_steps = set()
        for turn in conversation:
            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            all_steps.add(item.get("id", ""))

        for step_id in all_steps:
            if step_id not in state:
                state[step_id] = "not_started"

        return state

    def inject_initial_dst_state(
        self, segment_data: Dict[str, Any], segment_index: int
    ) -> Dict[str, Any]:
        """
        Add correct initial DST state to each conversation clip

        Args:
            segment_data: Video data for this segment
            segment_index: Index of this segment

        Returns:
            Updated segment data with initial DST state
        """
        # For first segment, no initial state needed
        if segment_index == 0:
            return segment_data

        # Get conversation and estimate split point
        conversation = segment_data.get("conversation", [])
        if not conversation:
            return segment_data

        # Estimate split point based on conversation length
        # For segments beyond the first, assume they start from a meaningful point
        split_point = len(conversation) // 2  # Simplified approach

        # This method would need access to the full conversation to work correctly
        # For now, check if we have transition info in the segment itself
        dst_transitions = segment_data.get("_dst_transitions", [])
        if dst_transitions:
            # Calculate state based on transitions in this segment
            initial_state = self._calculate_state_from_transitions(
                dst_transitions, segment_index
            )
        else:
            # Fallback: look for initial state in metadata
            initial_state = segment_data.get("initial_dst_state", {})

        # Add initial DST state to metadata
        if "metadata" not in segment_data:
            segment_data["metadata"] = {}

        segment_data["metadata"]["initial_dst_state"] = initial_state
        segment_data["metadata"]["has_initial_dst_state"] = len(initial_state) > 0

        # Optionally add to system prompt if present
        if conversation and conversation[0].get("role") == "system":
            system_content = conversation[0].get("content", "")
            if "Dialogue Context:" not in system_content and initial_state:
                dst_context = self._format_dst_state_context(initial_state)
                updated_content = (
                    system_content + f"\n\nDialogue Context:\n{dst_context}"
                )
                conversation[0]["content"] = updated_content

        self.logger.debug(
            f"Injected initial DST state for segment {segment_index}: {initial_state}"
        )
        return segment_data

    def _calculate_state_from_transitions(
        self, transitions: List[Dict[str, Any]], segment_index: int
    ) -> Dict[str, str]:
        """
        Calculate initial state for a segment based on its transitions

        Args:
            transitions: List of DST transitions for this segment
            segment_index: Index of the segment

        Returns:
            Dictionary mapping step IDs to states
        """
        # For segments beyond the first, assume some previous state
        # This is a simplified implementation
        state = {}

        # Apply transitions that would have occurred before this segment
        for transition in transitions[
            :segment_index
        ]:  # Apply transitions from previous segments
            step_id = transition["step_id"]
            transition_type = transition["transition"]

            if transition_type == "start":
                state[step_id] = "in_progress"
            elif transition_type == "complete":
                state[step_id] = "completed"

        # Fill in missing steps as not_started
        all_steps = set(t["step_id"] for t in transitions)
        for step_id in all_steps:
            if step_id not in state:
                state[step_id] = "not_started"

        return state

    def validate_state_consistency(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure DST transitions follow logical rules

        Args:
            video_data: Video data with DST transitions

        Returns:
            Validation results
        """
        transitions = video_data.get("_dst_transitions", [])
        if not transitions:
            return {"valid": True, "errors": [], "warnings": []}

        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Track state for each step
        step_states = {}

        for transition in transitions:
            step_id = transition["step_id"]
            transition_type = transition["transition"]
            timestamp = transition["timestamp"]

            # Get current state (default to not_started)
            current_state = step_states.get(step_id, "not_started")

            # Check transition validity
            if not self._is_valid_transition(current_state, transition_type):
                validation_result["errors"].append(
                    f"Invalid transition for step {step_id} at {timestamp}s: "
                    f"{current_state} -> {transition_type}"
                )
                validation_result["valid"] = False

            # Update state
            if transition_type == "start":
                step_states[step_id] = "in_progress"
            elif transition_type == "complete":
                step_states[step_id] = "completed"

        return validation_result

    def validate_transition_rules(
        self, transitions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Check transition validity (not_started → in_progress → completed)

        Args:
            transitions: List of DST transition events

        Returns:
            Validation results
        """
        validation_result = {"valid": True, "errors": [], "warnings": []}

        # Track state changes for each step
        step_history = {}

        for i, transition in enumerate(transitions):
            step_id = transition["step_id"]
            transition_type = transition["transition"]

            # Initialize step history if not seen before
            if step_id not in step_history:
                step_history[step_id] = []

            # Get previous state (default to not_started)
            if step_history[step_id]:
                prev_state = step_history[step_id][-1]
            else:
                prev_state = "not_started"

            # Check if transition is valid
            if not self._is_valid_transition(prev_state, transition_type):
                validation_result["errors"].append(
                    f"Invalid transition at index {i} for step {step_id}: "
                    f"{prev_state} -> {transition_type}"
                )
                validation_result["valid"] = False

            # Record new state
            if transition_type == "start":
                step_history[step_id].append("in_progress")
            elif transition_type == "complete":
                step_history[step_id].append("completed")

        return validation_result

    def _is_valid_transition(self, current_state: str, transition_type: str) -> bool:
        """
        Check if a state transition is valid

        Args:
            current_state: Current state of the step
            transition_type: Type of transition (start, complete)

        Returns:
            True if transition is valid
        """
        # Define valid transitions
        valid_transitions = {
            "not_started": ["start"],
            "in_progress": ["complete"],
            "completed": [],  # No valid transitions from completed
        }

        return transition_type in valid_transitions.get(current_state, [])

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

    def get_state_summary(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get a summary of DST state tracking for debugging

        Args:
            video_data: Video data with tracked transitions

        Returns:
            Summary of DST state tracking
        """
        transitions = video_data.get("_dst_transitions", [])
        state_history = video_data.get("_dst_state_history", [])

        summary = {
            "total_transitions": len(transitions),
            "unique_steps": len(set(t["step_id"] for t in transitions)),
            "timeline": [],
            "final_state": {},
        }

        # Build timeline
        for transition in transitions:
            summary["timeline"].append(
                {
                    "timestamp": transition["timestamp"],
                    "step_id": transition["step_id"],
                    "transition": transition["transition"],
                }
            )

        # Get final state
        if state_history:
            summary["final_state"] = state_history[-1]["state"]
        else:
            # Build final state from transitions
            final_state = {}
            for transition in reversed(transitions):
                step_id = transition["step_id"]
                if step_id not in final_state:
                    final_state[step_id] = "not_started"
                    if transition["transition"] == "start":
                        final_state[step_id] = "in_progress"
                    elif transition["transition"] == "complete":
                        final_state[step_id] = "completed"
            summary["final_state"] = final_state

        return summary
