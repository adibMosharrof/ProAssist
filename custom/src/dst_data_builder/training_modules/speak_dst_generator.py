"""
SpeakDST Generator Module

This module transforms DST data into enhanced SPEAK/DST_UPDATE format and creates
training conversation structure with system prompts and proper formatting for ProAssist compatibility.
"""

import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from omegaconf import DictConfig
from dataclasses import dataclass

from dst_data_builder.training_modules.system_prompts import (
    get_system_prompt_variations,
)


@dataclass
class DSTNode:
    """Represents a DST node with step information"""

    id: str
    name: str
    start_ts: float
    end_ts: float
    type: str = "step"


class SpeakDSTGenerator:
    """Transform DST data and generate enhanced conversation structure for training data"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Training creation configuration
        self.training_config = cfg.get("training_creation", {})
        self.conversation_format = self.training_config.get(
            "conversation_format", "proassist_training"
        )
        self.include_system_prompt = self.training_config.get(
            "include_system_prompt", True
        )
        self.system_prompt_variations = get_system_prompt_variations()

        # SpeakDST transformation configuration
        self.speak_dst_ratio = self.training_config.get("speak_dst_ratio", 0.6)
        self.include_state_snapshots = self.training_config.get(
            "include_state_snapshots", True
        )
        self.include_dst_updates = self.training_config.get("include_dst_updates", True)

    def create_training_conversation(
        self, video_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Create training-ready conversation structure from enhanced DST data

        Args:
            video_data: Enhanced DST data with conversation

        Returns:
            Updated video data with training conversation structure
        """
        self.logger.debug("Creating training conversation structure")

        conversation = video_data.get("conversation", [])
        if not conversation:
            self.logger.warning("No conversation found in video data")
            return video_data

        # Create training conversation with system prompt
        training_conversation = []

        # Check if conversation already has a system prompt
        has_system_prompt = any(turn.get("role") == "system" for turn in conversation)

        # Add system prompt if enabled and not already present
        if self.include_system_prompt and not has_system_prompt:
            system_prompt = self._create_system_prompt(video_data)
            training_conversation.append({"role": "system", "content": system_prompt})

        # Add DST_UPDATE events from DST spans if available
        dst_events = self._create_dst_update_events(video_data)
        if dst_events:
            training_conversation.extend(dst_events)
            self.logger.debug(
                f"Added {len(dst_events)} DST_UPDATE events to conversation"
            )

        # Process existing conversation turns
        for turn in conversation:
            processed_turn = self._process_conversation_turn(turn, video_data)
            if processed_turn:
                training_conversation.append(processed_turn)
            elif processed_turn is None:
                # Skip turns that should be filtered out
                continue

        # Sort conversation by timestamp to ensure proper chronological order
        training_conversation = self._sort_conversation_by_time(training_conversation)

        # Update video data with training conversation
        video_data["conversation"] = training_conversation

        self.logger.info(
            f"Created training conversation with {len(training_conversation)} turns"
        )
        return video_data

    def _create_system_prompt(self, video_data: Dict[str, Any]) -> str:
        """
        Create appropriate system prompt for the conversation

        Args:
            video_data: Video data with metadata

        Returns:
            System prompt string
        """
        # Select random prompt variation for diversity
        base_prompt = random.choice(self.system_prompt_variations)

        # Add knowledge context if available
        knowledge = video_data.get("knowledge", [])
        if knowledge and self.training_config.get("add_knowledge", True):
            knowledge_text = self._format_knowledge_context(knowledge)
            base_prompt += f"\n\nTask Knowledge:\n{knowledge_text}"

        # Add DST-specific context if available
        dst_context = self._extract_dst_context(video_data)
        if dst_context:
            base_prompt += f"\n\nDialogue Context:\n{dst_context}"

        return base_prompt

    def _format_knowledge_context(self, knowledge: List[str]) -> str:
        """
        Format knowledge context for system prompt

        Args:
            knowledge: List of knowledge items

        Returns:
            Formatted knowledge text
        """
        if not knowledge:
            return "No specific task knowledge available."

        formatted_knowledge = []
        for i, item in enumerate(knowledge, 1):
            formatted_knowledge.append(f"{i}. {item}")

        return "\n".join(formatted_knowledge)

    def _extract_dst_context(self, video_data: Dict[str, Any]) -> str:
        """
        Extract relevant DST context for system prompt

        Args:
            video_data: Video data

        Returns:
            DST context string
        """
        # Look for existing DST context in metadata
        if "metadata" in video_data and "dst_context" in video_data["metadata"]:
            return video_data["metadata"]["dst_context"]

        # Try to extract from conversation
        conversation = video_data.get("conversation", [])
        dst_updates = [
            turn for turn in conversation if turn.get("role") == "DST_UPDATE"
        ]

        if dst_updates:
            # Summarize DST state changes
            step_states = {}
            for turn in dst_updates:
                content = turn.get("content", [])
                for item in content:
                    step_id = item.get("id", "unknown")
                    transition = item.get("transition", "unknown")
                    step_states[step_id] = transition

            if step_states:
                context_parts = []
                for step_id, state in step_states.items():
                    context_parts.append(f"Step {step_id}: {state}")

                return f"Current step states - " + ", ".join(context_parts)

        return ""

    def _process_conversation_turn(
        self, turn: Dict[str, Any], video_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Process individual conversation turn for training format

        Args:
            turn: Original conversation turn
            video_data: Video data context

        Returns:
            Processed turn or None if should be skipped
        """
        role = turn.get("role", "")

        # Process different turn types
        if role == "user":
            return self._process_user_turn(turn)
        elif role == "assistant":
            return self._process_assistant_turn(turn)
        elif role == "system":
            return self._process_system_turn(turn)
        elif role == "DST_UPDATE":
            return self._process_dst_update_turn(turn)
        else:
            # Unknown turn type, include as-is but log warning
            self.logger.warning(f"Unknown turn type: {role}")
            return turn

    def _process_user_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Process USER turn for training format"""
        processed_turn = {"role": "user", "content": turn.get("content", "")}

        # Add timestamp if available
        if "time" in turn:
            processed_turn["time"] = turn["time"]

        return processed_turn

    def _process_assistant_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Process assistant turn for training format"""
        processed_turn = {"role": "assistant", "content": turn.get("content", "")}

        # Add timestamp if available
        if "time" in turn:
            processed_turn["time"] = turn["time"]

        # Add labels if available and should include them
        if self.training_config.get("enable_dst_labels", True) and "labels" in turn:
            processed_turn["labels"] = turn["labels"]

        return processed_turn

    def _process_system_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Process system turn for training format"""
        processed_turn = {"role": "system", "content": turn.get("content", "")}

        # Add timestamp if available
        if "time" in turn:
            processed_turn["time"] = turn["time"]

        return processed_turn

    def _process_dst_update_turn(self, turn: Dict[str, Any]) -> Dict[str, Any]:
        """Process DST_UPDATE turn for training format"""
        # Keep DST_UPDATE turns as-is, just ensure proper structure
        processed_turn = {"role": "DST_UPDATE", "content": turn.get("content", [])}

        # Add timestamp if available
        if "time" in turn:
            processed_turn["time"] = turn["time"]

        return processed_turn

    def _generate_dst_description(self, content: List[Dict[str, Any]]) -> str:
        """
        Generate human-readable description of DST state changes

        Args:
            content: DST update content

        Returns:
            Descriptive text for the state changes
        """
        if not content:
            return "Dialogue state updated."

        descriptions = []
        for item in content:
            step_id = item.get("id", "unknown")
            transition = item.get("transition", "unknown")

            if transition == "start":
                descriptions.append(f"Step {step_id} has been started.")
            elif transition == "complete":
                descriptions.append(f"Step {step_id} has been completed.")
            else:
                descriptions.append(f"Step {step_id} state changed to {transition}.")

        return " ".join(descriptions)

    def _generate_dst_labels(self, content: List[Dict[str, Any]]) -> str:
        """
        Generate appropriate labels for DST_UPDATE turns

        Args:
            content: DST update content

        Returns:
            Labels string for DST updates
        """
        if not content:
            return "initiative|dst_update"

        # Base label for all DST updates
        labels = ["initiative|dst_update"]

        # Add transition-specific labels
        transitions = [item.get("transition", "") for item in content]

        if "start" in transitions and "complete" in transitions:
            labels.append("dst_multiple")
        elif "start" in transitions:
            labels.append("dst_start")
        elif "complete" in transitions:
            labels.append("dst_complete")

        return ",".join(labels)

    def add_training_metadata(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add training-specific metadata to video data

        Args:
            video_data: Video data

        Returns:
            Updated video data with training metadata
        """
        conversation = video_data.get("conversation", [])

        # Calculate conversation statistics
        total_turns = len(conversation)
        user_turns = len([t for t in conversation if t.get("role") == "user"])
        assistant_turns = len([t for t in conversation if t.get("role") == "assistant"])
        system_turns = len([t for t in conversation if t.get("role") == "system"])

        # Add training metadata
        training_metadata = {
            "conversation_stats": {
                "total_turns": total_turns,
                "user_turns": user_turns,
                "assistant_turns": assistant_turns,
                "system_turns": system_turns,
                "has_system_prompt": system_turns > 0,
            },
            "training_format": self.conversation_format,
            "processed_for_training": True,
        }

        # Update metadata section
        if "metadata" not in video_data:
            video_data["metadata"] = {}

        video_data["metadata"].update(training_metadata)

        self.logger.debug(
            f"Added training metadata: {total_turns} total turns, "
            f"{user_turns} user, {assistant_turns} assistant, {system_turns} system"
        )

        return video_data

    def validate_training_format(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the conversation follows training format requirements

        Args:
            video_data: Video data to validate

        Returns:
            Validation results
        """
        conversation = video_data.get("conversation", [])

        validation_result = {"valid": True, "errors": [], "warnings": [], "stats": {}}

        if not conversation:
            validation_result["valid"] = False
            validation_result["errors"].append("No conversation found")
            return validation_result

        # Check conversation structure
        roles = [turn.get("role", "") for turn in conversation]

        # Count role types
        role_counts = {}
        for role in roles:
            role_counts[role] = role_counts.get(role, 0) + 1

        validation_result["stats"] = role_counts

        # Validate role sequence (should be system -> user -> assistant pattern)
        expected_sequence = ["system", "user", "assistant"]

        if roles and roles[0] != "system" and self.include_system_prompt:
            validation_result["warnings"].append(
                "Expected system prompt at the beginning but none found"
            )

        # Check for required fields in each turn
        for i, turn in enumerate(conversation):
            turn_role = turn.get("role", "")
            if not turn_role:
                validation_result["errors"].append(f"Turn {i} missing role")
                continue

            if turn_role not in ["system", "user", "assistant"]:
                validation_result["errors"].append(
                    f"Turn {i} has invalid role: {turn_role}"
                )

            if "content" not in turn or not turn["content"]:
                validation_result["errors"].append(f"Turn {i} missing content")

        # Update validity
        validation_result["valid"] = len(validation_result["errors"]) == 0

        return validation_result

    def _create_dst_update_events(
        self, video_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Create DST_UPDATE events from DST spans for inclusion in conversation

        Args:
            video_data: Video data containing DST spans

        Returns:
            List of DST_UPDATE conversation turns
        """
        dst_events = []

        # Get DST spans from video data
        dst_spans = video_data.get("dst", [])
        if not dst_spans:
            return dst_events

        # Get existing DST_UPDATE events to avoid duplication
        existing_dst_events = set()
        conversation = video_data.get("conversation", [])
        for turn in conversation:
            if turn.get("role") == "DST_UPDATE":
                content = turn.get("content", [])
                timestamp = turn.get("time", 0)
                for item in content:
                    if isinstance(item, dict):
                        step_id = item.get("id", "")
                        transition = item.get("transition", "")
                        if step_id and transition:
                            existing_dst_events.add((step_id, transition, timestamp))

        # Convert each DST span to DST_UPDATE events (start and complete)
        for span in dst_spans:
            step_id = span.get("id", "")
            start_ts = span.get("start_ts", 0)
            end_ts = span.get("end_ts", 0)

            # Validate that start_ts <= end_ts (temporal validation should have fixed this, but double-check)
            if start_ts > end_ts:
                self.logger.warning(
                    f"Invalid span for {step_id}: start_ts ({start_ts}) > end_ts ({end_ts}). Skipping complete event."
                )
                end_ts = start_ts  # Fallback: don't create complete event if timestamps are invalid

            # Create start event (if not already present)
            if (step_id, "start", start_ts) not in existing_dst_events:
                start_event = {
                    "role": "DST_UPDATE",
                    "time": start_ts,
                    "content": [{"id": step_id, "transition": "start"}],
                }
                dst_events.append(start_event)

            # Create complete event only if end_ts > start_ts (valid span duration)
            if (
                end_ts > start_ts
                and (step_id, "complete", end_ts) not in existing_dst_events
            ):
                complete_event = {
                    "role": "DST_UPDATE",
                    "time": end_ts,
                    "content": [{"id": step_id, "transition": "complete"}],
                }
                dst_events.append(complete_event)

        # Sort DST events by time
        dst_events.sort(key=lambda x: x.get("time", 0))

        self.logger.debug(
            f"Created {len(dst_events)} DST_UPDATE events from {len(dst_spans)} DST spans (start + complete transitions)"
        )
        return dst_events

    def _sort_conversation_by_time(
        self, conversation: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort conversation turns by timestamp to ensure chronological order

        Args:
            conversation: List of conversation turns

        Returns:
            Sorted conversation turns
        """

        def get_sort_key(turn):
            # System prompts should come first
            if turn.get("role") == "system":
                return (0, 0)
            # Use timestamp for other turns
            return (1, turn.get("time", 0))

        sorted_conversation = sorted(conversation, key=get_sort_key)
        return sorted_conversation
