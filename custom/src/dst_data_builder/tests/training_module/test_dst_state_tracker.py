"""
Test DST State Tracker Module

Comprehensive tests for the DSTStateTracker class that validates:
1. DST transition tracking and extraction
2. State computation at timestamps and split points
3. Initial state injection for conversation segments
4. State consistency and transition rule validation
5. State history building and summary generation
6. Error handling and edge cases
"""

import pytest
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.training_modules.dst_state_tracker import DSTStateTracker


class TestDSTStateTracker:
    """Comprehensive tests for DSTStateTracker functionality"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return OmegaConf.create({
            "training_creation": {
                "validate_transitions": True,
            }
        })

    @pytest.fixture
    def sample_conversation(self):
        """Create sample conversation with DST transitions"""
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "time": 7.8,
                "content": "I want to disassemble this toy water tanker truck.",
            },
            {
                "role": "assistant",
                "time": 7.9,
                "content": "Great goal! Let's get started.",
            },
            {
                "role": "DST_UPDATE",
                "time": 15.5,
                "content": [{"id": "S1", "transition": "start"}],
            },
            {
                "role": "assistant",
                "time": 32.1,
                "content": "Beginning assembly...",
            },
            {
                "role": "DST_UPDATE",
                "time": 45.0,
                "content": [{"id": "S2", "transition": "start"}],
            },
            {
                "role": "assistant",
                "time": 58.7,
                "content": "Working on both steps...",
            },
            {
                "role": "DST_UPDATE",
                "time": 67.8,
                "content": [
                    {"id": "S1", "transition": "complete"},
                    {"id": "S2", "transition": "complete"}
                ],
            },
        ]

    @pytest.fixture
    def video_data(self, sample_conversation):
        """Create sample video data"""
        return {
            "video_uid": "test_video_001",
            "conversation": sample_conversation,
            "metadata": {
                "user_type": "talk_some",
            }
        }

    def test_initialization(self, config):
        """Test DSTStateTracker initialization"""
        tracker = DSTStateTracker(config)

        assert tracker is not None
        assert tracker.cfg == config
        assert tracker.training_config == config.training_creation
        assert tracker.validate_transitions is True
        assert hasattr(tracker, "STATE_VALUES")
        assert tracker.STATE_VALUES["not_started"] == 0
        assert tracker.STATE_VALUES["in_progress"] == 1
        assert tracker.STATE_VALUES["completed"] == 2

    def test_initialization_minimal_config(self):
        """Test initialization with minimal configuration"""
        minimal_config = OmegaConf.create({
            "training_creation": {}
        })
        tracker = DSTStateTracker(minimal_config)

        assert tracker is not None
        assert tracker.validate_transitions is True  # Default value

    def test_track_dst_transitions(self, config, video_data):
        """Test tracking DST transitions from conversation"""
        tracker = DSTStateTracker(config)

        result = tracker.track_dst_transitions(video_data)

        # Check that transitions were extracted
        assert "_dst_transitions" in result
        transitions = result["_dst_transitions"]

        # Should have 4 transitions: S1 start, S2 start, S1 complete, S2 complete
        assert len(transitions) == 4

        # Check transition details
        assert transitions[0]["timestamp"] == 15.5
        assert transitions[0]["step_id"] == "S1"
        assert transitions[0]["transition"] == "start"

        assert transitions[1]["timestamp"] == 45.0
        assert transitions[1]["step_id"] == "S2"
        assert transitions[1]["transition"] == "start"

        assert transitions[2]["timestamp"] == 67.8
        assert transitions[2]["step_id"] == "S1"
        assert transitions[2]["transition"] == "complete"

        # Check that state history was built
        assert "_dst_state_history" in result
        state_history = result["_dst_state_history"]
        assert len(state_history) == 4

    def test_track_dst_transitions_empty_conversation(self, config):
        """Test tracking transitions with empty conversation"""
        tracker = DSTStateTracker(config)

        video_data = {"conversation": []}
        result = tracker.track_dst_transitions(video_data)

        # Keys are only added if there are transitions
        assert "_dst_transitions" not in result or result["_dst_transitions"] == []
        assert "_dst_state_history" not in result or result["_dst_state_history"] == []

    def test_track_dst_transitions_no_dst_events(self, config):
        """Test tracking transitions with no DST events"""
        tracker = DSTStateTracker(config)

        conversation = [
            {"role": "system", "content": "Hello"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "How can I help?"},
        ]
        video_data = {"conversation": conversation}

        result = tracker.track_dst_transitions(video_data)

        assert result["_dst_transitions"] == []
        assert result["_dst_state_history"] == []

    def test_extract_dst_transitions(self, config, sample_conversation):
        """Test extracting DST transitions from conversation"""
        tracker = DSTStateTracker(config)

        transitions = tracker._extract_dst_transitions(sample_conversation)

        assert len(transitions) == 4

        # Check chronological ordering
        assert transitions[0]["timestamp"] == 15.5
        assert transitions[1]["timestamp"] == 45.0
        assert transitions[2]["timestamp"] == 67.8

        # Check turn indices
        assert transitions[0]["turn_index"] == 0
        assert transitions[1]["turn_index"] == 1
        assert transitions[2]["turn_index"] == 2

    def test_extract_dst_transitions_multiple_per_turn(self, config):
        """Test extracting multiple transitions from single DST_UPDATE turn"""
        tracker = DSTStateTracker(config)

        conversation = [
            {
                "role": "DST_UPDATE",
                "time": 10.0,
                "content": [
                    {"id": "S1", "transition": "start"},
                    {"id": "S2", "transition": "start"},
                    {"id": "S3", "transition": "complete"}
                ],
            }
        ]

        transitions = tracker._extract_dst_transitions(conversation)

        assert len(transitions) == 3
        assert transitions[0]["step_id"] == "S1"
        assert transitions[1]["step_id"] == "S2"
        assert transitions[2]["step_id"] == "S3"

    def test_build_state_history(self, config):
        """Test building state history from transitions"""
        tracker = DSTStateTracker(config)

        transitions = [
            {"timestamp": 10.0, "step_id": "S1", "transition": "start"},
            {"timestamp": 20.0, "step_id": "S2", "transition": "start"},
            {"timestamp": 30.0, "step_id": "S1", "transition": "complete"},
        ]

        state_history = tracker._build_state_history(transitions)

        assert len(state_history) == 3

        # Check first state (after S1 start)
        assert state_history[0]["timestamp"] == 10.0
        assert state_history[0]["step_id"] == "S1"
        assert state_history[0]["transition"] == "start"
        assert state_history[0]["state"] == {"S1": "in_progress"}

        # Check second state (after S2 start)
        assert state_history[1]["timestamp"] == 20.0
        assert state_history[1]["step_id"] == "S2"
        assert state_history[1]["transition"] == "start"
        assert state_history[1]["state"] == {"S1": "in_progress", "S2": "in_progress"}

        # Check third state (after S1 complete)
        assert state_history[2]["timestamp"] == 30.0
        assert state_history[2]["step_id"] == "S1"
        assert state_history[2]["transition"] == "complete"
        assert state_history[2]["state"] == {"S1": "completed", "S2": "in_progress"}

    def test_compute_state_at_timestamp(self, config, video_data):
        """Test computing DST state at specific timestamps"""
        tracker = DSTStateTracker(config)

        # First track transitions
        tracker.track_dst_transitions(video_data)

        # Test state at different timestamps
        state_at_0 = tracker.compute_state_at_timestamp(video_data, 0.0)
        assert state_at_0 == {}  # No transitions yet

        state_at_20 = tracker.compute_state_at_timestamp(video_data, 20.0)
        assert state_at_20 == {"S1": "in_progress"}  # S1 started at 15.5

        state_at_50 = tracker.compute_state_at_timestamp(video_data, 50.0)
        assert state_at_50 == {"S1": "in_progress", "S2": "in_progress"}  # Both started

        state_at_70 = tracker.compute_state_at_timestamp(video_data, 70.0)
        assert state_at_70 == {"S1": "completed", "S2": "completed"}  # Both completed

    def test_compute_state_at_timestamp_no_transitions(self, config):
        """Test computing state when no transitions exist"""
        tracker = DSTStateTracker(config)

        video_data = {"conversation": []}
        tracker.track_dst_transitions(video_data)

        state = tracker.compute_state_at_timestamp(video_data, 10.0)
        assert state == {}

    def test_compute_state_at_split_point(self, config, sample_conversation):
        """Test computing DST state at conversation split points"""
        tracker = DSTStateTracker(config)

        # Test split at different points
        state_at_0 = tracker.compute_state_at_split_point(sample_conversation, 0)
        assert state_at_0 == {"S1": "not_started", "S2": "not_started"}  # All discovered steps are not_started

        state_at_4 = tracker.compute_state_at_split_point(sample_conversation, 4)
        assert state_at_4 == {"S1": "in_progress", "S2": "not_started"}  # S1 start processed, S2 not yet seen

        state_at_6 = tracker.compute_state_at_split_point(sample_conversation, 6)
        assert state_at_6 == {"S1": "in_progress", "S2": "in_progress"}  # Both starts processed

        state_at_8 = tracker.compute_state_at_split_point(sample_conversation, 8)
        assert state_at_8 == {"S1": "completed", "S2": "completed"}  # Both completes processed

    def test_compute_state_at_split_point_with_all_steps(self, config):
        """Test that split point computation includes all discovered steps"""
        tracker = DSTStateTracker(config)

        conversation = [
            {"role": "DST_UPDATE", "time": 10, "content": [{"id": "S1", "transition": "start"}]},
            {"role": "assistant", "content": "Working"},
            {"role": "DST_UPDATE", "time": 20, "content": [{"id": "S2", "transition": "start"}]},
            {"role": "assistant", "content": "More work"},
            {"role": "DST_UPDATE", "time": 30, "content": [{"id": "S1", "transition": "complete"}]},
        ]

        # Split after S2 start (turn index 3)
        state = tracker.compute_state_at_split_point(conversation, 3)
        assert state == {"S1": "in_progress", "S2": "in_progress"}

        # Split after S1 complete (turn index 5)
        state = tracker.compute_state_at_split_point(conversation, 5)
        assert state == {"S1": "completed", "S2": "in_progress"}

    def test_inject_initial_dst_state_first_segment(self, config):
        """Test injecting initial state for first segment (should be empty)"""
        tracker = DSTStateTracker(config)

        segment_data = {"conversation": []}

        result = tracker.inject_initial_dst_state(segment_data, 0)

        # First segment should not have initial state
        assert result == segment_data
        assert "metadata" not in result or "initial_dst_state" not in result.get("metadata", {})

    def test_inject_initial_dst_state_subsequent_segment(self, config):
        """Test injecting initial state for subsequent segments"""
        tracker = DSTStateTracker(config)

        # Create segment with some transitions
        segment_data = {
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "DST_UPDATE", "time": 10, "content": [{"id": "S1", "transition": "start"}]},
            ],
            "_dst_transitions": [
                {"timestamp": 10.0, "step_id": "S1", "transition": "start"}
            ]
        }

        result = tracker.inject_initial_dst_state(segment_data, 1)

        # Should have initial state in metadata
        assert "metadata" in result
        assert "initial_dst_state" in result["metadata"]
        assert "has_initial_dst_state" in result["metadata"]
        assert result["metadata"]["has_initial_dst_state"] is True

    def test_inject_initial_dst_state_with_system_prompt(self, config):
        """Test injecting initial state into system prompt"""
        tracker = DSTStateTracker(config)

        segment_data = {
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."},
            ],
            "initial_dst_state": {"S1": "completed", "S2": "in_progress"}  # Set directly, not in metadata
        }

        result = tracker.inject_initial_dst_state(segment_data, 1)

        # System prompt should be updated
        system_turn = result["conversation"][0]
        # The method checks if "Dialogue Context:" is NOT already in content, then adds it
        assert "Dialogue Context:" in system_turn["content"]
        assert "Step S1: completed" in system_turn["content"]
        assert "Step S2: in_progress" in system_turn["content"]

    def test_validate_state_consistency_valid(self, config, video_data):
        """Test validating state consistency with valid transitions"""
        tracker = DSTStateTracker(config)

        # Track transitions first
        tracker.track_dst_transitions(video_data)

        # Validate consistency
        result = tracker.validate_state_consistency(video_data)

        assert result["valid"] is True
        assert result["errors"] == []
        assert result["warnings"] == []

    def test_validate_state_consistency_invalid_transition(self, config):
        """Test validating state consistency with invalid transitions"""
        tracker = DSTStateTracker(config)

        # Create video data with invalid transition (complete -> start)
        video_data = {
            "conversation": [
                {"role": "DST_UPDATE", "time": 10, "content": [{"id": "S1", "transition": "complete"}]},
                {"role": "DST_UPDATE", "time": 20, "content": [{"id": "S1", "transition": "start"}]},
            ]
        }

        tracker.track_dst_transitions(video_data)
        result = tracker.validate_state_consistency(video_data)

        assert result["valid"] is False
        assert len(result["errors"]) == 2  # Two invalid transitions
        assert all("Invalid transition" in error for error in result["errors"])

    def test_validate_transition_rules_valid_sequence(self, config):
        """Test validating transition rules with valid sequence"""
        tracker = DSTStateTracker(config)

        transitions = [
            {"step_id": "S1", "transition": "start"},
            {"step_id": "S1", "transition": "complete"},
            {"step_id": "S2", "transition": "start"},
        ]

        result = tracker.validate_transition_rules(transitions)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_validate_transition_rules_invalid_sequence(self, config):
        """Test validating transition rules with invalid sequence"""
        tracker = DSTStateTracker(config)

        transitions = [
            {"step_id": "S1", "transition": "complete"},  # Invalid: complete before start
            {"step_id": "S1", "transition": "start"},
        ]

        result = tracker.validate_transition_rules(transitions)

        assert result["valid"] is False
        assert len(result["errors"]) == 2  # Two invalid transitions

    def test_validate_transition_rules_multiple_steps(self, config):
        """Test validating transition rules with multiple steps"""
        tracker = DSTStateTracker(config)

        transitions = [
            {"step_id": "S1", "transition": "start"},
            {"step_id": "S2", "transition": "start"},
            {"step_id": "S1", "transition": "complete"},
            {"step_id": "S2", "transition": "complete"},
        ]

        result = tracker.validate_transition_rules(transitions)

        assert result["valid"] is True
        assert result["errors"] == []

    def test_is_valid_transition(self, config):
        """Test individual transition validation"""
        tracker = DSTStateTracker(config)

        # Valid transitions
        assert tracker._is_valid_transition("not_started", "start") is True
        assert tracker._is_valid_transition("in_progress", "complete") is True

        # Invalid transitions
        assert tracker._is_valid_transition("not_started", "complete") is False
        assert tracker._is_valid_transition("completed", "start") is False
        assert tracker._is_valid_transition("in_progress", "start") is False

    def test_format_dst_state_context(self, config):
        """Test formatting DST state for system prompts"""
        tracker = DSTStateTracker(config)

        # Test with states
        state = {"S1": "completed", "S2": "in_progress"}
        context = tracker._format_dst_state_context(state)

        assert "Step S1: completed" in context
        assert "Step S2: in_progress" in context

        # Test with empty state
        empty_context = tracker._format_dst_state_context({})
        assert empty_context == "No previous dialogue state."

    def test_get_state_summary(self, config, video_data):
        """Test getting state summary for debugging"""
        tracker = DSTStateTracker(config)

        # Track transitions first
        tracker.track_dst_transitions(video_data)

        summary = tracker.get_state_summary(video_data)

        assert "total_transitions" in summary
        assert "unique_steps" in summary
        assert "timeline" in summary
        assert "final_state" in summary

        assert summary["total_transitions"] == 4
        assert summary["unique_steps"] == 2  # S1 and S2
        assert len(summary["timeline"]) == 4
        assert summary["final_state"]["S1"] == "completed"
        assert summary["final_state"]["S2"] == "completed"

    def test_get_state_summary_no_transitions(self, config):
        """Test getting state summary with no transitions"""
        tracker = DSTStateTracker(config)

        video_data = {"conversation": []}
        tracker.track_dst_transitions(video_data)

        summary = tracker.get_state_summary(video_data)

        assert summary["total_transitions"] == 0
        assert summary["unique_steps"] == 0
        assert summary["timeline"] == []
        assert summary["final_state"] == {}

    def test_error_handling_invalid_transition_data(self, config):
        """Test error handling with malformed transition data"""
        tracker = DSTStateTracker(config)

        # Test with missing transition field - should raise KeyError
        transitions = [{"step_id": "S1"}]  # Missing transition
        with pytest.raises(KeyError):
            tracker.validate_transition_rules(transitions)

    def test_chronological_transition_ordering(self, config):
        """Test that transitions are processed in chronological order"""
        tracker = DSTStateTracker(config)

        # Create transitions that would be out of order without sorting
        conversation = [
            {"role": "DST_UPDATE", "time": 30.0, "content": [{"id": "S1", "transition": "complete"}]},
            {"role": "DST_UPDATE", "time": 10.0, "content": [{"id": "S1", "transition": "start"}]},
            {"role": "DST_UPDATE", "time": 20.0, "content": [{"id": "S2", "transition": "start"}]},
        ]

        transitions = tracker._extract_dst_transitions(conversation)

        # Should be sorted by timestamp
        assert transitions[0]["timestamp"] == 10.0
        assert transitions[1]["timestamp"] == 20.0
        assert transitions[2]["timestamp"] == 30.0

    def test_state_persistence_across_multiple_transitions(self, config):
        """Test that state persists correctly across multiple transitions"""
        tracker = DSTStateTracker(config)

        transitions = [
            {"timestamp": 10.0, "step_id": "S1", "transition": "start"},
            {"timestamp": 20.0, "step_id": "S2", "transition": "start"},
            {"timestamp": 30.0, "step_id": "S1", "transition": "complete"},
            {"timestamp": 40.0, "step_id": "S3", "transition": "start"},
        ]

        state_history = tracker._build_state_history(transitions)

        # Check that state accumulates correctly
        assert state_history[0]["state"] == {"S1": "in_progress"}
        assert state_history[1]["state"] == {"S1": "in_progress", "S2": "in_progress"}
        assert state_history[2]["state"] == {"S1": "completed", "S2": "in_progress"}
        assert state_history[3]["state"] == {"S1": "completed", "S2": "in_progress", "S3": "in_progress"}


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys
    import pytest

    # Run the tests
    sys.exit(pytest.main([__file__, "-v", "-s"]))