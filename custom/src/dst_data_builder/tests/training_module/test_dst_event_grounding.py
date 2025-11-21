"""
Test DST Event Grounding Module

Comprehensive tests for the DSTEventGrounding class that validates:
1. Frame information embedding in conversation events (add_frames_to_conversation_events)
2. Event frame range calculation (calculate_event_frame_ranges)
3. Frame availability validation (validate_frame_availability)
4. DST frame alignment validation (validate_dst_frame_alignment)
5. DST context computation at each turn (compute_dst_context_at_turn)
6. Label generation for different event types
7. Grounding statistics updates
"""

import pytest
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.training_modules.dst_event_grounding import DSTEventGrounding


class TestDSTEventGrounding:
    """Comprehensive tests for DSTEventGrounding functionality"""

    @pytest.fixture
    def config(self):
        """Create test configuration"""
        return OmegaConf.create({
            "training_creation": {
                "fps": 2,
                "dst_frame_duration": 1,
                "enable_dst_labels": True,
            }
        })

    @pytest.fixture
    def sample_conversation(self):
        """Create sample conversation for testing - based on real data from MD docs"""
        return [
            {
                "role": "system",
                "content": "You are a helpful assistant.",
            },
            {
                "role": "user",
                "time": 7.8,
                "content": "I want to disassemble this toy water tanker truck into its component parts.",
            },
            {
                "role": "assistant",
                "time": 7.9,
                "content": "Great goal! Let's get started. First, we need to detach the wheels.",
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
        """Test DSTEventGrounding initialization"""
        grounder = DSTEventGrounding(config)

        assert grounder is not None
        assert grounder.cfg == config
        assert grounder.training_config == config.training_creation
        assert grounder.dst_frame_duration == 1
        assert grounder.enable_dst_labels is True

    def test_calculate_event_frame_ranges(self, config):
        """Test calculate_event_frame_ranges functionality - convert timestamps to frame indices"""
        grounder = DSTEventGrounding(config)

        # Test various timestamps with fps=2
        test_cases = [
            (0.0, (0, 1)),      # Start of video: center=0, range [0,1] (1 frame)
            (5.0, (9, 11)),     # 5 seconds: center=10, range [9,11] (2 frames)
            (10.5, (20, 22)),   # 10.5 seconds: center=21, range [20,22] (2 frames)
            (15.0, (29, 31)),   # 15 seconds: center=30, range [29,31] (2 frames)
            (15.5, (30, 32)),   # 15.5 seconds: center=31, range [30,32] (2 frames)
        ]

        for timestamp, expected_range in test_cases:
            start_frame, end_frame = grounder._calculate_frame_range_for_timestamp(timestamp)
            assert start_frame == expected_range[0], f"Timestamp {timestamp}: expected start {expected_range[0]}, got {start_frame}"
            assert end_frame == expected_range[1], f"Timestamp {timestamp}: expected end {expected_range[1]}, got {end_frame}"
            assert end_frame > start_frame, f"Timestamp {timestamp}: end_frame should be > start_frame"
            # Frame range varies: 1 frame for timestamp 0.0, 2 frames for others
            expected_frames = expected_range[1] - expected_range[0]
            assert end_frame - start_frame == expected_frames, f"Timestamp {timestamp}: frame range should be {expected_frames} frames"

    def test_validate_frame_availability(self, config, video_data):
        """Test validate_frame_availability - ensure calculated frames exist within video bounds"""
        grounder = DSTEventGrounding(config)

        # Add frames to conversation
        video_data_with_frames = grounder.add_frames_and_labels(video_data)

        # Check that all calculated frames are non-negative
        for turn in video_data_with_frames["conversation"]:
            if "start_frame" in turn and "end_frame" in turn:
                assert turn["start_frame"] >= 0, f"start_frame should be non-negative, got {turn['start_frame']}"
                assert turn["end_frame"] >= 0, f"end_frame should be non-negative, got {turn['end_frame']}"
                assert turn["end_frame"] > turn["start_frame"], f"end_frame should be > start_frame"

    def test_validate_dst_frame_alignment(self, config, video_data):
        """Test validate_dst_frame_alignment - ensure DST frames align with conversation temporal flow"""
        grounder = DSTEventGrounding(config)

        # Add frames to conversation
        video_data_with_frames = grounder.add_frames_and_labels(video_data)

        # Validate frame alignment
        validation_result = grounder.validate_dst_frame_alignment(video_data_with_frames["conversation"])

        # Should pass validation
        assert validation_result["valid"] is True, f"Frame alignment validation failed: {validation_result.get('errors', [])}"

        # Should have statistics
        assert "statistics" in validation_result
        stats = validation_result["statistics"]
        assert "total_events" in stats
        assert "dst_events" in stats
        assert stats["total_events"] > 0, "Should have some events to validate"

    def test_compute_dst_context_at_turn(self, config, sample_conversation):
        """Test compute_dst_context_at_turn - calculate current DST state for each conversation turn"""
        grounder = DSTEventGrounding(config)

        # Test state at different points in conversation
        # Note: State is computed BEFORE the current turn's DST updates are applied
        test_cases = [
            (0, {}, "System prompt - no DST state yet"),
            (1, {}, "User input - no DST updates yet"),
            (2, {}, "Assistant response - no DST updates yet"),
            (3, {"S1": "not_started"}, "At DST_UPDATE S1 start - state before this update (S1 discovered but not yet started)"),
            (4, {"S1": "in_progress"}, "During SPEAK event - S1 was started at turn 3"),
            (5, {"S1": "in_progress", "S2": "not_started"}, "At DST_UPDATE S2 start - state before this update (S2 discovered but not yet started)"),
            (6, {"S1": "in_progress", "S2": "in_progress"}, "During SPEAK event - both steps in progress"),
            (7, {"S1": "in_progress", "S2": "in_progress"}, "At DST_UPDATE completion - state before this update (both steps were in progress)"),
        ]

        for turn_index, expected_state, description in test_cases:
            current_turn = sample_conversation[turn_index]
            state = grounder._compute_dst_context_at_turn(current_turn, sample_conversation)
            assert state == expected_state, f"{description}: expected {expected_state}, got {state}"

    def test_add_frames_to_conversation_events(self, config, video_data):
        """Test add_frames_to_conversation_events - add start_frame and end_frame keys to events"""
        grounder = DSTEventGrounding(config)

        # Process video data
        result = grounder.add_frames_and_labels(video_data)

        # Check that frames were added to appropriate events
        conversation = result["conversation"]
        for turn in conversation:
            role = turn.get("role", "")
            if role in ["SPEAK", "DST_UPDATE", "user", "assistant"]:
                assert "start_frame" in turn, f"Missing start_frame in {role} event"
                assert "end_frame" in turn, f"Missing end_frame in {role} event"
                assert isinstance(turn["start_frame"], int), f"start_frame should be int for {role}"
                assert isinstance(turn["end_frame"], int), f"end_frame should be int for {role}"
                assert turn["start_frame"] >= 0, f"start_frame should be >= 0 for {role}"
                assert turn["end_frame"] > turn["start_frame"], f"end_frame should be > start_frame for {role}"

    def test_dst_state_with_transitions(self, config):
        """Test DST state computation with various transition types"""
        grounder = DSTEventGrounding(config)

        conversation = [
            {"role": "DST_UPDATE", "time": 0, "content": [{"id": "S1", "transition": "start"}]},
            {"role": "assistant", "time": 5, "content": "Working..."},
            {"role": "DST_UPDATE", "time": 10, "content": [{"id": "S1", "transition": "pause"}]},
            {"role": "assistant", "time": 15, "content": "Paused"},
            {"role": "DST_UPDATE", "time": 20, "content": [{"id": "S1", "transition": "resume"}]},
            {"role": "assistant", "time": 25, "content": "Resumed"},
            {"role": "DST_UPDATE", "time": 30, "content": [{"id": "S1", "transition": "complete"}]},
        ]

        # Test state progression
        test_cases = [
            (0, {"S1": "not_started"}, "Initial state - S1 discovered but not started"),
            (1, {"S1": "in_progress"}, "After start"),
            (3, {"S1": "paused"}, "After pause"),
            (5, {"S1": "in_progress"}, "After resume"),
            (6, {"S1": "in_progress"}, "At DST_UPDATE complete - state before this update"),
        ]

        for turn_index, expected_state, description in test_cases:
            current_turn = conversation[turn_index]
            state = grounder._compute_dst_context_at_turn(current_turn, conversation)
            assert state == expected_state, f"{description}: expected {expected_state}, got {state}"

    def test_label_generation_speak_events(self, config):
        """Test label generation for SPEAK events"""
        grounder = DSTEventGrounding(config)

        test_cases = [
            (
                {"role": "SPEAK", "content": "Let's start the task"},
                "initiative|instruction,initiation",
            ),
            (
                {"role": "SPEAK", "content": "Good job! Well done"},
                "initiative|instruction,feedback",
            ),
            (
                {"role": "SPEAK", "content": "You need to explain this because..."},
                "initiative|instruction,info_sharing",
            ),
            (
                {"role": "SPEAK", "content": "Wait, actually you should..."},
                "initiative|instruction,correction",
            ),
        ]

        for turn, expected_labels in test_cases:
            labels = grounder._generate_event_labels("SPEAK", turn)
            assert labels == expected_labels, f"Expected '{expected_labels}', got '{labels}'"

    def test_label_generation_dst_update_events(self, config):
        """Test label generation for DST_UPDATE events"""
        grounder = DSTEventGrounding(config)

        test_cases = [
            (
                {"content": [{"id": "S1", "transition": "start"}]},
                "initiative|dst_update,dst_start,steps_1",
            ),
            (
                {"content": [{"id": "S1", "transition": "complete"}]},
                "initiative|dst_update,dst_complete,steps_1",
            ),
            (
                {"content": [
                    {"id": "S1", "transition": "complete"},
                    {"id": "S2", "transition": "start"}
                ]},
                "initiative|dst_update,dst_multiple,steps_2",
            ),
            (
                {"content": [{"id": "S1", "transition": "pause"}]},
                "initiative|dst_update,dst_state_change,steps_1",
            ),
        ]

        for turn, expected_labels in test_cases:
            labels = grounder._generate_event_labels("DST_UPDATE", turn)
            assert labels == expected_labels, f"Expected '{expected_labels}', got '{labels}'"

    def test_label_generation_user_events(self, config):
        """Test label generation for USER events"""
        grounder = DSTEventGrounding(config)

        test_cases = [
            (
                {"content": "Help me with this"},
                "user|input,help_request",
            ),
            (
                {"content": "What should I do next?"},
                "user|input,question",
            ),
            (
                {"content": "Please perform the task"},
                "user|input,action_request",
            ),
            (
                {"content": "Thank you very much"},
                "user|input,gratitude",
            ),
        ]

        for turn, expected_labels in test_cases:
            labels = grounder._generate_event_labels("USER", turn)
            assert labels == expected_labels, f"Expected '{expected_labels}', got '{labels}'"

    def test_label_generation_generic_events(self, config):
        """Test label generation for generic events"""
        grounder = DSTEventGrounding(config)

        test_cases = [
            ("system", "system|generic"),
            ("unknown", "unknown|generic"),
            ("custom_role", "custom_role|generic"),
        ]

        for role, expected_labels in test_cases:
            turn = {"content": "test content"}
            labels = grounder._generate_event_labels(role, turn)
            assert labels == expected_labels, f"Role '{role}': expected '{expected_labels}', got '{labels}'"

    def test_labels_disabled(self, config, video_data):
        """Test that labels are not generated when disabled"""
        disabled_config = OmegaConf.create({
            "training_creation": {
                "enable_dst_labels": False,
            }
        })
        grounder = DSTEventGrounding(disabled_config)

        result = grounder.add_frames_and_labels(video_data)

        # Check that labels are not added to turns when disabled
        for turn in result["conversation"]:
            role = turn.get("role", "")
            if role in ["SPEAK", "DST_UPDATE", "USER"]:
                # Labels should not be present or should be empty
                labels = turn.get("labels", "")
                assert labels == "", f"Labels should be empty when disabled for {role} turn, got '{labels}'"

    def test_add_frames_and_labels_basic(self, config, video_data):
        """Test basic frame and label addition"""
        grounder = DSTEventGrounding(config)
        result = grounder.add_frames_and_labels(video_data)

        # Check that conversation was processed
        assert "conversation" in result
        conversation = result["conversation"]
        assert len(conversation) == len(video_data["conversation"])

        # Check that frames were added to appropriate turns
        for turn in conversation:
            role = turn.get("role", "")
            if role in ["system", "user", "assistant", "SPEAK", "DST_UPDATE"]:
                assert "start_frame" in turn, f"Missing start_frame in {role} turn"
                assert "end_frame" in turn, f"Missing end_frame in {role} turn"
                assert isinstance(turn["start_frame"], int)
                assert isinstance(turn["end_frame"], int)

    def test_add_frames_and_labels_dst_state(self, config, video_data):
        """Test that DST state is added to conversation turns"""
        grounder = DSTEventGrounding(config)
        result = grounder.add_frames_and_labels(video_data)

        conversation = result["conversation"]

        # Check DST state is added to turns when there are active DST states
        for turn in conversation:
            role = turn.get("role", "")
            dst_state = turn.get("dst_state")
            if dst_state is not None:
                assert isinstance(dst_state, dict), f"dst_state should be dict, got {type(dst_state)}"
                # DST state should only be added when there are actual states
                assert len(dst_state) > 0, f"DST state should not be empty when present: {dst_state}"

    def test_add_frames_and_labels_with_labels(self, config, video_data):
        """Test that labels are added when enabled"""
        grounder = DSTEventGrounding(config)
        result = grounder.add_frames_and_labels(video_data)

        conversation = result["conversation"]

        # Check labels are added to appropriate turns
        for turn in conversation:
            role = turn.get("role", "")
            if role in ["SPEAK", "DST_UPDATE", "USER"]:
                assert "labels" in turn, f"Missing labels in {role} turn"
                assert isinstance(turn["labels"], str), f"Labels should be string, got {type(turn['labels'])}"
                assert turn["labels"], f"Labels should not be empty for {role} turn"

    def test_frame_alignment_validation(self, config, video_data):
        """Test frame alignment validation"""
        grounder = DSTEventGrounding(config)

        # Add frames first
        video_data_with_frames = grounder.add_frames_and_labels(video_data)

        # Validate alignment
        validation_result = grounder.validate_dst_frame_alignment(video_data_with_frames["conversation"])

        assert validation_result["valid"] is True, f"Validation failed: {validation_result.get('errors', [])}"
        assert "statistics" in validation_result
        assert validation_result["statistics"]["total_events"] > 0

    def test_grounding_statistics_update(self, config, video_data):
        """Test that grounding statistics are properly updated"""
        grounder = DSTEventGrounding(config)

        # Process video data
        result = grounder.add_frames_and_labels(video_data)

        # Check metadata was updated
        assert "metadata" in result
        metadata = result["metadata"]
        assert "grounding_stats" in metadata

        grounding_stats = metadata["grounding_stats"]
        assert "event_counts" in grounding_stats
        assert "label_counts" in grounding_stats
        assert "frame_statistics" in grounding_stats
        assert grounding_stats["processed_for_grounding"] is True

    def test_empty_conversation_handling(self, config):
        """Test handling of empty conversation"""
        grounder = DSTEventGrounding(config)

        video_data = {"conversation": []}
        result = grounder.add_frames_and_labels(video_data)

        assert result["conversation"] == []

    def test_missing_conversation_handling(self, config):
        """Test handling of missing conversation"""
        grounder = DSTEventGrounding(config)

        video_data = {}
        result = grounder.add_frames_and_labels(video_data)

        assert result == {}

    def test_frame_statistics_calculation(self, config, video_data):
        """Test frame statistics calculation"""
        grounder = DSTEventGrounding(config)

        # Process data
        result = grounder.add_frames_and_labels(video_data)

        # Check frame statistics
        frame_stats = result["metadata"]["grounding_stats"]["frame_statistics"]
        assert "total_frames_used" in frame_stats
        assert "unique_frame_ranges" in frame_stats
        assert "avg_frames_per_event" in frame_stats
        assert "overall_start_frame" in frame_stats
        assert "overall_end_frame" in frame_stats

        # Verify statistics are reasonable
        assert frame_stats["total_frames_used"] >= 0
        assert frame_stats["unique_frame_ranges"] >= 0

    def test_turn_processing_with_existing_frames(self, config):
        """Test processing turns that already have frame information"""
        grounder = DSTEventGrounding(config)

        conversation = [
            {
                "role": "SPEAK",
                "time": 5.0,
                "content": "Test",
                "start_frame": 100,  # Pre-existing frames
                "end_frame": 110,
            }
        ]

        result = grounder._add_frames_and_labels_to_turn(conversation[0], conversation)

        # Should preserve existing frames
        assert result["start_frame"] == 100
        assert result["end_frame"] == 110

    def test_dst_state_discovery(self, config):
        """Test that DST state discovery works correctly"""
        grounder = DSTEventGrounding(config)

        conversation = [
            {"role": "DST_UPDATE", "time": 0, "content": [{"id": "S1", "transition": "start"}]},
            {"role": "DST_UPDATE", "time": 0, "content": [{"id": "S2", "transition": "start"}]},
            {"role": "assistant", "time": 5, "content": "Working"},
            {"role": "DST_UPDATE", "time": 10, "content": [{"id": "S1", "transition": "complete"}]},
        ]

        # Test state at assistant turn (should include both steps)
        assistant_turn = conversation[2]
        state = grounder._compute_dst_context_at_turn(assistant_turn, conversation)

        assert "S1" in state
        assert "S2" in state
        assert state["S1"] == "in_progress"
        assert state["S2"] == "in_progress"

    def test_frame_range_edge_cases(self, config):
        """Test frame range calculation for edge cases"""
        grounder = DSTEventGrounding(config)

        # Test timestamp 0
        start, end = grounder._calculate_frame_range_for_timestamp(0.0)
        assert start == 0
        assert end == 1

        # Test very small timestamp
        start, end = grounder._calculate_frame_range_for_timestamp(0.1)
        assert start >= 0
        assert end > start

        # Test large timestamp
        start, end = grounder._calculate_frame_range_for_timestamp(1000.0)
        assert start > 0
        assert end > start
        assert end - start == 2  # Should always be 2 frames (1 second at 2fps)


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys
    import pytest

    # Run the tests
    sys.exit(pytest.main([__file__, "-v", "-s"]))