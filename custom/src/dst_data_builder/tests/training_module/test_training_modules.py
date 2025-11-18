#!/usr/bin/env python3
"""
Test for training modules integration

This test verifies that the training modules work together correctly.
"""

import sys
import os
from pathlib import Path
from dst_data_builder.training_modules.speak_dst_generator import (
    SpeakDSTGenerator,
)
from dst_data_builder.training_modules.dst_event_grounding import (
    DSTEventGrounding,
)
from dst_data_builder.training_modules.conversation_splitter import (
    ConversationSplitter,
)
from omegaconf import DictConfig, OmegaConf

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dst_data_builder.training_modules.dataset_metadata_generator import (
    DatasetMetadataGenerator,
)
from dst_data_builder.training_modules.sequence_length_calculator import (
    SequenceLengthCalculator,
)
from dst_data_builder.training_modules.frame_integration import (
    FrameIntegration,
)
import pytest


def create_sample_config():
    """Create a sample configuration for testing"""
    config_dict = {
        "training_creation": {
            "fps": 2,
            "frames_subdir": "frames",
            "dst_frame_duration": 1,
            "max_seq_len": 4096,
            "num_tokens_per_img": 1,
            "tokenizer_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "special_tokens_count": 10,
            "enable_conversation_splitting": True,
            "keep_context_length": [5, 20],
            "conversation_format": "proassist_training",
            "include_system_prompt": True,
            "enable_dst_labels": True,
            "include_quality_metrics": True,
            "add_knowledge": True,
        },
        "data_source": {"frames_subdir": "frames"},
    }
    return OmegaConf.create(config_dict)


def create_sample_video_data():
    """Create sample video data for testing"""
    return {
        "video_uid": "test_video_001",
        "user_id": 0,
        "conversation": [
            {
                "role": "USER",
                "time": 0.0,
                "content": "Hello, I need help with this task",
            },
            {
                "role": "SPEAK",
                "time": 2.5,
                "content": "I can help you with that task. Let me start by analyzing the current state.",
            },
            {
                "role": "DST_UPDATE",
                "time": 5.0,
                "content": [{"id": "S1", "transition": "start"}],
            },
            {
                "role": "SPEAK",
                "time": 8.0,
                "content": "Good! Step 1 is now in progress. What would you like to do next?",
            },
            {
                "role": "DST_UPDATE",
                "time": 12.0,
                "content": [{"id": "S1", "transition": "complete"}],
            },
        ],
        "knowledge": ["Step 1: Initial analysis", "Step 2: Complete task"],
        "metadata": {
            "user_type": "talk_some",
            "task_goal": "Complete the assembly task",
        },
    }


class TestTrainingModules:
    """Test class for training modules integration"""

    @pytest.fixture
    def config(self):
        """Create a sample configuration for testing"""
        return create_sample_config()

    @pytest.fixture
    def video_data(self):
        """Create sample video data for testing"""
        return create_sample_video_data()

    def test_module_integration(self, config, video_data):
        """Test that modules can be imported and initialized"""
        print("🧪 Testing training modules integration...")

        # Initialize modules
        try:
            metadata_generator = DatasetMetadataGenerator(config)
            sequence_calculator = SequenceLengthCalculator(config)
            frame_integrator = FrameIntegration(config)
            conversation_splitter = ConversationSplitter(config)
            print("✅ All modules initialized successfully")
        except Exception as e:
            pytest.fail(f"Module initialization failed: {e}")

        # Test each module
        print("\n📊 Testing individual modules:")

        # Test frame integration
        try:
            video_data_with_frames = frame_integrator.add_frame_metadata(
                video_data.copy(), "test_dataset"
            )
            print(
                f"✅ Frame integration: Added frames to {len(video_data_with_frames['conversation'])} turns"
            )
            print(
                f"   Frame range: {video_data_with_frames.get('start_frame_idx', 'N/A')}-{video_data_with_frames.get('end_frame_idx', 'N/A')}"
            )
        except Exception as e:
            pytest.fail(f"Frame integration failed: {e}")

        # Test conversation splitting
        try:
            conversation_segments = conversation_splitter.split_conversations(
                video_data_with_frames.copy()
            )
            print(
                f"✅ Conversation splitting: Created {len(conversation_segments)} segments"
            )

            # Test context overlap preservation
            for i, segment in enumerate(conversation_segments):
                segment_turns = len(segment.get("conversation", []))
                print(f"   Segment {i}: {segment_turns} turns")
        except Exception as e:
            pytest.fail(f"Conversation splitting failed: {e}")

        # Test sequence calculation
        try:
            video_data_with_sequence = sequence_calculator.add_sequence_metadata(
                video_data_with_frames.copy()
            )
            seq_len = video_data_with_sequence.get("seq_len", 0)
            print(f"✅ Sequence calculation: Total tokens = {seq_len}")
        except Exception as e:
            pytest.fail(f"Sequence calculation failed: {e}")

        # Test metadata generation
        try:
            final_video_data = metadata_generator.add_training_metadata(
                video_data_with_sequence, "test_dataset", "train", clip_idx=0
            )
            user_type = final_video_data.get("metadata", {}).get("user_type", "N/A")
            clip_idx = final_video_data.get("clip_idx", "N/A")
            print(f"✅ Metadata generation: user_type={user_type}, clip_idx={clip_idx}")
        except Exception as e:
            pytest.fail(f"Metadata generation failed: {e}")

        # Test data integrity validation
        try:
            is_valid = metadata_generator.validate_data_integrity(final_video_data)
            print(f"✅ Data integrity validation: {'PASSED' if is_valid else 'FAILED'}")
            assert is_valid, "Data integrity validation failed"
        except Exception as e:
            pytest.fail(f"Data integrity validation failed: {e}")

        # Test sequence validation
        try:
            conversation = final_video_data.get("conversation", [])
            validation = sequence_calculator.validate_sequence_length(conversation)
            is_valid_seq = validation.get("valid", False)
            length = validation.get("current_length", 0)
            print(
                f"✅ Sequence validation: {'PASSED' if is_valid_seq else 'FAILED'} (length={length})"
            )
            assert is_valid_seq, f"Sequence validation failed (length={length})"
        except Exception as e:
            pytest.fail(f"Sequence validation failed: {e}")

        print("\n🎉 All tests passed! Training modules are working correctly.")

    def test_frame_integration_only(self, config, video_data):
        """Test frame integration module specifically"""
        print("🧪 Testing frame integration only...")

        frame_integrator = FrameIntegration(config)
        video_data_with_frames = frame_integrator.add_frame_metadata(
            video_data.copy(), "test_dataset"
        )

        # Verify frame metadata was added
        assert "start_frame_idx" in video_data_with_frames
        assert "end_frame_idx" in video_data_with_frames
        assert video_data_with_frames["start_frame_idx"] == 0
        assert video_data_with_frames["end_frame_idx"] == 25  # Last turn at 12.0s * 2 fps + 1

        print("✅ Frame integration test passed")

    def test_conversation_splitter_only(self, config, video_data):
        """Test conversation splitter module specifically"""
        print("🧪 Testing conversation splitter...")

        # Test 1: Short conversation (no splitting needed)
        print("\n📝 Test 1: Short conversation")

        # First add frame metadata
        frame_integrator = FrameIntegration(config)
        video_data_with_frames = frame_integrator.add_frame_metadata(
            video_data.copy(), "test_dataset"
        )

        # Test conversation splitting
        conversation_splitter = ConversationSplitter(config)
        conversation_segments = conversation_splitter.split_conversations(
            video_data_with_frames.copy()
        )

        # Verify splitting worked (no splitting for short conversation)
        assert (
            len(conversation_segments) == 1
        )  # Short conversation, no splitting needed
        assert len(conversation_segments[0]["conversation"]) == 5
        print(
            f"   ✅ Short conversation: {len(conversation_segments)} segment, {len(conversation_segments[0]['conversation'])} turns"
        )

        # Test 2: Long conversation (splitting required)
        print("\n📝 Test 2: Long conversation")

        # Create a very long conversation that will exceed token limits
        long_conversation = []
        base_time = 0.0

        # Add many long turns to exceed token limit
        for i in range(50):  # Create 50 turns
            long_content = (
                "This is a very long conversation turn with lots of detailed content that will definitely exceed the token limit when combined together. "
                * 10
            )  # Repeat to make it long
            turn = {
                "role": "USER" if i % 2 == 0 else "SPEAK",
                "time": base_time + i * 2.0,
                "content": long_content,
            }
            if i % 10 == 5:  # Add some DST updates
                turn["role"] = "DST_UPDATE"
                turn["content"] = [
                    {
                        "id": f"S{i//10}",
                        "transition": "start" if i % 20 == 5 else "complete",
                    }
                ]
            long_conversation.append(turn)

        long_video_data = {
            "video_uid": "test_long_video",
            "user_id": 1,
            "conversation": long_conversation,
            "knowledge": ["Step 1: Initial analysis"] * 20,  # Lots of knowledge
            "metadata": {
                "user_type": "talk_some",
                "task_goal": "Complete the assembly task",
            },
        }

        # Add frame metadata for long conversation
        long_video_data_with_frames = frame_integrator.add_frame_metadata(
            long_video_data, "test_dataset"
        )

        # Test conversation splitting on long conversation
        long_conversation_segments = conversation_splitter.split_conversations(
            long_video_data_with_frames.copy()
        )

        print(f"   Long conversation: {len(long_conversation)} turns")
        print(f"   Split into: {len(long_conversation_segments)} segments")

        # Verify splitting actually occurred
        assert (
            len(long_conversation_segments) > 1
        ), f"Expected splitting but got {len(long_conversation_segments)} segments"

        # Verify each segment has reasonable turn counts
        total_turns = sum(
            len(seg.get("conversation", [])) for seg in long_conversation_segments
        )

        # Count DST_CONTEXT turns added during splitting for state continuity
        dst_context_turns = sum(
            1 for seg in long_conversation_segments
            for turn in seg.get("conversation", [])
            if turn.get("role") == "DST_CONTEXT"
        )

        # Original turns should be preserved, plus DST_CONTEXT turns added during splitting
        assert total_turns == len(long_conversation) + dst_context_turns, \
            f"All original turns should be preserved (got {total_turns}, expected {len(long_conversation) + dst_context_turns})"

        # Verify segments have clip indices
        for i, segment in enumerate(long_conversation_segments):
            assert "clip_idx" in segment, f"Segment {i} missing clip_idx"
            assert segment["clip_idx"] == i, f"Segment {i} has wrong clip_idx"

            # Verify initial DST state injection for segments beyond the first
            if i > 0:  # Segments after the first should have initial DST state
                metadata = segment.get("metadata", {})
                assert "initial_dst_state" in metadata, f"Segment {i} missing initial_dst_state in metadata"
                assert "has_initial_dst_state" in metadata, f"Segment {i} missing has_initial_dst_state flag"
                assert metadata["has_initial_dst_state"], f"Segment {i} should have initial DST state"
                # The initial state should be a dict mapping step IDs to states
                initial_state = metadata["initial_dst_state"]
                assert isinstance(initial_state, dict), f"Segment {i} initial_dst_state should be a dict"

        print(
            f"   ✅ Long conversation: Successfully split {len(long_conversation)} turns into {len(long_conversation_segments)} segments"
        )

        print(
            "\n✅ Conversation splitter test passed - both short and long conversations handled correctly"
        )

    def test_metadata_generation_only(self, config, video_data):
        """Test metadata generation module specifically"""
        print("🧪 Testing metadata generation only...")

        metadata_generator = DatasetMetadataGenerator(config)
        final_video_data = metadata_generator.add_training_metadata(
            video_data, "test_dataset", "train", clip_idx=0
        )

        # Verify metadata was added
        assert "metadata" in final_video_data
        assert final_video_data["metadata"]["user_type"] == "talk_some"
        assert final_video_data["clip_idx"] == 0

        print("✅ Metadata generation test passed")


if __name__ == "__main__":
    # Allow running as standalone script
    import pytest

    # Run the tests
    sys.exit(pytest.main([__file__, "-v", "-s"]))
