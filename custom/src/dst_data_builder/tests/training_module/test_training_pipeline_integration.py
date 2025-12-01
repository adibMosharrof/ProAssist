"""
Test script to verify the DST training pipeline integration works correctly.

This test verifies:
1. All training modules can be imported
2. DSTStateTrackerModule works correctly
3. SimpleDSTGenerator initializes with training modules
4. Configuration is properly loaded
"""

import sys
import os
import json
import tempfile
from pathlib import Path

# Add the custom src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from omegaconf import DictConfig, OmegaConf


def test_training_modules_import():
    """Test that all training modules can be imported correctly"""
    print("🔄 Testing training modules import...")

    try:
        from dst_data_builder.training_modules import (
            FrameIntegration,
            SequenceLengthCalculator,
            ConversationSplitter,
            DSTStateTracker,
            SpeakDSTGenerator,
            DSTEventGrounding,
            DatasetMetadataGenerator,
        )

        print("✅ All training modules imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import training modules: {e}")
        return False


def test_dst_state_tracker():
    """Test DSTStateTracker functionality"""
    print("🔄 Testing DSTStateTracker...")

    try:
        from dst_data_builder.training_modules import DSTStateTracker
        from omegaconf import DictConfig

        # Create test config
        config_dict = {"training_creation": {"validate_transitions": True}}
        cfg = DictConfig(config_dict)

        # Initialize module
        tracker = DSTStateTracker(cfg)
        print("✅ DSTStateTracker initialized successfully")

        # Test with sample data
        test_video_data = {
            "conversation": [
                {
                    "role": "DST_UPDATE",
                    "time": 10.0,
                    "content": [{"id": "S1", "transition": "start"}],
                },
                {
                    "role": "DST_UPDATE",
                    "time": 25.0,
                    "content": [{"id": "S1", "transition": "complete"}],
                },
            ]
        }

        # Test transition tracking
        tracked_data = tracker.track_dst_transitions(test_video_data)
        assert "_dst_transitions" in tracked_data
        assert len(tracked_data["_dst_transitions"]) == 2
        print("✅ DST transition tracking works correctly")

        # Define test cases as (timestamp, expected_state, message)
        test_cases = [
            (5.0, {}, "Not Started State computation works correctly"),
            (
                15.0,
                {"S1": "in_progress"},
                "In Progress State computation works correctly",
            ),
            (30.0, {"S1": "completed"}, "Complete State computation works correctly"),
        ]

        for ts, expected, msg in test_cases:
            state = tracker.compute_state_at_timestamp(tracked_data, ts)
            assert state == expected, f"At {ts}: expected {expected}, got {state}"
            print(f"✅ {msg}")

        return True

    except Exception as e:
        print(f"❌ DSTStateTracker test failed: {e}")
        return False


def test_simple_dst_generator_integration():
    """Test SimpleDSTGenerator with training modules"""
    print("🔄 Testing SimpleDSTGenerator integration...")

    try:
        # Create test configuration
        config_dict = {
            "model": {"name": "gpt-4o", "temperature": 0.7, "max_tokens": 1000},
            "generation": {
                "enable_multiprocessing": False,
            },
            "training_creation": {
                "enable_training_creation": True,
                "validate_transitions": True,
                "fps": 2,
                "max_seq_len": 4096,
            },
            "max_retries": 1,
        }

        cfg = DictConfig(config_dict)

        # This will test imports and initialization
        from dst_data_builder.simple_dst_generator import SimpleDSTGenerator

        generator = SimpleDSTGenerator(cfg)

        # Check that training modules were initialized
        assert generator.training_modules is not None
        assert "dst_state_tracker" in generator.training_modules
        print("✅ SimpleDSTGenerator initialized with training modules")

        return True

    except Exception as e:
        print(f"❌ SimpleDSTGenerator integration test failed: {e}")
        return False


def test_configuration_loading():
    """Test that the configuration file loads correctly"""
    print("🔄 Testing configuration loading...")

    try:
        config_path = (
            Path(__file__).parent.parent.parent.parent
            / "config"
            / "dst_data_generator"
            / "simple_dst_generator.yaml"
        )

        if not config_path.exists():
            print(f"⚠️ Config file not found at {config_path}")
            return False

        # Load and validate configuration
        cfg = OmegaConf.load(config_path)

        # Check required sections exist
        assert "training_creation" in cfg
        assert "output" in cfg
        assert "generation" in cfg

        print("✅ Configuration loaded and validated successfully")
        print(f"   - Training creation configured: {hasattr(cfg, 'training_creation')}")
        print("   - Training format is default output (no config needed)")

        return True

    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False


def test_training_format_creation():
    """Test the training format creation pipeline"""
    print("🔄 Testing training format creation...")

    try:
        from dst_data_builder.training_modules import (
            DSTStateTracker,
            FrameIntegration,
            DSTEventGrounding,
            SpeakDSTGenerator,
        )
        from omegaconf import DictConfig

        # Create comprehensive config for testing
        config_dict = {
            "training_creation": {
                "validate_transitions": True,
                "fps": 2,
                "max_seq_len": 4096,
                "dst_frame_duration": 1,
            },
            "data_source": {
                "data_path": "data",
                "frames_subdir": "frames",
                "datasets": ["assembly101", "ego4d"],
            },
        }
        cfg = DictConfig(config_dict)

        # Initialize all modules
        tracker = DSTStateTracker(cfg)
        frame_integration = FrameIntegration(cfg)
        dst_grounding = DSTEventGrounding(cfg)
        speak_dst_generator = SpeakDSTGenerator(cfg)

        # Create sample enhanced data with timestamps
        sample_data = {
            "video_uid": "nusar-2021_action_both_9011-a01_9011_user_id_2021-02-01_153724__HMC_84355350_mono10bit",
            "conversation": [
                {
                    "role": "DST_UPDATE",
                    "time": 15.5,
                    "content": [{"id": "S1", "transition": "start"}],
                },
                {"role": "SPEAK", "time": 32.1, "content": "Beginning assembly..."},
                {
                    "role": "DST_UPDATE",
                    "time": 45.0,
                    "content": [{"id": "S2", "transition": "start"}],
                },
            ],
        }

        # Test 1: Frame integration - add start_frame and end_frame to conversation events
        print("   🔍 Testing frame integration...")

        # Simulate the loop over datasets from SimpleDSTGenerator
        for dataset_idx, dataset_name in enumerate(
            config_dict["data_source"]["datasets"]
        ):
            print(f"   🔍 Testing dataset {dataset_idx + 1}: {dataset_name}")
            frame_data = frame_integration.add_frame_metadata(
                sample_data.copy(), dataset_name
            )

            # Note: frames_file is NOT added here - it will be resolved from video_uid during training
            # Verify video_uid is present for frame resolution
            assert "video_uid" in frame_data, "Missing video_uid in video data"
            print(f"   ✅ video_uid present for {dataset_name}: {frame_data['video_uid']}")

        # Use the last dataset's processed data for remaining tests
        frame_data = frame_integration.add_frame_metadata(
            sample_data, config_dict["data_source"]["datasets"][-1]
        )

        # Verify frame information is embedded in conversation events
        for event in frame_data["conversation"]:
            assert "start_frame" in event, f"Missing start_frame in {event}"
            assert "end_frame" in event, f"Missing end_frame in {event}"
            assert isinstance(
                event["start_frame"], int
            ), f"start_frame should be int, got {type(event['start_frame'])}"
            assert isinstance(
                event["end_frame"], int
            ), f"end_frame should be int, got {type(event['end_frame'])}"

        # Verify frame calculations use the sophisticated frame range approach
        # The module calculates center frame using int(timestamp * fps)
        # Then creates a range around it based on frame_duration
        fps = config_dict["training_creation"]["fps"]  # 2
        frame_duration = config_dict["training_creation"]["dst_frame_duration"]  # 1

        # For timestamp T, the range is [T - frame_duration/2, T + frame_duration/2]
        # Converted to frames: [int((T - 0.5) * fps), int((T + 0.5) * fps)]

        # Event 1: time=15.5, range=[15.0, 16.0] -> frames=[30, 32]
        # Event 2: time=32.1, range=[31.6, 32.6] -> frames=[63, 65]
        # Event 3: time=45.0, range=[44.5, 45.5] -> frames=[89, 91]
        expected_frames = [
            (30, 32),  # 15.5 seconds with fps=2
            (63, 65),  # 32.1 seconds with fps=2
            (89, 91),  # 45.0 seconds with fps=2
        ]

        for i, (expected_start, expected_end) in enumerate(expected_frames):
            actual_start = frame_data["conversation"][i]["start_frame"]
            actual_end = frame_data["conversation"][i]["end_frame"]
            assert (
                actual_start == expected_start
            ), f"Event {i}: expected start {expected_start}, got {actual_start}"
            assert (
                actual_end == expected_end
            ), f"Event {i}: expected end {expected_end}, got {actual_end}"
        print("   ✅ Frame integration works correctly")

        # Test 2: DST state tracking
        print("   🔍 Testing DST state tracking...")
        tracked_data = tracker.track_dst_transitions(frame_data)
        state = tracker.compute_state_at_timestamp(tracked_data, 20.0)

        # Verify state tracking works
        assert "S1" in state
        assert state["S1"] == "in_progress"
        print("   ✅ DST state tracking works correctly")

        # Test 3: DST event grounding (labels and validation)
        print("   🔍 Testing DST event grounding...")
        grounded_data = dst_grounding.add_frames_and_labels(tracked_data)

        # Verify labels are added to events
        for event in grounded_data["conversation"]:
            if event["role"] == "DST_UPDATE":
                assert "labels" in event, f"Missing labels in DST_UPDATE event: {event}"
            elif event["role"] == "SPEAK":
                assert "labels" in event, f"Missing labels in SPEAK event: {event}"
        print("   ✅ DST event grounding works correctly")

        # Test 4: Training conversation structure creation
        print("   🔍 Testing training conversation structure...")
        training_data = speak_dst_generator.create_training_conversation(grounded_data)

        # Verify training structure has system prompt and proper metadata
        assert "conversation" in training_data
        if len(training_data["conversation"]) > 0:
            first_event = training_data["conversation"][0]
            assert (
                first_event["role"] == "system"
            ), "First event should be system prompt"
            assert "content" in first_event
        print("   ✅ Training conversation structure works correctly")

        print("✅ Training format creation pipeline works correctly")
        print("   ✅ Frame information embedded directly in conversation events")
        print("   ✅ DST state tracking functional")
        print("   ✅ Event grounding and labeling functional")
        print("   ✅ Training structure creation functional")
        return True

    except Exception as e:
        print(f"❌ Training format creation test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("🚀 Starting DST Training Pipeline Integration Tests")
    print("=" * 60)

    tests = [
        test_training_modules_import,
        test_dst_state_tracker,
        test_simple_dst_generator_integration,
        test_configuration_loading,
        test_training_format_creation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
        print()

    print("=" * 60)
    print(f"📊 Test Results: {passed} passed, {failed} failed")

    if failed == 0:
        print(
            "🎉 All tests passed! Training pipeline integration is working correctly."
        )
        return True
    else:
        print("⚠️ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
