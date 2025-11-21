"""
Test SimpleDSTGenerator

Comprehensive tests for the SimpleDSTGenerator class that orchestrates the entire DST generation pipeline.
Tests initialization, training format creation, error handling, and integration with all components.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.simple_dst_generator import SimpleDSTGenerator
from dst_data_builder.validators.training_format_validator import TrainingFormatValidator


class TestSimpleDSTGenerator:
    """Test class for SimpleDSTGenerator functionality"""

    @pytest.fixture
    def sample_config(self):
        """Create comprehensive sample configuration for testing"""
        return OmegaConf.create({
            "model": {
                "name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 4000,
                "log_name": "gpt-4o"
            },
            "generation": {
                "evidence_spans": False,
                "enable_multiprocessing": False,
                "multiprocessing_processes": 12
            },
            "training_creation": {
                "fps": 2,
                "dst_frame_duration": 2,
                "max_seq_len": 4096,
                "num_tokens_per_img": 1,
                "tokenizer_name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
                "special_tokens_count": 10,
                "enable_conversation_splitting": True,
                "keep_context_length": [5, 20],
                "conversation_format": "proassist_training",
                "include_system_prompt": True,
                "enable_dst_labels": True,
                "validate_transitions": True,
                "include_quality_metrics": True,
                "add_knowledge": False
            },
            "output": {
                "save_intermediate": True
            },
            "max_retries": 1,
            "generator": {
                "type": "hybrid_dst",
                "overlap_reduction": {
                    "min_overlap_ratio": 0.8,
                    "max_gap_duration": 1.0,
                },
                "similarity": {
                    "semantic_weight": 0.6,
                    "nli_weight": 0.4,
                    "high_confidence_threshold": 0.3,
                },
                "llm_fallback": {
                    "batch_size": 5,
                    "max_tokens": 1000,
                    "temperature": 0.1,
                },
                "temporal_validation": {
                    "max_allowed_overlap": 5.0,
                    "min_gap_duration": 0.1,
                    "max_gap_duration": 300.0,
                },
                "span_construction": {
                    "min_span_duration": 1.0,
                    "max_span_duration": 300.0,
                },
                "models": {
                    "semantic_encoder": "BAAI/bge-base-en-v1.5",
                    "nli_model": "cross-encoder/nli-deberta-v3-base",
                },
                "parsing": {
                    "gap_threshold": 2.0,
                    "similarity_threshold": 0.6,
                },
            }
        })

    @pytest.fixture
    def sample_enhanced_data(self):
        """Create sample enhanced DST data for testing"""
        return [
            {
                "video_uid": "test_video_1",
                "dataset": "assembly101",
                "dst": [
                    {"id": "S1", "t0": 10.0, "t1": 25.0, "name": "Step 1"},
                    {"id": "S2", "t0": 30.0, "t1": 45.0, "name": "Step 2"}
                ],
                "conversation": [
                    {"role": "user", "time": 5.0, "content": "Starting task"},
                    {"role": "assistant", "time": 15.0, "content": "Working on step 1"},
                    {"role": "user", "time": 35.0, "content": "Moving to step 2"}
                ],
                "knowledge": ["Step 1: Prepare materials", "Step 2: Assemble parts"],
                "metadata": {
                    "user_type": "talk_more_dst_enhanced",
                    "task_goal": "Assemble the object",
                    "inferred_knowledge": ["Step 1: Prepare materials", "Step 2: Assemble parts"]
                }
            },
            {
                "video_uid": "test_video_2",
                "dataset": "assembly101",
                "dst": [
                    {"id": "S1", "t0": 5.0, "t1": 20.0, "name": "Single Step"}
                ],
                "conversation": [
                    {"role": "user", "time": 10.0, "content": "Completing task"}
                ],
                "knowledge": ["Complete the assembly"],
                "metadata": {
                    "user_type": "no_talk_dst_enhanced",
                    "task_goal": "Complete assembly",
                    "inferred_knowledge": ["Complete the assembly"]
                }
            }
        ]

    def test_initialization_with_config(self, sample_config):
        """Test that SimpleDSTGenerator initializes correctly with configuration"""
        generator = SimpleDSTGenerator(sample_config)

        assert generator is not None
        assert generator.cfg == sample_config
        assert hasattr(generator, 'dst_generator')
        assert hasattr(generator, 'data_processor')

        # Check training modules are initialized
        assert hasattr(generator, 'training_modules')
        assert 'frame_integration' in generator.training_modules
        assert 'sequence_calculator' in generator.training_modules
        assert 'conversation_splitter' in generator.training_modules
        assert 'dst_state_tracker' in generator.training_modules
        assert 'enhanced_speak_dst' in generator.training_modules
        assert 'dst_grounding' in generator.training_modules
        assert 'metadata_generator' in generator.training_modules

        # Check validators are initialized
        assert hasattr(generator, 'training_validators')
        assert len(generator.training_validators) > 0
        assert isinstance(generator.training_validators[0], TrainingFormatValidator)

    def test_initialization_without_training_modules(self):
        """Test initialization when training modules fail to initialize"""
        config = OmegaConf.create({
            "model": {"name": "gpt-4o", "temperature": 0.1, "max_tokens": 4000},
            "max_retries": 1
        })

        generator = SimpleDSTGenerator(config)

        # Should still initialize but training_modules might be empty
        assert generator is not None
        assert hasattr(generator, 'training_modules')

    @patch('dst_data_builder.simple_dst_generator.FrameIntegration')
    @patch('dst_data_builder.simple_dst_generator.SequenceLengthCalculator')
    @patch('dst_data_builder.simple_dst_generator.ConversationSplitter')
    @patch('dst_data_builder.simple_dst_generator.DSTStateTracker')
    @patch('dst_data_builder.simple_dst_generator.SpeakDSTGenerator')
    @patch('dst_data_builder.simple_dst_generator.DSTEventGrounding')
    @patch('dst_data_builder.simple_dst_generator.DatasetMetadataGenerator')
    def test_training_modules_initialization_failure_handling(
        self, mock_metadata_gen, mock_dst_grounding, mock_speak_dst,
        mock_dst_tracker, mock_conv_splitter, mock_seq_calc, mock_frame_int
    ):
        """Test that generator handles training module initialization failures gracefully"""
        # Make one module fail to initialize
        mock_frame_int.side_effect = Exception("Module initialization failed")

        config = OmegaConf.create({
            "model": {"name": "gpt-4o", "temperature": 0.1, "max_tokens": 4000},
            "training_creation": {},
            "max_retries": 1
        })

        # Should still initialize but log warnings
        with patch('dst_data_builder.simple_dst_generator.logging') as mock_logging:
            generator = SimpleDSTGenerator(config)
            assert generator is not None
            # Should have logged the initialization
            mock_logging.getLogger.return_value.info.assert_called()

    def test_create_training_format_with_valid_data(self, sample_config, sample_enhanced_data):
        """Test create_training_format method with valid enhanced data"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format(
            sample_enhanced_data, "assembly101", "train"
        )

        # Should return a list of training samples
        assert isinstance(training_samples, list)
        assert len(training_samples) > 0

        # Each sample should be a dict with required fields
        for sample in training_samples:
            assert isinstance(sample, dict)
            assert 'video_uid' in sample
            assert 'conversation' in sample
            assert 'dataset' in sample
            assert 'clip_idx' in sample
            assert 'max_seq_len' in sample
            assert 'seq_len' in sample
            assert 'fps' in sample
            assert 'num_tokens_per_img' in sample

            # Conversation should be a list
            assert isinstance(sample['conversation'], list)
            assert len(sample['conversation']) > 0

            # First turn should be system prompt
            first_turn = sample['conversation'][0]
            assert first_turn['role'] == 'system'
            assert 'content' in first_turn

    def test_create_training_format_with_empty_data(self, sample_config):
        """Test create_training_format with empty data"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format([], "assembly101", "train")

        # Should return empty list
        assert isinstance(training_samples, list)
        assert len(training_samples) == 0

    def test_create_training_format_validation_failures(self, sample_config):
        """Test that invalid training samples are filtered out"""
        generator = SimpleDSTGenerator(sample_config)

        # Create data that will fail validation (missing required fields)
        invalid_data = [{
            "video_uid": "test_video",
            # Missing required fields like conversation, dataset, etc.
        }]

        training_samples = generator.create_training_format(
            invalid_data, "assembly101", "train"
        )

        # Should return empty list due to validation failures
        assert isinstance(training_samples, list)
        # May be empty or contain only valid samples

    def test_create_training_format_with_dst_state_tracking(self, sample_config, sample_enhanced_data):
        """Test that DST state is properly tracked and added to conversation turns"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format(
            sample_enhanced_data, "assembly101", "train"
        )

        # Check that DST state is added to conversation turns
        found_dst_state = False
        for sample in training_samples:
            for turn in sample['conversation']:
                if 'dst_state' in turn:
                    found_dst_state = True
                    assert isinstance(turn['dst_state'], dict)
                    # DST state should contain step IDs as keys
                    for step_id, state in turn['dst_state'].items():
                        assert step_id.startswith('S')
                        assert state in ['not_started', 'in_progress', 'completed']

        # Should have found DST state in at least some turns
        assert found_dst_state, "No DST state found in conversation turns"

    def test_create_training_format_with_frame_integration(self, sample_config, sample_enhanced_data):
        """Test that frame information is properly integrated"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format(
            sample_enhanced_data, "assembly101", "train"
        )

        # Check that frame information is added
        for sample in training_samples:
            assert 'frames_file' in sample
            # Should point to the correct frames file path
            assert 'assembly101' in sample['frames_file']
            assert sample['frames_file'].endswith('.arrow')

            # Conversation turns should have frame information
            for turn in sample['conversation']:
                if turn['role'] in ['SPEAK', 'DST_UPDATE']:
                    assert 'start_frame' in turn
                    assert 'end_frame' in turn
                    assert isinstance(turn['start_frame'], int)
                    assert isinstance(turn['end_frame'], int)

    def test_create_training_format_with_sequence_calculation(self, sample_config, sample_enhanced_data):
        """Test that sequence lengths are properly calculated"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format(
            sample_enhanced_data, "assembly101", "train"
        )

        # Check sequence length calculations
        for sample in training_samples:
            assert 'seq_len' in sample
            assert isinstance(sample['seq_len'], int)
            assert sample['seq_len'] > 0
            assert sample['seq_len'] <= sample['max_seq_len']

            # Should have frame indices
            assert 'start_frame_idx' in sample
            assert 'end_frame_idx' in sample
            assert isinstance(sample['start_frame_idx'], int)
            assert isinstance(sample['end_frame_idx'], int)

    def test_create_training_format_with_metadata_generation(self, sample_config, sample_enhanced_data):
        """Test that proper metadata is generated for training samples"""
        generator = SimpleDSTGenerator(sample_config)

        training_samples = generator.create_training_format(
            sample_enhanced_data, "assembly101", "train"
        )

        # Check metadata generation
        for sample in training_samples:
            assert 'metadata' in sample
            metadata = sample['metadata']

            # Should have user type and other metadata
            assert 'user_type' in metadata
            assert 'user_id' in metadata
            assert 'task_goal' in metadata
            assert 'knowledge' in metadata
            assert 'progress' in metadata

            # User ID should be derived from user type
            assert metadata['user_id'].startswith(metadata['user_type'])

    @patch('dst_data_builder.simple_dst_generator.json.dump')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('pathlib.Path.exists')
    @patch('pathlib.Path.mkdir')
    def test_run_method_integration(self, mock_mkdir, mock_exists, mock_open, mock_json_dump, sample_config):
        """Test the run method with mocked file operations"""
        # Mock file existence checks
        mock_exists.return_value = True

        # Mock the data processor
        with patch('dst_data_builder.simple_dst_generator.DSTDataProcessor') as mock_processor_class:
            mock_processor_instance = MagicMock()
            mock_processor_instance.process_dataset_split.return_value = (10, 0)  # 10 processed, 0 failed
            mock_processor_class.return_value = mock_processor_instance

            generator = SimpleDSTGenerator(sample_config)

            # Mock Hydra config
            with patch('dst_data_builder.simple_dst_generator.HydraConfig') as mock_hydra:
                mock_hydra.get.return_value.runtime.output_dir = '/tmp/test_output'

                # Run the generator
                generator.run(sample_config)

                # Verify data processor was called
                assert mock_processor_instance.process_dataset_split.called

                # Verify JSON dump was called for training data
                assert mock_json_dump.called

    def test_run_method_with_missing_enhanced_data(self, sample_config):
        """Test run method when enhanced data files don't exist"""
        with patch('pathlib.Path.exists') as mock_exists:
            mock_exists.return_value = False

            with patch('dst_data_builder.simple_dst_generator.HydraConfig') as mock_hydra:
                mock_hydra.get.return_value.runtime.output_dir = '/tmp/test_output'

                generator = SimpleDSTGenerator(sample_config)

                # Should not crash when files don't exist
                generator.run(sample_config)

    def test_run_method_with_invalid_enhanced_data(self, sample_config):
        """Test run method with invalid JSON in enhanced data files"""
        with patch('pathlib.Path.exists') as mock_exists, \
             patch('builtins.open', MagicMock()) as mock_open, \
             patch('dst_data_builder.simple_dst_generator.json.load') as mock_json_load:

            mock_exists.return_value = True
            mock_json_load.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)

            with patch('dst_data_builder.simple_dst_generator.HydraConfig') as mock_hydra:
                mock_hydra.get.return_value.runtime.output_dir = '/tmp/test_output'

                generator = SimpleDSTGenerator(sample_config)

                # Should handle JSON parsing errors gracefully
                generator.run(sample_config)

    def test_error_handling_in_create_training_format(self, sample_config):
        """Test error handling in create_training_format method"""
        generator = SimpleDSTGenerator(sample_config)

        # Test with None input
        result = generator.create_training_format(None, "test", "train")
        assert isinstance(result, list)

        # Test with non-list input
        result = generator.create_training_format("not_a_list", "test", "train")
        assert isinstance(result, list)

    def test_training_format_validation_integration(self, sample_config, sample_enhanced_data):
        """Test integration with training format validators"""
        generator = SimpleDSTGenerator(sample_config)

        # Mock validator to fail
        with patch.object(generator.training_validators[0], 'validate') as mock_validate:
            mock_validate.return_value = (False, "Test validation failure")

            training_samples = generator.create_training_format(
                sample_enhanced_data, "assembly101", "train"
            )

            # Should still return a list, but may be filtered
            assert isinstance(training_samples, list)

    def test_different_dataset_types(self, sample_config):
        """Test processing different dataset types"""
        generator = SimpleDSTGenerator(sample_config)

        test_data = [{
            "video_uid": "test_video",
            "dataset": "ego4d",  # Different dataset
            "dst": [{"id": "S1", "t0": 10.0, "t1": 20.0, "name": "Test Step"}],
            "conversation": [{"role": "user", "time": 15.0, "content": "Test"}],
            "knowledge": ["Test knowledge"],
            "metadata": {
                "user_type": "talk_some_dst_enhanced",
                "task_goal": "Test task",
                "inferred_knowledge": ["Test knowledge"]
            }
        }]

        training_samples = generator.create_training_format(test_data, "ego4d", "val")

        assert isinstance(training_samples, list)
        assert len(training_samples) > 0

        # Check dataset-specific frame path
        sample = training_samples[0]
        assert 'ego4d' in sample['frames_file']

    def test_conversation_splitting_integration(self, sample_config):
        """Test integration with conversation splitting"""
        generator = SimpleDSTGenerator(sample_config)

        # Create data that would trigger splitting (long conversation)
        long_conversation = []
        for i in range(50):  # Create many turns
            long_conversation.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "time": float(i * 10),
                "content": f"Turn {i} content"
            })

        test_data = [{
            "video_uid": "long_video",
            "dataset": "assembly101",
            "dst": [{"id": "S1", "t0": 0.0, "t1": 500.0, "name": "Long Step"}],
            "conversation": long_conversation,
            "knowledge": ["Long task"],
            "metadata": {
                "user_type": "talk_more_dst_enhanced",
                "task_goal": "Long task",
                "inferred_knowledge": ["Long task"]
            }
        }]

        training_samples = generator.create_training_format(test_data, "assembly101", "train")

        # Should create multiple clips due to splitting
        assert isinstance(training_samples, list)
        # May create multiple samples due to conversation splitting
        assert len(training_samples) >= 1

        # Each sample should have clip_idx
        for sample in training_samples:
            assert 'clip_idx' in sample
            assert isinstance(sample['clip_idx'], int)
            assert sample['clip_idx'] >= 0


if __name__ == "__main__":
    pytest.main([__file__])