"""
Test Hybrid DST Modules

Comprehensive tests for hybrid DST components including the main generator.
"""

import pytest
import logging
from unittest.mock import Mock, patch, MagicMock
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.hybrid_dst.hybrid_dst_generator import HybridDSTLabelGenerator
from dst_data_builder.hybrid_dst.utils import (
    parse_blocks,
    extract_steps,
    convert_spans_to_rows,
)


class TestHybridDSTModules:
    """Test class for hybrid DST modules integration"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create(
            {
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
                "two_phase_algorithm": {
                    "high_confidence_threshold": 0.3,
                    "semantic_similarity_weight": 0.6,
                    "nli_score_weight": 0.4,
                },
            }
        )

    def test_hybrid_dst_generator_initialization(self, sample_config):
        """Test that HybridDSTLabelGenerator can be initialized"""
        model_cfg = {
            "name": "gpt4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=sample_config,
            model_cfg=model_cfg,
        )
        assert generator is not None
        assert generator.generator_type == "hybrid_dst"
        assert hasattr(generator, "overlap_reducer")
        assert hasattr(generator, "span_constructor")
        assert hasattr(generator, "temporal_validator")

    def test_utils_parse_blocks(self):
        """Test utility function for parsing blocks"""
        step_descriptions = "[0.0-5.0] User says hello\n[5.0-10.0] Assistant responds"
        blocks = parse_blocks(step_descriptions)

        assert len(blocks) == 2
        assert blocks[0]["text"] == "User says hello"
        assert blocks[0]["start_time"] == 0.0
        assert blocks[0]["end_time"] == 5.0
        assert blocks[1]["text"] == "Assistant responds"
        assert blocks[1]["start_time"] == 5.0
        assert blocks[1]["end_time"] == 10.0

    def test_utils_extract_steps(self):
        """Test utility function for extracting steps"""
        inferred_knowledge = "1. Prepare workspace\n2. Gather tools\n3. Start assembly"
        steps = extract_steps(inferred_knowledge)

        assert len(steps) == 3
        assert steps[0] == "Prepare workspace"
        assert steps[1] == "Gather tools"
        assert steps[2] == "Start assembly"

    def test_utils_convert_spans_to_rows(self):
        """Test utility function for converting spans to rows"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1", "conf": 1.0},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2", "conf": 0.9},
        ]
        rows = convert_spans_to_rows(spans)

        assert len(rows) == 2
        assert rows[0]["type"] == "step"
        assert rows[0]["id"] == "S1"
        assert rows[0]["start_ts"] == 0.0
        assert rows[0]["end_ts"] == 5.0
        assert rows[0]["name"] == "Step 1"

    def test_module_integration(self, sample_config):
        """Test full pipeline integration"""
        # This is a basic integration test - in practice you'd want more comprehensive tests
        model_cfg = {
            "name": "gpt4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=sample_config,
            model_cfg=model_cfg,
        )

        # Test that all components are properly initialized
        assert generator.overlap_reducer is not None
        assert generator.span_constructor is not None
        assert generator.temporal_validator is not None

        # Test basic utility functions
        step_descriptions = "[0.0-5.0] User says hello\n[5.0-10.0] Assistant responds"
        inferred_knowledge = "1. Greet user\n2. Provide assistance"

        blocks = parse_blocks(step_descriptions)
        steps = extract_steps(inferred_knowledge)

        assert len(blocks) > 0
        assert len(steps) > 0

    def test_single_system_prompt(self):
        """Test that only one system prompt is added even if conversation already has one"""
        from dst_data_builder.training_modules.speak_dst_generator import SpeakDSTGenerator
        from omegaconf import DictConfig

        # Create a mock config
        config = DictConfig({"training_creation": {"include_system_prompt": True}})

        generator = SpeakDSTGenerator(config)

        # Test conversation that already has a system prompt
        conversation_with_system = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"}
        ]

        video_data = {"conversation": conversation_with_system}
        result = generator.create_training_conversation(video_data)

        # Should still have only one system prompt
        system_turns = [turn for turn in result["conversation"] if turn.get("role") == "system"]
        assert len(system_turns) == 1
        assert system_turns[0]["content"] == "You are a helpful assistant."

    def test_dst_state_in_conversation_turns(self):
        """Test that DST state is added to each conversation turn"""
        from dst_data_builder.training_modules.dst_event_grounding import DSTEventGrounding
        from omegaconf import DictConfig

        config = DictConfig({"training_creation": {"enable_dst_labels": True}})
        grounder = DSTEventGrounding(config)

        # Mock conversation with DST updates
        conversation = [
            {"role": "DST_UPDATE", "time": 0, "content": [{"id": 1, "transition": "start"}]},
            {"role": "assistant", "time": 5, "content": "Starting step 1"},
            {"role": "DST_UPDATE", "time": 10, "content": [{"id": 1, "transition": "complete"}]},
        ]

        video_data = {"conversation": conversation}
        result = grounder.add_frames_and_labels(video_data)

        # Check that DST state is added to turns
        for turn in result["conversation"]:
            assert "dst_state" in turn
            assert isinstance(turn["dst_state"], dict)

    def test_incremental_dst_ids_sorted_by_time(self):
        """Test that DST spans have incremental IDs sorted by start time"""
        from dst_data_builder.hybrid_dst.hybrid_dst_generator import HybridDSTLabelGenerator

        # Create spans with non-incremental IDs and wrong temporal order
        spans = [
            {"id": 5, "name": "Step 5", "t0": 20.0, "t1": 25.0, "conf": 1.0},
            {"id": 2, "name": "Step 2", "t0": 5.0, "t1": 10.0, "conf": 1.0},
            {"id": 8, "name": "Step 8", "t0": 15.0, "t1": 20.0, "conf": 1.0},
        ]

        # Convert to TSV format first (simulating what the pipeline does)
        tsv_rows = []
        for span in spans:
            tsv_rows.append({
                "type": "step",
                "id": f'S{span["id"]}',
                "start_ts": span["t0"],
                "end_ts": span["t1"],
                "name": span["name"],
            })

        # Create generator and test the conversion
        model_cfg = {
            "name": "gpt4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        }
        # Create a minimal config for testing
        test_config = OmegaConf.create({
            "overlap_reduction": {"min_overlap_ratio": 0.8},
            "similarity": {"semantic_weight": 0.6, "nli_weight": 0.4},
            "span_construction": {"min_span_duration": 1.0, "max_span_duration": 300.0},
            "temporal_validation": {"max_allowed_overlap": 5.0, "min_gap_duration": 0.1, "max_gap_duration": 300.0},
            "two_phase_algorithm": {"high_confidence_threshold": 0.3, "semantic_similarity_weight": 0.6, "nli_score_weight": 0.4},
        })

        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=test_config,
            model_cfg=model_cfg,
        )

        result = generator._convert_to_dst_structure(tsv_rows)

        # Should be sorted by start time and have incremental IDs
        assert len(result["steps"]) == 3

        # Check temporal ordering
        assert result["steps"][0]["t0"] == 5.0   # Step 2
        assert result["steps"][1]["t0"] == 15.0  # Step 8
        assert result["steps"][2]["t0"] == 20.0  # Step 5

        # Check incremental IDs
        assert result["steps"][0]["id"] == 1
        assert result["steps"][1]["id"] == 2
        assert result["steps"][2]["id"] == 3


if __name__ == "__main__":
    # Run tests manually for basic validation
    test_instance = TestHybridDSTModules()

    config = OmegaConf.create(
        {
            "overlap_reduction": {"min_overlap_ratio": 0.8},
            "similarity": {"semantic_weight": 0.6, "nli_weight": 0.4},
            "span_construction": {"min_span_duration": 1.0, "max_span_duration": 300.0},
            "temporal_validation": {"max_allowed_overlap": 5.0, "min_gap_duration": 0.1, "max_gap_duration": 300.0},
            "two_phase_algorithm": {"high_confidence_threshold": 0.3, "semantic_similarity_weight": 0.6, "nli_score_weight": 0.4},
        }
    )

    try:
        # Test existing functionality
        test_instance.test_hybrid_dst_generator_initialization(config)
        test_instance.test_utils_parse_blocks()
        test_instance.test_utils_extract_steps()
        test_instance.test_utils_convert_spans_to_rows()
        test_instance.test_module_integration(config)
        test_instance.test_single_system_prompt()
        test_instance.test_dst_state_in_conversation_turns()
        test_instance.test_incremental_dst_ids_sorted_by_time()


        print("✅ All comprehensive tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()
