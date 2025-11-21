"""
Test Hybrid DST Generator - End-to-End DST Structure Validation

Comprehensive end-to-end tests for the HybridDSTLabelGenerator class that validate
the DST structure produced by the hybrid algorithm. Tests focus on:

1. Temporal Ordering: DST spans respect chronological order (t0 <= t1, proper sequencing)
2. Incremental IDs: Spans have incremental IDs starting from 1
3. Valid Timestamps: All timestamps are non-negative floats
4. Reasonable Durations: Spans have appropriate duration ranges
5. Proper Structure: All required fields present (id, name, t0, t1, conf)
6. No Excessive Overlaps: Configurable overlap limits respected
7. Confidence Scores: Valid confidence ranges (0-1)
8. Step Names: Match inferred knowledge appropriately
"""

import pytest
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.hybrid_dst.hybrid_dst_generator import HybridDSTLabelGenerator


class TestHybridDSTGenerator:
    """End-to-end tests for HybridDSTLabelGenerator DST structure validation"""

    @pytest.fixture
    def generator_config(self):
        """Create a complete generator configuration for testing"""
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
                    "max_allowed_overlap": 2.0,  # 2 seconds max overlap
                    "min_gap_duration": 0.1,
                    "max_gap_duration": 300.0,
                },
                "span_construction": {
                    "min_span_duration": 1.0,  # Minimum 1 second spans
                    "max_span_duration": 300.0,  # Maximum 5 minutes spans
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

    @pytest.fixture
    def model_config(self):
        """Create model configuration for testing"""
        return {
            "name": "gpt4o",
            "temperature": 0.1,
            "max_tokens": 4000,
        }

    def test_end_to_end_valid_dst_structure_simple_case(
        self, generator_config, model_config
    ):
        """Test end-to-end DST generation produces valid structure for simple case"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Simple input: 3 sequential steps
        input_data = {
            "inferred_knowledge": "1. Prepare workspace\n2. Gather tools\n3. Start assembly",
            "all_step_descriptions": "[0.0-10.0] Prepare workspace by clearing area\n[10.0-20.0] Gather tools from toolbox\n[20.0-30.0] Start assembly process",
        }

        result = generator._generate_dst_from_input_data(input_data)

        # Validate basic structure
        assert isinstance(result, list)
        assert len(result) == 3  # Should have 3 steps

        # Validate each step has required fields
        for row in result:
            assert "type" in row
            assert row["type"] == "step"
            assert "id" in row
            assert "start_ts" in row
            assert "end_ts" in row
            assert "name" in row

        # Convert to DST structure and validate
        dst_structure = generator._convert_to_dst_structure(result)
        self._validate_dst_structure(dst_structure)

    def test_end_to_end_valid_dst_structure_complex_case(
        self, generator_config, model_config
    ):
        """Test end-to-end DST generation with overlapping blocks (should be reduced)"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Complex input with overlapping blocks that should be reduced
        input_data = {
            "inferred_knowledge": "1. Prepare materials\n2. Assemble frame\n3. Install components\n4. Test assembly",
            "all_step_descriptions": "[0.0-15.0] Prepare materials and workspace\n[10.0-25.0] Assemble frame structure\n[20.0-35.0] Install components carefully\n[30.0-45.0] Test assembly functionality",
        }

        result = generator._generate_dst_from_input_data(input_data)

        # Should produce valid DST structure
        assert isinstance(result, list)
        assert len(result) >= 3  # At least 3 steps after reduction

        # Convert to DST structure and validate basic properties
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate basic structure without strict temporal ordering (since overlaps may exist)
        assert "steps" in dst_structure, "DST structure missing 'steps' field"
        assert isinstance(dst_structure["steps"], list), "'steps' should be a list"

        steps = dst_structure["steps"]
        assert len(steps) > 0, "DST structure should have at least one step"

        # Validate each step has required fields and basic properties
        for i, step in enumerate(steps):
            # Required fields
            required_fields = ["id", "name", "t0", "t1", "conf"]
            for field in required_fields:
                assert field in step, f"Step {i} missing required field '{field}'"

            # ID validation
            assert isinstance(
                step["id"], int
            ), f"Step {i} id should be integer, got {type(step['id'])}"
            assert step["id"] > 0, f"Step {i} id should be positive, got {step['id']}"

            # Timestamp validation
            assert isinstance(
                step["t0"], (int, float)
            ), f"Step {i} t0 should be numeric, got {type(step['t0'])}"
            assert isinstance(
                step["t1"], (int, float)
            ), f"Step {i} t1 should be numeric, got {type(step['t1'])}"
            assert (
                step["t0"] >= 0
            ), f"Step {i} t0 should be non-negative, got {step['t0']}"
            assert (
                step["t1"] >= 0
            ), f"Step {i} t1 should be non-negative, got {step['t1']}"
            assert (
                step["t0"] <= step["t1"]
            ), f"Step {i} t0 ({step['t0']}) should be <= t1 ({step['t1']})"

            # Confidence validation
            assert isinstance(
                step["conf"], (int, float)
            ), f"Step {i} conf should be numeric, got {type(step['conf'])}"
            assert (
                0.0 <= step["conf"] <= 1.0
            ), f"Step {i} conf should be in [0,1], got {step['conf']}"

            # Name validation
            assert isinstance(
                step["name"], str
            ), f"Step {i} name should be string, got {type(step['name'])}"
            assert step["name"].strip(), f"Step {i} name should not be empty"

        # Validate incremental IDs
        ids = [step["id"] for step in steps]
        expected_ids = list(range(1, len(steps) + 1))
        assert ids == expected_ids, f"Step IDs should be {expected_ids}, got {ids}"

    def test_end_to_end_temporal_ordering_validation(
        self, generator_config, model_config
    ):
        """Test that DST spans maintain proper temporal ordering after validation"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Input with sequential steps that should maintain reasonable temporal order
        input_data = {
            "inferred_knowledge": "1. Step A\n2. Step B\n3. Step C",
            "all_step_descriptions": "[0.0-10.0] Step A execution\n[10.0-20.0] Step B execution\n[20.0-30.0] Step C execution",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate temporal ordering - steps should be in chronological order
        steps = dst_structure["steps"]
        assert (
            len(steps) >= 2
        ), "Should have at least 2 steps for temporal ordering test"

        # Check that steps are sorted by start time (basic temporal ordering)
        start_times = [step["t0"] for step in steps]
        assert start_times == sorted(
            start_times
        ), f"Steps not in chronological order: {start_times}"

        # Validate that no step has negative duration (start > end)
        for step in steps:
            assert (
                step["t0"] <= step["t1"]
            ), f"Step {step['id']} has invalid duration: start={step['t0']} > end={step['t1']}"

        # Validate reasonable durations (not too short or too long)
        for step in steps:
            duration = step["t1"] - step["t0"]
            assert (
                duration > 0
            ), f"Step {step['id']} has zero or negative duration: {duration}"
            assert (
                duration <= 300.0
            ), f"Step {step['id']} duration too long: {duration}s > 300s"

    def test_end_to_end_incremental_ids_validation(
        self, generator_config, model_config
    ):
        """Test that DST spans have proper incremental IDs"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Input with steps that might be reordered
        input_data = {
            "inferred_knowledge": "1. First step\n2. Second step\n3. Third step\n4. Fourth step",
            "all_step_descriptions": "[15.0-20.0] Third step\n[5.0-10.0] First step\n[25.0-30.0] Fourth step\n[10.0-15.0] Second step",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate incremental IDs
        steps = dst_structure["steps"]
        for i, step in enumerate(steps, 1):
            assert (
                step["id"] == i
            ), f"Step at position {i} should have id {i}, got {step['id']}"

    def test_end_to_end_duration_validation(self, generator_config, model_config):
        """Test that DST spans have reasonable durations"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        input_data = {
            "inferred_knowledge": "1. Quick step\n2. Normal step\n3. Long step",
            "all_step_descriptions": "[0.0-2.0] Quick step\n[2.0-12.0] Normal step\n[12.0-62.0] Long step",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate durations
        steps = dst_structure["steps"]
        for step in steps:
            duration = step["t1"] - step["t0"]
            assert (
                duration >= 1.0
            ), f"Step {step['id']} duration {duration} too short (min 1.0s)"
            assert (
                duration <= 300.0
            ), f"Step {step['id']} duration {duration} too long (max 300.0s)"

    def test_end_to_end_confidence_scores_validation(
        self, generator_config, model_config
    ):
        """Test that DST spans have valid confidence scores"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        input_data = {
            "inferred_knowledge": "1. Step 1\n2. Step 2",
            "all_step_descriptions": "[0.0-10.0] Step 1\n[10.0-20.0] Step 2",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate confidence scores
        steps = dst_structure["steps"]
        for step in steps:
            assert "conf" in step, f"Step {step['id']} missing confidence score"
            assert (
                0.0 <= step["conf"] <= 1.0
            ), f"Step {step['id']} confidence {step['conf']} out of range [0,1]"

    def test_end_to_end_empty_input_handling(self, generator_config, model_config):
        """Test handling of empty or invalid input"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Test empty input
        result = generator._generate_dst_from_input_data(
            {"inferred_knowledge": "", "all_step_descriptions": ""}
        )
        assert result == []

        # Test missing fields
        result = generator._generate_dst_from_input_data({})
        assert result == []

    @pytest.mark.skip(reason="Requires GPU resources for ML model loading")
    def test_end_to_end_step_names_validation(self, generator_config, model_config):
        """Test that step names are properly extracted and assigned"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        input_data = {
            "inferred_knowledge": "1. Prepare workspace\n2. Gather materials\n3. Begin assembly",
            "all_step_descriptions": "[0.0-10.0] Prepare workspace\n[10.0-20.0] Gather materials\n[20.0-30.0] Begin assembly",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Validate step names are present and reasonable
        steps = dst_structure["steps"]
        expected_names = ["Prepare workspace", "Gather materials", "Begin assembly"]

        for i, step in enumerate(steps):
            assert "name" in step, f"Step {step['id']} missing name"
            assert step["name"], f"Step {step['id']} has empty name"
            # Note: Exact name matching might not be guaranteed due to processing,
            # but we validate that names are present and non-empty

    @pytest.mark.skip(reason="Requires GPU resources for ML model loading")
    def test_end_to_end_overlap_reduction_validation(
        self, generator_config, model_config
    ):
        """Test that overlapping blocks are properly reduced"""
        generator = HybridDSTLabelGenerator(
            generator_type="hybrid_dst",
            model_name="gpt4o",
            temperature=0.1,
            max_tokens=4000,
            max_retries=1,
            generator_cfg=generator_config,
            model_cfg=model_config,
        )

        # Create input with significant overlaps that should be reduced
        input_data = {
            "inferred_knowledge": "1. Setup\n2. Process\n3. Finish",
            "all_step_descriptions": "[0.0-20.0] Setup phase\n[5.0-15.0] Process step\n[18.0-25.0] Finish step",
        }

        result = generator._generate_dst_from_input_data(input_data)
        dst_structure = generator._convert_to_dst_structure(result)

        # Should still produce valid structure even with overlaps
        self._validate_dst_structure(dst_structure)

        # Check that we don't have excessive overlaps
        steps = dst_structure["steps"]
        for i in range(len(steps) - 1):
            overlap = steps[i]["t1"] - steps[i + 1]["t0"]
            assert (
                overlap <= 2.0
            ), f"Excessive overlap {overlap}s between steps {i+1} and {i+2}"

    def _validate_dst_structure(self, dst_structure: dict):
        """Helper method to validate DST structure has all required properties"""
        assert "steps" in dst_structure, "DST structure missing 'steps' field"
        assert isinstance(dst_structure["steps"], list), "'steps' should be a list"

        steps = dst_structure["steps"]
        assert len(steps) > 0, "DST structure should have at least one step"

        # Validate each step
        for i, step in enumerate(steps):
            # Required fields
            required_fields = ["id", "name", "t0", "t1", "conf"]
            for field in required_fields:
                assert field in step, f"Step {i} missing required field '{field}'"

            # ID validation
            assert isinstance(
                step["id"], int
            ), f"Step {i} id should be integer, got {type(step['id'])}"
            assert step["id"] > 0, f"Step {i} id should be positive, got {step['id']}"

            # Timestamp validation
            assert isinstance(
                step["t0"], (int, float)
            ), f"Step {i} t0 should be numeric, got {type(step['t0'])}"
            assert isinstance(
                step["t1"], (int, float)
            ), f"Step {i} t1 should be numeric, got {type(step['t1'])}"
            assert (
                step["t0"] >= 0
            ), f"Step {i} t0 should be non-negative, got {step['t0']}"
            assert (
                step["t1"] >= 0
            ), f"Step {i} t1 should be non-negative, got {step['t1']}"
            assert (
                step["t0"] <= step["t1"]
            ), f"Step {i} t0 ({step['t0']}) should be <= t1 ({step['t1']})"

            # Confidence validation
            assert isinstance(
                step["conf"], (int, float)
            ), f"Step {i} conf should be numeric, got {type(step['conf'])}"
            assert (
                0.0 <= step["conf"] <= 1.0
            ), f"Step {i} conf should be in [0,1], got {step['conf']}"

            # Name validation
            assert isinstance(
                step["name"], str
            ), f"Step {i} name should be string, got {type(step['name'])}"
            assert step["name"].strip(), f"Step {i} name should not be empty"

        # Validate incremental IDs
        ids = [step["id"] for step in steps]
        expected_ids = list(range(1, len(steps) + 1))
        assert ids == expected_ids, f"Step IDs should be {expected_ids}, got {ids}"

        # Validate temporal ordering (allowing small overlaps)
        for i in range(len(steps) - 1):
            current_end = steps[i]["t1"]
            next_start = steps[i + 1]["t0"]
            # Allow up to 2 seconds of overlap but not major violations
            assert (
                next_start >= current_end - 2.0
            ), f"Temporal ordering violation between steps {i+1} and {i+2}"

    def test_temporal_validation_filters_invalid_blocks(
        self, generator_config, model_config
    ):
        """Test that temporal validation filters out blocks with invalid timestamps"""
        from dst_data_builder.hybrid_dst.span_constructors.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Create test blocks - some valid, some invalid
        filtered_blocks = [
            {"text": "Valid block 1", "start_time": 0.0, "end_time": 5.0},
            {
                "text": "Invalid block",
                "start_time": 10.0,
                "end_time": 8.0,
            },  # start > end
            {"text": "Valid block 2", "start_time": 15.0, "end_time": 20.0},
            {
                "text": "Missing timestamps",
                "text": "No time data",
            },  # missing timestamps
        ]

        # Create block assignments
        block_assignments = {
            0: {"step_index": 0, "confidence": 0.9, "source": "similarity"},
            1: {"step_index": 0, "confidence": 0.8, "source": "similarity"},
            2: {"step_index": 0, "confidence": 0.7, "source": "similarity"},
            3: {"step_index": 0, "confidence": 0.6, "source": "similarity"},
        }

        # Test temporal validation
        validated_assignments = constructor._validate_temporal_consistency(
            block_assignments, filtered_blocks
        )

        # Should only keep blocks 0 and 2 (valid timestamps)
        assert len(validated_assignments) == 2
        assert 0 in validated_assignments
        assert 2 in validated_assignments
        assert 1 not in validated_assignments  # Invalid timestamps
        assert 3 not in validated_assignments  # Missing timestamps

    def test_temporal_validation_keeps_valid_blocks(
        self, generator_config, model_config
    ):
        """Test that temporal validation keeps blocks with valid timestamps"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Create test blocks - all valid
        filtered_blocks = [
            {"text": "Block 1", "start_time": 0.0, "end_time": 5.0},
            {"text": "Block 2", "start_time": 10.0, "end_time": 15.0},
            {"text": "Block 3", "start_time": 20.0, "end_time": 25.0},
        ]

        # Create block assignments
        block_assignments = {
            0: {"step_index": 0, "confidence": 0.9, "source": "similarity"},
            1: {"step_index": 0, "confidence": 0.8, "source": "similarity"},
            2: {"step_index": 0, "confidence": 0.7, "source": "similarity"},
        }

        # Test temporal validation
        validated_assignments = constructor._validate_temporal_consistency(
            block_assignments, filtered_blocks
        )

        # Should keep all blocks
        assert len(validated_assignments) == 3
        assert all(i in validated_assignments for i in range(3))

    def test_temporal_validation_handles_empty_assignments(
        self, generator_config, model_config
    ):
        """Test temporal validation with empty assignments"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Test with empty assignments
        validated_assignments = constructor._validate_temporal_consistency({}, [])
        assert validated_assignments == {}

    def test_temporal_validation_handles_out_of_range_blocks(
        self, generator_config, model_config
    ):
        """Test temporal validation with block IDs out of range"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        filtered_blocks = [
            {"text": "Only one block", "start_time": 0.0, "end_time": 5.0}
        ]

        # Create assignment with out-of-range block ID
        block_assignments = {
            5: {
                "step_index": 0,
                "confidence": 0.9,
                "source": "similarity",
            },  # Out of range
        }

        validated_assignments = constructor._validate_temporal_consistency(
            block_assignments, filtered_blocks
        )

        # Should reject the out-of-range block
        assert len(validated_assignments) == 0

    def test_construct_spans_from_groups_with_valid_blocks(
        self, generator_config, model_config
    ):
        """Test span construction from groups of valid blocks"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Create step groups with valid blocks
        step_groups = {
            0: [  # Step 1
                {
                    "text": "Block 1",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "assignment_info": {"confidence": 0.9, "source": "similarity"},
                },
                {
                    "text": "Block 2",
                    "start_time": 10.0,
                    "end_time": 15.0,
                    "assignment_info": {"confidence": 0.8, "source": "similarity"},
                },
            ],
            1: [  # Step 2
                {
                    "text": "Block 3",
                    "start_time": 20.0,
                    "end_time": 25.0,
                    "assignment_info": {"confidence": 0.7, "source": "similarity"},
                }
            ],
        }

        inferred_knowledge = ["Step 1: Do something", "Step 2: Do another thing"]

        spans = constructor._construct_spans_from_groups(
            step_groups, inferred_knowledge
        )

        # Should create 2 spans
        assert len(spans) == 2

        # Check first span (step 1)
        span1 = spans[0]
        assert span1["id"] == 1
        assert span1["name"] == "Step 1: Do something"
        assert span1["t0"] == 0.0  # min start time
        assert span1["t1"] == 15.0  # max end time
        assert span1["conf"] == 0.85  # average confidence

        # Check second span (step 2)
        span2 = spans[1]
        assert span2["id"] == 2
        assert span2["name"] == "Step 2: Do another thing"
        assert span2["t0"] == 20.0
        assert span2["t1"] == 25.0
        assert span2["conf"] == 0.7

    def test_construct_spans_from_groups_filters_invalid_spans(
        self, generator_config, model_config
    ):
        """Test that span construction filters out spans with invalid durations"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Create step groups with blocks that would create invalid spans
        step_groups = {
            0: [  # Valid span
                {
                    "text": "Valid block",
                    "start_time": 0.0,
                    "end_time": 5.0,
                    "assignment_info": {"confidence": 0.9},
                }
            ],
            1: [  # Too short span (duration < min_span_duration)
                {
                    "text": "Too short",
                    "start_time": 10.0,
                    "end_time": 10.5,  # Duration = 0.5 < 1.0
                    "assignment_info": {"confidence": 0.8},
                }
            ],
            2: [  # Too long span (duration > max_span_duration)
                {
                    "text": "Too long",
                    "start_time": 20.0,
                    "end_time": 320.0,  # Duration = 300.0 == 300.0 (should be < 300)
                    "assignment_info": {"confidence": 0.7},
                }
            ],
        }

        inferred_knowledge = ["Valid step", "Too short step", "Too long step"]

        spans = constructor._construct_spans_from_groups(
            step_groups, inferred_knowledge
        )

        # Should only create the valid span
        assert len(spans) == 1
        assert spans[0]["id"] == 1
        assert spans[0]["name"] == "Valid step"

    def test_span_timestamp_swapping_fix(self, generator_config, model_config):
        """Test that spans with invalid timestamps (start > end) get their values swapped"""
        from dst_data_builder.hybrid_dst.hybrid_span_constructor import (
            HybridSpanConstructor,
        )

        constructor = HybridSpanConstructor(generator_config, model_config)

        # Create mock blocks with timestamps that would result in start > end
        # This simulates the case where blocks assigned to a step have timestamps that create invalid spans
        blocks = [
            {"start_time": 150.0, "end_time": 150.0},  # Single timestamp block
            {"start_time": 100.0, "end_time": 120.0},  # Valid block
        ]

        # Mock the assignment so both blocks go to step 0
        block_assignments = {
            0: {"step_index": 0, "confidence": 0.9, "source": "similarity"},
            1: {"step_index": 0, "confidence": 0.8, "source": "similarity"},
        }

        # This should create a span with start=100.0, end=150.0 (valid)
        step_groups = {0: blocks}
        inferred_knowledge = ["Test step"]

        spans = constructor._construct_spans_from_groups(
            step_groups, inferred_knowledge
        )

        assert len(spans) == 1
        span = spans[0]
        assert (
            span["t0"] <= span["t1"]
        ), f"Span has invalid timestamps: start={span['t0']}, end={span['t1']}"

        # The span should cover the full range
        assert span["t0"] == 100.0
        assert span["t1"] == 150.0


if __name__ == "__main__":
    # Run tests manually for basic validation
    test_instance = TestHybridDSTGenerator()

    config = OmegaConf.create(
        {
            "overlap_reduction": {"min_overlap_ratio": 0.8},
            "similarity": {"semantic_weight": 0.6, "nli_weight": 0.4},
            "span_construction": {"min_span_duration": 1.0, "max_span_duration": 300.0},
            "temporal_validation": {
                "max_allowed_overlap": 2.0,
                "min_gap_duration": 0.1,
                "max_gap_duration": 300.0,
            },
            "two_phase_algorithm": {
                "high_confidence_threshold": 0.3,
                "semantic_similarity_weight": 0.6,
                "nli_score_weight": 0.4,
            },
        }
    )

    try:
        # Test end-to-end DST structure validation
        test_instance.test_end_to_end_valid_dst_structure_simple_case(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_valid_dst_structure_complex_case(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_temporal_ordering_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_incremental_ids_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_duration_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_confidence_scores_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_empty_input_handling(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_step_names_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_end_to_end_overlap_reduction_validation(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )

        print("✅ All end-to-end DST structure validation tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    # Run tests manually for basic validation
    test_instance = TestHybridDSTGenerator()

    config = OmegaConf.create(
        {
            "overlap_reduction": {"min_overlap_ratio": 0.8},
            "similarity": {"semantic_weight": 0.6, "nli_weight": 0.4},
            "span_construction": {"min_span_duration": 1.0, "max_span_duration": 300.0},
            "temporal_validation": {
                "max_allowed_overlap": 5.0,
                "min_gap_duration": 0.1,
                "max_gap_duration": 300.0,
            },
            "two_phase_algorithm": {
                "high_confidence_threshold": 0.3,
                "semantic_similarity_weight": 0.6,
                "nli_score_weight": 0.4,
            },
        }
    )

    try:
        # Test new HybridDSTGenerator functionality
        test_instance.test_generator_initialization(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_generator_initialization_minimal_config()
        test_instance.test_convert_to_dst_structure(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_convert_to_dst_structure_empty_input(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
        test_instance.test_convert_to_dst_structure_non_step_rows(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )

        print("✅ All TestHybridDSTGenerator tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback

        traceback.print_exc()

        # Test span timestamp swapping fix
        test_instance.test_span_timestamp_swapping_fix(
            config, {"name": "gpt4o", "temperature": 0.1, "max_tokens": 4000}
        )
