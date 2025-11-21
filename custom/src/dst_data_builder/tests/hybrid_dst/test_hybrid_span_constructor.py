"""
Test Hybrid Span Constructor

Comprehensive tests for the HybridSpanConstructor component that orchestrates
the two-phase DST span construction process combining similarity scoring with LLM fallback.
"""

import pytest
from unittest.mock import Mock, patch
from omegaconf import OmegaConf
import numpy as np

from dst_data_builder.hybrid_dst.span_constructors import (
    HybridSpanConstructor,
    SpanConstructionResult,
    ClassificationResult,
    SimilarityResult,
    LLMDecision,
    LLMHandlingResult,
)


class TestHybridSpanConstructor:
    """Test class for HybridSpanConstructor functionality"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create(
            {
                "similarity": {
                    "semantic_weight": 0.6,
                    "nli_weight": 0.4,
                    "high_confidence_threshold": 0.3,
                },
                "span_construction": {
                    "min_span_duration": 1.0,
                    "max_span_duration": 300.0,
                },
            }
        )

    @pytest.fixture
    def model_config(self):
        """Create sample model configuration"""
        return OmegaConf.create(
            {
                "name": "gpt4o",
                "temperature": 0.1,
                "max_tokens": 4000,
            }
        )

    @pytest.fixture
    def sample_blocks(self):
        """Create sample filtered blocks for testing"""
        return [
            {
                "text": "Prepare workspace",
                "start_time": 0.0,
                "end_time": 10.0,
                "merged_blocks": 1,
                "original_id": 0,
            },
            {
                "text": "Gather tools",
                "start_time": 10.0,
                "end_time": 20.0,
                "merged_blocks": 1,
                "original_id": 1,
            },
            {
                "text": "Start assembly",
                "start_time": 20.0,
                "end_time": 30.0,
                "merged_blocks": 1,
                "original_id": 2,
            },
        ]

    @pytest.fixture
    def sample_knowledge(self):
        """Create sample inferred knowledge"""
        return [
            "Prepare workspace",
            "Gather tools",
            "Start assembly",
        ]

    def test_initialization(self, sample_config, model_config):
        """Test that HybridSpanConstructor initializes correctly"""
        constructor = HybridSpanConstructor(sample_config, model_config)

        assert constructor.hybrid_config == sample_config
        assert constructor.model_config == model_config
        assert hasattr(constructor, "similarity_calculator")
        assert hasattr(constructor, "llm_handler")
        assert constructor.min_span_duration == 1.0
        assert constructor.max_span_duration == 300.0

    def test_empty_input(self, sample_config, model_config):
        """Test handling of empty input"""
        constructor = HybridSpanConstructor(sample_config, model_config)

        result = constructor.construct_spans([], [])
        assert result.dst_spans == []
        assert result.clear_blocks_used == 0
        assert result.ambiguous_blocks_resolved == 0
        assert result.total_blocks_processed == 0

    def test_empty_knowledge(self, sample_config, model_config, sample_blocks):
        """Test handling when no knowledge is provided"""
        constructor = HybridSpanConstructor(sample_config, model_config)

        result = constructor.construct_spans(sample_blocks, [])
        assert result.dst_spans == []
        assert result.clear_blocks_used == 0
        assert result.ambiguous_blocks_resolved == 0
        assert result.total_blocks_processed == 3
        assert "error" in result.construction_statistics

    @patch(
        "dst_data_builder.hybrid_dst.global_similarity_calculator.GlobalSimilarityCalculator"
    )
    def test_clear_blocks_only(
        self,
        mock_similarity_calculator,
        sample_config,
        model_config,
        sample_blocks,
        sample_knowledge,
    ):
        """Test processing when all blocks are clear (no LLM needed)"""
        # Mock similarity calculator to return all clear blocks
        mock_calc = Mock()
        mock_calc.score_blocks.return_value = ClassificationResult(
            clear_blocks=[
                SimilarityResult(
                    block_id=0,
                    similarity_scores=[0.9, 0.1, 0.0],
                    nli_scores=[0.8, 0.0, -0.2],
                    combined_scores=[0.9, 0.1, 0.0],
                    confidence=0.9,
                    is_clear=True,
                ),
                SimilarityResult(
                    block_id=1,
                    similarity_scores=[0.0, 0.95, 0.05],
                    nli_scores=[-0.1, 0.9, 0.0],
                    combined_scores=[0.0, 0.95, 0.05],
                    confidence=0.95,
                    is_clear=True,
                ),
                SimilarityResult(
                    block_id=2,
                    similarity_scores=[0.0, 0.0, 0.98],
                    nli_scores=[0.0, 0.0, 0.9],
                    combined_scores=[0.0, 0.0, 0.98],
                    confidence=0.98,
                    is_clear=True,
                ),
            ],
            ambiguous_blocks=[],
            clear_count=3,
            ambiguous_count=0,
            total_blocks=3,
        )

        constructor = HybridSpanConstructor(sample_config, model_config)
        constructor.similarity_calculator = mock_calc

        result = constructor.construct_spans(sample_blocks, sample_knowledge)

        # Verify results
        assert len(result.dst_spans) == 3
        assert result.clear_blocks_used == 3
        assert result.ambiguous_blocks_resolved == 0
        assert result.total_blocks_processed == 3

        # Check spans
        spans = result.dst_spans
        assert spans[0]["id"] == 1  # Step 1 (0-indexed + 1)
        assert spans[0]["name"] == "Prepare workspace"
        assert spans[0]["t0"] == 0.0
        assert spans[0]["t1"] == 10.0
        assert spans[0]["conf"] == 0.9

        assert spans[1]["id"] == 2
        assert spans[1]["name"] == "Gather tools"
        assert spans[1]["t0"] == 10.0
        assert spans[1]["t1"] == 20.0

        assert spans[2]["id"] == 3
        assert spans[2]["name"] == "Start assembly"
        assert spans[2]["t0"] == 20.0
        assert spans[2]["t1"] == 30.0

    @patch(
        "dst_data_builder.hybrid_dst.span_constructors.global_similarity_calculator.GlobalSimilarityCalculator"
    )
    @patch(
        "dst_data_builder.hybrid_dst.span_constructors.llm_ambiguous_handler.LLMAmbiguousHandler"
    )
    def test_mixed_clear_and_ambiguous_blocks(
        self,
        mock_llm_handler,
        mock_similarity_calculator,
        sample_config,
        model_config,
        sample_blocks,
        sample_knowledge,
    ):
        """Test processing with both clear and ambiguous blocks requiring LLM"""
        # Mock similarity calculator
        mock_calc = Mock()
        mock_calc.score_blocks.return_value = ClassificationResult(
            clear_blocks=[
                SimilarityResult(
                    block_id=0,
                    similarity_scores=[0.9, 0.1, 0.0],
                    nli_scores=[0.8, 0.0, -0.2],
                    combined_scores=[0.9, 0.1, 0.0],
                    confidence=0.9,
                    is_clear=True,
                ),
            ],
            ambiguous_blocks=[
                SimilarityResult(
                    block_id=1,
                    similarity_scores=[0.3, 0.4, 0.3],
                    nli_scores=[0.2, 0.3, 0.4],
                    combined_scores=[0.3, 0.4, 0.3],  # Ambiguous scores
                    confidence=0.15,
                    is_clear=False,
                ),
                SimilarityResult(
                    block_id=2,
                    similarity_scores=[0.2, 0.2, 0.6],
                    nli_scores=[0.1, 0.1, 0.5],
                    combined_scores=[0.2, 0.2, 0.6],  # Ambiguous scores
                    confidence=0.18,
                    is_clear=False,
                ),
            ],
            clear_count=1,
            ambiguous_count=2,
            total_blocks=3,
        )

        # Mock LLM handler
        mock_llm = Mock()
        mock_llm.resolve_ambiguous_blocks.return_value = LLMHandlingResult(
            decisions=[
                LLMDecision(
                    block_id=1,
                    chosen_step_index=1,  # Correctly assigned to step 2
                    confidence=0.8,
                    reasoning="Block content matches 'Gather tools'",
                ),
                LLMDecision(
                    block_id=2,
                    chosen_step_index=2,  # Correctly assigned to step 3
                    confidence=0.85,
                    reasoning="Block content matches 'Start assembly'",
                ),
            ],
            total_llm_calls=2,
            success_count=2,
            failure_count=0,
            total_cost_estimate=0.02,
        )

        constructor = HybridSpanConstructor(sample_config, model_config)
        constructor.similarity_calculator = mock_calc
        constructor.llm_handler = mock_llm

        result = constructor.construct_spans(sample_blocks, sample_knowledge)

        # Verify results
        assert len(result.dst_spans) == 3
        assert result.clear_blocks_used == 1
        assert result.ambiguous_blocks_resolved == 2
        assert result.total_blocks_processed == 3

        # Check that LLM was called
        mock_llm.resolve_ambiguous_blocks.assert_called_once()

    def test_span_duration_validation(self, sample_config, model_config):
        """Test that spans are validated for duration constraints"""
        # Create config with very restrictive duration limits
        restrictive_config = OmegaConf.create(
            {
                "similarity": {
                    "semantic_weight": 0.6,
                    "nli_weight": 0.4,
                    "high_confidence_threshold": 0.3,
                },
                "span_construction": {
                    "min_span_duration": 50.0,  # Very high minimum
                    "max_span_duration": 60.0,
                },
            }
        )

        constructor = HybridSpanConstructor(restrictive_config, model_config)

        # Create blocks with spans that are too short
        short_blocks = [
            {
                "text": "Quick action",
                "start_time": 0.0,
                "end_time": 5.0,  # Only 5 seconds - too short
                "merged_blocks": 1,
                "original_id": 0,
            },
        ]

        knowledge = ["Quick action"]

        # Mock similarity calculator
        with patch(
            "dst_data_builder.hybrid_dst.global_similarity_calculator.GlobalSimilarityCalculator"
        ) as mock_calc:
            mock_calc.score_blocks.return_value = ClassificationResult(
                clear_blocks=[
                    SimilarityResult(
                        block_id=0,
                        similarity_scores=[0.9],
                        nli_scores=[0.8],
                        combined_scores=[0.9],
                        confidence=0.9,
                        is_clear=True,
                    ),
                ],
                ambiguous_blocks=[],
                clear_count=1,
                ambiguous_count=0,
                total_blocks=1,
            )

            constructor.similarity_calculator = mock_calc

            result = constructor.construct_spans(short_blocks, knowledge)

            # Span should be rejected due to duration constraints
            assert len(result.dst_spans) == 0

    def test_statistics_generation(
        self, sample_config, model_config, sample_blocks, sample_knowledge
    ):
        """Test that comprehensive statistics are generated"""
        with patch(
            "dst_data_builder.hybrid_dst.span_constructors.global_similarity_calculator.GlobalSimilarityCalculator"
        ) as mock_calc, patch(
            "dst_data_builder.hybrid_dst.span_constructors.llm_ambiguous_handler.LLMAmbiguousHandler"
        ) as mock_llm:

            # Setup mocks
            mock_calc.score_blocks.return_value = ClassificationResult(
                clear_blocks=[
                    SimilarityResult(
                        block_id=0,
                        similarity_scores=[0.9, 0.1, 0.0],
                        nli_scores=[0.8, 0.0, -0.2],
                        combined_scores=[0.9, 0.1, 0.0],
                        confidence=0.9,
                        is_clear=True,
                    ),
                ],
                ambiguous_blocks=[],
                clear_count=1,
                ambiguous_count=0,
                total_blocks=3,
            )
            mock_calc.get_similarity_statistics.return_value = {
                "similarity_metric": 0.85
            }

            constructor = HybridSpanConstructor(sample_config, model_config)
            constructor.similarity_calculator = mock_calc

            result = constructor.construct_spans(sample_blocks, sample_knowledge)

            # Check statistics structure
            stats = result.construction_statistics
            assert "similarity_phase" in stats
            assert "llm_phase" in stats
            assert "span_construction" in stats
            assert "hybrid_processing" in stats

            # Check hybrid processing stats
            hybrid_stats = stats["hybrid_processing"]
            assert hybrid_stats["total_blocks_input"] == 3
            assert hybrid_stats["clear_blocks"] == 1
            assert hybrid_stats["ambiguous_blocks"] == 0
            assert hybrid_stats["final_spans_created"] == 1

    def test_time_extraction_methods(self, sample_config, model_config):
        """Test time extraction utility methods"""
        constructor = HybridSpanConstructor(sample_config, model_config)

        # Test start time extraction
        block_with_start = {"start_time": 10.5}
        assert constructor._extract_start_time(block_with_start) == 10.5

        block_with_t0 = {"t0": 20.0}
        assert constructor._extract_start_time(block_with_t0) == 20.0

        block_with_timestamp = {"timestamp": 5.5}
        assert constructor._extract_start_time(block_with_timestamp) == 5.5

        # Test end time extraction
        block_with_end = {"end_time": 15.5}
        assert constructor._extract_end_time(block_with_end) == 15.5

        block_with_t1 = {"t1": 25.0}
        assert constructor._extract_end_time(block_with_t1) == 25.0

        # Test invalid data
        assert constructor._extract_start_time({"invalid": "data"}) is None
        assert constructor._extract_end_time({"invalid": "data"}) is None

    def test_block_grouping_and_span_construction(self, sample_config, model_config):
        """Test the internal methods for grouping blocks and constructing spans"""
        constructor = HybridSpanConstructor(sample_config, model_config)

        # Test block grouping
        block_assignments = {
            0: {"step_index": 0, "confidence": 0.9, "source": "similarity"},
            1: {"step_index": 1, "confidence": 0.8, "source": "llm"},
            2: {
                "step_index": 0,
                "confidence": 0.7,
                "source": "similarity",
            },  # Another block for step 0
        }

        filtered_blocks = [
            {"text": "Block 0", "start_time": 0.0, "end_time": 5.0},
            {"text": "Block 1", "start_time": 10.0, "end_time": 15.0},
            {"text": "Block 2", "start_time": 20.0, "end_time": 25.0},
        ]

        step_groups = constructor._group_blocks_by_step(
            block_assignments, filtered_blocks
        )

        assert 0 in step_groups  # Step 0 has 2 blocks
        assert 1 in step_groups  # Step 1 has 1 block
        assert len(step_groups[0]) == 2
        assert len(step_groups[1]) == 1

        # Test span construction
        knowledge = ["Step 1", "Step 2"]
        spans = constructor._construct_spans_from_groups(step_groups, knowledge)

        assert len(spans) == 2

        # Check step 0 span (combines blocks 0 and 2)
        step0_span = next(span for span in spans if span["id"] == 1)
        assert step0_span["t0"] == 0.0  # Min of block 0 and 2
        assert step0_span["t1"] == 25.0  # Max of block 0 and 2
        assert step0_span["source_blocks"] == 2

        # Check step 1 span
        step1_span = next(span for span in spans if span["id"] == 2)
        assert step1_span["t0"] == 10.0
        assert step1_span["t1"] == 15.0
        assert step1_span["source_blocks"] == 1

    def test_real_proassist_data_processing(self, sample_config, model_config):
        """Test with real ProAssist data from the documentation"""
        # Real filtered blocks after overlap-aware reduction (from the doc)
        real_blocks = [
            {
                "text": "attach interior to chassis",
                "start_time": 94.4,
                "end_time": 105.2,
            },
            {"text": "attach wheel to chassis", "start_time": 105.2, "end_time": 153.6},
            {
                "text": "attach arm to turntable top",
                "start_time": 153.6,
                "end_time": 171.7,
            },
            {"text": "attach hook to arm", "start_time": 171.7, "end_time": 187.1},
            {
                "text": "attach turntable top to chassis",
                "start_time": 187.1,
                "end_time": 203.7,
            },
            {
                "text": "attach cabin to interior",
                "start_time": 203.7,
                "end_time": 213.1,
            },
            {
                "text": "demonstrate functionality",
                "start_time": 213.1,
                "end_time": 232.0,
            },
        ]

        # Real inferred knowledge (from the doc)
        real_knowledge = [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
            "Attach the body to the chassis.",
            "Add the cabin window to the chassis.",
            "Finalize the assembly and demonstrate the toy's functionality.",
        ]

        constructor = HybridSpanConstructor(sample_config, model_config)

        # Mock similarity calculator to simulate realistic scoring
        from unittest.mock import Mock

        mock_calc = Mock()
        # Simulate realistic similarity scoring where blocks map to knowledge steps
        mock_calc.score_blocks.return_value = ClassificationResult(
            clear_blocks=[
                # Block 0 -> Step 0 (chassis assembly)
                SimilarityResult(
                    block_id=0,
                    similarity_scores=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                    nli_scores=[0.8, 0.0, -0.2, -0.1, -0.1, -0.1],
                    combined_scores=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                    confidence=0.9,
                    is_clear=True,
                ),
                # Block 1 -> Step 1 (wheel attachment)
                SimilarityResult(
                    block_id=1,
                    similarity_scores=[0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
                    nli_scores=[-0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                    combined_scores=[0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
                    confidence=0.95,
                    is_clear=True,
                ),
                # Block 2 -> Step 2 (arm assembly)
                SimilarityResult(
                    block_id=2,
                    similarity_scores=[0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                    nli_scores=[0.0, 0.0, 0.85, 0.1, 0.0, 0.0],
                    combined_scores=[0.0, 0.0, 0.9, 0.0, 0.0, 0.0],
                    confidence=0.9,
                    is_clear=True,
                ),
                # Block 3 -> Step 2 (hook attachment, part of arm assembly)
                SimilarityResult(
                    block_id=3,
                    similarity_scores=[
                        0.0,
                        0.0,
                        0.88,
                        0.05,
                        0.0,
                        0.0,
                    ],  # Highest for step 2
                    nli_scores=[0.0, 0.0, 0.82, 0.12, 0.0, 0.0],
                    combined_scores=[0.0, 0.0, 0.88, 0.05, 0.0, 0.0],
                    confidence=0.88,
                    is_clear=True,
                ),
                # Block 4 -> Step 2 (turntable attachment, part of arm assembly)
                SimilarityResult(
                    block_id=4,
                    similarity_scores=[
                        0.0,
                        0.0,
                        0.85,
                        0.08,
                        0.0,
                        0.0,
                    ],  # Highest for step 2
                    nli_scores=[0.0, 0.0, 0.78, 0.15, 0.0, 0.0],
                    combined_scores=[0.0, 0.0, 0.85, 0.08, 0.0, 0.0],
                    confidence=0.85,
                    is_clear=True,
                ),
                # Block 5 -> Step 3 (cabin attachment)
                SimilarityResult(
                    block_id=5,
                    similarity_scores=[0.0, 0.0, 0.0, 0.9, 0.05, 0.0],
                    nli_scores=[0.0, 0.0, 0.0, 0.85, 0.1, 0.0],
                    combined_scores=[0.0, 0.0, 0.0, 0.9, 0.05, 0.0],
                    confidence=0.9,
                    is_clear=True,
                ),
                # Block 6 -> Step 5 (demonstration/finalization)
                SimilarityResult(
                    block_id=6,
                    similarity_scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
                    nli_scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.9],
                    combined_scores=[0.0, 0.0, 0.0, 0.0, 0.0, 0.95],
                    confidence=0.95,
                    is_clear=True,
                ),
            ],
            ambiguous_blocks=[],  # All blocks are clear in this example
            clear_count=7,
            ambiguous_count=0,
            total_blocks=7,
        )

        constructor.similarity_calculator = mock_calc

        result = constructor.construct_spans(real_blocks, real_knowledge)

        # Verify results with real data expectations
        assert len(result.dst_spans) == 5  # Should create 5 distinct spans

        # Check that spans are properly constructed
        spans = result.dst_spans

        # Step 0: Chassis assembly
        step0 = next(span for span in spans if span["id"] == 1)
        assert step0["name"] == real_knowledge[0]
        assert step0["t0"] == 94.4
        assert step0["t1"] == 105.2

        # Step 1: Wheel attachment
        step1 = next(span for span in spans if span["id"] == 2)
        assert step1["name"] == real_knowledge[1]
        assert step1["t0"] == 105.2
        assert step1["t1"] == 153.6

        # Step 2: Arm assembly (combines blocks 2, 3, 4)
        step2 = next(span for span in spans if span["id"] == 3)
        assert step2["name"] == real_knowledge[2]
        assert step2["t0"] == 153.6  # Earliest block in group
        assert step2["t1"] == 203.7  # Latest block in group
        assert step2["source_blocks"] == 3  # Three blocks combined

        # Step 3: Body/cabin attachment
        step3 = next(span for span in spans if span["id"] == 4)
        assert step3["name"] == real_knowledge[3]
        assert step3["t0"] == 203.7
        assert step3["t1"] == 213.1

        # Step 5: Finalization/demonstration
        step5 = next(span for span in spans if span["id"] == 5)
        assert step5["name"] == real_knowledge[5]
        assert step5["t0"] == 213.1
        assert step5["t1"] == 232.0

        # Verify statistics
        assert result.clear_blocks_used == 7
        assert result.ambiguous_blocks_resolved == 0
        assert result.total_blocks_processed == 7

    def test_real_data_with_ambiguous_blocks(self, sample_config, model_config):
        """Test real data with some ambiguous blocks requiring LLM"""
        real_blocks = [
            {
                "text": "attach interior to chassis",
                "start_time": 94.4,
                "end_time": 105.2,
            },
            {"text": "attach wheel to chassis", "start_time": 105.2, "end_time": 153.6},
            {
                "text": "attach arm to turntable top",
                "start_time": 153.6,
                "end_time": 171.7,
            },
            {"text": "attach hook to arm", "start_time": 171.7, "end_time": 187.1},
            {
                "text": "attach turntable top to chassis",
                "start_time": 187.1,
                "end_time": 203.7,
            },
            {
                "text": "attach cabin to interior",
                "start_time": 203.7,
                "end_time": 213.1,
            },
            {
                "text": "demonstrate functionality",
                "start_time": 213.1,
                "end_time": 232.0,
            },
        ]

        real_knowledge = [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
            "Attach the body to the chassis.",
            "Add the cabin window to the chassis.",
            "Finalize the assembly and demonstrate the toy's functionality.",
        ]

        constructor = HybridSpanConstructor(sample_config, model_config)

        with patch(
            "dst_data_builder.hybrid_dst.span_constructors.global_similarity_calculator.GlobalSimilarityCalculator"
        ) as mock_calc, patch(
            "dst_data_builder.hybrid_dst.span_constructors.llm_ambiguous_handler.LLMAmbiguousHandler"
        ) as mock_llm:

            # Mock similarity calculator with some ambiguous blocks
            mock_calc.score_blocks.return_value = ClassificationResult(
                clear_blocks=[
                    SimilarityResult(
                        block_id=0,
                        similarity_scores=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                        nli_scores=[0.8, 0.0, -0.2, -0.1, -0.1, -0.1],
                        combined_scores=[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
                        confidence=0.9,
                        is_clear=True,
                    ),
                    SimilarityResult(
                        block_id=1,
                        similarity_scores=[0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
                        nli_scores=[-0.1, 0.9, 0.0, 0.0, 0.0, 0.0],
                        combined_scores=[0.0, 0.95, 0.0, 0.0, 0.0, 0.0],
                        confidence=0.95,
                        is_clear=True,
                    ),
                ],
                ambiguous_blocks=[
                    # Ambiguous: could be step 2 or 3
                    SimilarityResult(
                        block_id=2,
                        similarity_scores=[0.0, 0.0, 0.4, 0.35, 0.0, 0.0],
                        nli_scores=[0.0, 0.0, 0.3, 0.4, 0.0, 0.0],
                        combined_scores=[0.0, 0.0, 0.4, 0.35, 0.0, 0.0],
                        confidence=0.15,
                        is_clear=False,
                    ),
                    # Ambiguous: could be step 3 or 4
                    SimilarityResult(
                        block_id=3,
                        similarity_scores=[0.0, 0.0, 0.0, 0.45, 0.4, 0.0],
                        nli_scores=[0.0, 0.0, 0.0, 0.4, 0.45, 0.0],
                        combined_scores=[0.0, 0.0, 0.0, 0.45, 0.4, 0.0],
                        confidence=0.18,
                        is_clear=False,
                    ),
                ],
                clear_count=2,
                ambiguous_count=2,
                total_blocks=7,
            )

            # Mock LLM handler
            mock_llm.resolve_ambiguous_blocks.return_value = LLMHandlingResult(
                decisions=[
                    LLMDecision(
                        block_id=2,
                        chosen_step_index=2,
                        confidence=0.8,
                        reasoning="Arm assembly step",
                    ),
                    LLMDecision(
                        block_id=3,
                        chosen_step_index=3,
                        confidence=0.75,
                        reasoning="Body attachment step",
                    ),
                ],
                total_llm_calls=2,
                success_count=2,
                failure_count=0,
                total_cost_estimate=0.02,
            )

            constructor.similarity_calculator = mock_calc
            constructor.llm_handler = mock_llm

            result = constructor.construct_spans(real_blocks, real_knowledge)

            # Should have spans for steps 0, 1, 2, 3
            assert len(result.dst_spans) == 4
            assert result.clear_blocks_used == 2
            assert result.ambiguous_blocks_resolved == 2

            # Verify LLM was called for ambiguous blocks
            mock_llm.resolve_ambiguous_blocks.assert_called_once()


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

    pytest.main([__file__, "-v"])
