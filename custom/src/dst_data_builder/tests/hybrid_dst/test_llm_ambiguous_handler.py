"""
Test LLM Ambiguous Handler

Comprehensive tests for the LLMAmbiguousHandler component that handles LLM fallback
for ambiguous blocks in the hybrid DST generation pipeline.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
import json
from omegaconf import OmegaConf

from dst_data_builder.hybrid_dst.span_constructors import (
    LLMAmbiguousHandler,
    AmbiguousBlock,
    LLMDecision,
    LLMHandlingResult,
)


class TestLLMAmbiguousHandler:
    """Test class for LLMAmbiguousHandler functionality"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create(
            {
                "temperature": 0.1,
                "max_tokens": 1000,
                "batch_size": 3,  # Small batch for testing
            }
        )

    @pytest.fixture
    def sample_ambiguous_blocks(self):
        """Create sample ambiguous blocks for testing"""
        return [
            AmbiguousBlock(
                block_id=0,
                block_data={
                    "text": "attach interior to chassis",
                    "start_time": 94.4,
                    "end_time": 105.2,
                },
                similarity_scores=[0.4, 0.35, 0.3],
                confidence=0.15,
                top_alternatives=[(0, 0.4), (1, 0.35), (2, 0.3)],
            ),
            AmbiguousBlock(
                block_id=1,
                block_data={
                    "text": "attach wheel to chassis",
                    "start_time": 105.2,
                    "end_time": 153.6,
                },
                similarity_scores=[0.2, 0.45, 0.4],
                confidence=0.18,
                top_alternatives=[(1, 0.45), (2, 0.4), (0, 0.2)],
            ),
        ]

    @pytest.fixture
    def sample_knowledge(self):
        """Create sample inferred knowledge"""
        return [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
        ]

    def test_initialization(self, sample_config):
        """Test that LLMAmbiguousHandler initializes correctly"""
        handler = LLMAmbiguousHandler(sample_config)

        assert handler.config == sample_config
        assert handler.temperature == 0.1
        assert handler.max_tokens == 1000
        assert handler.batch_size == 3
        assert handler.total_cost_estimate == 0.0
        assert hasattr(handler, "llm_client")
        assert hasattr(handler, "logger")

    def test_empty_input_handling(self, sample_config):
        """Test handling of empty inputs"""
        handler = LLMAmbiguousHandler(sample_config)

        result = handler.resolve_ambiguous_blocks([], ["step 1"])
        assert result.decisions == []
        assert result.total_llm_calls == 0
        assert result.success_count == 0
        assert result.failure_count == 0
        assert result.total_cost_estimate == 0.0

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_resolve_ambiguous_blocks_success(
        self,
        mock_client_class,
        sample_config,
        sample_ambiguous_blocks,
        sample_knowledge,
    ):
        """Test successful resolution of ambiguous blocks"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful LLM responses
        mock_client.generate_completion = AsyncMock(
            side_effect=[
                (
                    True,
                    '{"chosen_step": 1, "confidence": 0.8, "reasoning": "Best matches chassis assembly"}',
                ),
                (
                    True,
                    '{"chosen_step": 2, "confidence": 0.9, "reasoning": "Clearly wheel attachment"}',
                ),
            ]
        )

        handler = LLMAmbiguousHandler(sample_config)
        result = handler.resolve_ambiguous_blocks(
            sample_ambiguous_blocks, sample_knowledge
        )

        # Verify results
        assert len(result.decisions) == 2
        assert result.total_llm_calls == 2
        assert result.success_count == 2
        assert result.failure_count == 0

        # Check decisions
        assert result.decisions[0].block_id == 0
        assert (
            result.decisions[0].chosen_step_index == 0
        )  # 1-based to 0-based conversion
        assert result.decisions[0].confidence == 0.8
        assert "chassis assembly" in result.decisions[0].reasoning

        assert result.decisions[1].block_id == 1
        assert (
            result.decisions[1].chosen_step_index == 1
        )  # 2-based to 1-based conversion
        assert result.decisions[1].confidence == 0.9
        assert "wheel attachment" in result.decisions[1].reasoning

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_resolve_ambiguous_blocks_partial_failure(
        self,
        mock_client_class,
        sample_config,
        sample_ambiguous_blocks,
        sample_knowledge,
    ):
        """Test handling of partial LLM failures"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock mixed success/failure responses
        mock_client.generate_completion = AsyncMock(
            side_effect=[
                (True, '{"chosen_step": 1, "confidence": 0.8, "reasoning": "Success"}'),
                (False, "API Error"),  # Failure
            ]
        )

        handler = LLMAmbiguousHandler(sample_config)
        result = handler.resolve_ambiguous_blocks(
            sample_ambiguous_blocks, sample_knowledge
        )

        # Verify results
        assert len(result.decisions) == 2
        assert result.total_llm_calls == 2
        assert result.success_count == 1  # Only first succeeded
        assert result.failure_count == 1

        # Check decisions
        assert result.decisions[0].chosen_step_index == 0  # Success
        assert result.decisions[1].chosen_step_index == -1  # Failure fallback

    def test_context_preparation(
        self, sample_config, sample_ambiguous_blocks, sample_knowledge
    ):
        """Test preparation of LLM contexts"""
        handler = LLMAmbiguousHandler(sample_config)

        contexts = handler._prepare_contexts(sample_ambiguous_blocks, sample_knowledge)

        assert len(contexts) == 2

        # Check first context
        context1 = contexts[0]
        assert "attach interior to chassis" in context1
        assert "Assemble the chassis" in context1
        assert "Attach wheels" in context1
        assert "Assemble the arm" in context1
        assert "chosen_step" in context1
        assert "confidence" in context1
        assert "reasoning" in context1

    def test_context_formatting(self, sample_config, sample_knowledge):
        """Test formatting of individual ambiguous block contexts"""
        handler = LLMAmbiguousHandler(sample_config)

        block = AmbiguousBlock(
            block_id=0,
            block_data={"text": "test block content"},
            similarity_scores=[0.5, 0.3, 0.2],
            confidence=0.15,
            top_alternatives=[(0, 0.5), (1, 0.3), (2, 0.2)],
        )

        context = handler._format_ambiguous_block_context(block, sample_knowledge)

        # Verify structure
        assert 'BLOCK CONTENT: "test block content"' in context
        assert "SIMILARITY SCORES:" in context
        assert "1. Assemble the chassis" in context
        assert "2. Attach wheels" in context
        assert "3. Assemble the arm" in context
        assert "chosen_step" in context
        assert "confidence" in context
        assert "reasoning" in context

    def test_text_extraction(self, sample_config):
        """Test text extraction from block data"""
        handler = LLMAmbiguousHandler(sample_config)

        # Test with "text" field
        assert handler._extract_block_text({"text": "content"}) == "content"

        # Test with "content" field
        assert handler._extract_block_text({"content": "alt content"}) == "alt content"

        # Test with neither field
        assert handler._extract_block_text({"other": "field"}) == ""

        # Test with non-string content
        assert handler._extract_block_text({"text": 123}) == "123"

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_batch_query_processing(self, mock_client_class, sample_config):
        """Test batch processing of LLM queries"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.generate_completion = AsyncMock(return_value=(True, "response"))

        handler = LLMAmbiguousHandler(sample_config)

        # Test with 5 contexts and batch size 3
        contexts = [f"context {i}" for i in range(5)]
        responses = handler._batch_query_llm(contexts)

        # Should have made 2 batches: [0,1,2] and [3,4]
        assert len(responses) == 5
        assert all(r == "response" for r in responses)

        # Verify generate_completion was called 5 times
        assert mock_client.generate_completion.call_count == 5

    def test_parse_decisions_success(
        self, sample_config, sample_ambiguous_blocks, sample_knowledge
    ):
        """Test successful parsing of LLM decisions"""
        handler = LLMAmbiguousHandler(sample_config)

        responses = [
            '{"chosen_step": 1, "confidence": 0.8, "reasoning": "Best match"}',
            '{"chosen_step": 2, "confidence": 0.9, "reasoning": "Clear choice"}',
        ]

        decisions = handler._parse_decisions(
            responses, sample_ambiguous_blocks, sample_knowledge
        )

        assert len(decisions) == 2

        assert decisions[0].block_id == 0
        assert decisions[0].chosen_step_index == 0  # 1-based to 0-based
        assert decisions[0].confidence == 0.8
        assert decisions[0].reasoning == "Best match"

        assert decisions[1].block_id == 1
        assert decisions[1].chosen_step_index == 1  # 2-based to 1-based
        assert decisions[1].confidence == 0.9
        assert decisions[1].reasoning == "Clear choice"

    def test_parse_decisions_json_variations(
        self, sample_config, sample_ambiguous_blocks, sample_knowledge
    ):
        """Test parsing of various JSON response formats"""
        handler = LLMAmbiguousHandler(sample_config)

        # Test with markdown JSON blocks
        responses = [
            '```json\n{"chosen_step": 1, "confidence": 0.8, "reasoning": "Markdown format"}\n```',
            '{"chosen_step": 2, "confidence": 0.9, "reasoning": "Plain JSON"}',
        ]

        decisions = handler._parse_decisions(
            responses, sample_ambiguous_blocks[:2], sample_knowledge
        )

        assert len(decisions) == 2
        assert decisions[0].reasoning == "Markdown format"
        assert decisions[1].reasoning == "Plain JSON"

    def test_parse_decisions_fallback(
        self, sample_config, sample_ambiguous_blocks, sample_knowledge
    ):
        """Test fallback parsing when JSON is invalid"""
        handler = LLMAmbiguousHandler(sample_config)

        # Invalid JSON responses
        responses = [
            "Invalid JSON response",
            "",  # Empty response
        ]

        decisions = handler._parse_decisions(
            responses, sample_ambiguous_blocks[:2], sample_knowledge
        )

        assert len(decisions) == 2

        # First decision should use fallback to highest similarity
        assert decisions[0].chosen_step_index == 0  # Highest similarity step
        assert decisions[0].confidence == 0.3  # Lower confidence for fallback
        assert "parse error" in decisions[0].reasoning.lower()

        # Second decision should fail completely
        assert decisions[1].chosen_step_index == -1
        assert decisions[1].confidence == 0.0

    def test_parse_decisions_invalid_step_index(
        self, sample_config, sample_ambiguous_blocks, sample_knowledge
    ):
        """Test handling of invalid step indices in LLM responses"""
        handler = LLMAmbiguousHandler(sample_config)

        # Response with invalid step index
        responses = [
            '{"chosen_step": 99, "confidence": 0.8, "reasoning": "Invalid step"}',  # Way out of range
        ]

        decisions = handler._parse_decisions(
            responses, sample_ambiguous_blocks[:1], sample_knowledge
        )

        # Should default to step 0
        assert decisions[0].chosen_step_index == 0
        assert decisions[0].confidence == 0.8

    def test_cost_estimation(self, sample_config):
        """Test cost estimation for LLM usage"""
        handler = LLMAmbiguousHandler(sample_config)

        # Test cost calculation
        cost = handler.get_cost_estimate(1000)
        expected_cost = 1000 * 0.00001  # $0.01 per 1K tokens
        assert abs(cost - expected_cost) < 1e-6

        # Test accumulation
        total_before = handler.total_cost_estimate
        handler.get_cost_estimate(500)
        assert handler.total_cost_estimate == total_before + (500 * 0.00001)

    def test_handling_statistics(self, sample_config):
        """Test generation of handling statistics"""
        handler = LLMAmbiguousHandler(sample_config)

        # Create sample result
        decisions = [
            LLMDecision(
                block_id=0, chosen_step_index=0, confidence=0.8, reasoning="Good"
            ),
            LLMDecision(
                block_id=1, chosen_step_index=1, confidence=0.9, reasoning="Better"
            ),
            LLMDecision(
                block_id=2, chosen_step_index=-1, confidence=0.0, reasoning="Failed"
            ),
        ]

        result = LLMHandlingResult(
            decisions=decisions,
            total_llm_calls=3,
            success_count=2,
            failure_count=1,
            total_cost_estimate=0.03,
        )

        stats = handler.get_handling_statistics(result)

        assert stats["total_ambiguous_blocks"] == 3
        assert stats["llm_calls_made"] == 3
        assert stats["successful_decisions"] == 2
        assert stats["failed_decisions"] == 1
        assert abs(stats["success_rate"] - 66.67) < 0.01
        assert abs(stats["average_confidence"] - 0.85) < 0.01  # (0.8 + 0.9) / 2
        assert stats["estimated_total_cost"] == 0.03
        assert abs(stats["cost_per_block"] - 0.01) < 1e-6

    def test_statistics_empty_result(self, sample_config):
        """Test statistics generation with empty results"""
        handler = LLMAmbiguousHandler(sample_config)

        empty_result = LLMHandlingResult([], 0, 0, 0, 0.0)
        stats = handler.get_handling_statistics(empty_result)

        assert stats == {"error": "No decisions to analyze"}

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_real_proassist_data_resolution(self, mock_client_class, sample_config):
        """Test LLM resolution with real ProAssist ambiguous blocks"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful LLM responses for real ambiguous blocks
        mock_client.generate_completion = AsyncMock(
            side_effect=[
                (
                    True,
                    '{"chosen_step": 3, "confidence": 0.85, "reasoning": "Body attachment step - cabin interior"}',
                ),
                (
                    True,
                    '{"chosen_step": 4, "confidence": 0.9, "reasoning": "Window addition step"}',
                ),
            ]
        )

        handler = LLMAmbiguousHandler(sample_config)

        # Real ambiguous blocks from ProAssist (could be arm assembly or body attachment)
        real_ambiguous_blocks = [
            AmbiguousBlock(
                block_id=3,
                block_data={
                    "text": "attach cabin to interior",
                    "start_time": 203.7,
                    "end_time": 213.1,
                },
                similarity_scores=[
                    0.0,
                    0.0,
                    0.3,
                    0.4,
                    0.35,
                    0.0,
                ],  # Similar to steps 3 and 4
                confidence=0.18,
                top_alternatives=[(3, 0.4), (4, 0.35), (2, 0.3)],
            ),
            AmbiguousBlock(
                block_id=4,
                block_data={
                    "text": "add window to cabin",
                    "start_time": 213.1,
                    "end_time": 220.0,
                },
                similarity_scores=[
                    0.0,
                    0.0,
                    0.2,
                    0.45,
                    0.4,
                    0.0,
                ],  # Similar to steps 3 and 4
                confidence=0.19,
                top_alternatives=[(3, 0.45), (4, 0.4), (2, 0.2)],
            ),
        ]

        real_knowledge = [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
            "Attach the body to the chassis.",
            "Add the cabin window to the chassis.",
            "Finalize the assembly and demonstrate the toy's functionality.",
        ]

        result = handler.resolve_ambiguous_blocks(real_ambiguous_blocks, real_knowledge)

        # Verify results
        assert len(result.decisions) == 2
        assert result.total_llm_calls == 2
        assert result.success_count == 2
        assert result.failure_count == 0

        # Check that LLM made appropriate decisions
        assert result.decisions[0].chosen_step_index == 2  # Step 3 (0-based)
        assert result.decisions[0].confidence == 0.85
        assert "body attachment" in result.decisions[0].reasoning.lower()

        assert result.decisions[1].chosen_step_index == 3  # Step 4 (0-based)
        assert result.decisions[1].confidence == 0.9
        assert "window" in result.decisions[1].reasoning.lower()

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_concurrent_batch_processing(self, mock_client_class, sample_config):
        """Test that batch processing works concurrently"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.generate_completion = AsyncMock(return_value=(True, "response"))

        handler = LLMAmbiguousHandler(sample_config)

        # Test with multiple batches
        contexts = [
            f"context {i}" for i in range(7)
        ]  # 7 contexts with batch_size=3 = 3 batches
        responses = handler._batch_query_llm(contexts)

        # Should process all 7 contexts
        assert len(responses) == 7
        assert mock_client.generate_completion.call_count == 7

    @patch("dst_data_builder.hybrid_dst.llm_ambiguous_handler.OpenAIAPIClient")
    def test_error_handling_in_batch_processing(self, mock_client_class, sample_config):
        """Test error handling during batch processing"""
        # Mock the LLM client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock mixed success and exceptions
        mock_client.generate_completion = AsyncMock(
            side_effect=[
                (True, "success response"),
                Exception("API timeout"),
                (False, "API error"),
                (True, "another success"),
            ]
        )

        handler = LLMAmbiguousHandler(sample_config)

        contexts = [f"context {i}" for i in range(4)]
        responses = handler._batch_query_llm(contexts)

        # Should handle all cases gracefully
        assert len(responses) == 4
        assert responses[0] == "success response"
        assert responses[1] == ""  # Exception converted to empty string
        assert responses[2] == ""  # Failure converted to empty string
        assert responses[3] == "another success"


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

    pytest.main([__file__, "-v"])
