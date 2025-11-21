"""
Test Bidirectional Span Constructor

Basic test structure for the BidirectionalSpanConstructor POC.
Tests will be added after debugging the implementation.
"""

import pytest
from omegaconf import OmegaConf

from dst_data_builder.hybrid_dst.span_constructors import (
    BidirectionalSpanConstructor,
    BidirectionalSpanConstructionResult,
    DirectionalAssignment,
)


class TestBidirectionalSpanConstructor:
    """Test class for BidirectionalSpanConstructor POC"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create(
            {
                "high_confidence_threshold": 0.3,
                "semantic_weight": 0.6,
                "nli_weight": 0.4,
                "min_span_duration": 1.0,
                "max_span_duration": 300.0,
            }
        )

    @pytest.fixture
    def model_config(self):
        """Create sample model configuration"""
        return OmegaConf.create(
            {
                "name": "gpt-4o",
                "temperature": 0.1,
                "max_tokens": 4000,
            }
        )

    @pytest.fixture
    def constructor(self, sample_config, model_config):
        """Create BidirectionalSpanConstructor instance"""
        return BidirectionalSpanConstructor(sample_config, model_config)

    # TODO: Add tests after debugging the implementation
    # def test_basic_construction(self, constructor):
    #     """Test basic bidirectional construction"""

    # def test_forward_pass(self, constructor):
    #     """Test forward pass logic"""

    # def test_backward_pass(self, constructor):
    #     """Test backward pass logic"""

    # def test_conflict_detection(self, constructor):
    #     """Test conflict detection between passes"""

    # def test_span_construction(self, constructor):
    #     """Test final span construction from assignments"""
