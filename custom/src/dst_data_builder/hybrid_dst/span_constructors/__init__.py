"""
Span Constructors Package

This package contains the span construction classes for DST processing.
"""

from dst_data_builder.hybrid_dst.span_constructors.base_span_constructor import (
    BaseSpanConstructor,
)
from dst_data_builder.hybrid_dst.span_constructors.simple_span_constructor import (
    SimpleSpanConstructor,
)
from dst_data_builder.hybrid_dst.span_constructors.hybrid_span_constructor import (
    HybridSpanConstructor,
)
from dst_data_builder.hybrid_dst.span_constructors.bidirectional_span_constructor import (
    BidirectionalSpanConstructor,
    BidirectionalSpanConstructionResult,
    DirectionalAssignment,
)
from dst_data_builder.hybrid_dst.span_constructors.llm_span_constructor import (
    LLMSpanConstructor,
    LLMSpanConstructionResult,
)
from dst_data_builder.hybrid_dst.span_constructors.global_similarity_calculator import (
    GlobalSimilarityCalculator,
    ClassificationResult,
    SimilarityResult,
)
from dst_data_builder.hybrid_dst.span_constructors.llm_ambiguous_handler import (
    LLMAmbiguousHandler,
    AmbiguousBlock,
    LLMDecision,
    LLMHandlingResult,
)

__all__ = [
    "BaseSpanConstructor",
    "SimpleSpanConstructor",
    "HybridSpanConstructor",
    "BidirectionalSpanConstructor",
    "BidirectionalSpanConstructionResult",
    "DirectionalAssignment",
    "LLMSpanConstructor",
    "LLMSpanConstructionResult",
    "GlobalSimilarityCalculator",
    "ClassificationResult",
    "SimilarityResult",
    "LLMAmbiguousHandler",
    "AmbiguousBlock",
    "LLMDecision",
    "LLMHandlingResult",
]
