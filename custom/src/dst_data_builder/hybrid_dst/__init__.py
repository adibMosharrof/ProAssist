"""
Hybrid DST Package

This package implements the hybrid two-stage DST label generation algorithm that combines
global similarity scoring for high-confidence cases with LLM fallback for ambiguous cases.
"""

from dst_data_builder.hybrid_dst.hybrid_dst_generator import HybridDSTLabelGenerator
from dst_data_builder.hybrid_dst.overlap_aware_reducer import OverlapAwareBlockReducer
from dst_data_builder.hybrid_dst.span_constructors import (
    BaseSpanConstructor,
    SimpleSpanConstructor,
    HybridSpanConstructor,
    GlobalSimilarityCalculator,
    LLMAmbiguousHandler,
)
from dst_data_builder.hybrid_dst.temporal_validator import TemporalOrderingValidator
from dst_data_builder.hybrid_dst import utils

__all__ = [
    "HybridDSTLabelGenerator",
    "OverlapAwareBlockReducer",
    "BaseSpanConstructor",
    "SimpleSpanConstructor",
    "HybridSpanConstructor",
    "GlobalSimilarityCalculator",
    "LLMAmbiguousHandler",
    "TemporalOrderingValidator",
    "utils",
]

__version__ = "1.0.0"
