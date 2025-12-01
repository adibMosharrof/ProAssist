"""
Simple Span Constructor Module

This module implements the SimpleSpanConstructor that provides direct timestamp assignment
when the number of filtered blocks exactly matches the number of inferred knowledge steps.
"""

import logging
from typing import List, Dict, Any
from dataclasses import dataclass
from omegaconf import DictConfig

from dst_data_builder.hybrid_dst.span_constructors.base_span_constructor import (
    BaseSpanConstructor,
)


@dataclass
class SimpleSpanConstructionResult:
    """Result of simple span construction"""

    dst_spans: List[Dict[str, Any]]
    total_blocks_processed: int
    construction_statistics: Dict[str, Any]


class SimpleSpanConstructor(BaseSpanConstructor):
    """Simple Span Constructor: Direct timestamp assignment when block count matches step count"""

    def __init__(self, config: DictConfig):
        super().__init__(config)

        # Additional configuration specific to simple constructor
        self.default_point_duration = config.get("default_point_duration", 5.0)

    def construct_spans(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> SimpleSpanConstructionResult:
        """
        Direct timestamp assignment when block count matches step count

        Args:
            filtered_blocks: Block list from Stage 1
            inferred_knowledge: Step descriptions

        Returns:
            SimpleSpanConstructionResult with direct timestamp mapping
        """
        # Handle empty inputs gracefully
        if not filtered_blocks or not inferred_knowledge:
            return SimpleSpanConstructionResult(
                [],
                len(filtered_blocks),
                {
                    "error": "Empty inputs",
                    "blocks": len(filtered_blocks),
                    "steps": len(inferred_knowledge),
                },
            )

        self._validate_construct_spans_inputs(filtered_blocks, inferred_knowledge)

        total_blocks = len(filtered_blocks)
        total_steps = len(inferred_knowledge)

        self.logger.debug(
            "🔄 Simple span construction: %d filtered blocks, %d inferred steps",
            total_blocks,
            total_steps,
        )

        # Ensure equal lengths for direct mapping
        if total_blocks != total_steps:
            self.logger.warning(
                "SimpleSpanConstructor requires equal number of blocks and steps. "
                "Got %d blocks and %d steps. Use HybridSpanConstructor instead.",
                total_blocks,
                total_steps,
            )
            return SimpleSpanConstructionResult(
                [],
                total_blocks,
                {
                    "error": "Unequal counts",
                    "blocks": total_blocks,
                    "steps": total_steps,
                },
            )

        dst_spans = []

        for i, (block, step) in enumerate(zip(filtered_blocks, inferred_knowledge)):
            # Extract timestamps from block
            start_time = self._extract_start_time(block)
            end_time = self._extract_end_time(block)

            # Handle point blocks (single timestamp)
            if end_time is None:
                # For point blocks, assign a reasonable duration or use default
                end_time = start_time + self.default_point_duration
                self.logger.debug(
                    "Point block %d: assigned default duration %.1fs",
                    i,
                    self.default_point_duration,
                )

            # CRITICAL VALIDATION: Ensure start < end before any processing
            if start_time >= end_time:
                self.logger.error(
                    f"❌ CRITICAL: Block {i} has invalid timestamps "
                    f"(start={start_time:.2f} >= end={end_time:.2f}). "
                    f"This indicates fundamental data corruption in block data. "
                    f"Skipping this corrupted span."
                )
                continue

            # Validate and fix span duration and timestamps
            start_time, end_time = self._validate_and_fix_span_duration(
                start_time, end_time, i + 1
            )
            start_time, end_time = self._validate_and_fix_timestamps(
                start_time, end_time, i + 1
            )

            # Create span using base class method
            span = self._create_common_span_structure(
                span_id=i + 1,
                name=step,
                start_time=start_time,
                end_time=end_time,
                confidence=1.0,  # Maximum confidence for direct assignment
                source="simple_span",
                source_blocks=1,
                original_block_id=i,
            )
            dst_spans.append(span)

        # Ensure temporal ordering and reassign sequential IDs
        dst_spans = self._sort_spans_temporally(dst_spans)

        # Create statistics
        construction_stats = self._create_construction_statistics(
            dst_spans, total_blocks
        )

        self.logger.debug(
            "✅ Simple span construction complete: %d final DST spans", len(dst_spans)
        )

        return SimpleSpanConstructionResult(
            dst_spans=dst_spans,
            total_blocks_processed=total_blocks,
            construction_statistics=construction_stats,
        )

    def _create_construction_statistics(
        self, spans: List[Dict[str, Any]], total_blocks: int
    ) -> Dict[str, Any]:
        """
        Create statistics about the simple construction process

        Args:
            spans: Final constructed spans
            total_blocks: Total blocks processed

        Returns:
            Dictionary with construction statistics
        """
        if not spans:
            return {"error": "No spans created"}

        # Get common temporal statistics from base class
        temporal_stats = self._calculate_temporal_statistics(spans)

        return {
            "construction_type": "simple_span_constructor",
            "total_blocks_input": total_blocks,
            "total_spans_created": len(spans),
            "point_blocks_handled": sum(
                1 for span in spans if span["source"] == "simple_span"
            ),
            "duration_statistics": temporal_stats["duration_stats"],
            "temporal_range": temporal_stats["temporal_range"],
            "efficiency": {
                "processing_type": "direct_assignment",
                "complexity": "O(n)",
                "llm_calls": 0,
                "model_calls": 0,
            },
        }
