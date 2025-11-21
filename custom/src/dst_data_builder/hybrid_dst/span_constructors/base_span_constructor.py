"""
Base Span Constructor Module

This module provides the base functionality for span construction classes,
containing common methods for time extraction, span validation, and statistics.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from omegaconf import DictConfig


class BaseSpanConstructor:
    """
    Base class for span constructors providing common functionality

    This class contains shared methods for:
    - Time extraction from blocks
    - Span validation and fixing
    - Temporal ordering
    - Statistics creation
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Common configuration - subclasses should override if needed
        self.min_span_duration = config.get("min_span_duration", 1.0)
        self.max_span_duration = config.get("max_span_duration", 300.0)

    def _extract_start_time(self, block: Dict[str, Any]) -> Optional[float]:
        """Extract start time from block"""
        for time_field in ["start_time", "t0", "timestamp"]:
            if time_field in block:
                try:
                    return float(block[time_field])
                except (ValueError, TypeError):
                    continue
        return None

    def _extract_end_time(self, block: Dict[str, Any]) -> Optional[float]:
        """Extract end time from block"""
        for time_field in ["end_time", "t1"]:
            if time_field in block:
                try:
                    return float(block[time_field])
                except (ValueError, TypeError):
                    continue
        return None

    def _validate_and_fix_span_duration(
        self, start_time: float, end_time: float, span_id: int = None
    ) -> Tuple[float, float]:
        """
        Validate span duration and fix if necessary

        Args:
            start_time: Original start time
            end_time: Original end time
            span_id: Optional span identifier for logging

        Returns:
            Tuple of (fixed_start_time, fixed_end_time)
        """
        duration = end_time - start_time

        # Check minimum duration
        if duration < self.min_span_duration:
            span_desc = f"Span {span_id}" if span_id else "Span"
            self.logger.warning(
                "%s duration %.2fs is below minimum %.1fs, adjusting",
                span_desc,
                duration,
                self.min_span_duration,
            )
            end_time = start_time + self.min_span_duration

        # Check maximum duration
        elif duration > self.max_span_duration:
            span_desc = f"Span {span_id}" if span_id else "Span"
            self.logger.warning(
                "%s duration %.2fs exceeds maximum %.1fs, adjusting",
                span_desc,
                duration,
                self.max_span_duration,
            )
            end_time = start_time + self.max_span_duration

        return start_time, end_time

    def _validate_and_fix_timestamps(
        self, start_time: float, end_time: float, span_id: int = None
    ) -> Tuple[float, float]:
        """
        Validate and fix timestamp ordering

        Args:
            start_time: Original start time
            end_time: Original end time
            span_id: Optional span identifier for logging

        Returns:
            Tuple of (fixed_start_time, fixed_end_time)
        """
        if start_time >= end_time:
            span_desc = f"Span {span_id}" if span_id else "Span"
            self.logger.warning(
                "%s has invalid timestamps (start=%.2f >= end=%.2f), swapping",
                span_desc,
                start_time,
                end_time,
            )
            start_time, end_time = end_time, start_time

        return start_time, end_time

    def _sort_spans_temporally(
        self, spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Sort spans by start time and reassign sequential IDs

        Args:
            spans: List of span dictionaries

        Returns:
            Sorted spans with sequential IDs
        """
        spans.sort(key=lambda x: x["start_ts"])
        for i, span in enumerate(spans, 1):
            span["id"] = i
        return spans

    def _create_common_span_structure(
        self,
        span_id: int,
        name: str,
        start_time: float,
        end_time: float,
        confidence: float,
        source: str,
        source_blocks: int = 1,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Create a standardized span dictionary structure

        Args:
            span_id: Sequential span ID
            name: Span name/description
            start_time: Start timestamp
            end_time: End timestamp
            confidence: Confidence score
            source: Source of the span (e.g., "simple_span", "hybrid")
            source_blocks: Number of source blocks used
            **kwargs: Additional fields to include

        Returns:
            Standardized span dictionary
        """
        span = {
            "id": span_id,
            "name": name,
            "start_ts": start_time,
            "end_ts": end_time,
            "conf": confidence,
            "source_blocks": source_blocks,
            "source": source,
        }
        span.update(kwargs)
        return span

    def _calculate_temporal_statistics(
        self, spans: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Calculate common temporal statistics from spans

        Args:
            spans: List of span dictionaries

        Returns:
            Dictionary with temporal statistics
        """
        if not spans:
            return {"error": "No spans to analyze"}

        durations = [span["end_ts"] - span["start_ts"] for span in spans]

        return {
            "total_spans": len(spans),
            "duration_stats": {
                "mean": float(sum(durations) / len(durations)),
                "min": float(min(durations)),
                "max": float(max(durations)),
            },
            "temporal_range": {
                "start": float(min(span["start_ts"] for span in spans)),
                "end": float(max(span["end_ts"] for span in spans)),
                "total_duration": float(
                    max(span["end_ts"] for span in spans)
                    - min(span["start_ts"] for span in spans)
                ),
            },
        }

    def _validate_construct_spans_inputs(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ):
        if not filtered_blocks:
            raise ValueError("No filtered blocks provided for span construction")
        if not inferred_knowledge:
            raise ValueError("No inferred knowledge provided for span construction")
