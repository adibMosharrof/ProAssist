"""
Temporal Validator Module

This module implements temporal ordering validation for DST spans in the hybrid DST algorithm.
It detects temporal ordering violations (overlaps, regressions, gaps) but does not apply fixes.
Violations are reported for handling by other components in the pipeline. Overlaps are allowed
for DST spans since assembly steps can happen in parallel.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

# Import DST enums for consistency
from dst_data_builder.training_modules.dst_enums import DSTTransition, DSTState


@dataclass
class TemporalViolation:
    """Represents a temporal ordering violation"""

    violation_type: str  # 'overlap', 'regression', 'gap', 'invalid_order'
    span1_id: int
    span2_id: int
    span1_end: float
    span2_start: float
    severity: str  # 'error', 'warning', 'info'
    description: str
    suggested_fix: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of temporal validation"""

    is_valid: bool
    violations: List[TemporalViolation]
    sorted_spans: List[Dict[str, Any]]  # Sorted by start time, otherwise unchanged
    span_count: int
    violation_count: int


class TemporalOrderingValidator:
    """
    Temporal Ordering Validator: Detect temporal ordering issues in DST spans

    This class detects temporal ordering violations in DST spans (overlaps, regressions, gaps)
    but does not apply fixes. Violations are reported for handling by other components
    in the pipeline (span construction, etc.). Overlaps are allowed for DST spans since
    assembly steps can happen in parallel.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration for validation
        self.max_allowed_overlap = config.get("max_allowed_overlap", 5.0)  # seconds
        self.min_gap_duration = config.get("min_gap_duration", 0.1)  # seconds
        self.max_gap_duration = config.get(
            "max_gap_duration", 300.0
        )  # 5 minutes max gap

    def validate_temporal_ordering(
        self, dst_spans: List[Dict[str, Any]]
    ) -> ValidationResult:
        """
        Detect temporal ordering issues in DST spans

        This method detects violations but does not apply fixes. The returned spans
        are sorted by start time but otherwise unchanged. Fixes should be handled
        by other components in the pipeline.

        Args:
            dst_spans: List of DST span dictionaries

        Returns:
            ValidationResult with detected violations and sorted spans (no fixes applied)
        """
        if not dst_spans:
            self.logger.warning("No DST spans provided for temporal validation")
            return ValidationResult(True, [], [], 0, 0)


        # Extract temporal information and collect extraction violations
        spans_with_time, extraction_violations = self._extract_temporal_info(dst_spans)

        if len(spans_with_time) < 2:
            # Check if there are any error violations
            has_errors = any(v.severity == "error" for v in extraction_violations)
            return ValidationResult(
                not has_errors,
                extraction_violations,
                dst_spans,
                len(dst_spans),
                len(extraction_violations),
            )

        # Sort by start time for proper ordering
        sorted_spans = sorted(spans_with_time, key=lambda x: x["start_ts"])

        # Detect temporal violations
        ordering_violations = self._detect_violations(sorted_spans)
        all_violations = extraction_violations + ordering_violations

        # For DST spans, we only detect violations - no fixes applied
        # Return sorted spans unchanged
        final_violations = all_violations

        # Count results
        is_valid = len(final_violations) == 0 or all(
            v.severity != "error" for v in final_violations
        )
        original_count = len(dst_spans)
        violation_count = len(final_violations)

        # Log validation results
        self._log_validation_results(
            original_count, violation_count, final_violations
        )

        return ValidationResult(
            is_valid=is_valid,
            violations=final_violations,
            sorted_spans=sorted_spans,  # Sorted but otherwise unchanged
            span_count=original_count,
            violation_count=violation_count,
        )

    def _extract_temporal_info(
        self, dst_spans: List[Dict[str, Any]]
    ) -> Tuple[List[Dict[str, Any]], List[TemporalViolation]]:
        """
        Extract temporal information from DST spans

        Args:
            dst_spans: List of DST span dictionaries

        Returns:
            Tuple of (spans with extracted temporal info, extraction violations)
        """
        spans_with_time = []
        violations = []

        for span in dst_spans:
            # Extract time information
            t0 = self._extract_start_time(span)
            t1 = self._extract_end_time(span)

            if t0 is not None and t1 is not None and t1 > t0:
                span_with_time = {
                    "id": span.get("id", 0),
                    "name": span.get("name", ""),
                    "start_ts": t0,
                    "end_ts": t1,
                    "conf": span.get("conf", 0.5),
                    "original_span": span,
                }
                spans_with_time.append(span_with_time)
            else:
                # Create violation for invalid temporal data
                violation = TemporalViolation(
                    violation_type="invalid_temporal_data",
                    span1_id=span.get("id", 0),
                    span2_id=-1,
                    span1_end=t1 if t1 is not None else 0,
                    span2_start=t0 if t0 is not None else 0,
                    severity="error",
                    description=f"Invalid temporal data in span {span.get('id', 0)}: t0={t0}, t1={t1}",
                    suggested_fix="Ensure span has valid t0 < t1",
                )
                violations.append(violation)
                self.logger.warning("Invalid temporal data in span: %s", span)

        return spans_with_time, violations

    def _extract_start_time(self, span: Dict[str, Any]) -> Optional[float]:
        """Extract start time from span"""
        for time_field in ["t0", "start_time", "start_ts"]:
            if time_field in span:
                return float(span[time_field])
        return None

    def _extract_end_time(self, span: Dict[str, Any]) -> Optional[float]:
        """Extract end time from span"""
        for time_field in ["t1", "end_time", "end_ts"]:
            if time_field in span:
                return float(span[time_field])
        return None

    def _detect_violations(
        self, sorted_spans: List[Dict[str, Any]]
    ) -> List[TemporalViolation]:
        """
        Detect temporal ordering violations in sorted spans

        Args:
            sorted_spans: List of spans sorted by start time

        Returns:
            List of detected violations
        """
        violations = []

        for i in range(len(sorted_spans) - 1):
            current_span = sorted_spans[i]
            next_span = sorted_spans[i + 1]

            current_end = current_span["end_ts"]
            next_start = next_span["start_ts"]

            # Check for temporal violations
            if next_start < current_end:
                # Overlap violation - for DST spans, overlaps are generally not allowed
                overlap_duration = current_end - next_start
                violation = TemporalViolation(
                    violation_type="overlap",
                    span1_id=current_span["id"],
                    span2_id=next_span["id"],
                    span1_end=current_end,
                    span2_start=next_start,
                    severity="error",  # Treat all overlaps as errors for DST spans
                    description=f"Span {current_span['id']} overlaps with span {next_span['id']} by {overlap_duration:.2f}s",
                    suggested_fix="Adjust start/end times to eliminate overlap",
                )
                violations.append(violation)

            elif next_start > current_end + self.max_gap_duration:
                # Gap violation (too large)
                gap_duration = next_start - current_end
                violation = TemporalViolation(
                    violation_type="gap",
                    span1_id=current_span["id"],
                    span2_id=next_span["id"],
                    span1_end=current_end,
                    span2_start=next_start,
                    severity="warning",
                    description=f"Large gap of {gap_duration:.2f}s between spans {current_span['id']} and {next_span['id']}",
                    suggested_fix="Consider if gap is intentional or if steps should be merged",
                )
                violations.append(violation)

            # Check for regression (next span starts before current span)
            elif next_start < current_span["start_ts"]:
                violation = TemporalViolation(
                    violation_type="regression",
                    span1_id=current_span["id"],
                    span2_id=next_span["id"],
                    span1_end=current_end,
                    span2_start=next_start,
                    severity="error",
                    description=f"Regression: span {next_span['id']} starts before span {current_span['id']}",
                    suggested_fix="Reorder spans chronologically",
                )
                violations.append(violation)

        # Check for invalid span duration (negative or zero duration)
        for span in sorted_spans:
            duration = span["end_ts"] - span["start_ts"]
            if duration <= 0:
                violation = TemporalViolation(
                    violation_type="invalid_duration",
                    span1_id=span["id"],
                    span2_id=-1,
                    span1_end=span["end_ts"],
                    span2_start=span["start_ts"],
                    severity="error",
                    description=f"Invalid duration: span {span['id']} has duration {duration:.2f}s",
                    suggested_fix="Ensure end_time > start_time",
                )
                violations.append(violation)

        return violations



    def _log_validation_results(
        self,
        span_count: int,
        violation_count: int,
        violations: List[TemporalViolation],
    ):
        """Log validation results"""
        if violation_count > 0:
            error_count = len([v for v in violations if v.severity == "error"])
            warning_count = len([v for v in violations if v.severity == "warning"])

            self.logger.warning(
                "⚠️ Temporal validation issues: %d violations (%d errors, %d warnings)",
                violation_count,
                error_count,
                warning_count,
            )

            # Log first few violations for debugging
            for i, violation in enumerate(violations[:3]):
                self.logger.debug("Violation %d: %s", i + 1, violation.description)

            if violation_count > 3:
                self.logger.debug("... and %d more violations", violation_count - 3)

        # Note: This validator detects violations but does not apply fixes
        # Fixes should be handled by other components in the pipeline

    def get_validation_summary(self, result: ValidationResult) -> Dict[str, Any]:
        """
        Get a summary of validation results

        Args:
            result: Result from validate_temporal_ordering

        Returns:
            Dictionary with validation summary
        """
        if not result.violations:
            return {
                "status": "valid",
                "spans_count": result.span_count,
                "violations_count": 0,
                "message": "All spans have valid temporal ordering",
            }

        # Count violations by type and severity
        violation_summary = {}
        for violation in result.violations:
            key = f"{violation.violation_type}_{violation.severity}"
            violation_summary[key] = violation_summary.get(key, 0) + 1

        return {
            "status": "invalid" if not result.is_valid else "warning",
            "spans_count": result.span_count,
            "violations_count": result.violation_count,
            "violation_types": violation_summary,
            "message": f"Found {result.violation_count} temporal violations",
        }

    def validate_single_span(self, span: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """
        Validate a single span for temporal consistency

        Args:
            span: Single DST span

        Returns:
            Tuple of (is_valid, error_message)
        """
        t0 = self._extract_start_time(span)
        t1 = self._extract_end_time(span)

        if t0 is None or t1 is None:
            return False, "Missing temporal information"

        if t1 <= t0:
            return False, f"Invalid duration: end_time ({t1}) <= start_time ({t0})"

        duration = t1 - t0
        if duration < self.min_gap_duration:
            return (
                False,
                f"Duration too short: {duration:.3f}s < {self.min_gap_duration}s",
            )

        if duration > self.max_gap_duration:
            return (
                False,
                f"Duration too long: {duration:.1f}s > {self.max_gap_duration}s",
            )

        return True, None
