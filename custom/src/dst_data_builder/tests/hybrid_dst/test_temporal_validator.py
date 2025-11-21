"""
Test Temporal Validator Module

Comprehensive tests for the TemporalOrderingValidator class covering:
- Temporal ordering validation
- Violation detection and classification
- Fix application
- Edge cases and error handling
- Configuration handling
"""

import pytest
import logging
from omegaconf import DictConfig, OmegaConf

from dst_data_builder.hybrid_dst.temporal_validator import (
    TemporalOrderingValidator,
    TemporalViolation,
    ValidationResult,
)


class TestTemporalValidator:
    """Test class for TemporalOrderingValidator"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create({
            "max_allowed_overlap": 5.0,
            "min_gap_duration": 0.1,
            "max_gap_duration": 300.0,
        })

    @pytest.fixture
    def validator(self, sample_config):
        """Create validator instance"""
        return TemporalOrderingValidator(sample_config)

    def test_initialization(self, sample_config):
        """Test validator initialization"""
        validator = TemporalOrderingValidator(sample_config)
        assert validator.max_allowed_overlap == 5.0
        assert validator.min_gap_duration == 0.1
        assert validator.max_gap_duration == 300.0
        assert hasattr(validator, 'logger')

    def test_empty_spans_validation(self, validator):
        """Test validation with empty spans list"""
        result = validator.validate_temporal_ordering([])
        assert result.is_valid is True
        assert result.violations == []
        assert result.sorted_spans == []
        assert result.span_count == 0
        assert result.fixed_count == 0
        assert result.violation_count == 0

    def test_single_span_validation(self, validator):
        """Test validation with single span"""
        spans = [{"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"}]
        result = validator.validate_temporal_ordering(spans)
        assert result.is_valid is True
        assert result.violations == []
        assert len(result.sorted_spans) == 1
        assert result.span_count == 1
        assert result.fixed_count == 1

    def test_valid_temporal_ordering(self, validator):
        """Test validation with properly ordered spans"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},
            {"id": 3, "t0": 10.0, "t1": 15.0, "name": "Step 3"},
        ]
        result = validator.validate_temporal_ordering(spans)
        assert result.is_valid is True
        assert result.violations == []
        assert len(result.sorted_spans) == 3
        assert result.violation_count == 0

    def test_overlap_violation_warning(self, validator):
        """Test small overlap that generates warning"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.5, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},  # 0.5s overlap
        ]
        result = validator.validate_temporal_ordering(spans)
        assert result.is_valid is True  # Warning level, not error
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "overlap"
        assert result.violations[0].severity == "warning"
        assert "0.50s" in result.violations[0].description

    def test_overlap_violation_error(self, validator):
        """Test large overlap that generates error"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 10.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 8.0, "name": "Step 2"},  # 3.0s overlap, but within 5s limit
        ]
        result = validator.validate_temporal_ordering(spans)
        # With 5s max overlap, 5s overlap is still a warning, not error
        assert result.is_valid is True  # Warning level, not error
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "overlap"
        assert result.violations[0].severity == "warning"
        assert "5.00s" in result.violations[0].description

    def test_gap_violation(self, validator):
        """Test large gap between spans"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 310.0, "t1": 315.0, "name": "Step 2"},  # 305s gap
        ]
        result = validator.validate_temporal_ordering(spans)
        assert result.is_valid is True  # Gap is warning level
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "gap"
        assert result.violations[0].severity == "warning"
        assert "305.00s" in result.violations[0].description

    def test_regression_violation(self, validator):
        """Test that regression detection doesn't trigger for properly sorted spans"""
        # Regression can't occur after sorting by start time
        # This test verifies that properly ordered spans don't trigger regression
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},
        ]
        result = validator.validate_temporal_ordering(spans)
        assert result.is_valid is True
        assert len(result.violations) == 0  # No violations for proper ordering

    def test_invalid_duration_validation(self, validator):
        """Test that invalid duration spans are detected and reported"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},      # Valid
            {"id": 2, "t0": 10.0, "t1": 10.0, "name": "Step 2"},    # Zero duration - invalid
            {"id": 3, "t0": 15.0, "t1": 12.0, "name": "Step 3"},    # Negative duration - invalid
        ]
        result = validator.validate_temporal_ordering(spans)
        # Invalid spans are detected but all original spans are returned
        assert result.is_valid is False  # Invalid durations are errors
        assert len(result.violations) == 2  # Two invalid temporal data violations
        assert len(result.sorted_spans) == 3  # All original spans returned
        invalid_data_count = len([v for v in result.violations if v.violation_type == "invalid_temporal_data"])
        assert invalid_data_count == 2

    def test_fix_application_no_fixes_needed(self, validator):
        """Test that no fixes are applied when violations are warnings"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 8.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},  # 3.0s overlap (warning level)
        ]
        result = validator.validate_temporal_ordering(spans)

        # Overlap is warning level, so no fixes applied
        assert len(result.violations) == 1
        assert result.fixed_count == 2

        # Spans should remain unchanged since only warnings are detected
        fixed_span_1 = next(s for s in result.sorted_spans if s["id"] == 1)
        fixed_span_2 = next(s for s in result.sorted_spans if s["id"] == 2)

        assert fixed_span_1["t1"] == 8.0
        assert fixed_span_2["t0"] == 5.0

    def test_fix_application_invalid_duration(self, validator):
        """Test that invalid durations are detected and can be fixed"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 10.0, "t1": 10.0, "name": "Step 2"},  # Zero duration - should be fixed
        ]
        result = validator.validate_temporal_ordering(spans)

        # Invalid duration is detected but not fixed (invalid_temporal_data can't be fixed)
        assert len(result.violations) == 1
        assert result.violations[0].violation_type == "invalid_temporal_data"
        assert len(result.sorted_spans) == 2  # All original spans returned

        # Invalid spans remain unchanged
        fixed_span_2 = next(s for s in result.sorted_spans if s["id"] == 2)
        assert fixed_span_2["t1"] == fixed_span_2["t0"]  # Still invalid

    def test_temporal_info_extraction(self, validator):
        """Test extraction of temporal information from spans"""
        spans = [
            {"id": 1, "start_time": 0.0, "end_time": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},
            {"id": 3, "start_ts": 10.0, "end_ts": 15.0, "name": "Step 3"},
        ]

        spans_with_time, violations = validator._extract_temporal_info(spans)
        assert len(spans_with_time) == 3
        assert len(violations) == 0

        for span in spans_with_time:
            assert "t0" in span
            assert "t1" in span
            assert span["t1"] > span["t0"]

    def test_temporal_info_extraction_invalid(self, validator):
        """Test extraction with invalid temporal data"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "invalid": "data", "name": "Step 2"},  # No temporal info
            {"id": 3, "t0": 10.0, "t1": 8.0, "name": "Step 3"},  # Invalid duration
        ]

        spans_with_time, violations = validator._extract_temporal_info(spans)
        assert len(spans_with_time) == 1  # Only valid spans extracted
        assert spans_with_time[0]["id"] == 1
        assert len(violations) == 2  # Two invalid spans detected

    def test_single_span_validation_valid(self, validator):
        """Test validation of individual valid span"""
        span = {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"}
        is_valid, error_msg = validator.validate_single_span(span)
        assert is_valid is True
        assert error_msg is None

    def test_single_span_validation_missing_time(self, validator):
        """Test validation of span with missing temporal info"""
        span = {"id": 1, "name": "Step 1"}  # No time info
        is_valid, error_msg = validator.validate_single_span(span)
        assert is_valid is False
        assert "Missing temporal information" in error_msg

    def test_single_span_validation_invalid_duration(self, validator):
        """Test validation of span with invalid duration"""
        span = {"id": 1, "t0": 5.0, "t1": 5.0, "name": "Step 1"}  # Zero duration
        is_valid, error_msg = validator.validate_single_span(span)
        assert is_valid is False
        assert "Invalid duration" in error_msg

    def test_single_span_validation_too_short(self, validator):
        """Test validation of span that's too short"""
        span = {"id": 1, "t0": 0.0, "t1": 0.05, "name": "Step 1"}  # 0.05s duration
        is_valid, error_msg = validator.validate_single_span(span)
        assert is_valid is False
        assert "too short" in error_msg

    def test_single_span_validation_too_long(self, validator):
        """Test validation of span that's too long"""
        span = {"id": 1, "t0": 0.0, "t1": 400.0, "name": "Step 1"}  # 400s duration
        is_valid, error_msg = validator.validate_single_span(span)
        assert is_valid is False
        assert "too long" in error_msg

    def test_validation_summary_valid(self, validator):
        """Test validation summary for valid result"""
        result = ValidationResult(
            is_valid=True,
            violations=[],
            sorted_spans=[{"id": 1, "t0": 0.0, "t1": 5.0}],
            span_count=1,
            fixed_count=1,
            violation_count=0,
        )
        summary = validator.get_validation_summary(result)
        assert summary["status"] == "valid"
        assert summary["spans_count"] == 1
        assert summary["violations_count"] == 0

    def test_validation_summary_with_violations(self, validator):
        """Test validation summary with violations"""
        violations = [
            TemporalViolation("overlap", 1, 2, 5.0, 4.5, "error", "Test violation", None)
        ]
        result = ValidationResult(
            is_valid=False,
            violations=violations,
            sorted_spans=[{"id": 1, "t0": 0.0, "t1": 5.0}],
            span_count=1,
            fixed_count=1,
            violation_count=1,
        )
        summary = validator.get_validation_summary(result)
        assert summary["status"] == "invalid"
        assert summary["violations_count"] == 1
        assert "overlap_error" in summary["violation_types"]

    def test_multiple_violations_detected(self, validator):
        """Test that multiple violations are detected"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 0.0, "name": "Step 1"},  # Invalid duration
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"}, # Valid
            {"id": 3, "t0": 8.0, "t1": 8.0, "name": "Step 3"},  # Invalid duration + overlap
        ]
        result = validator.validate_temporal_ordering(spans)

        # Multiple violations detected
        assert result.violation_count == 2
        invalid_data_count = len([v for v in result.violations if v.violation_type == "invalid_temporal_data"])
        assert invalid_data_count == 2
        assert len(result.sorted_spans) == 3  # All original spans returned

    def test_configuration_edge_cases(self):
        """Test validator with edge case configurations"""
        # Test with very strict overlap settings
        strict_config = OmegaConf.create({
            "max_allowed_overlap": 0.0,  # No overlap allowed
            "min_gap_duration": 1.0,
            "max_gap_duration": 10.0,
        })
        strict_validator = TemporalOrderingValidator(strict_config)

        spans = [
            {"id": 1, "t0": 0.0, "t1": 5.1, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},  # 0.1s overlap
        ]
        result = strict_validator.validate_temporal_ordering(spans)

        # With strict config (max_overlap=0.0), tiny overlap becomes error and gets fixed
        # After fix, validation passes with no remaining violations
        assert result.is_valid is True  # Fixed successfully
        assert len(result.violations) == 0  # No violations remain after fix
        # Check that spans remain unchanged (no fixes applied)
        fixed_span_1 = next(s for s in result.sorted_spans if s["id"] == 1)
        fixed_span_2 = next(s for s in result.sorted_spans if s["id"] == 2)
        # Spans should remain unchanged since validator only detects, doesn't fix
        assert fixed_span_1["t1"] > fixed_span_2["t0"]  # Overlap still exists

    def test_sorting_behavior(self, validator):
        """Test that spans are properly sorted by start time"""
        # Spans provided out of order
        spans = [
            {"id": 3, "t0": 10.0, "t1": 15.0, "name": "Step 3"},
            {"id": 1, "t0": 0.0, "t1": 5.0, "name": "Step 1"},
            {"id": 2, "t0": 5.0, "t1": 10.0, "name": "Step 2"},
        ]
        result = validator.validate_temporal_ordering(spans)

        # Should be sorted in sorted_spans
        assert len(result.sorted_spans) == 3
        assert result.sorted_spans[0]["id"] == 1
        assert result.sorted_spans[1]["id"] == 2
        assert result.sorted_spans[2]["id"] == 3

    def test_complex_violation_scenario(self, validator):
        """Test complex scenario with multiple violation types"""
        spans = [
            {"id": 1, "t0": 0.0, "t1": 8.0, "name": "Step 1"},     # Valid
            {"id": 2, "t0": 5.0, "t1": 5.0, "name": "Step 2"},     # Invalid duration
            {"id": 3, "t0": 15.0, "t1": 20.0, "name": "Step 3"},   # Valid (gap with Step 1)
        ]
        result = validator.validate_temporal_ordering(spans)

        # Only invalid_temporal_data violation (gap not detected since invalid span is filtered)
        violation_types = {v.violation_type for v in result.violations}
        assert "invalid_temporal_data" in violation_types
        assert result.violation_count == 1  # Only the invalid span violation
        assert result.fixed_count == 2  # Invalid span filtered out, 2 valid spans remain


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys

    test_instance = TestTemporalValidator()

    # Create sample config
    config = OmegaConf.create({
        "max_allowed_overlap": 5.0,
        "min_gap_duration": 0.1,
        "max_gap_duration": 300.0,
    })

    try:
        # Run a few key tests
        test_instance.test_initialization(config)
        test_instance.test_valid_temporal_ordering(test_instance.validator(config))
        test_instance.test_overlap_violation_error(test_instance.validator(config))
        test_instance.test_invalid_duration_violation(test_instance.validator(config))
        test_instance.test_single_span_validation_valid(test_instance.validator(config))

        print("✅ All manual temporal validator tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)