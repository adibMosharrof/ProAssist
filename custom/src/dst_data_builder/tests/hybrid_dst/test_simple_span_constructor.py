"""
Test SimpleSpanConstructor Implementation

This module tests the simple span constructor that provides direct timestamp assignment
when the number of filtered blocks exactly matches the number of inferred knowledge steps.
"""

import pytest
from omegaconf import OmegaConf
from dst_data_builder.hybrid_dst.span_constructors.simple_span_constructor import (
    SimpleSpanConstructor,
    SimpleSpanConstructionResult,
)


class TestSimpleSpanConstructor:
    """Test class for SimpleSpanConstructor"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create(
            {
                "default_point_duration": 5.0,
                "min_span_duration": 1.0,
                "max_span_duration": 300.0,
            }
        )

    @pytest.fixture
    def constructor(self, sample_config):
        """Create SimpleSpanConstructor instance"""
        return SimpleSpanConstructor(sample_config)

    @pytest.fixture
    def sample_blocks_equal(self):
        """Create sample filtered blocks for equal count tests"""
        return [
            {"start_time": 10.0, "end_time": 20.0, "has_end_time": True},
            {"start_time": 30.0, "end_time": 40.0, "has_end_time": True},
            {"start_time": 50.0, "end_time": 60.0, "has_end_time": True},
        ]

    @pytest.fixture
    def sample_knowledge_equal(self):
        """Create sample inferred knowledge for equal count tests"""
        return ["Step 1", "Step 2", "Step 3"]

    @pytest.fixture
    def sample_blocks_point(self):
        """Create sample filtered blocks with point blocks"""
        return [
            {"start_time": 10.0, "end_time": 20.0, "has_end_time": True},
            {
                "start_time": 25.0,
                "end_time": None,
                "has_end_time": False,
            },  # Point block
        ]

    @pytest.fixture
    def sample_knowledge_point(self):
        """Create sample inferred knowledge for point block tests"""
        return ["Step 1", "Step 2"]

    @pytest.fixture
    def sample_blocks_two(self):
        """Create sample filtered blocks with 2 items"""
        return [
            {"start_time": 10.0, "end_time": 20.0, "has_end_time": True},
            {"start_time": 30.0, "end_time": 40.0, "has_end_time": True},
        ]

    @pytest.fixture
    def sample_knowledge_one(self):
        """Create sample inferred knowledge with 1 item"""
        return ["Step 1"]

    @pytest.fixture
    def sample_knowledge_two(self):
        """Create sample inferred knowledge with 2 items"""
        return ["Step 1", "Step 2"]

    @pytest.fixture
    def sample_blocks_out_of_order(self):
        """Create sample filtered blocks in out-of-order timestamps"""
        return [
            {"start_time": 50.0, "end_time": 60.0, "has_end_time": True},
            {"start_time": 10.0, "end_time": 20.0, "has_end_time": True},
            {"start_time": 30.0, "end_time": 40.0, "has_end_time": True},
        ]

    @pytest.fixture
    def real_blocks_base(self):
        """Base real data filtered blocks from documentation"""
        return [
            {
                "start_time": 94.4,
                "end_time": 105.2,
                "has_end_time": True,
                "text": "attach interior to chassis",
            },
            {
                "start_time": 105.2,
                "end_time": 153.6,
                "has_end_time": True,
                "text": "attach wheel to chassis",
            },
            {
                "start_time": 153.6,
                "end_time": 171.7,
                "has_end_time": True,
                "text": "attach arm to turntable top",
            },
            {
                "start_time": 171.7,
                "end_time": 187.1,
                "has_end_time": True,
                "text": "attach hook to arm",
            },
            {
                "start_time": 187.1,
                "end_time": 203.7,
                "has_end_time": True,
                "text": "attach turntable top to chassis",
            },
            {
                "start_time": 203.7,
                "end_time": 213.1,
                "has_end_time": True,
                "text": "attach cabin to interior",
            },
            {
                "start_time": 213.1,
                "end_time": 232.0,
                "has_end_time": True,
                "text": "demonstrate functionality",
            },
        ]

    @pytest.fixture
    def real_knowledge_six(self):
        """Real inferred knowledge with 6 steps"""
        return [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
            "Attach the body to the chassis.",
            "Add the cabin window to the chassis.",
            "Finalize the assembly and demonstrate the toy's functionality.",
        ]

    @pytest.fixture
    def real_blocks_with_points(self):
        """Real blocks with point blocks for equal count test (6 blocks)"""
        return [
            {
                "start_time": 94.4,
                "end_time": 105.2,
                "has_end_time": True,
                "text": "attach interior to chassis",
            },
            {
                "start_time": 105.2,
                "end_time": 153.6,
                "has_end_time": True,
                "text": "attach wheel to chassis",
            },
            {
                "start_time": 164.2,
                "end_time": None,  # Point block
                "has_end_time": False,
                "text": "screw turntable top with screwdriver",
            },
            {
                "start_time": 171.7,
                "end_time": 187.1,
                "has_end_time": True,
                "text": "attach hook to arm",
            },
            {
                "start_time": 180.5,
                "end_time": None,  # Point block
                "has_end_time": False,
                "text": "screw hook with hand",
            },
            {
                "start_time": 187.1,
                "end_time": 203.7,
                "has_end_time": True,
                "text": "attach turntable top to chassis",
            },
        ]

    @pytest.fixture
    def real_knowledge_with_points(self):
        """Real knowledge with point block steps"""
        return [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Screw turntable top securely.",
            "Attach the hook to arm.",
            "Screw the hook in place.",
            "Finalize the assembly.",
        ]

    @pytest.fixture
    def real_knowledge_four(self):
        """Real inferred knowledge with 4 steps"""
        return [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Assemble the arm and attach it to the chassis.",
            "Finalize the assembly.",
        ]

    @pytest.fixture
    def real_blocks_unequal(self):
        """Real blocks for unequal count test (7 blocks)"""
        return [
            {
                "start_time": 94.4,
                "end_time": 105.2,
                "has_end_time": True,
                "text": "attach interior to chassis",
            },
            {
                "start_time": 105.2,
                "end_time": 153.6,
                "has_end_time": True,
                "text": "attach wheel to chassis",
            },
            {
                "start_time": 153.6,
                "end_time": 171.7,
                "has_end_time": True,
                "text": "attach arm to turntable top",
            },
            {
                "start_time": 164.2,
                "end_time": None,  # Point block
                "has_end_time": False,
                "text": "screw turntable top with screwdriver",
            },
            {
                "start_time": 171.7,
                "end_time": 187.1,
                "has_end_time": True,
                "text": "attach hook to arm",
            },
            {
                "start_time": 180.5,
                "end_time": None,  # Point block
                "has_end_time": False,
                "text": "screw hook with hand",
            },
            {
                "start_time": 187.1,
                "end_time": 203.7,
                "has_end_time": True,
                "text": "attach turntable top to chassis",
            },
        ]

    @pytest.fixture
    def real_blocks_out_of_order(self):
        """Real blocks in out-of-order for temporal validation"""
        return [
            {
                "start_time": 213.1,
                "end_time": 232.0,
                "has_end_time": True,
                "text": "demonstrate functionality",
            },
            {
                "start_time": 94.4,
                "end_time": 105.2,
                "has_end_time": True,
                "text": "attach interior to chassis",
            },
            {
                "start_time": 105.2,
                "end_time": 153.6,
                "has_end_time": True,
                "text": "attach wheel to chassis",
            },
            {
                "start_time": 171.7,
                "end_time": 187.1,
                "has_end_time": True,
                "text": "attach hook to arm",
            },
        ]

    @pytest.fixture
    def real_knowledge_four_temporal(self):
        """Real knowledge for temporal validation test"""
        return [
            "Assemble the chassis by attaching and screwing the chassis parts together.",
            "Attach wheels to the chassis.",
            "Attach the hook to arm.",
            "Demonstrate functionality.",
        ]

    def test_equal_counts_simple_case(
        self, constructor, sample_blocks_equal, sample_knowledge_equal
    ):
        """Test simple case with equal counts"""
        result = constructor.construct_spans(
            sample_blocks_equal, sample_knowledge_equal
        )

        assert result.total_blocks_processed == 3
        assert len(result.dst_spans) == 3

        # Check first span
        span1 = result.dst_spans[0]
        assert span1["id"] == 1
        assert span1["name"] == "Step 1"
        assert span1["t0"] == 10.0
        assert span1["t1"] == 20.0
        assert span1["conf"] == 1.0
        assert span1["source"] == "simple_span"

        # Check temporal ordering
        assert result.dst_spans[0]["t1"] <= result.dst_spans[1]["t0"]

    def test_point_blocks_handling(
        self, constructor, sample_blocks_point, sample_knowledge_point
    ):
        """Test handling of point blocks (single timestamp)"""
        result = constructor.construct_spans(
            sample_blocks_point, sample_knowledge_point
        )

        assert len(result.dst_spans) == 2

        # Point block should get default duration
        point_span = result.dst_spans[1]
        assert point_span["t0"] == 25.0
        assert point_span["t1"] == 30.0  # 25.0 + 5.0 (default_point_duration)

    def test_unequal_counts_error(
        self, constructor, sample_blocks_two, sample_knowledge_one
    ):
        """Test error handling for unequal counts"""
        result = constructor.construct_spans(sample_blocks_two, sample_knowledge_one)

        assert len(result.dst_spans) == 0
        assert result.construction_statistics["error"] == "Unequal counts"

    def test_empty_inputs(self, constructor):
        """Test handling of empty inputs"""

        # Test with empty blocks
        result1 = constructor.construct_spans([], ["Step 1"])
        assert len(result1.dst_spans) == 0

        # Test with empty knowledge
        result2 = constructor.construct_spans(
            [{"start_time": 10.0, "end_time": 20.0}], []
        )
        assert len(result2.dst_spans) == 0

    def test_temporal_ordering(
        self, constructor, sample_blocks_out_of_order, sample_knowledge_equal
    ):
        """Test that spans are properly ordered by start time"""
        result = constructor.construct_spans(
            sample_blocks_out_of_order, sample_knowledge_equal
        )

        assert len(result.dst_spans) == 3

        # Check that spans are sorted by start time
        for i in range(len(result.dst_spans) - 1):
            current_span = result.dst_spans[i]
            next_span = result.dst_spans[i + 1]
            assert (
                current_span["t0"] <= next_span["t0"]
            ), f"Spans not ordered: {current_span['t0']} > {next_span['t0']}"

        # Check that IDs are reassigned sequentially
        for i, span in enumerate(result.dst_spans, 1):
            assert span["id"] == i

    def test_construction_statistics(
        self, constructor, sample_blocks_two, sample_knowledge_two
    ):
        """Test that construction statistics are properly generated"""
        result = constructor.construct_spans(sample_blocks_two, sample_knowledge_two)

        stats = result.construction_statistics

        assert stats["construction_type"] == "simple_span_constructor"
        assert stats["total_blocks_input"] == 2
        assert stats["total_spans_created"] == 2
        assert "duration_statistics" in stats
        assert "temporal_range" in stats
        assert "efficiency" in stats

        # Check efficiency stats
        efficiency = stats["efficiency"]
        assert efficiency["processing_type"] == "direct_assignment"
        assert efficiency["complexity"] == "O(n)"
        assert efficiency["llm_calls"] == 0
        assert efficiency["model_calls"] == 0

    def test_real_data_format_from_documentation(
        self, constructor, real_blocks_base, real_knowledge_six
    ):
        """Test with real data format from the documentation - unequal counts should return empty"""
        result = constructor.construct_spans(real_blocks_base, real_knowledge_six)

        # SimpleSpanConstructor requires equal counts, so this should return empty
        assert len(result.dst_spans) == 0
        assert result.total_blocks_processed == 7
        assert result.construction_statistics["error"] == "Unequal counts"
        assert result.construction_statistics["blocks"] == 7
        assert result.construction_statistics["steps"] == 6

    def test_real_data_with_point_blocks(
        self, constructor, real_blocks_with_points, real_knowledge_with_points
    ):
        """Test with real data including point blocks"""
        result = constructor.construct_spans(
            real_blocks_with_points, real_knowledge_with_points
        )

        assert len(result.dst_spans) == 6
        assert result.total_blocks_processed == 6

        # Check point blocks got default duration
        point_span1 = result.dst_spans[2]
        assert point_span1["t0"] == 164.2
        assert point_span1["t1"] == 169.2  # 164.2 + 5.0 (default duration)

        point_span2 = result.dst_spans[4]
        assert point_span2["t0"] == 180.5
        assert point_span2["t1"] == 185.5  # 180.5 + 5.0 (default duration)

    def test_mixed_real_data_unequal_counts(
        self, constructor, real_blocks_unequal, real_knowledge_four
    ):
        """Test real data scenario where counts don't match"""
        result = constructor.construct_spans(real_blocks_unequal, real_knowledge_four)

        # Should return empty result since counts don't match
        assert len(result.dst_spans) == 0
        assert result.construction_statistics["error"] == "Unequal counts"
        assert result.construction_statistics["blocks"] == 7
        assert result.construction_statistics["steps"] == 4

    def test_temporal_validation_with_real_data(
        self, constructor, real_blocks_out_of_order, real_knowledge_four_temporal
    ):
        """Test temporal ordering with real data timestamps"""
        result = constructor.construct_spans(
            real_blocks_out_of_order, real_knowledge_four_temporal
        )

        # Check that spans are properly ordered by time
        assert len(result.dst_spans) == 4

        # Verify temporal ordering
        for i in range(len(result.dst_spans) - 1):
            current_span = result.dst_spans[i]
            next_span = result.dst_spans[i + 1]
            assert (
                current_span["t0"] < next_span["t0"]
            ), f"Spans not ordered: {current_span['t0']} >= {next_span['t0']}"

        # Verify sequential IDs
        for i, span in enumerate(result.dst_spans, 1):
            assert span["id"] == i, f"Span {i} has incorrect ID: {span['id']}"
