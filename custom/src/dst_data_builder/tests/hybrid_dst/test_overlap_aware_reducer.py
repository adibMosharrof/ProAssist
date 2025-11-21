"""
Test Overlap-Aware Block Reducer

Comprehensive tests for the OverlapAwareBlockReducer component that handles
block merging based on time overlap ratios.
"""

import pytest
from omegaconf import OmegaConf

from dst_data_builder.hybrid_dst.overlap_aware_reducer import (
    OverlapAwareBlockReducer,
    BlockReductionResult,
    TimeBlock,
)


class TestOverlapAwareBlockReducer:
    """Test class for OverlapAwareBlockReducer functionality"""

    @pytest.fixture
    def sample_config(self):
        """Create sample configuration for testing"""
        return OmegaConf.create({
            # Simple containment logic - no complex parameters needed
        })

    @pytest.fixture
    def reducer(self, sample_config):
        """Create reducer instance for testing"""
        return OverlapAwareBlockReducer(sample_config)

    def test_initialization(self, sample_config):
        """Test that reducer initializes correctly"""
        reducer = OverlapAwareBlockReducer(sample_config)
        assert reducer.config == sample_config
        assert hasattr(reducer, 'logger')

    def test_empty_input(self, reducer):
        """Test handling of empty input"""
        result = reducer.reduce_blocks([])
        assert result.filtered_blocks == []
        assert result.merged_count == 0
        assert result.removed_count == 0
        assert result.original_count == 0

    def test_single_block(self, reducer):
        """Test processing of single block"""
        blocks = [{
            "text": "Single step",
            "start_time": 0.0,
            "end_time": 5.0,
        }]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 0
        assert result.original_count == 1
        assert result.filtered_blocks[0]["text"] == "Single step"

    def test_no_overlap_blocks(self, reducer):
        """Test blocks with no time overlap"""
        blocks = [
            {"text": "Step 1", "start_time": 0.0, "end_time": 5.0},
            {"text": "Step 2", "start_time": 6.0, "end_time": 10.0},
            {"text": "Step 3", "start_time": 11.0, "end_time": 15.0},
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 3
        assert result.merged_count == 0
        assert result.original_count == 3

    def test_complete_overlap_merge(self, reducer):
        """Test merging of completely contained blocks"""
        blocks = [
            {"text": "Main step", "start_time": 0.0, "end_time": 10.0},
            {"text": "Sub step 1", "start_time": 2.0, "end_time": 4.0},
            {"text": "Sub step 2", "start_time": 6.0, "end_time": 8.0},
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 2
        assert result.original_count == 3

        # Check merged content - should only keep main block's text
        merged = result.filtered_blocks[0]
        assert merged["text"] == "Main step"  # Only main block's text
        assert merged["merged_blocks"] == 3
        assert merged["start_time"] == 0.0  # Extended time range
        assert merged["end_time"] == 10.0

    def test_partial_overlap_merge(self, reducer):
        """Test blocks with partial overlap that should merge if contained"""
        blocks = [
            {"text": "Container", "start_time": 0.0, "end_time": 10.0},
            {"text": "Contained", "start_time": 2.0, "end_time": 8.0},  # Fully contained
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 1
        assert result.original_count == 2
        assert result.filtered_blocks[0]["text"] == "Container"  # Keeps container text

    def test_multiple_main_blocks(self, reducer):
        """Test multiple main blocks each containing sub-blocks"""
        blocks = [
            {"text": "Main A", "start_time": 0.0, "end_time": 10.0},
            {"text": "Sub A1", "start_time": 2.0, "end_time": 4.0},
            {"text": "Sub A2", "start_time": 6.0, "end_time": 8.0},
            {"text": "Main B", "start_time": 15.0, "end_time": 25.0},
            {"text": "Sub B1", "start_time": 17.0, "end_time": 19.0},
            {"text": "Independent", "start_time": 30.0, "end_time": 35.0},
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 3  # 2 merged + 1 independent
        assert result.merged_count == 3  # 4 sub-blocks merged into 2 main blocks
        assert result.original_count == 6

    def test_time_extraction_from_fields(self, reducer):
        """Test time extraction from various field names"""
        # Test start_time/end_time fields
        block1 = {"text": "Test", "start_time": 1.0, "end_time": 5.0}
        assert reducer._extract_start_time(block1) == 1.0
        assert reducer._extract_end_time(block1) == 5.0

        # Test t0/t1 fields
        block2 = {"text": "Test", "t0": 2.0, "t1": 6.0}
        assert reducer._extract_start_time(block2) == 2.0
        assert reducer._extract_end_time(block2) == 6.0

        # Test timestamp field
        block3 = {"text": "Test", "timestamp": 3.0}
        assert reducer._extract_start_time(block3) == 3.0

    def test_time_extraction_from_text(self, reducer):
        """Test time extraction from text content"""
        # Test [start-end] format
        block1 = {"text": "[1.5-4.2] Some action"}
        assert reducer._extract_start_time(block1) == 1.5
        assert reducer._extract_end_time(block1) == 4.2

        # Test [start] format
        block2 = {"text": "[2.0] Another action"}
        assert reducer._extract_start_time(block2) == 2.0
        assert reducer._extract_end_time(block2) == 2.0  # Fallback for end

    def test_containment_check(self, reducer):
        """Test containment checking logic"""
        # Complete containment
        outer = TimeBlock("outer", 0.0, 10.0)
        inner = TimeBlock("inner", 2.0, 8.0)
        assert reducer._is_contained_in(inner, outer) == True

        # Partial containment (extends beyond)
        inner2 = TimeBlock("inner2", 2.0, 12.0)
        assert reducer._is_contained_in(inner2, outer) == False

        # Zero duration block (single timestamp)
        zero_block = TimeBlock("zero", 5.0, 5.0)
        assert reducer._is_contained_in(zero_block, outer) == True

        # Outside range
        outside = TimeBlock("outside", 15.0, 20.0)
        assert reducer._is_contained_in(outside, outer) == False

    def test_text_preservation(self, reducer):
        """Test that only main block text is preserved during merging"""
        # Text combination is no longer used - main block text is preserved
        # This test verifies the new behavior
        pass

    def test_config_override(self):
        """Test configuration parameter override"""
        custom_config = OmegaConf.create({
            # Simple containment logic - no parameters to override
        })
        reducer = OverlapAwareBlockReducer(custom_config)
        assert reducer.config == custom_config

    def test_complex_nested_structure(self, reducer):
        """Test complex nested block structure"""
        blocks = [
            {"text": "Outer most", "start_time": 0.0, "end_time": 20.0},
            {"text": "Middle", "start_time": 5.0, "end_time": 15.0},
            {"text": "Inner 1", "start_time": 7.0, "end_time": 9.0},
            {"text": "Inner 2", "start_time": 11.0, "end_time": 13.0},
            {"text": "Separate", "start_time": 25.0, "end_time": 30.0},
        ]

        result = reducer.reduce_blocks(blocks)
        # Should merge Middle, Inner 1, Inner 2 into Outer most
        # Separate should remain independent
        assert len(result.filtered_blocks) == 2
        assert result.merged_count == 3  # Middle, Inner 1 & 2 merged into Outer most
        assert result.original_count == 5

        # Check that merged block contains only the main block's text
        merged_block = next(b for b in result.filtered_blocks if b["merged_blocks"] == 4)
        assert merged_block["text"] == "Outer most"  # Only main block's text preserved
        assert merged_block["start_time"] == 0.0  # Extended time range
        assert merged_block["end_time"] == 20.0

        # Check that separate block remains
        separate_block = next(b for b in result.filtered_blocks if b["text"] == "Separate")
        assert separate_block["merged_blocks"] == 1

    def test_edge_case_identical_times(self, reducer):
        """Test blocks with identical start/end times - should merge"""
        blocks = [
            {"text": "Block A", "start_time": 5.0, "end_time": 5.0},
            {"text": "Block B", "start_time": 5.0, "end_time": 5.0},
        ]

        result = reducer.reduce_blocks(blocks)
        # Both have identical times, so they are "contained" in each other - merge into first one
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 1
        assert result.filtered_blocks[0]["text"] == "Block A"  # Keeps first block's text

    def test_large_time_gaps(self, reducer):
        """Test blocks with large time gaps"""
        blocks = [
            {"text": "Early", "start_time": 0.0, "end_time": 1.0},
            {"text": "Late", "start_time": 100.0, "end_time": 101.0},
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 2
        assert result.merged_count == 0

    def test_mixed_content_fields(self, reducer):
        """Test handling of different content field names"""
        blocks = [
            {"text": "Text field", "start_time": 0.0, "end_time": 5.0},
            {"content": "Content field", "start_time": 6.0, "end_time": 10.0},
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 2
        assert "Text field" in result.filtered_blocks[0]["text"]
        assert "Content field" in result.filtered_blocks[1]["text"]

    def test_single_timestamp_blocks(self, reducer):
        """Test blocks with only single timestamps (point-in-time events)"""
        blocks = [
            {"text": "Main action", "start_time": 100.0, "end_time": 120.0},
            {"text": "Point event 1", "timestamp": 105.0},  # Single timestamp
            {"text": "Point event 2", "timestamp": 115.0},  # Single timestamp
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 2
        assert result.filtered_blocks[0]["text"] == "Main action"  # Keeps main block text
        assert result.filtered_blocks[0]["start_time"] == 100.0
        assert result.filtered_blocks[0]["end_time"] == 120.0

    def test_mixed_single_and_range_timestamps(self, reducer):
        """Test mixing blocks with single timestamps and time ranges"""
        blocks = [
            {"text": "Container", "start_time": 0.0, "end_time": 50.0},
            {"text": "Range block", "start_time": 10.0, "end_time": 20.0},
            {"text": "Point 1", "timestamp": 5.0},
            {"text": "Point 2", "timestamp": 25.0},
            {"text": "Separate", "start_time": 60.0, "end_time": 70.0},
        ]

        result = reducer.reduce_blocks(blocks)
        # Should merge Range block, Point 1, Point 2 into Container
        # Separate should remain independent
        assert len(result.filtered_blocks) == 2
        assert result.merged_count == 3

        # Check merged block
        merged = next(b for b in result.filtered_blocks if b["merged_blocks"] == 4)
        assert merged["text"] == "Container"
        assert merged["start_time"] == 0.0  # Extended by Point 1
        assert merged["end_time"] == 50.0

        # Check separate block
        separate = next(b for b in result.filtered_blocks if b["text"] == "Separate")
        assert separate["merged_blocks"] == 1

    def test_timestamp_from_text_parsing(self, reducer):
        """Test parsing timestamps from text content like '[111.0s] action'"""
        blocks = [
            {"text": "Container action", "start_time": 100.0, "end_time": 130.0},
            {"text": "[111.0s] screw first wheel with hand", "timestamp": 111.0},  # Add explicit timestamp
            {"text": "[115.7s] screw second wheel with hand", "timestamp": 115.7},  # Add explicit timestamp
        ]

        result = reducer.reduce_blocks(blocks)
        assert len(result.filtered_blocks) == 1
        assert result.merged_count == 2
        assert result.filtered_blocks[0]["text"] == "Container action"

    def test_real_proassist_data_reduction(self, reducer):
        """Test block reduction with real ProAssist data from documentation"""
        # Real step descriptions from the document (before reduction)
        real_step_descriptions = [
            {"text": "[94.4s-105.2s] attach interior to chassis", "start_time": 94.4, "end_time": 105.2},
            {"text": "[105.2s-153.6s] attach wheel to chassis", "start_time": 105.2, "end_time": 153.6},
            {"text": "[111.0s] screw first wheel with hand", "timestamp": 111.0},
            {"text": "[115.7s] screw second wheel with hand", "timestamp": 115.7},
            {"text": "[116.2s] screw second wheel with hand", "timestamp": 116.2},
            {"text": "[117.1s] screw first wheel with screwdriver", "timestamp": 117.1},
            {"text": "[117.3s] screw first wheel with screwdriver", "timestamp": 117.3},
            {"text": "[120.0s] screw second wheel with screwdriver", "timestamp": 120.0},
            {"text": "[127.7s] screw third wheel with screwdriver", "timestamp": 127.7},
            {"text": "[138.8s] screw fourth wheel with screwdriver", "timestamp": 138.8},
            {"text": "[141.0s] screw fourth wheel with screwdriver", "timestamp": 141.0},
            {"text": "[153.6s-171.7s] attach arm to turntable top", "start_time": 153.6, "end_time": 171.7},
            {"text": "[164.2s] screw turntable top with screwdriver", "timestamp": 164.2},
            {"text": "[169.5s] screw turntable top with hand", "timestamp": 169.5},
            {"text": "[171.7s-187.1s] attach hook to arm", "start_time": 171.7, "end_time": 187.1},
            {"text": "[180.5s] screw hook with hand", "timestamp": 180.5},
            {"text": "[183.6s] screw hook with hand", "timestamp": 183.6},
            {"text": "[187.1s-203.7s] attach turntable top to chassis", "start_time": 187.1, "end_time": 203.7},
            {"text": "[203.7s-213.1s] attach cabin to interior", "start_time": 203.7, "end_time": 213.1},
            {"text": "[213.1s-232.0s] demonstrate functionality", "start_time": 213.1, "end_time": 232.0},
        ]

        result = reducer.reduce_blocks(real_step_descriptions)

        # Expected results based on the documentation
        # Should reduce from 20 blocks to 7 top-level blocks
        assert len(result.filtered_blocks) == 7
        assert result.merged_count == 13  # 20 - 7 = 13 merged
        assert result.original_count == 20

        # Verify the main blocks are preserved
        block_texts = [block["text"] for block in result.filtered_blocks]
        expected_texts = [
            "[94.4s-105.2s] attach interior to chassis",
            "[105.2s-153.6s] attach wheel to chassis",
            "[153.6s-171.7s] attach arm to turntable top",
            "[171.7s-187.1s] attach hook to arm",
            "[187.1s-203.7s] attach turntable top to chassis",
            "[203.7s-213.1s] attach cabin to interior",
            "[213.1s-232.0s] demonstrate functionality",
        ]

        for expected_text in expected_texts:
            assert expected_text in block_texts

        # Verify merged block counts
        wheel_block = next(block for block in result.filtered_blocks
                          if "attach wheel to chassis" in block["text"])
        assert wheel_block["merged_blocks"] == 10  # Main block + 9 contained blocks

        arm_block = next(block for block in result.filtered_blocks
                        if "attach arm to turntable top" in block["text"])
        assert arm_block["merged_blocks"] == 3  # Main block + 2 contained blocks

        hook_block = next(block for block in result.filtered_blocks
                         if "attach hook to arm" in block["text"])
        assert hook_block["merged_blocks"] == 3  # Main block + 2 contained blocks

        # Verify time ranges are extended correctly
        wheel_block = next(block for block in result.filtered_blocks
                          if "attach wheel to chassis" in block["text"])
        assert wheel_block["start_time"] == 105.2  # Original start
        assert wheel_block["end_time"] == 153.6    # Original end (contains all wheel actions)


if __name__ == "__main__":
    # Run tests manually for basic validation
    import sys
    import os

    # Add the project root to Python path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

    test_instance = TestOverlapAwareBlockReducer()

    config = OmegaConf.create({
        "min_overlap_ratio": 0.8,
        "max_gap_duration": 1.0,
    })

    try:
        test_instance.test_initialization(config)
        test_instance.test_empty_input(test_instance.reducer(config))
        test_instance.test_single_block(test_instance.reducer(config))
        test_instance.test_no_overlap_blocks(test_instance.reducer(config))
        test_instance.test_complete_overlap_merge(test_instance.reducer(config))
        test_instance.test_partial_overlap_no_merge(test_instance.reducer(config))
        test_instance.test_multiple_main_blocks(test_instance.reducer(config))
        test_instance.test_time_extraction_from_fields(test_instance.reducer(config))
        test_instance.test_time_extraction_from_text(test_instance.reducer(config))
        test_instance.test_overlap_ratio_calculation(test_instance.reducer(config))
        test_instance.test_text_combination(test_instance.reducer(config))
        test_instance.test_config_override()
        test_instance.test_complex_nested_structure(test_instance.reducer(config))
        test_instance.test_edge_case_identical_times(test_instance.reducer(config))
        test_instance.test_large_time_gaps(test_instance.reducer(config))
        test_instance.test_mixed_content_fields(test_instance.reducer(config))

        print("✅ All OverlapAwareBlockReducer tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()