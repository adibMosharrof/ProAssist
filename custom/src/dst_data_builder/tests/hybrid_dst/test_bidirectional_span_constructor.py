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
                "similarity": {
                    "high_confidence_threshold": 0.3,
                    "semantic_weight": 0.6,
                    "nli_weight": 0.4,
                },
                "models": {
                    "semantic_encoder": "BAAI/bge-base-en-v1.5",
                    "nli_model": "cross-encoder/ms-marco-MiniLM-L-12-v2",
                },
                "min_span_duration": 1.0,
                "max_span_duration": 300.0,
                "llm_max_retries": 3,
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
        """Create BidirectionalSpanConstructor instance with real components"""
        return BidirectionalSpanConstructor(sample_config, model_config)

    @pytest.fixture
    def complex_assembly_data(self):
        """
        Real-world assembly scenario with mistakes and corrections
        
        Steps: Assembly process with 6 main steps
        Blocks: 13 segments including mistakes and corrections
        """
        inferred_knowledge = [
            'Assemble the interior and chassis',
            'Attach wheels to the chassis',
            'Attach the body to the interior',
            'Attach the roof to the body',
            'Attach the engine cover to the body',
            'Finalize the assembly and test functionality'
        ]
        
        filtered_blocks = [
            {'text': 'screw interior', 'start_time': 88.0, 'end_time': 101.8, 'merged_blocks': 3, 'original_id': 0},
            {'text': 'attach wheel to chassis', 'start_time': 101.8, 'end_time': 137.3, 'merged_blocks': 9, 'original_id': 3},
            {'text': 'attach body to interior', 'start_time': 137.3, 'end_time': 143.0, 'merged_blocks': 1, 'original_id': 12},
            {'text': 'attach wheel to chassis', 'start_time': 143.0, 'end_time': 165.6, 'merged_blocks': 1, 'original_id': 13},
            {'text': 'inspect toy', 'start_time': 165.6, 'end_time': 171.7, 'merged_blocks': 1, 'original_id': 14},
            {'text': "detach body from interior (mistake: shouldn't have happened)", 'start_time': 171.7, 'end_time': 173.8, 'merged_blocks': 1, 'original_id': 15},
            {'text': 'attach body to interior (mistake: wrong position)', 'start_time': 173.8, 'end_time': 176.1, 'merged_blocks': 1, 'original_id': 16},
            {'text': 'detach body from interior (correction of "attach body to interior")', 'start_time': 176.1, 'end_time': 181.8, 'merged_blocks': 1, 'original_id': 17},
            {'text': 'attach body to interior', 'start_time': 181.8, 'end_time': 187.4, 'merged_blocks': 1, 'original_id': 18},
            {'text': 'screw wheel', 'start_time': 187.4, 'end_time': 198.3, 'merged_blocks': 3, 'original_id': 19},
            {'text': 'attach roof to body', 'start_time': 198.3, 'end_time': 213.8, 'merged_blocks': 5, 'original_id': 22},
            {'text': 'attach engine cover to body', 'start_time': 213.8, 'end_time': 223.7, 'merged_blocks': 1, 'original_id': 27},
            {'text': 'demonstrate functionality', 'start_time': 223.7, 'end_time': 226.2, 'merged_blocks': 1, 'original_id': 28}
        ]
        
        return inferred_knowledge, filtered_blocks

    def test_complex_assembly_with_mistakes(self, constructor, complex_assembly_data):
        """
        Test bidirectional span construction on complex assembly with mistakes
        
        This test verifies the constructor can handle:
        - Repeated actions (wheel attachment, body attachment)
        - Mistake segments (detach/reattach sequences)
        - Multiple blocks mapping to same step
        """
        inferred_knowledge, filtered_blocks = complex_assembly_data
        
        # Actually call the method and see what it returns
        result = constructor.construct_spans(filtered_blocks, inferred_knowledge)
        
        # Verify result structure
        assert isinstance(result, BidirectionalSpanConstructionResult)
        assert len(result.dst_spans) == len(inferred_knowledge), \
            f"Expected {len(inferred_knowledge)} spans, got {len(result.dst_spans)}"
        assert result.total_blocks_processed == len(filtered_blocks)
        
        # Verify all spans have required fields
        for span in result.dst_spans:
            assert 'id' in span
            assert 'name' in span
            assert 'start_ts' in span
            assert 'end_ts' in span
            assert 'conf' in span
            assert 'block_indices' in span
            assert span['start_ts'] < span['end_ts'], \
                f"Step {span['id']}: start_ts {span['start_ts']} >= end_ts {span['end_ts']}"
        
        # Verify all step IDs are present (1 to N)
        span_ids = {span['id'] for span in result.dst_spans}
        expected_ids = set(range(1, len(inferred_knowledge) + 1))
        assert span_ids == expected_ids, \
            f"Missing step IDs: {expected_ids - span_ids}"
        
        # Verify all blocks are assigned exactly once
        all_assigned_indices = []
        for span in result.dst_spans:
            all_assigned_indices.extend(span['block_indices'])
        
        assert len(all_assigned_indices) == len(filtered_blocks), \
            f"Total assignments {len(all_assigned_indices)} != total blocks {len(filtered_blocks)}"
        
        assigned_set = set(all_assigned_indices)
        expected_indices = set(range(len(filtered_blocks)))
        assert assigned_set == expected_indices, \
            f"Missing blocks: {expected_indices - assigned_set}, Extra: {assigned_set - expected_indices}"
        
        # Check for duplicates (each block assigned only once)
        assert len(all_assigned_indices) == len(assigned_set), \
            "Some blocks assigned to multiple steps"
        
        # Print results for analysis
        print(f"\n✅ Bidirectional span construction successful!")
        print(f"   Created {len(result.dst_spans)} spans from {result.total_blocks_processed} blocks")
        print(f"\nSpan assignments (by ID):")
        for span_id in sorted([s['id'] for s in result.dst_spans]):
            span = next(s for s in result.dst_spans if s['id'] == span_id)
            block_texts = [filtered_blocks[int(idx)]['text'] for idx in span['block_indices']]
            print(f"  Step {span['id']}: {span['name']}")
            print(f"    Time: {span['start_ts']:.1f}s - {span['end_ts']:.1f}s")
            print(f"    Blocks: {span['block_indices']}")
            print(f"    Block texts: {block_texts}")

    # TODO: Add more tests after debugging the implementation
    # def test_forward_pass(self, constructor):
    #     """Test forward pass logic"""

    # def test_backward_pass(self, constructor):
    #     """Test backward pass logic"""

    # def test_conflict_detection(self, constructor):
    #     """Test conflict detection between passes"""
