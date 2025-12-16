
import logging
import unittest
from unittest.mock import MagicMock
from dst_data_builder.dst_data_processor import DSTDataProcessor

# Mock imports/classes needed for instantiation
class MockGPTGenerator:
    pass

class TestFrameFiltering(unittest.TestCase):
    def setUp(self):
        self.processor = DSTDataProcessor(
            dst_generator=MockGPTGenerator(),
            logger=logging.getLogger("TestLogger")
        )
        
        # Mock _get_video_frame_count
        self.processor._get_video_frame_count = MagicMock(return_value=100)
        
    def test_filtering(self):
        video_uid = "test_video"
        dataset_name = "test_dataset"
        
        turns = [
            {"start_frame": 10, "end_frame": 50, "content": "valid turn"},
            {"start_frame": 80, "end_frame": 99, "content": "valid turn edge"},
            {"start_frame": 90, "end_frame": 120, "content": "truncated turn"},
            {"start_frame": 100, "end_frame": 150, "content": "dropped turn exact"},
            {"start_frame": 150, "end_frame": 200, "content": "dropped turn far"},
            {"role": "user", "content": "no frames turn"} # Should be kept
        ]
        
        filtered = self.processor._filter_conversation_turns(turns, video_uid, dataset_name)
        
        print(f"\nOriginal count: {len(turns)}")
        print(f"Filtered count: {len(filtered)}")
        
        self.assertEqual(len(filtered), 4)
        
        # Check specific turns
        self.assertEqual(filtered[0]["content"], "valid turn")
        self.assertEqual(filtered[1]["content"], "valid turn edge")
        
        # Check truncation
        self.assertEqual(filtered[2]["content"], "truncated turn")
        self.assertEqual(filtered[2]["end_frame"], 100)
        
        # Check no frames preservation
        self.assertEqual(filtered[3]["content"], "no frames turn")
        
        print("Test passed!")

if __name__ == '__main__':
    unittest.main()
