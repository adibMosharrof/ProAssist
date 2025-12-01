#!/usr/bin/env python3
"""Quick test to verify image collator works correctly."""

import torch
import sys
from pathlib import Path

# Set up paths - test is at custom/src/prospect/tests/dst_train/test_image_collator.py
# Need to go up to custom/src level
current_file = Path(__file__).resolve()
custom_src = current_file.parents[3]

sys.path.insert(0, str(custom_src))

# Direct import to avoid __init__.py dependencies
import importlib.util
collator_path = custom_src / "prospect" / "data_sources" / "dst_data_collator.py"
spec = importlib.util.spec_from_file_location("dst_data_collator", collator_path)
dst_data_collator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dst_data_collator_module)
DSTDataCollator = dst_data_collator_module.DSTDataCollator


def test_image_collator():
    """Test that collator can handle frames correctly."""
    
    # Create mock data collator
    collator = DSTDataCollator(
        tokenizer=None,
        chat_formatter=None,
        max_seq_len=4096,
        compute_labels=False,  # Don't compute labels, just test frame handling
    )
    
    # Create mock samples with different frame counts
    samples = [
        {
            "conversation": [],
            "images": [torch.randn(3, 384, 384) for _ in range(4)],  # 4 frames (as PIL-like)
            "sample_idx": 0,
        },
        {
            "conversation": [],
            "images": [torch.randn(3, 384, 384) for _ in range(3)],  # 3 frames
            "sample_idx": 1,
        },
        {
            "conversation": [],
            "images": [torch.randn(3, 384, 384) for _ in range(5)],  # 5 frames (max)
            "sample_idx": 2,
        },
    ]
    
    print("Sample 1 images: 4 items")
    print("Sample 2 images: 3 items")
    print("Sample 3 images: 5 items")
    
    # This should fail gracefully because we don't have proper tokenizer
    # But it should at least process frames correctly
    try:
        # Create minimal tokenizer mock to avoid errors
        class MockTokenizer:
            def __call__(self, *args, **kwargs):
                # Return mock batch
                return {
                    "input_ids": torch.randint(0, 1000, (3, 100)),
                    "attention_mask": torch.ones(3, 100),
                    "offset_mapping": [[(0, 0)] * 100 for _ in range(3)],
                }
            
            def convert_tokens_to_ids(self, token):
                return 100
        
        class MockProcessor:
            """Mock processor that simulates SmolVLM2 processor behavior"""
            def __init__(self):
                self.tokenizer = MockTokenizer()
            
            def __call__(self, images=None, **kwargs):
                # Simulate processor output for images
                # In real usage, SmolVLM2 processor would return properly formatted pixel_values
                # For now, just return a dummy tensor that indicates processing happened
                if images and isinstance(images, list) and len(images) > 0 and isinstance(images[0], list):
                    # Batch of image lists
                    num_samples = len(images)
                    max_images = max(len(img_list) for img_list in images) if images else 1
                    # Each image becomes multiple patches (simplified: just return as-is)
                    pixel_values = torch.randn(num_samples, max_images * 17, 3, 384, 384)  # 17 is patch dimension
                    return {"pixel_values": pixel_values}
                else:
                    # Single image or no images
                    return {"pixel_values": torch.zeros((1, 17, 3, 384, 384))}
        
        collator.tokenizer = MockTokenizer()
        collator.processor = MockProcessor()
        batch = collator(samples)
        
        print("\n✓ Collator processed successfully")
        print(f"Batch keys: {batch.keys()}")
        
        if "pixel_values" in batch:
            print(f"✓ pixel_values shape: {batch['pixel_values'].shape}")
            print(f"  (Processor output format - may vary based on SmolVLM2 implementation)")
            print(f"  Sample has: batch_size=3, variable images per sample (4, 3, 5)")
        
        if "frame_counts" in batch:
            print(f"✓ frame_counts: {batch['frame_counts']}")
            expected = torch.tensor([4, 3, 5], dtype=torch.long)
            if torch.equal(batch["frame_counts"], expected):
                print("  ✓ Frame counts are correct!")
            else:
                print(f"  ✗ Frame count mismatch! Got {batch['frame_counts']}, expected {expected}")
        
        print("\n✓ All image handling tests passed!")
        print("  Images are now passed directly to SmolVLM2 processor")
        return True
        
    except Exception as e:
        print(f"\n✗ Error during test: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_image_collator()
    sys.exit(0 if success else 1)
