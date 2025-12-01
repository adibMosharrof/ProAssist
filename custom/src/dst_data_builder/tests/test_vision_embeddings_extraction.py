#!/usr/bin/env python3
"""
Test script to verify vision embeddings extraction from frames.

This script tests the updated VisionEmbeddingsSaver that:
1. Extracts only specific frames from arrow files (based on start/end indices)
2. Extracts [CLS] tokens from SmolVLM2 vision encoder
3. Saves embeddings with correct 1152-dim shape
"""

import json
import logging
from pathlib import Path
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project to path
sys.path.insert(0, str(Path(__file__).parent / "custom" / "src"))

from dst_data_builder.vision_embeddings_saver import VisionEmbeddingsSaver


def test_vision_embeddings_extraction():
    """Test vision embeddings extraction for a sample clip."""
    
    logger.info("=" * 70)
    logger.info("Testing Vision Embeddings Extraction")
    logger.info("=" * 70)
    
    # Initialize saver
    logger.info("\n1. Initializing VisionEmbeddingsSaver...")
    saver = VisionEmbeddingsSaver(
        model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        frames_root="data/proassist/processed_data",
        device="cuda:0" if len(sys.argv) > 1 and sys.argv[1] == "--cuda" else "cpu",
    )
    logger.info(f"   ✓ Vision hidden size: {saver.vision_hidden_size}")
    
    # Create a sample clip data structure
    logger.info("\n2. Creating sample clip data...")
    sample_clip = {
        "video_uid": "9011-a01_9011_user_id_2021-02-01_153724__HMC_84355350_mono10bit",
        "start_frame_idx": 0,
        "end_frame_idx": 10,  # Extract only 11 frames (0-10 inclusive)
        "dataset": "assembly101",
    }
    logger.info(f"   Video UID: {sample_clip['video_uid']}")
    logger.info(f"   Frame range: [{sample_clip['start_frame_idx']}:{sample_clip['end_frame_idx']}]")
    
    # Test frame loading
    logger.info("\n3. Testing frame loading from arrow file...")
    try:
        # Build frame file path
        dataset_name = sample_clip["dataset"]
        video_uid = sample_clip["video_uid"]
        
        if dataset_name != "assembly101":
            frame_file = f"frames/{video_uid}.arrow"
        else:
            # For Assembly101, extract filename from video_uid
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        
        arrow_path = saver.frames_root / dataset_name / frame_file
        logger.info(f"   Arrow path: {arrow_path}")
        logger.info(f"   Exists: {arrow_path.exists()}")
        
        if not arrow_path.exists():
            logger.warning(f"   ⚠ Arrow file not found at {arrow_path}")
            logger.info("   ℹ This test requires the arrow file to exist.")
            logger.info("   ℹ Run this test after generating DST data on actual video files.")
            return
        
        # Load frames
        start_frame = sample_clip["start_frame_idx"]
        end_frame = sample_clip["end_frame_idx"]
        frame_indices = list(range(start_frame, end_frame + 1))
        
        logger.info(f"   Loading {len(frame_indices)} frames...")
        frames = saver.load_frames_from_arrow(arrow_path, frame_indices)
        logger.info(f"   ✓ Loaded {len(frames)} frames")
        
        # Test embedding extraction
        logger.info("\n4. Testing embedding extraction...")
        logger.info(f"   Processing {len(frames)} frames with vision encoder...")
        
        processor_output = saver.processor(images=frames, return_tensors="pt")
        pixel_values = processor_output["pixel_values"]
        logger.info(f"   Pixel values shape: {pixel_values.shape}")
        
        cls_tokens = saver.extract_cls_token(pixel_values)
        logger.info(f"   ✓ Extracted [CLS] tokens: {cls_tokens.shape}")
        
        # Verify dimensions
        logger.info("\n5. Verifying embedding dimensions...")
        batch_size = cls_tokens.shape[0]
        embedding_dim = cls_tokens.shape[1]
        
        logger.info(f"   Batch size: {batch_size} (expected {len(frames)})")
        logger.info(f"   Embedding dimension: {embedding_dim} (expected 1152)")
        
        assert batch_size == len(frames), f"Batch size mismatch: {batch_size} != {len(frames)}"
        assert embedding_dim == 1152, f"Embedding dimension mismatch: {embedding_dim} != 1152"
        
        logger.info("\n" + "=" * 70)
        logger.info("✓ Vision Embeddings Extraction Test PASSED")
        logger.info("=" * 70)
        logger.info("\nKey Results:")
        logger.info(f"  - Frame range loaded: [{start_frame}:{end_frame}]")
        logger.info(f"  - Frames extracted: {len(frames)}")
        logger.info(f"  - Embedding shape: {cls_tokens.shape}")
        logger.info(f"  - Vision hidden size: {saver.vision_hidden_size} (1152-dim)")
        
    except Exception as e:
        logger.error(f"\n✗ Test FAILED with error:")
        logger.error(f"  {type(e).__name__}: {e}")
        raise


if __name__ == "__main__":
    test_vision_embeddings_extraction()
