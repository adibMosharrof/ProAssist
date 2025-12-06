#!/usr/bin/env python3
"""
Test script to validate sparse sequence data loading and collator.

This script tests:
1. Loading sparse JSON data
2. Loading SigLIP embeddings
3. Collator building continuous sequences
4. Correct label generation
"""

import logging
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Direct imports to avoid __init__ issues
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

# Import directly from files
import importlib.util

# Load dst_proassist_dataset
spec = importlib.util.spec_from_file_location(
    "dst_proassist_dataset",
    Path(__file__).parent.parent / "data_sources" / "dst_proassist_dataset.py"
)
sparse_dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_dataset_module)
DSTProAssistDataset = sparse_dataset_module.DSTProAssistDataset

# Load dst_proassist_collator
spec = importlib.util.spec_from_file_location(
    "dst_proassist_collator",
    Path(__file__).parent.parent / "data_sources" / "dst_proassist_collator.py"
)
sparse_collator_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sparse_collator_module)
DSTProAssistCollator = sparse_collator_module.DSTProAssistCollator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_data_loading():
    """Test loading sparse sequence data and collator."""
    
    # Paths
    data_dir = Path("/u/siddique-d1/adib/ProAssist/custom/outputs/dst_generated/sparse_format/2025-12-06/05-27-35_gpt-4o_proassist_sparse")
    train_json = data_dir / "assembly101" / "train.json"
    siglip_features_dir = data_dir
    
    logger.info(f"Loading data from: {train_json}")
    logger.info(f"SigLIP features dir: {siglip_features_dir}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    # Add special tokens
    tokens_to_add = []
    if "<image>" not in tokenizer.get_vocab():
        tokens_to_add.append("<image>")
    if "[DST]" not in tokenizer.get_vocab():
        tokens_to_add.append("[DST]")
    if "[ASST]" not in tokenizer.get_vocab():
        tokens_to_add.append("[ASST]")
    
    if tokens_to_add:
        tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        logger.info(f"Added tokens: {tokens_to_add}")
    
    # Create dataset
    dataset = DSTProAssistDataset(
        data_path=str(train_json),
        dataset_name="assembly101"
    )
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    # Create collator
    collator = DSTProAssistCollator(
        tokenizer=tokenizer,
        siglip_features_dir=siglip_features_dir
    )
    
    logger.info("Collator created")
    
    # Test with first sample
    logger.info("\n=== Testing first sample ===")
    sample = dataset[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"Video UID: {sample['video_uid']}")
    logger.info(f"Clip ID: {sample.get('id', 'N/A')}")
    logger.info(f"Frame range: [{sample['start_frame']}, {sample['end_frame']})")
    logger.info(f"Number of events: {len(sample.get('events', []))}")
    
    # Test collator with batch of 2
    logger.info("\n=== Testing collator with batch of 2 ===")
    batch = collator([dataset[0], dataset[1]])
    
    logger.info(f"Batch keys: {batch.keys()}")
    logger.info(f"Input IDs shape: {batch['input_ids'].shape}")
    logger.info(f"Speaking gen labels shape: {batch['speaking_gen_labels'].shape}")
    logger.info(f"DST gen labels shape: {batch['dst_gen_labels'].shape}")
    logger.info(f"Speaking labels shape: {batch['speaking_labels'].shape}")
    logger.info(f"DST update labels shape: {batch['dst_update_labels'].shape}")
    logger.info(f"Number of embedding tensors: {len(batch['image_embeds'])}")
    logger.info(f"First embedding shape: {batch['image_embeds'][0].shape}")
    
    # Analyze first sample in batch
    logger.info("\n=== Analyzing first sample in batch ===")
    input_ids = batch['input_ids'][0]
    speaking_gen = batch['speaking_gen_labels'][0]
    dst_gen = batch['dst_gen_labels'][0]
    speaking_bin = batch['speaking_labels'][0]
    dst_bin = batch['dst_update_labels'][0]
    
    # Count tokens
    img_token_id = tokenizer.convert_tokens_to_ids("<image>")
    dst_token_id = tokenizer.convert_tokens_to_ids("[DST]")
    asst_token_id = tokenizer.convert_tokens_to_ids("[ASST]")
    
    num_img = (input_ids == img_token_id).sum().item()
    num_dst = (input_ids == dst_token_id).sum().item()
    num_asst = (input_ids == asst_token_id).sum().item()
    
    logger.info(f"Token counts:")
    logger.info(f"  <image>: {num_img}")
    logger.info(f"  [DST]: {num_dst}")
    logger.info(f"  [ASST]: {num_asst}")
    
    # Count labels
    speaking_gen_active = (speaking_gen != -100).sum().item()
    dst_gen_active = (dst_gen != -100).sum().item()
    speaking_bin_active = (speaking_bin != -100).sum().item()
    dst_bin_active = (dst_bin != -100).sum().item()
    
    logger.info(f"\nLabel counts (non -100):")
    logger.info(f"  Speaking gen: {speaking_gen_active}")
    logger.info(f"  DST gen: {dst_gen_active}")
    logger.info(f"  Speaking binary: {speaking_bin_active}")
    logger.info(f"  DST binary: {dst_bin_active}")
    
    # Check binary labels at image positions
    img_positions = (input_ids == img_token_id).nonzero(as_tuple=True)[0]
    logger.info(f"\nBinary labels at <image> positions:")
    logger.info(f"  Speaking: {speaking_bin[img_positions].tolist()[:10]}...")
    logger.info(f"  DST: {dst_bin[img_positions].tolist()[:10]}...")
    
    logger.info("\n✅ Test completed successfully!")
    return True


if __name__ == "__main__":
    try:
        test_data_loading()
    except Exception as e:
        logger.error(f"Test failed: {e}", exc_info=True)
        sys.exit(1)
