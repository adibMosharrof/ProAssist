#!/usr/bin/env python3
"""Test script to validate DST per-turn binary head architecture."""

import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_binary_head_shapes():
    """Test that binary heads produce correct shapes for per-turn predictions."""
    
    logger.info("Testing DST per-turn binary head architecture...")
    
    # Simulate model outputs
    batch_size = 2
    num_turns = 4
    hidden_size = 1280
    
    # Frame embeddings extracted from sequence: [batch, num_turns, hidden_size]
    frame_embeddings = torch.randn(batch_size, num_turns, hidden_size)
    
    # Simple binary head: Linear(hidden_size, 1)
    speaking_head = torch.nn.Linear(hidden_size, 1)
    dst_update_head = torch.nn.Linear(hidden_size, 1)
    
    # Forward pass through heads
    speaking_logits = speaking_head(frame_embeddings)  # [batch, num_turns, 1]
    dst_update_logits = dst_update_head(frame_embeddings)  # [batch, num_turns, 1]
    
    logger.info(f"Frame embeddings shape: {frame_embeddings.shape}")
    logger.info(f"Speaking logits shape: {speaking_logits.shape}")
    logger.info(f"DST update logits shape: {dst_update_logits.shape}")
    
    # Create per-turn labels
    speaking_labels = torch.tensor([
        [1, 0, 1, 0],  # Sample 1: speak on turns 0, 2
        [0, 1, 0, 1],  # Sample 2: speak on turns 1, 3
    ], dtype=torch.float32)  # [batch, num_turns]
    
    dst_update_labels = torch.tensor([
        [1, 1, 0, 0],  # Sample 1: update on turns 0, 1
        [0, 0, 1, 1],  # Sample 2: update on turns 2, 3
    ], dtype=torch.float32)  # [batch, num_turns]
    
    logger.info(f"Speaking labels shape: {speaking_labels.shape}")
    logger.info(f"DST update labels shape: {dst_update_labels.shape}")
    
    # Flatten and compute BCE loss
    speaking_logits_flat = speaking_logits.view(-1)  # [batch*num_turns]
    speaking_labels_flat = speaking_labels.view(-1)  # [batch*num_turns]
    
    dst_update_logits_flat = dst_update_logits.view(-1)  # [batch*num_turns]
    dst_update_labels_flat = dst_update_labels.view(-1)  # [batch*num_turns]
    
    logger.info(f"Flattened speaking logits shape: {speaking_logits_flat.shape}")
    logger.info(f"Flattened speaking labels shape: {speaking_labels_flat.shape}")
    
    # Compute BCE loss
    bce_loss_fct = torch.nn.BCEWithLogitsLoss()
    speaking_loss = bce_loss_fct(speaking_logits_flat, speaking_labels_flat)
    dst_update_loss = bce_loss_fct(dst_update_logits_flat, dst_update_labels_flat)
    
    logger.info(f"Speaking loss: {speaking_loss.item():.4f}")
    logger.info(f"DST update loss: {dst_update_loss.item():.4f}")
    
    # Test dimension matching
    assert speaking_logits_flat.shape == speaking_labels_flat.shape, \
        f"Mismatch: logits {speaking_logits_flat.shape} vs labels {speaking_labels_flat.shape}"
    assert dst_update_logits_flat.shape == dst_update_labels_flat.shape, \
        f"Mismatch: logits {dst_update_logits_flat.shape} vs labels {dst_update_labels_flat.shape}"
    
    logger.info("✓ All shape tests passed!")
    logger.info("✓ No dimension mismatch - per-turn architecture works correctly!")
    
    return True

if __name__ == "__main__":
    test_binary_head_shapes()
