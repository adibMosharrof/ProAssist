#!/usr/bin/env python3
"""
Test the [CLS] token forward pass with sample data.

This script:
1. Creates synthetic conversations and images
2. Passes them through the data collator
3. Runs the model forward pass
4. Monitors token counts and memory usage
"""

import sys
import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader

# Setup path
sys.path.insert(0, '/u/siddique-d1/adib/ProAssist/custom/src')
sys.path.insert(0, '/u/siddique-d1/adib/ProAssist')

from transformers import AutoProcessor, AutoTokenizer, AutoConfig
from prospect.models.dst_smolvlm_with_strategies import DSTSmolVLMWithStrategies
from prospect.data_sources.dst_data_collator import DSTDataCollator

def create_sample_conversation(num_turns=3, num_images=2):
    """Create a synthetic conversation with DST labels."""
    conversation = []
    
    for i in range(num_turns):
        # User turn
        conversation.append({
            "role": "user",
            "content": f"Turn {i+1}: What action is being performed? Can you describe the current state?"
        })
        
        # Assistant turn
        conversation.append({
            "role": "assistant",
            "content": f"In turn {i+1}, the person is performing action X. The state has changed from Y to Z."
        })
        
        # DST_UPDATE turn
        conversation.append({
            "role": "DST_UPDATE",
            "content": f'[{{"task": "action_{i+1}", "state": "completed", "confidence": 0.95}}]'
        })
    
    return conversation

def create_sample_images(num_images=2, size=(256, 256)):
    """Create random PIL images for testing."""
    images = []
    colors = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
    
    for i in range(num_images):
        # Create image with different colors
        img = Image.new('RGB', size, color=colors[i % len(colors)])
        images.append(img)
    
    return images

def test_forward_pass():
    """Test the model forward pass with sample data."""
    
    print("=" * 80)
    print("Testing SmolVLM2 [CLS] Token Strategy Forward Pass")
    print("=" * 80)
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model components
    print("\n[1] Loading model components...")
    processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    config = AutoConfig.from_pretrained("HuggingFaceTB/SmolVLM2-2.2B-Instruct")
    
    # Add config for [CLS] token strategy
    config.use_img_cls_token = True
    config.vision_hidden_size = 1152
    config.img_token_id = tokenizer.convert_tokens_to_ids("<image>")
    
    print(f"    ✓ Loaded processor, tokenizer, config")
    print(f"    ✓ use_img_cls_token: {config.use_img_cls_token}")
    print(f"    ✓ Image token ID: {config.img_token_id}")
    
    # Load model (use smaller dtype to save memory)
    print("\n[2] Loading model...")
    model = DSTSmolVLMWithStrategies.from_pretrained(
        "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        config=config,
        dtype=torch.float32,
        device_map="auto",
    )
    model.eval()
    print(f"    ✓ Model loaded on {device}")
    
    # Create data collator
    print("\n[3] Creating data collator...")
    collator = DSTDataCollator(
        tokenizer=tokenizer,
        compute_labels=True,
    )
    print(f"    ✓ Data collator created")
    
    # Create sample data with varying configurations
    test_configs = [
        {"num_samples": 1, "num_turns": 2, "num_images": 1, "desc": "1 sample, 2 turns, 1 image"},
        {"num_samples": 1, "num_turns": 3, "num_images": 2, "desc": "1 sample, 3 turns, 2 images"},
        {"num_samples": 2, "num_turns": 2, "num_images": 1, "desc": "2 samples, 2 turns, 1 image"},
    ]
    
    for config_idx, test_config in enumerate(test_configs):
        print(f"\n{'='*80}")
        print(f"Test {config_idx + 1}: {test_config['desc']}")
        print(f"{'='*80}")
        
        num_samples = test_config["num_samples"]
        num_turns = test_config["num_turns"]
        num_images = test_config["num_images"]
        
        # Create samples
        print(f"\n[A] Creating {num_samples} sample(s)...")
        samples = []
        for s_idx in range(num_samples):
            sample = {
                "sample_idx": s_idx,
                "conversation": create_sample_conversation(num_turns=num_turns, num_images=num_images),
                "images": create_sample_images(num_images=num_images),
                "neg_frame_sampling_rate": 1.0,
            }
            samples.append(sample)
            print(f"    ✓ Sample {s_idx}: {len(sample['conversation'])} turns, {len(sample['images'])} images")
        
        # Collate batch
        print(f"\n[B] Collating batch...")
        try:
            batch = collator(samples)
            print(f"    ✓ Batch collated successfully")
            print(f"    Batch keys: {list(batch.keys())}")
            print(f"    input_ids shape: {batch['input_ids'].shape}")
            print(f"    attention_mask shape: {batch['attention_mask'].shape}")
            if 'pixel_values' in batch:
                print(f"    pixel_values shape: {batch['pixel_values'].shape}")
            if 'frame_counts' in batch:
                print(f"    frame_counts: {batch['frame_counts']}")
            
            # Calculate total tokens
            total_tokens = batch['input_ids'].shape[0] * batch['input_ids'].shape[1]
            print(f"    Total text tokens: {total_tokens}")
            
            # Count image tokens
            if 'input_ids' in batch:
                image_token_id = config.img_token_id
                num_image_tokens = (batch['input_ids'] == image_token_id).sum().item()
                print(f"    Image tokens in text: {num_image_tokens}")
        except Exception as e:
            print(f"    ✗ Error during collation: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        # Move to device
        print(f"\n[C] Moving batch to device...")
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        print(f"    ✓ Batch moved to {device}")
        
        # Forward pass
        print(f"\n[D] Running model forward pass...")
        try:
            with torch.no_grad():
                # Measure memory before
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.synchronize()
                    mem_before = torch.cuda.memory_allocated()
                
                # Forward pass
                outputs = model(
                    input_ids=batch['input_ids'],
                    pixel_values=batch.get('pixel_values'),
                    attention_mask=batch['attention_mask'],
                    labels=batch.get('labels'),
                    speaking_labels=batch.get('speaking_labels'),
                    dst_update_labels=batch.get('dst_update_labels'),
                    dst_gen_labels=batch.get('dst_gen_labels'),
                )
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    mem_after = torch.cuda.memory_allocated()
                    mem_used = (mem_after - mem_before) / 1e9  # Convert to GB
                
                print(f"    ✓ Forward pass successful!")
                print(f"    Output keys: {list(outputs.keys())}")
                print(f"    logits shape: {outputs['logits'].shape if 'logits' in outputs else 'N/A'}")
                if 'loss' in outputs and outputs['loss'] is not None:
                    print(f"    loss: {outputs['loss'].item():.4f}")
                
                if torch.cuda.is_available():
                    print(f"    Memory used: {mem_used:.2f} GB")
                    peak_memory = torch.cuda.max_memory_allocated() / 1e9
                    print(f"    Peak memory: {peak_memory:.2f} GB")
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"    ✗ OUT OF MEMORY ERROR!")
                print(f"    Error: {e}")
                
                # Analyze what caused OOM
                print(f"\n    [Analysis]")
                print(f"    Batch size: {num_samples}")
                print(f"    Sequence length: {batch['input_ids'].shape[1]}")
                print(f"    Total tokens per sample: {batch['input_ids'].shape[1]}")
                
                if torch.cuda.is_available():
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    print(f"    GPU allocated: {allocated:.2f} GB")
                    print(f"    GPU reserved: {reserved:.2f} GB")
                    print(f"    GPU total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print(f"    ✗ Runtime error: {e}")
                import traceback
                traceback.print_exc()
            
            continue
        except Exception as e:
            print(f"    ✗ Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        print(f"\n[✓] Test {config_idx + 1} PASSED")
    
    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)

if __name__ == "__main__":
    test_forward_pass()
