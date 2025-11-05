"""Test VLM runner integration with custom model"""

import sys
import logging
from pathlib import Path
from PIL import Image
import torch

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prospect.runners.vlm_stream_runner import VLMStreamRunner

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_runner_initialization():
    """Test 1: Initialize runner with custom model"""
    print("\n" + "="*60)
    print("Test 1: Runner Initialization with Custom Model")
    print("="*60)
    
    try:
        runner = VLMStreamRunner(
            model_name="HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            eval_name="test_custom",
            device="cuda",
            torch_dtype="bfloat16",
            max_new_tokens=50,
            use_kv_cache=True,
            context_strategy_type="drop_all",
            max_seq_len=4096,
            reserved_seq_len=128,
        )
        
        print(f"✅ Runner initialized successfully")
        print(f"   Model type: {type(runner.model).__name__}")
        print(f"   Processor type: {type(runner.processor).__name__}")
        print(f"   Has joint_embed: {hasattr(runner.model, 'joint_embed')}")
        print(f"   Has fast_greedy_generate: {hasattr(runner.model, 'fast_greedy_generate')}")
        print(f"   Max seq len: {runner.model.max_seq_len}")
        print(f"   Context strategy: {runner.model.exceed_context_handling}")
        
        return runner
        
    except Exception as e:
        print(f"❌ Runner initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_frame_generation(runner):
    """Test 2: Generate dialogue for single frame"""
    print("\n" + "="*60)
    print("Test 2: Single Frame Generation")
    print("="*60)
    
    try:
        # Create dummy frame
        frame = Image.new("RGB", (384, 384), color="blue")
        
        # Mock prompt with proper SmolVLM2 format (includes <image> tag)
        runner.dialogue_generation_prompt = "<image>Previous step: {prev_substep}. Current step: {curr_substep}. Generate helpful dialogue."
        
        # Generate with cache
        dialogue = runner._generate_dialogue_with_cache(
            frame=frame,
            prev_substep="gathering ingredients",
            curr_substep="mixing batter"
        )
        
        print(f"✅ Generation successful")
        print(f"   Generated dialogue: {dialogue}")
        print(f"   KV cache size: {runner._get_cache_length()} tokens")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multi_frame_accumulation(runner):
    """Test 3: Multi-frame KV cache accumulation"""
    print("\n" + "="*60)
    print("Test 3: Multi-frame KV Cache Accumulation")
    print("="*60)
    
    try:
        # Reset cache
        runner.past_key_values = None
        
        # Temporarily disable context strategy for this test
        original_strategy = runner.context_strategy
        runner.context_strategy = None
        
        frames = [
            ("gathering ingredients", "mixing batter"),
            ("mixing batter", "pouring into pan"),
            ("pouring into pan", "baking"),
        ]
        
        cache_sizes = []
        
        for i, (prev_step, curr_step) in enumerate(frames):
            frame = Image.new("RGB", (384, 384), color="blue")
            
            dialogue = runner._generate_dialogue_with_cache(
                frame=frame,
                prev_substep=prev_step,
                curr_substep=curr_step
            )
            
            cache_len = runner._get_cache_length()
            cache_sizes.append(cache_len)
            print(f"   Frame {i}: KV cache = {cache_len} tokens")
        
        # Restore context strategy
        runner.context_strategy = original_strategy
        
        # Verify cache is growing
        is_growing = all(cache_sizes[i] < cache_sizes[i+1] for i in range(len(cache_sizes)-1))
        
        if is_growing:
            print(f"✅ KV cache accumulation working")
            print(f"   Cache sizes: {cache_sizes}")
        else:
            print(f"❌ KV cache not accumulating properly")
            print(f"   Cache sizes: {cache_sizes}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Multi-frame test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("VLM Runner Integration Tests")
    print("="*60)
    
    # Test 1: Initialize
    runner = test_runner_initialization()
    if runner is None:
        print("\n❌ TESTS FAILED - Could not initialize runner")
        return
    
    # Test 2: Single frame
    if not test_single_frame_generation(runner):
        print("\n❌ TESTS FAILED - Single frame generation error")
        return
    
    # Test 3: Multi-frame accumulation
    if not test_multi_frame_accumulation(runner):
        print("\n❌ TESTS FAILED - Multi-frame accumulation error")
        return
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED")
    print("="*60)


if __name__ == "__main__":
    main()
