"""
Shared test fixtures and utilities for PROSPECT tests.

This conftest.py provides:
- Reusable model fixtures (mocked and real)
- Test data fixtures (sample images, videos, annotations)
- Mock factories for VLM models and processors
- Helper functions for common assertions
- Shared test utilities
"""

import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch
from PIL import Image
from omegaconf import OmegaConf


# ============================================================================
# Configuration Fixtures
# ============================================================================

@pytest.fixture
def basic_prospect_config():
    """Basic PROSPECT configuration for testing"""
    return OmegaConf.create({
        "model": {
            "name": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "dtype": "bfloat16",
            "device": "cuda",
        },
        "context_strategy": {
            "type": "drop_all",
            "max_seq_len": 4096,
            "reserved_seq_len": 128,
        },
        "data_source": {
            "type": "proassist_dst",
            "video_ids": ["test_video"],
        },
        "generator": {
            "type": "baseline",
        },
        "output_dir": "./test_outputs",
    })


@pytest.fixture
def context_strategy_configs():
    """Dictionary of context strategy configurations"""
    return {
        "none": {"type": "none"},
        "drop_all": {
            "type": "drop_all",
            "max_seq_len": 4096,
            "reserved_seq_len": 128,
        },
        "drop_middle": {
            "type": "drop_middle",
            "max_seq_len": 4096,
            "reserved_seq_len": 128,
            "last_keep_len": 512,
        },
        "summarize_and_drop": {
            "type": "summarize_and_drop",
            "max_seq_len": 4096,
            "reserved_seq_len": 128,
            "summary_max_length": 512,
        },
    }


# ============================================================================
# Sample Data Fixtures
# ============================================================================

@pytest.fixture
def sample_image():
    """Create a sample PIL image for testing"""
    # Create a 224x224 RGB image with gradient
    img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(img_array)


@pytest.fixture
def sample_images(sample_image):
    """Create list of sample images"""
    return [sample_image for _ in range(5)]


@pytest.fixture
def sample_video_frames(tmp_path, sample_images):
    """Create temporary directory with video frames"""
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    
    frame_paths = []
    for i, img in enumerate(sample_images):
        frame_path = frames_dir / f"frame_{i:04d}.jpg"
        img.save(frame_path)
        frame_paths.append(frame_path)
    
    return frames_dir, frame_paths


@pytest.fixture
def sample_dst_annotations():
    """Sample DST annotations for testing"""
    return {
        "video_id": "test_video",
        "steps": [
            {
                "step_id": "S1",
                "name": "Prepare components",
                "start_ts": 0.0,
                "end_ts": 10.0,
                "substeps": [
                    {
                        "sub_id": "S1.1",
                        "name": "Pick up first component",
                        "start_ts": 0.0,
                        "end_ts": 5.0,
                    },
                    {
                        "sub_id": "S1.2",
                        "name": "Pick up second component",
                        "start_ts": 5.0,
                        "end_ts": 10.0,
                    },
                ],
            },
        ],
    }


@pytest.fixture
def sample_conversation():
    """Sample conversation/dialogue for testing"""
    return [
        {
            "from": "assistant",
            "value": "Let's prepare the components.",
            "timestamp": 0.5,
        },
        {
            "from": "assistant",
            "value": "Now pick up the second component.",
            "timestamp": 5.5,
        },
    ]


# ============================================================================
# Model Fixtures (Mocked)
# ============================================================================

@pytest.fixture
def mock_custom_smolvlm_model():
    """Mock CustomSmolVLM model for testing"""
    model = MagicMock()
    model.config = MagicMock()
    model.config.max_seq_len = 4096
    model.config.eos_token_id = 2
    model.config.pad_token_id = 0
    model.config.image_token_id = 32000
    model.device = "cpu"
    
    # Mock joint_embed
    def mock_joint_embed(input_ids, pixel_values=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden_dim = 2048
        return torch.randn(batch_size, seq_len, hidden_dim)
    
    model.joint_embed = Mock(side_effect=mock_joint_embed)
    
    # Mock fast_greedy_generate
    def mock_fast_greedy_generate(inputs_embeds, past_key_values=None, max_length=50, **kwargs):
        batch_size = inputs_embeds.shape[0]
        # Generate some random tokens
        output_ids = torch.randint(10, 1000, (batch_size, max_length))
        # Create fake KV cache
        num_layers = 28
        num_heads = 32
        head_dim = 64
        cache_seq_len = inputs_embeds.shape[1] + max_length
        fake_cache = tuple([
            (
                torch.randn(batch_size, num_heads, cache_seq_len, head_dim),
                torch.randn(batch_size, num_heads, cache_seq_len, head_dim)
            )
            for _ in range(num_layers)
        ])
        return output_ids, fake_cache
    
    model.fast_greedy_generate = Mock(side_effect=mock_fast_greedy_generate)
    
    return model


@pytest.fixture
def mock_processor():
    """Mock processor for testing"""
    processor = MagicMock()
    
    # Mock tokenizer
    processor.tokenizer = MagicMock()
    processor.tokenizer.eos_token_id = 2
    processor.tokenizer.pad_token_id = 0
    
    def mock_tokenize(text, **kwargs):
        # Return fake token IDs
        return {"input_ids": torch.randint(10, 1000, (1, 20))}
    
    processor.tokenizer.side_effect = mock_tokenize
    
    def mock_decode(token_ids, **kwargs):
        return "This is a generated response."
    
    processor.tokenizer.decode = Mock(side_effect=mock_decode)
    processor.decode = Mock(side_effect=mock_decode)
    
    # Mock image processor
    processor.image_processor = MagicMock()
    
    # Mock __call__ for processing images + text
    def mock_process(images=None, text=None, **kwargs):
        result = {}
        if text is not None:
            result["input_ids"] = torch.randint(10, 1000, (1, 20))
        if images is not None:
            result["pixel_values"] = torch.randn(1, 3, 224, 224)
            result["pixel_attention_mask"] = torch.ones(1, 224, 224)
        return result
    
    processor.side_effect = mock_process
    
    return processor


# ============================================================================
# KV Cache Fixtures
# ============================================================================

@pytest.fixture
def sample_kv_cache():
    """Create sample KV cache for testing"""
    num_layers = 28
    batch_size = 1
    num_heads = 32
    seq_len = 1000
    head_dim = 64
    
    cache = tuple([
        (
            torch.randn(batch_size, num_heads, seq_len, head_dim),
            torch.randn(batch_size, num_heads, seq_len, head_dim)
        )
        for _ in range(num_layers)
    ])
    
    return cache


@pytest.fixture
def large_kv_cache():
    """Create large KV cache that exceeds threshold"""
    num_layers = 28
    batch_size = 1
    num_heads = 32
    seq_len = 4500  # Exceeds 4096 threshold
    head_dim = 64
    
    cache = tuple([
        (
            torch.randn(batch_size, num_heads, seq_len, head_dim),
            torch.randn(batch_size, num_heads, seq_len, head_dim)
        )
        for _ in range(num_layers)
    ])
    
    return cache


# ============================================================================
# Helper Functions
# ============================================================================

def assert_kv_cache_format(cache):
    """Assert KV cache has correct format"""
    assert isinstance(cache, tuple), "KV cache must be tuple"
    assert len(cache) > 0, "KV cache must have at least one layer"
    
    for layer_cache in cache:
        assert isinstance(layer_cache, tuple), "Each layer must be tuple of (keys, values)"
        assert len(layer_cache) == 2, "Each layer must have keys and values"
        keys, values = layer_cache
        assert isinstance(keys, torch.Tensor), "Keys must be tensors"
        assert isinstance(values, torch.Tensor), "Values must be tensors"
        assert keys.shape == values.shape, "Keys and values must have same shape"


def assert_kv_cache_size(cache, expected_seq_len):
    """Assert KV cache has expected sequence length"""
    actual_seq_len = cache[0][0].shape[2]
    assert actual_seq_len == expected_seq_len, \
        f"Expected KV cache seq_len={expected_seq_len}, got {actual_seq_len}"


def create_mock_frame_output(gen="", ref="", timestamp=0.0):
    """Create mock FrameOutput for testing"""
    from prospect.runners.vlm_stream_runner import FrameOutput
    return FrameOutput(
        gen=gen,
        ref=ref,
        image=None,
        frame_idx_in_stream=0,
        timestamp_in_stream=timestamp,
    )
