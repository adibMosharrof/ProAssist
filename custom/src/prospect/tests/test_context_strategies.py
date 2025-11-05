"""
Tests for context overflow strategies

This module tests all context overflow strategies:
- drop_all: Drop all KV cache on overflow
- drop_middle: Keep initial + recent context
- summarize_and_drop: Generate summary and drop all
"""

import pytest
import torch
from prospect.context_strategies.drop_all import DropAllStrategy
from prospect.context_strategies.drop_middle import DropMiddleStrategy
from prospect.context_strategies.summarize_and_drop import SummarizeAndDropStrategy


class TestDropAllStrategy:
    """Tests for drop_all strategy"""
    
    def test_initialization(self):
        """Test strategy initializes with correct parameters"""
        strategy = DropAllStrategy(
            max_seq_len=4096,
            reserved_seq_len=128
        )
        
        assert strategy.max_seq_len == 4096
        assert strategy.reserved_seq_len == 128
        assert strategy.ctxlen_to_reduce == 4096 - 128
        assert strategy.name == "drop_all"
    
    def test_should_reduce_cache_below_threshold(self):
        """Test should_reduce_cache returns False below threshold"""
        strategy = DropAllStrategy(max_seq_len=4096, reserved_seq_len=128)
        
        assert not strategy.should_reduce_cache(3000)
        assert not strategy.should_reduce_cache(3967)  # Just below threshold
    
    def test_should_reduce_cache_at_threshold(self):
        """Test should_reduce_cache returns True at/above threshold"""
        strategy = DropAllStrategy(max_seq_len=4096, reserved_seq_len=128)
        
        assert strategy.should_reduce_cache(3968)
        assert strategy.should_reduce_cache(4000)
        assert strategy.should_reduce_cache(5000)
    
    def test_handle_overflow_drops_all(self, sample_kv_cache):
        """Test handle_overflow returns None (drops all cache)"""
        strategy = DropAllStrategy(max_seq_len=4096, reserved_seq_len=128)
        
        new_cache, last_msg = strategy.handle_overflow(
            past_key_values=sample_kv_cache,
            last_msg="Previous message"
        )
        
        assert new_cache is None
        assert last_msg == "Previous message"


class TestDropMiddleStrategy:
    """Tests for drop_middle strategy"""
    
    def test_initialization(self):
        """Test strategy initializes with correct parameters"""
        strategy = DropMiddleStrategy(
            max_seq_len=4096,
            reserved_seq_len=128,
            last_keep_len=512
        )
        
        assert strategy.max_seq_len == 4096
        assert strategy.reserved_seq_len == 128
        assert strategy.last_keep_len == 512
        assert strategy.name == "drop_middle"
        assert strategy.init_kv_cache is None
    
    def test_set_initial_cache(self, sample_kv_cache):
        """Test storing initial KV cache"""
        strategy = DropMiddleStrategy(max_seq_len=4096)
        
        strategy.set_initial_cache(sample_kv_cache)
        
        assert strategy.init_kv_cache is not None
        assert strategy.init_kv_cache[0][0].shape == sample_kv_cache[0][0].shape
    
    def test_handle_overflow_no_initial_cache(self, sample_kv_cache):
        """Test handle_overflow falls back to drop_all without initial cache"""
        strategy = DropMiddleStrategy(max_seq_len=4096)
        
        new_cache, last_msg = strategy.handle_overflow(
            past_key_values=sample_kv_cache,
            last_msg="test"
        )
        
        # Should fall back to drop_all behavior
        assert new_cache is None
        assert last_msg == "test"
    
    def test_handle_overflow_keeps_init_and_recent(self):
        """Test handle_overflow keeps initial + recent tokens"""
        strategy = DropMiddleStrategy(
            max_seq_len=4096,
            last_keep_len=100
        )
        
        # Create initial cache (500 tokens)
        init_cache = self._create_cache(seq_len=500)
        strategy.set_initial_cache(init_cache)
        
        # Create current cache (2000 tokens)
        curr_cache = self._create_cache(seq_len=2000)
        
        new_cache, last_msg = strategy.handle_overflow(
            past_key_values=curr_cache,
            last_msg="test"
        )
        
        # Should keep init (500) + recent (100) = 600 tokens
        assert new_cache is not None
        new_len = new_cache[0][0].shape[2]
        assert new_len == 600
    
    @staticmethod
    def _create_cache(seq_len):
        """Helper to create KV cache with specific sequence length"""
        num_layers = 28
        return tuple([
            (
                torch.randn(1, 32, seq_len, 64),
                torch.randn(1, 32, seq_len, 64)
            )
            for _ in range(num_layers)
        ])


class TestSummarizeAndDropStrategy:
    """Tests for summarize_and_drop strategy"""
    
    def test_initialization(self):
        """Test strategy initializes with correct parameters"""
        strategy = SummarizeAndDropStrategy(
            max_seq_len=4096,
            reserved_seq_len=128,
            summary_max_length=512,
            summary_prompt="Summarize progress."
        )
        
        assert strategy.max_seq_len == 4096
        assert strategy.reserved_seq_len == 128
        assert strategy.summary_max_length == 512
        assert strategy.summary_prompt == "Summarize progress."
        assert strategy.name == "summarize_and_drop"
    
    def test_handle_overflow_missing_context(self, sample_kv_cache):
        """Test handle_overflow falls back when required context is missing"""
        strategy = SummarizeAndDropStrategy(max_seq_len=4096)
        
        # Call without required context (model, processor, etc.)
        new_cache, last_msg = strategy.handle_overflow(
            past_key_values=sample_kv_cache,
            last_msg="test"
        )
        
        # Should fall back to drop_all behavior
        assert new_cache is None
        assert last_msg == "test"
    
    def test_handle_overflow_with_mock_model(
        self,
        sample_kv_cache,
        mock_custom_smolvlm_model,
        mock_processor,
        sample_image
    ):
        """Test handle_overflow generates summary with mocked model"""
        strategy = SummarizeAndDropStrategy(
            max_seq_len=4096,
            summary_prompt="Summarize."
        )
        
        # Create mock chat formatter
        class MockChatFormatter:
            @staticmethod
            def apply_chat_template(messages):
                return messages[0]["content"] if messages else ""
        
        # Prepare context
        context = {
            'model': mock_custom_smolvlm_model,
            'processor': mock_processor,
            'current_frame': {
                'input_ids': torch.randint(10, 1000, (1, 20)),
                'pixel_values': torch.randn(1, 3, 224, 224),
            },
            'num_frames': 1,
            'chat_formatter': MockChatFormatter(),
        }
        
        new_cache, summary_msg = strategy.handle_overflow(
            past_key_values=sample_kv_cache,
            last_msg="test",
            **context
        )
        
        # Should drop all cache
        assert new_cache is None
        # Should return a summary message
        assert summary_msg is not None
        assert isinstance(summary_msg, str)


class TestStrategyComparison:
    """Comparative tests across strategies"""
    
    @pytest.mark.parametrize("strategy_class,expected_name", [
        (DropAllStrategy, "drop_all"),
        (DropMiddleStrategy, "drop_middle"),
        (SummarizeAndDropStrategy, "summarize_and_drop"),
    ])
    def test_strategy_names(self, strategy_class, expected_name):
        """Test all strategies have correct names"""
        strategy = strategy_class(max_seq_len=4096)
        assert strategy.name == expected_name
    
    @pytest.mark.parametrize("strategy_class", [
        DropAllStrategy,
        DropMiddleStrategy,
        SummarizeAndDropStrategy,
    ])
    def test_all_strategies_have_required_methods(self, strategy_class):
        """Test all strategies implement required interface"""
        strategy = strategy_class(max_seq_len=4096)
        
        assert hasattr(strategy, 'should_reduce_cache')
        assert hasattr(strategy, 'handle_overflow')
        assert hasattr(strategy, 'name')
        assert callable(strategy.should_reduce_cache)
        assert callable(strategy.handle_overflow)
