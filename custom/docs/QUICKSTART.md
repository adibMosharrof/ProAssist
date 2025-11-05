# Quick Start Guide - ProAssist with KV Cache Compression

**Last Updated:** November 4, 2025

## Overview

This guide shows you how to run ProAssist streaming dialogue generation with KV cache compression strategies.

## Prerequisites

```bash
# Python 3.10+
# CUDA-capable GPU (tested on RTX 3090)
# ~8GB VRAM for drop_middle strategy

# Install dependencies
pip install -r requirements.txt
```

## Basic Usage

### 1. Run Single Strategy Test

Test a specific compression strategy on one video:

```bash
cd /u/siddique-d1/adib/ProAssist

# Test drop_middle strategy (recommended)
./custom/src/prospect/tests/run_tests.sh e2e drop_middle

# Test other strategies
./custom/src/prospect/tests/run_tests.sh e2e drop_all
./custom/src/prospect/tests/run_tests.sh e2e none
```

**Output:**
```
✅ Strategy drop_middle completed!
Results:
  Generated: 6
  Matched: 2
  Precision: 0.3333
  Recall: 0.0513
  F1: 0.0889
  Cache Compression: 4726→2087 tokens
  Time/Frame: 0.13s
```

### 2. Run All Strategies

Compare all strategies on the same video:

```bash
./custom/src/prospect/tests/run_tests.sh e2e all
```

### 3. Python API Usage

```python
from prospect.runners import VLMStreamRunner
from prospect.context_strategies import ContextStrategyFactory

# Initialize strategy
strategy = ContextStrategyFactory.create_strategy(
    strategy_type='drop_middle',
    max_seq_len=4096,
    reserved_seq_len=128,
    last_keep_len=512,
)

# Initialize runner
runner = VLMStreamRunner(
    model_name='HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    device='cuda',
    torch_dtype='bfloat16',
    context_strategy_type='drop_middle',
    context_strategy_config={'last_keep_len': 512},
    use_kv_cache=True,
)

# Run inference
video_data = {
    'video_id': '9011-c03f',
    'frames': frames,  # List of PIL Images
    'conversation': gt_conversation,
    'fps': 2.0,
}

results = runner.run_inference_on_video(video_data)
```

## Strategies Explained

### 1. None (Stateless)
- **No KV cache** - each turn starts fresh
- **Memory:** Constant (~3GB)
- **Speed:** Slowest (no context reuse)
- **Quality:** Baseline

```python
strategy_type='none'
```

### 2. Drop All
- **Clear cache** when exceeds threshold
- **Memory:** Grows then resets
- **Speed:** Fast (cache reuse until reset)
- **Quality:** Loses all context on reset

```python
strategy_type='drop_all'
# Cache: 0 → 4726 → 0 → 4726 → ...
```

### 3. Drop Middle (Recommended)
- **Keep initial + recent** tokens, drop middle
- **Memory:** Bounded (~7.5GB peak)
- **Speed:** Fast (always has context)
- **Quality:** Best (preserves task + recent activity)

```python
strategy_type='drop_middle'
context_strategy_config={
    'last_keep_len': 512,  # Recent tokens to keep
}
# Cache: 4726 → 2087 (init 1575 + recent 512)
```

### 4. Summarize and Drop
- **Generate summary** of cached context, then clear
- **Memory:** Grows then resets
- **Speed:** Slow (summarization overhead)
- **Quality:** Experimental

```python
strategy_type='summarize_and_drop'
# Note: Currently being updated to use fast_greedy_generate
```

## Configuration Options

### Strategy Parameters

```python
strategy_config = {
    'max_seq_len': 4096,       # Maximum sequence length
    'reserved_seq_len': 128,   # Reserve for new input
    'last_keep_len': 512,      # [drop_middle only] Recent tokens
}
```

**Tuning Guidelines:**
- **max_seq_len:** Model's maximum (4096 for SmolVLM)
- **reserved_seq_len:** Typical input size (128-256)
- **last_keep_len:** Trade-off memory vs context
  - Larger = better quality, more memory
  - Smaller = less memory, may lose context
  - Recommended: 512 (works well)

### Runner Parameters

```python
runner = VLMStreamRunner(
    model_name='HuggingFaceTB/SmolVLM2-2.2B-Instruct',
    device='cuda',
    torch_dtype='bfloat16',      # or 'float16'
    max_new_tokens=100,          # Max tokens per turn
    temperature=0.0,             # Greedy sampling (0.0)
    use_kv_cache=True,           # Enable caching
    context_strategy_type='drop_middle',
)
```

## Expected Performance

### drop_middle Strategy

**Video:** 230 seconds, 39 ground truth dialogues

| Metric | Value |
|--------|-------|
| Generated | 6 dialogues |
| Matched | 2 dialogues |
| Precision | 0.33 |
| Recall | 0.05 |
| F1 | 0.09 |
| Cache Size | 2087 tokens (bounded) |
| Compression | 4726→2087 (55.8% reduction) |
| Time/Frame | 0.13s |
| Peak Memory | 7.5GB |

### Comparison (Expected)

| Strategy | Memory | Speed | F1 | Comments |
|----------|--------|-------|-----|----------|
| none | 3GB | 1.0x | ~0.05 | Baseline |
| drop_all | 3-15GB | 2.5x | ~0.07 | Fast but loses context |
| drop_middle | 7.5GB | 1.7x | ~0.09 | Best balance |
| summarize | 3-15GB | 0.8x | TBD | Experimental |

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution 1:** Use smaller model
```python
model_name='HuggingFaceTB/SmolVLM-500M-Instruct'  # Smaller
```

**Solution 2:** Reduce cache size
```python
context_strategy_config={'last_keep_len': 256}  # Smaller recent context
```

**Solution 3:** Use float16
```python
torch_dtype='float16'  # Less memory than bfloat16
```

### Issue: "Generated 0 dialogues"

**Cause:** Model not detecting transitions or generating empty output

**Debug:**
```python
# Enable verbose logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Check transition detection
runner._detect_substep_transition(frame, history)
```

### Issue: "Cache not compressing"

**Cause:** Threshold not reached or initial_cache not set

**Debug:**
```python
# Check cache length
cache_len = runner.cache_manager.get_cache_length()
print(f"Cache length: {cache_len}")

# Check threshold
print(f"Threshold: {strategy.ctxlen_to_reduce}")

# Verify initial cache (for drop_middle)
print(f"Initial cache set: {strategy.init_kv_cache is not None}")
```

## Testing

### Unit Tests

```bash
# Test strategies
pytest custom/src/prospect/tests/test_context_strategies.py

# Test cache manager
pytest custom/src/prospect/tests/test_cache_manager.py

# Test runner
pytest custom/src/prospect/tests/test_runner_integration.py
```

### E2E Tests

```bash
# Single strategy
./custom/src/prospect/tests/run_tests.sh e2e drop_middle

# All strategies
./custom/src/prospect/tests/run_tests.sh e2e all

# With custom config
./custom/src/prospect/tests/run_tests.sh e2e drop_middle --config custom_config.yaml
```

## Output Files

Results are saved to:
```
custom/outputs/single_strategy_tests/
├── drop_middle/
│   ├── drop_middle_predictions.json       # Model predictions
│   ├── drop_middle_metrics.json           # Evaluation metrics
│   └── drop_middle_20251104_125618.log    # Detailed log
```

**Metrics JSON:**
```json
{
  "frames_processed": 461,
  "dialogues_generated": 6,
  "dialogues_matched": 2,
  "precision": 0.3333,
  "recall": 0.0513,
  "f1": 0.0889,
  "cache_compressions": 3,
  "avg_cache_size": 2087,
  "time_per_frame": 0.13,
  "peak_memory_mb": 7514.56
}
```

## Advanced Usage

### Custom Strategy

Create your own compression strategy:

```python
from prospect.context_strategies import BaseContextStrategy

class CustomStrategy(BaseContextStrategy):
    def should_reduce_cache(self, current_seq_len: int) -> bool:
        """Decide when to compress"""
        return current_seq_len >= self.ctxlen_to_reduce
    
    def handle_overflow(self, past_key_values, last_msg, **context):
        """Implement custom compression logic"""
        # Your compression algorithm here
        compressed_cache = my_compression(past_key_values)
        return compressed_cache, last_msg
    
    @property
    def name(self) -> str:
        return "custom"
```

### Batch Processing

Process multiple videos:

```python
from prospect.generators import BaselineGenerator

generator = BaselineGenerator(
    runner=runner,
    dataset=video_dataset,
    output_dir='outputs/my_experiment',
)

metrics = generator.run()  # Process all videos
```

## Next Steps

1. **Test all strategies:** Run comparison on your videos
2. **Tune parameters:** Optimize for your use case
3. **Analyze results:** Compare F1 scores and memory usage
4. **Deploy:** Integrate into your pipeline

## Support

For issues or questions:
- Check [pipeline_architecture.md](pipeline_architecture.md) for implementation details
- Review test files for usage examples
- See [test_coverage_analysis.md](test_coverage_analysis.md) for test coverage

## References

- **ProAssist Paper:** [Link]
- **SmolVLM Model:** https://huggingface.co/HuggingFaceTB/SmolVLM2-2.2B-Instruct
- **Full Documentation:** `custom/docs/pipeline_architecture.md`
