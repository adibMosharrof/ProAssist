# PROSPECT Tests

Comprehensive test suite for PROSPECT (Proactive Assistant Evaluation).

## Quick Start

```bash
# Run all tests
./run_tests.sh all

# Run specific test suite
./run_tests.sh custom_model    # Custom SmolVLM2 tests
./run_tests.sh integration     # VLM runner integration tests
./run_tests.sh strategy         # Context strategy tests
./run_tests.sh quick            # Quick smoke tests only
```

## Test Organization

Following the pattern from `dst_data_builder/tests/`, this test suite provides:

- **Shared fixtures** (`conftest.py`) - Reusable test data, mocks, and helpers
- **Modular test files** - Each file tests a specific component
- **Main runner** (`run_tests.py`) - Python-based test orchestration
- **Shell wrapper** (`run_tests.sh`) - Handles bash profile sourcing and environment setup

### Test Files

```
tests/
├── conftest.py                    # Shared fixtures and utilities
├── run_tests.py                   # Main Python test runner
├── run_tests.sh                   # Shell wrapper with bash profile sourcing
├── test_custom_model.py           # Custom SmolVLM2 model tests (5 tests)
├── test_runner_integration.py     # VLM runner integration tests (3 tests)
├── test_context_strategies.py     # Context strategy unit tests
└── test_e2e_strategies.py         # End-to-end strategy comparison
```

## Test Suites

### 1. Custom Model Tests (`test_custom_model.py`)

Tests for the custom SmolVLM2 model with KV cache management.

**Tests**:
- ✅ `test_model_loading` - Verify model loads without errors
- ✅ `test_joint_embed` - Test joint_embed() produces correct embeddings
- ✅ `test_fast_greedy_generate` - Test generation with KV cache
- ✅ `test_kv_cache_accumulation` - Verify cache grows across frames
- ✅ `test_processor` - Test processor input sequence creation

**Run**:
```bash
./run_tests.sh custom_model
# or
pytest test_custom_model.py -v
```

### 2. Integration Tests (`test_runner_integration.py`)

Integration tests for VLM stream runner with custom model.

**Tests**:
- ✅ `test_runner_initialization` - Runner initializes with custom model
- ✅ `test_single_frame_generation` - Generate dialogue with KV cache
- ✅ `test_multi_frame_accumulation` - Cache accumulates, overflow triggers

**Run**:
```bash
./run_tests.sh integration
# or
pytest test_runner_integration.py -v
```

### 3. Strategy Tests (`test_context_strategies.py`)

Unit tests for all context overflow strategies.

**Tests**:
- `TestDropAllStrategy` - Tests for drop_all strategy
- `TestDropMiddleStrategy` - Tests for drop_middle strategy
- `TestSummarizeAndDropStrategy` - Tests for summarize_and_drop strategy
- `TestStrategyComparison` - Comparative tests across strategies

**Run**:
```bash
./run_tests.sh strategy
# or
pytest test_context_strategies.py -v
```

### 4. E2E Tests (`test_e2e_strategies.py`)

End-to-end comparison of all strategies on real video data.

**Features**:
- Tests all 4 strategies (none, drop_all, drop_middle, summarize_and_drop)
- Uses BaselineGenerator + StreamEvaluator (ProAssist's framework)
- Generates metrics: F1, BLEU, CIDEr, METEOR, semantic similarity
- Creates comparison CSV and detailed JSON

**Run**:
```bash
# E2E tests take 15-20 minutes
python test_e2e_strategies.py
```

**⚠️ Note**: E2E tests require ~2GB disk space for sentence-transformers model download.

## Shared Fixtures (conftest.py)

Following the pattern from `dst_data_builder/tests/conftest.py`, we provide reusable fixtures:

### Configuration Fixtures
- `basic_prospect_config` - Base PROSPECT configuration
- `context_strategy_configs` - All strategy configurations

### Sample Data Fixtures
- `sample_image` / `sample_images` - Test images
- `sample_video_frames` - Temporary directory with frames
- `sample_dst_annotations` - DST annotations
- `sample_conversation` - Sample dialogue

### Model Fixtures (Mocked)
- `mock_custom_smolvlm_model` - Mocked CustomSmolVLM
- `mock_processor` - Mocked processor

### KV Cache Fixtures
- `sample_kv_cache` - Standard KV cache (1000 tokens)
- `large_kv_cache` - Large cache that exceeds threshold (4500 tokens)

### Helper Functions
- `assert_kv_cache_format(cache)` - Validate KV cache structure
- `assert_kv_cache_size(cache, expected_len)` - Check cache sequence length
- `create_mock_frame_output(...)` - Create FrameOutput for testing

## Environment Setup

The shell wrapper (`run_tests.sh`) handles environment setup:

1. **Sources bash profile** - Fixes HOME directory for disk space
2. **Activates conda environment** - Uses `.venv`
3. **Sets PYTHONPATH** - Adds `custom/src` to path
4. **Checks pytest** - Ensures pytest is installed

This follows the pattern from `run_dst_generator.sh`.

## Usage Examples

### Run All Tests
```bash
cd /u/siddique-d1/adib/ProAssist
./custom/src/prospect/tests/run_tests.sh all
```

### Run Specific Suite
```bash
# Custom model tests only
./custom/src/prospect/tests/run_tests.sh custom_model

# Integration tests only
./custom/src/prospect/tests/run_tests.sh integration

# Strategy tests only
./custom/src/prospect/tests/run_tests.sh strategy
```

### Run Quick Tests (Skip Slow Tests)
```bash
./custom/src/prospect/tests/run_tests.sh quick
```

### Use Python Runner Directly
```bash
cd /u/siddique-d1/adib/ProAssist
source ~/.bash_profile
conda activate .venv
export PYTHONPATH="custom/src:$PYTHONPATH"

python custom/src/prospect/tests/run_tests.py --suite all
```

### Use pytest Directly
```bash
cd /u/siddique-d1/adib/ProAssist
source ~/.bash_profile
conda activate .venv
export PYTHONPATH="custom/src:$PYTHONPATH"

# Run all tests
pytest custom/src/prospect/tests/ -v

# Run specific file
pytest custom/src/prospect/tests/test_custom_model.py -v

# Run specific test
pytest custom/src/prospect/tests/test_custom_model.py::test_model_loading -v

# Run with coverage
pytest custom/src/prospect/tests/ --cov=prospect --cov-report=html
```

## Test Markers

Tests can be marked with pytest markers for selective running:

```python
@pytest.mark.slow
def test_long_running_operation():
    ...

@pytest.mark.integration
def test_full_pipeline():
    ...
```

Run only quick tests (skip slow):
```bash
pytest -m "not slow"
```

Run only integration tests:
```bash
pytest -m integration
```

## Troubleshooting

### Disk Space Errors

If you see "No space left on device":

```bash
# Check disk usage
df -h

# Clean HuggingFace cache
rm -rf ~/.cache/huggingface/hub/*

# Or set HF_HOME to existing cache
export HF_HOME=/path/to/existing/cache
```

The shell wrapper sources bash profile which may fix HOME directory issues.

### Import Errors

If you see import errors:

```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"

# Or use the shell wrapper which sets it automatically
./run_tests.sh all
```

### CUDA Errors

If you see CUDA errors on CPU-only machines:

```bash
# Set device to CPU
export CUDA_VISIBLE_DEVICES=""

# Or modify test to use CPU
pytest --device=cpu
```

## Test Development

### Adding New Tests

1. Create test file: `test_<component>.py`
2. Import fixtures from conftest:
   ```python
   def test_something(sample_image, mock_processor):
       # Use fixtures
       ...
   ```
3. Follow naming convention: `test_<what_it_tests>`
4. Add docstrings explaining what is tested
5. Run tests: `./run_tests.sh all`

### Adding New Fixtures

1. Add to `conftest.py`:
   ```python
   @pytest.fixture
   def my_fixture():
       """Describe what this fixture provides"""
       return test_data
   ```
2. Document in fixture docstring
3. Use in tests by name

### Best Practices

Following patterns from `dst_data_builder/tests/`:

1. **Use fixtures for common setup** - Avoid duplication
2. **Mock external dependencies** - Don't call real APIs
3. **Test one thing per test** - Keep tests focused
4. **Use descriptive names** - `test_kv_cache_accumulation` not `test_1`
5. **Add docstrings** - Explain what is tested and why
6. **Parametrize similar tests** - Use `@pytest.mark.parametrize`

## CI/CD Integration

### Running in CI

```yaml
# Example GitHub Actions
- name: Run PROSPECT tests
  run: |
    source ~/.bash_profile || true
    ./custom/src/prospect/tests/run_tests.sh quick
```

### Test Coverage

```bash
# Generate coverage report
pytest custom/src/prospect/tests/ \
  --cov=prospect \
  --cov-report=html \
  --cov-report=term

# View report
open htmlcov/index.html
```

## References

- **Test Pattern**: `custom/src/dst_data_builder/tests/` - Similar structure
- **Implementation Status**: `custom/docs/updates/17-implementation_status.md`
- **Error Analysis**: `custom/docs/updates/16-e2e_test_errors_and_fixes.md`
- **Custom Model**: `custom/src/prospect/models/custom_smolvlm.py`

## Current Status

**Unit Tests**: ✅ 8/8 PASSING
- Custom model: 5/5 passing
- Integration: 3/3 passing

**E2E Tests**: ⚠️ BLOCKED
- Issue: Disk space exhaustion
- Solution: Free ~2GB or source bash profile

**Coverage**: ~80% (estimated)
- Core model: Fully tested
- Strategies: Partially tested
- E2E: Needs completion
