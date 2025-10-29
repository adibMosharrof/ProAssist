# Test Coverage Analysis & Recommendations

## Executive Summary

**Current Test Status**: 6/24 tests passing (25% pass rate)  
**Major Issues**: 
- 5 tests have file path errors (missing test data)
- 8 tests have incorrect async handling
- 5 tests use wrong API expectations
- Several tests are outdated

## Test Results Breakdown

### ✅ PASSING Tests (6/24)
1. `test_single_gpt_generator_initialization` - Generator creates successfully
2. `test_initialization_with_config` - SimpleDSTGenerator initializes
3. `test_metadata_generation` - Metadata counting logic works
4. `test_structure_validator_simple_pass` - Structure validation works
5. `test_timestamps_validator_pass_and_fail` - Timestamp validation works
6. `test_id_validator_*` (3 tests) - ID validation works

**Analysis**: Validators are well-tested and working correctly. Basic initialization tests pass.

---

## ❌ FAILING Tests by Category

### Category 1: Missing Test Data Files (5 tests - CRITICAL)

**Affected Tests:**
- `test_dst_prompt_creation`
- `test_timestamp_parsing`
- `test_dst_structure_validation`
- `test_action_timestamp_containment`
- `test_retry_with_detailed_error_messages`
- `test_dst_gen_manual_data_validation`

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 
'data/proassist_dst_manual_data/assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.json'
'data/proassist_dst_manual_data/ego4d_grp-cec778f9-9b54-4b67-b013-116378fd7a85.json'
```

**Root Cause**: Tests depend on external data files that don't exist in test environment.

**Recommendation**: 
- ❌ **REMOVE** these tests OR
- ✅ **REPLACE** with inline JSON fixtures using `tmp_path`
- Tests should be **self-contained** and not depend on external data

**Action Required**:
```python
# BAD (current approach)
def test_example(sample_data_assembly):
    data_path = Path("data/proassist_dst_manual_data/file.json")
    with open(data_path, "r") as f:
        return json.load(f)

# GOOD (recommended approach)
def test_example(tmp_path):
    test_data = {
        "video_uid": "test_123",
        "inferred_knowledge": "test knowledge",
        "parsed_video_anns": {
            "all_step_descriptions": "[0.0s-10.0s] Step 1"
        }
    }
    test_file = tmp_path / "test.json"
    test_file.write_text(json.dumps(test_data))
    # ... rest of test
```

---

### Category 2: Empty Dataset Errors (4 tests)

**Affected Tests:**
- `test_manual_dataloader`
- `test_dst_data_module`
- `test_dst_generator_dataloader_integration`
- `test_output_validation`

**Error:**
```
ValueError: num_samples should be a positive integer value, but got num_samples=0
```

**Root Cause**: Tests try to create DataLoaders from empty directories.

**Recommendation**: 
- ✅ **FIX** by creating temporary test data in `tmp_path`
- Tests should create minimal valid datasets before testing DataLoader

**Example Fix**:
```python
def test_manual_dataloader(tmp_path):
    # Create test data
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    test_file = test_dir / "sample.json"
    test_file.write_text(json.dumps({
        "video_uid": "test",
        "inferred_knowledge": "test",
        "parsed_video_anns": {"all_step_descriptions": "test"}
    }))
    
    # Now test with non-empty dataset
    dataset = ManualDSTDataset(data_path=str(test_dir), num_rows=None)
    assert len(dataset) > 0
    # ... rest of test
```

---

### Category 3: Wrong API Usage (3 tests)

**Test**: `test_rebatch_retries`  
**Error**: `assert None == {"ok": True}` - Result is None instead of expected dict

**Root Cause**: Mocked methods don't match actual implementation:
1. `_parse_batch_results` returns `None` for failures
2. Test expects dict but gets None

**Fix**: Update mock to return actual expected format

---

**Test**: `test_single_generator_retries_on_validator_rejection`  
**Error**: `assert None == {...}` - Similar issue

**Root Cause**: Mock doesn't properly simulate retry behavior

---

**Test**: `test_required_fields_validation`  
**Error**: `TypeError: argument of type 'coroutine' is not iterable`

**Root Cause**: Test doesn't `await` async method!

**Fix**:
```python
# Change from:
results = generator.generate_and_save_dst_outputs(...)
assert "dummy_path" in results

# To:
results = await generator.generate_and_save_dst_outputs(...)
assert "dummy_path" in results
```

---

### Category 4: Missing Configuration (1 test)

**Test**: `test_proassist_dataloader`, `test_data_source_factory`  
**Error**: `TypeError: expected str, bytes or os.PathLike object, not NoneType`

**Root Cause**: ProAssistDSTDataset requires `data_path` but test passes None

**Fix**: Provide valid test path in configuration

---

## Components Tested vs Not Tested

### ✅ Well-Tested Components
| Component | Test Count | Status |
|-----------|-----------|--------|
| `StructureValidator` | 1 | ✅ Passing |
| `TimestampsValidator` | 1 | ✅ Passing |
| `IdValidator` | 4 | ✅ Passing |
| `SimpleDSTGenerator.__init__` | 1 | ✅ Passing |
| `SingleGPTGenerator.__init__` | 1 | ✅ Passing |

### ⚠️ Partially Tested Components
| Component | Test Count | Issues |
|-----------|-----------|--------|
| `SimpleDSTGenerator.run()` | 2 | Missing await, wrong mocks |
| `ManualDSTDataset` | 2 | Empty dataset errors |
| `DSTDataModule` | 2 | Empty dataset errors |
| `BatchGPTGenerator` | 1 | Mock doesn't match API |
| `SingleGPTGenerator` | 1 | Mock doesn't match API |

### ❌ UNTESTED Components (NEW CODE!)
| Component | Lines | Priority | Risk |
|-----------|-------|----------|------|
| `JSONParsingValidator` | 153 | **CRITICAL** | HIGH |
| `OpenAIAPIClient` | 89 | **CRITICAL** | HIGH |
| `GPTGeneratorFactory` | 101 | HIGH | MEDIUM |
| `BaseGPTGenerator._try_generate_and_validate()` | ~50 | HIGH | HIGH |
| `BaseGPTGenerator.generate_and_save_dst_outputs()` | ~100 | **CRITICAL** | HIGH |
| `dst_generation_prompt.create_dst_prompt()` | 126 | MEDIUM | LOW |
| `DSTOutput` | 86 | MEDIUM | LOW |
| `ProAssistDSTDataset` | ~80 | LOW | LOW |

---

## Critical Gaps - NEW TESTS NEEDED

### 1. **JSONParsingValidator Tests** ⚠️ CRITICAL

This is brand new code and has **zero test coverage**!

**Tests Needed:**
```python
def test_json_parsing_valid_json():
    """Test parsing valid JSON string"""
    validator = JSONParsingValidator()
    raw = '{"steps": [{"step_id": "S1"}]}'
    ok, msg = validator.validate(raw)
    assert ok
    assert validator.parsed_result == {"steps": [{"step_id": "S1"}]}

def test_json_parsing_with_markdown_fence():
    """Test parsing JSON wrapped in markdown```"""
    validator = JSONParsingValidator()
    raw = '```json\n{"steps": []}\n```'
    ok, msg = validator.validate(raw)
    assert ok
    assert validator.parsed_result == {"steps": []}

def test_json_parsing_invalid_json():
    """Test error handling for invalid JSON"""
    validator = JSONParsingValidator()
    raw = '{"steps": [unclosed'
    ok, msg = validator.validate(raw)
    assert not ok
    assert "line" in msg.lower() or "column" in msg.lower()

def test_json_parsing_trailing_commas():
    """Test cleaning trailing commas"""
    validator = JSONParsingValidator()
    raw = '{"steps": [{"id": "S1",}],}'  # Trailing commas
    ok, msg = validator.validate(raw)
    assert ok  # Should clean and parse successfully

def test_json_parsing_dict_input():
    """Test backward compatibility with dict input"""
    validator = JSONParsingValidator()
    data = {"steps": []}
    ok, msg = validator.validate(data)
    assert ok
    assert validator.parsed_result == data
```

**Why Critical**: This validator processes ALL GPT responses. Bugs here affect every DST generation.

---

### 2. **OpenAIAPIClient Tests** ⚠️ CRITICAL

Brand new code with **zero test coverage**!

**Tests Needed:**
```python
async def test_api_client_initialization():
    """Test client initialization"""
    client = OpenAIAPIClient(api_key="test_key", base_url="https://test.com")
    assert client.api_key == "test_key"
    assert client.base_url == "https://test.com"
    assert client.client is not None

async def test_api_client_invalid_key():
    """Test handling of invalid API key"""
    client = OpenAIAPIClient(api_key=None, base_url=None)
    assert client.client is None

async def test_generate_completion_success(monkeypatch):
    """Test successful API call"""
    client = OpenAIAPIClient(api_key="test_key")
    
    # Mock the API response
    async def fake_create(**kwargs):
        class FakeResponse:
            class Choice:
                class Message:
                    content = '{"test": "response"}'
                message = Message()
            choices = [Choice()]
        return FakeResponse()
    
    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    
    success, result = await client.generate_completion(
        prompt="test", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    assert success
    assert "test" in result

async def test_generate_completion_api_error(monkeypatch):
    """Test API error handling"""
    client = OpenAIAPIClient(api_key="test_key")
    
    async def fake_create(**kwargs):
        raise Exception("API Error")
    
    monkeypatch.setattr(client.client.chat.completions, "create", fake_create)
    
    success, result = await client.generate_completion(
        prompt="test", model="gpt-4o", temperature=0.1, max_tokens=1000
    )
    assert not success
    assert "api_error" in result.lower()
```

**Why Critical**: All API communication goes through this class. Network errors, timeouts, and rate limits must be handled correctly.

---

### 3. **GPTGeneratorFactory Tests** ⚠️ HIGH PRIORITY

**Tests Needed:**
```python
def test_factory_creates_single_generator(monkeypatch):
    """Test factory creates SingleGPTGenerator"""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test_key")
    gen = GPTGeneratorFactory.create_generator(
        generator_type="single",
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=1000,
    )
    assert isinstance(gen, SingleGPTGenerator)

def test_factory_creates_batch_generator(monkeypatch):
    """Test factory creates BatchGPTGenerator"""
    monkeypatch.setenv("OPENAI_API_KEY", "test_key")
    gen = GPTGeneratorFactory.create_generator(
        generator_type="batch",
        model_name="gpt-4o",
        temperature=0.1,
        max_tokens=1000,
    )
    assert isinstance(gen, BatchGPTGenerator)

def test_factory_invalid_type():
    """Test factory raises error for invalid type"""
    with pytest.raises(ValueError, match="Unsupported generator type"):
        GPTGeneratorFactory.create_generator(
            generator_type="invalid",
            model_name="gpt-4o",
            temperature=0.1,
            max_tokens=1000,
        )

def test_factory_with_custom_validators():
    """Test factory accepts custom validators"""
    custom_validators = [StructureValidator()]
    gen = GPTGeneratorFactory.create_generator(
        generator_type="single",
        generator_cfg={"validators": custom_validators},
    )
    assert len(gen.validators) == 1
```

---

### 4. **Global Retry Logic Tests** ⚠️ HIGH PRIORITY

The core retry mechanism in `generate_and_save_dst_outputs()` has **no dedicated tests**!

**Tests Needed:**
```python
async def test_global_retry_all_succeed_first_attempt(monkeypatch):
    """Test no retries needed when all succeed"""
    # Mock _execute_generation_round to succeed for all items
    # Verify only called once

async def test_global_retry_partial_failure_then_success(monkeypatch):
    """Test retry of failed items"""
    # First call: 2 succeed, 1 fails
    # Second call: 1 succeeds
    # Verify called twice with correct items

async def test_global_retry_max_retries_exceeded(monkeypatch):
    """Test giving up after max retries"""
    # All attempts fail
    # Verify stops after max_retries

async def test_global_retry_incremental_save(tmp_path, monkeypatch):
    """Test successful items saved immediately"""
    # Verify save_intermediate creates files after each round
```

---

### 5. **Integration Tests** ⚠️ MEDIUM PRIORITY

No end-to-end tests exist!

**Tests Needed:**
```python
async def test_end_to_end_single_file(tmp_path, monkeypatch):
    """Test complete generation for one file"""
    # Create input file
    # Mock API response
    # Run generator
    # Verify output file created with correct structure

async def test_end_to_end_batch(tmp_path, monkeypatch):
    """Test batch processing"""
    # Create multiple input files
    # Mock batch API
    # Run generator
    # Verify all outputs created
```

---

## Outdated Tests to Remove

### 1. `test_dst_gen_manual_single.py` - **ENTIRE FILE**

**Why Remove**:
- Tests rely on external data files
- Tests duplicate coverage from `test_simple_dst_generator.py`
- `test_output_validation` checks for specific directory structure that may not exist

**Action**: ❌ **DELETE FILE** - functionality tested elsewhere

---

### 2. Tests Using Real File Paths

**In `test_simple_dst_generator.py`**:
- `sample_data_assembly` fixture (lines 91-95)
- `sample_data_ego4d` fixture (lines 98-102)

**Action**: ❌ **REPLACE** with inline test data

---

## Summary of Required Actions

### Immediate (Critical - Do First)
1. ✅ **CREATE** `test_json_parsing_validator.py` with 5-6 tests
2. ✅ **CREATE** `test_openai_api_client.py` with 4-5 tests
3. ✅ **CREATE** `test_gpt_generator_factory.py` with 4 tests
4. ✅ **FIX** `test_required_fields_validation` - add `await`
5. ✅ **FIX** `test_rebatch_retries` - update mock to match API
6. ✅ **FIX** `test_single_generator_retries` - update mock to match API

### High Priority (Do Next)
7. ✅ **CREATE** `test_global_retry.py` with 4-5 tests for retry logic
8. ✅ **FIX** dataloader tests to use `tmp_path` with valid data
9. ✅ **REMOVE** `test_dst_gen_manual_single.py` entirely
10. ✅ **REPLACE** file fixtures in `test_simple_dst_generator.py`

### Medium Priority (Nice to Have)
11. ✅ **CREATE** end-to-end integration tests
12. ✅ **ADD** tests for `DSTOutput.from_data_and_dst()`
13. ✅ **ADD** tests for prompt creation (`create_dst_prompt`)

---

## Test Quality Issues

### Missing Assertions
Some tests have weak assertions:
```python
# BAD - too vague
assert results is not None

# GOOD - specific
assert len(results) == 2
assert "S1" in results["steps"][0]["step_id"]
```

### Poor Mocking
Tests mock entire methods instead of external dependencies:
```python
# BAD - mocks internal logic
monkeypatch.setattr(gen, "_execute_generation_round", fake_execute)

# GOOD - mocks external API
monkeypatch.setattr(gen.api_client, "generate_completion", fake_api)
```

### No Edge Case Testing
Missing tests for:
- Empty inputs
- Malformed JSON
- Network timeouts
- Rate limiting
- Concurrent access

---

## Recommended Test Structure

```
tests/
├── unit/
│   ├── test_validators.py ✅ (exists, working)
│   ├── test_json_parsing_validator.py ❌ (CREATE)
│   ├── test_openai_api_client.py ❌ (CREATE)
│   ├── test_gpt_generator_factory.py ❌ (CREATE)
│   ├── test_dst_output.py ❌ (CREATE)
│   └── test_datasets.py ✅ (fix existing)
│
├── integration/
│   ├── test_single_generator.py ✅ (fix existing)
│   ├── test_batch_generator.py ✅ (fix existing)
│   ├── test_global_retry.py ❌ (CREATE)
│   └── test_end_to_end.py ❌ (CREATE)
│
└── fixtures/
    ├── sample_data.py (shared test data)
    └── mocks.py (shared mocks)
```

---

## Metrics

### Before Fixes
- **Total Tests**: 24
- **Passing**: 6 (25%)
- **Failing**: 18 (75%)
- **Code Coverage (estimated)**: ~30%
- **Critical Components Untested**: 3

### After Fixes (Projected)
- **Total Tests**: ~40 (add 16 new)
- **Passing**: ~38 (95%)
- **Failing**: ~2 (5%)
- **Code Coverage (estimated)**: ~80%
- **Critical Components Untested**: 0

---

## Next Steps

1. **Review this document** with your team
2. **Prioritize** which tests to write first
3. **Create a branch** for test improvements
4. **Implement** tests in order of priority
5. **Run coverage tool** to verify improvement:
   ```bash
   pytest --cov=dst_data_builder --cov-report=html
   ```

Would you like me to:
1. Start writing the critical missing tests?
2. Fix the existing failing tests?
3. Remove outdated tests?
4. All of the above?
