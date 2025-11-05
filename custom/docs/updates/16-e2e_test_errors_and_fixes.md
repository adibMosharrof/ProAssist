# E2E Test Run: Errors Found and Fixes Applied

**Date**: November 3, 2025  
**Status**: Critical disk space issue blocking evaluation

## Executive Summary

The E2E test successfully launched and tested all 4 context strategies (none, drop_all, drop_middle, summarize_and_drop), but all failed during the evaluation phase due to **disk space exhaustion** when trying to download the `sentence-transformers/all-mpnet-base-v2` model needed for semantic similarity scoring.

Additionally, several code bugs were discovered and fixed:
1. ✅ **Missing decode() method** in CustomSmolVLMProcessor - FIXED
2. ✅ **Incorrect image handling** in summarize_and_drop - FIXED  
3. ⚠️ **drop_middle KV cache format issue** - Needs investigation (secondary to disk space)

## Detailed Error Analysis

### 1. Disk Space Exhaustion (CRITICAL - BLOCKS ALL TESTS)

**Error**:
```
RuntimeError: Data processing error: CAS service error : IO Error: No space left on device (os error 28)
OSError: [Errno 28] No space left on device
```

**Location**: StreamEvaluator tries to download sentence-transformers model  
**Impact**: ALL strategies fail - no metrics can be computed  
**Root Cause**: Machine `/u/siddique-d1/adib/` has no disk space left

**Solution Required**:
```bash
# User needs to clean up disk space, for example:
# 1. Check disk usage
df -h

# 2. Find large files/directories
du -sh ~/* | sort -rh | head -20

# 3. Possible cleanup candidates:
#    - HuggingFace cache: ~/.cache/huggingface/
#    - Old model checkpoints
#    - Temporary outputs
```

**Workaround** (if model already exists elsewhere):
Set `HF_HOME` environment variable to point to existing cache:
```bash
export HF_HOME=/path/to/existing/hf/cache
```

---

### 2. CustomSmolVLMProcessor Missing decode() Method (FIXED ✅)

**Error**:
```
AttributeError: 'CustomSmolVLMProcessor' object has no attribute 'decode'
```

**Location**: `vlm_stream_runner.py` lines 300, 328  
**Impact**: Strategy "none" failed - no dialogues generated (0 generated vs 39 expected)

**Fix Applied**:
Added decode methods to CustomSmolVLMProcessor:
```python
def decode(self, *args, **kwargs):
    """Delegate decode to tokenizer"""
    return self.tokenizer.decode(*args, **kwargs)

def batch_decode(self, *args, **kwargs):
    """Delegate batch_decode to tokenizer"""
    return self.tokenizer.batch_decode(*args, **kwargs)
```

**Status**: ✅ FIXED in `custom/src/prospect/models/processing_custom_smolvlm.py`

---

### 3. Summarize and Drop - Incorrect Image Handling (FIXED ✅)

**Error**:
```
ValueError: We detected 1 tokens in the text but no images/videos were passed
```

**Location**: `summarize_and_drop.py` - `_generate_summary()`  
**Impact**: Summarization fails, falls back to "Task in progress."

**Root Cause**: 
- Summary prompt included `<image>` token but passed `pixel_values=None`
- Summarization should be text-only (no image needed for text summary)

**Fix Applied**:
```python
# OLD (incorrect):
summary_text = f"<image>{self.summary_prompt}"
inputs_embeds = model.joint_embed(
    input_ids=summary_tokens,
    pixel_values=summary_inputs.get('pixel_values'),  # ❌ Causes error
    ...
)

# NEW (correct):
summary_text = self.summary_prompt  # No <image> token
inputs_embeds = model.joint_embed(
    input_ids=summary_tokens,
    pixel_values=None,  # ✅ Text-only summarization
    ...
)
```

**Status**: ✅ FIXED in `custom/src/prospect/context_strategies/summarize_and_drop.py`

---

### 4. drop_middle - KV Cache Format Issue (NEEDS INVESTIGATION ⚠️)

**Error**:
```
AttributeError: 'tuple' object has no attribute 'get_seq_length'
```

**Location**: After `drop_middle` modifies KV cache and returns it  
**Impact**: After first overflow, subsequent generations fail

**Root Cause** (hypothesis):
- `drop_middle` returns modified KV cache as tuple (correct format)
- BUT: Idefics2/SmolVLM2 may internally convert to DynamicCache
- After modification, something breaks in the conversion

**Evidence**:
- ✅ `drop_all` works (returns None, no cache to convert)
- ✅ ProAssist uses identical tuple format for drop_middle
- ❌ `drop_middle` fails after first modification

**Generated Output**:
- Strategy generated 1 dialogue successfully
- Then failed on all subsequent frames (6 failures)

**Next Steps**:
1. Check if SmolVLM2 uses DynamicCache internally
2. Compare with ProAssist's LlamaForCausalLM (works fine with tuples)
3. Possible fix: Convert modified tuple back to DynamicCache if needed
4. Or: Investigate if Idefics2 has different KV cache requirements

**Status**: ⚠️ NEEDS INVESTIGATION (secondary priority - disk space is blocker)

---

## Test Results Before Disk Space Failure

### Strategy: none (stateless)
- **Duration**: ~35 seconds  
- **Dialogues Generated**: 0 (expected 39)
- **Issue**: Missing decode() method ✅ NOW FIXED
- **Result**: FAILED - no output due to decode error

### Strategy: drop_all  
- **Duration**: ~21 seconds
- **Dialogues Generated**: 3 (expected 39)
- **KV Cache Overflows**: 3 times (correctly dropped each time)
- **Cache After Drop**: 0 tokens ✅
- **Issue**: Evaluation failed due to disk space
- **Result**: PARTIAL - generation worked, evaluation failed

### Strategy: drop_middle
- **Duration**: ~11 seconds
- **Dialogues Generated**: 1 (expected 39)
- **KV Cache Overflows**: 1 time (correctly modified: 4528 tokens)
- **Cache After Drop**: 4528 tokens (kept initial + recent)
- **Issue**: Subsequent generations failed with 'get_seq_length' error
- **Result**: PARTIAL - first generation worked, then broke

### Strategy: summarize_and_drop
- **Duration**: ~23 seconds
- **Dialogues Generated**: 3 (expected 39)
- **KV Cache Overflows**: 3 times  
- **Summaries Generated**: 3 (all fell back to "Task in progress.")
- **Issue**: Image handling error ✅ NOW FIXED
- **Result**: PARTIAL - generation worked with fallback, evaluation failed

---

## Next Actions

### IMMEDIATE (User Action Required)
1. **Free up disk space** on `/u/siddique-d1/adib/`
   - Check `df -h` and `du -sh ~/* | sort -rh | head -20`
   - Clean up HuggingFace cache: `~/.cache/huggingface/`
   - Or set `HF_HOME` to existing cache location

### AFTER DISK SPACE FIXED
2. **Re-run E2E tests** with fixes:
   ```bash
   cd /u/siddique-d1/adib/ProAssist
   ./.venv/bin/python custom/src/prospect/test_e2e_strategies.py 2>&1 | tee custom/outputs/e2e_test_run.log
   ```

3. **Investigate drop_middle issue** if it persists:
   - Check Idefics2 KV cache internals
   - Compare with ProAssist's implementation
   - May need DynamicCache compatibility layer

### SUCCESS CRITERIA
- ✅ All 4 strategies complete without errors
- ✅ Metrics computed: F1, BLEU, CIDEr, METEOR, semantic similarity
- ✅ Comparison CSV generated with valid data
- ✅ Can determine which strategy performs best

---

## Code Changes Applied

### File 1: `custom/src/prospect/models/processing_custom_smolvlm.py`
**Change**: Added decode() and batch_decode() methods  
**Lines**: After line 38  
**Status**: ✅ Committed

### File 2: `custom/src/prospect/context_strategies/summarize_and_drop.py`  
**Change**: Removed image from summarization (text-only)  
**Lines**: 158-177  
**Status**: ✅ Committed

---

## Lessons Learned

1. **Always check disk space** before long-running tests
2. **Wrapper classes need complete interface** - CustomSmolVLMProcessor needed decode()
3. **VLM models are picky about image tokens** - Must match text/image carefully
4. **KV cache format matters** - Different model architectures may have different requirements
5. **Test incrementally** - Each strategy should be unit tested before E2E

---

## References

- E2E Test Script: `custom/src/prospect/test_e2e_strategies.py`
- Test Output Log: `custom/outputs/e2e_test_run.log`
- Custom Model: `custom/src/prospect/models/custom_smolvlm.py`
- Context Strategies: `custom/src/prospect/context_strategies/`

