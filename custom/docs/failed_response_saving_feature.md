# Failed Response Saving Feature

## Overview
Added functionality to automatically save raw GPT responses when JSON parsing fails, enabling post-mortem analysis of generation failures.

## Changes Made

### 1. Modified `base_gpt_generator.py`

#### Added `_save_failed_response()` method
- Creates `failed_responses/` subdirectory in output directory
- Saves failed responses as `failed_{filename}.txt`
- Includes metadata header with:
  - Input file path
  - Error reason
  - Raw response text
- Handles UTF-8 encoding properly
- Includes error logging for save failures

#### Modified `generate_and_save_dst_outputs()`
- Added `failure_info` tracking: `Dict[str, Tuple[str, Any]]`
- Accumulates failure information across all retry attempts
- After all retries complete, saves responses for JSON parse failures
- Logs count of saved responses

#### Modified `generate_dst_outputs()`
- Now returns `Tuple[Dict[str, Optional[DSTOutput]], Dict[str, Tuple[str, Any]]]`
- Returns both outputs and failure_info

#### Modified `generate_multiple_dst_structures()`
- Now returns `Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[str, Any]]]`
- Tracks failure info from `_execute_generation_round()`
- Stores `(error_reason, raw_content)` for each failure

### 2. Updated Tests

#### `test_single_retries.py`
- Updated to unpack tuple return value from `generate_multiple_dst_structures()`

#### `test_simple_dst_generator.py` (2 locations)
- Updated `test_required_fields_validation` to unpack tuple
- Updated `test_dst_structure_validation` to unpack tuple

## Behavior

### When JSON Parsing Fails
1. System attempts generation (up to `max_retries + 1` times)
2. If JSON parsing fails, raw response is captured
3. After all attempts exhausted, system saves:
   - Only responses where error reason contains "JSON Parse Error"
   - Ignores validator rejections (structural issues)
4. Saves to: `{output_dir}/failed_responses/failed_{filename}.txt`

### File Format
```
=== FAILURE INFO ===
Input file: /path/to/input.json
Error: JSON Parse Error at line 490, column 123: Expecting ',' delimiter

=== RAW RESPONSE ===
{actual raw GPT-4o response text}
```

## Usage

No code changes required for existing users. The feature automatically activates when:
- A file fails with a JSON parse error
- All retry attempts are exhausted
- The raw response is available

## Testing

All 55 tests pass:
```bash
pytest tests/ -v
# Result: 55 passed, 1 warning
```

Manual verification confirmed:
- Failed responses directory is created
- Files are saved with correct naming
- Content includes metadata and raw response
- Logging shows count of saved responses

## Benefits

1. **Post-mortem analysis**: Can analyze patterns in failed responses
2. **Prompt engineering**: Identify common failure modes in GPT-4o output
3. **Error recovery**: Potentially implement custom parsers for common malformations
4. **Debugging**: Understand what GPT-4o is actually generating
5. **Metrics**: Track types of failures over time

## Example Output

When processing 4,482 files with ~950 JSON parse failures:
```
💾 Saved 950 failed responses to /path/to/output/failed_responses/
```

Each failed file gets its raw response saved for analysis.
