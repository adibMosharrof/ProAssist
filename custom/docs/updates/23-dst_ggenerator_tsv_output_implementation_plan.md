# DST Generator TSV Output Implementation Plan

## Overview

This document outlines the implementation plan to update the `simple_dst_generator.py` script to generate TSV (Tab-Separated Values) output instead of JSON. The change addresses GPT model limitations with valid JSON generation while maintaining the existing modular, object-oriented design with separation of concerns.

## Requirements

- **Input**: Same as current (video files, configurations via Hydra).
- **Output**: TSV files with format: `type\tid\tstart_ts\tend_ts\tname` (one row per node: STEP, SUBSTEP, ACTION).
- **Processing**: Individual file processing (no batching for now).
- **Error Handling**: Skip invalid rows; raise errors for critical failures (no fallbacks).
- **Validators**: Already disabled; focus on output generation pipeline.
- **Dependencies**: Install any required packages (e.g., for TSV handling).
- **Design**: Maintain OO design, separation of concerns, no nested methods, imports at top of files.

## Implementation Details

### 1. TSV Format Specification
- **Columns**: `type`, `id`, `start_ts`, `end_ts`, `name`
- **Rows**: One per hierarchical node (STEP, SUBSTEP, ACTION)
- **IDs**: Hierarchical flattening with unique IDs (e.g., S1 for steps, S1.1 for substeps, S1.1.a for actions)
- **Data Types**: `type` (string: STEP/SUBSTEP/ACTION), `id` (string), `start_ts`/`end_ts` (float), `name` (string)

### 2. GPT Prompt Modification
- Update GPT prompts in generator classes to request TSV output instead of JSON.
- Example prompt structure: "Generate a TSV with columns: type, id, start_ts, end_ts, name. Each row represents a STEP, SUBSTEP, or ACTION node with hierarchical IDs."
- Parse GPT response as TSV text (split by lines/tabs).

### 3. Output Saving Logic
- Replace JSON dumping with TSV writing.
- Use Python's `csv` module (tab delimiter) to write rows.
- File naming: `{video_uid}.tsv` (same as before).
- Output directory: `dst_outputs/` with summary file.

### 4. Processing Changes
- **No Batching**: Process files individually in the main loop.
- **Error Handling**: Skip malformed TSV rows from GPT; raise exceptions for file I/O or critical parsing errors.
- **Hierarchical Flattening**: Convert any internal hierarchical structures to flat TSV rows during output.

### 5. Code Reuse and Modularity
- Leverage existing `GPTGeneratorFactory`, `DSTDataModule`, and generator classes.
- Keep logic separated: data loading in modules, generation in factories, output in dedicated methods.
- No nested methods; all imports at file tops.
- OO design: Classes for generators, data modules, etc.

### 6. Testing
- **Sample Data Test Case**: Use provided JSON/TSV sample pair to validate flattening and output.
- **Unit Tests**: Test TSV parsing, writing, and error skipping.
- **Integration Test**: Run full pipeline on sample data, verify TSV matches expected format.

### 7. Dependencies
- Standard library: `csv` for TSV handling.
- If additional packages needed (e.g., for advanced TSV parsing), install via pip/requirements.

## Design Principles

- **Separation of Concerns**: Data loading (DSTDataModule), generation (GPTGeneratorFactory), output (saving methods).
- **Object-Oriented**: Classes with clear responsibilities, no monolithic functions.
- **Error-First**: Fail fast on critical errors; skip non-critical issues (e.g., invalid rows).
- **Minimal Changes**: Reuse existing code; only modify output format and processing logic.
- **No Backward Compatibility**: No JSON fallbacks or legacy support.

## Implementation Checklist

- [ ] Update GPT prompts in generator classes for TSV output.
- [ ] Modify response parsing to handle TSV text.
- [ ] Replace JSON saving with TSV writing logic.
- [ ] Remove/disable batch processing in main loop.
- [ ] Add error handling for invalid TSV rows (skip them).
- [ ] Update summary generation to reflect TSV output.
- [ ] Add sample data test case.
- [ ] Test with sample JSON/TSV pair.
- [ ] Install any new dependencies.
- [ ] Update logging messages for TSV.

## Risks and Considerations

- **Data Integrity**: Ensure hierarchical flattening preserves all necessary information from JSON structure.
- **GPT Output Quality**: TSV may be more reliable than JSON, but monitor for formatting issues.
- **Performance**: Individual file processing may be slower; optimize if needed later.
- **Future Validators**: Re-enabling validators will require careful integration with TSV format.

## Timeline

- **Phase 1**: Update prompts and parsing (1-2 days).
- **Phase 2**: Implement TSV saving and error handling (1 day).
- **Phase 3**: Testing and validation (1-2 days).
- **Total**: 3-5 days for implementation and testing.

---

**This document serves as the reference for implementation. Review and approve before proceeding with code changes.**