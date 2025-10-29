Project context - ProAssist DST generator
=======================================

This document summarizes the current structure, goals, and how to run the DST generator in this repository.

Overview
--------
- Purpose: Generate structured DST (Dialog State Tracking) outputs from dataset items using LLMs (GPT-family). Supports both single-request processing and OpenAI-style batch API for efficient large-scale runs.
- Key features: config-driven (Hydra), modular generators (single / batch), PyTorch Dataset/DataLoader data sources, per-row observability, retry/re-batching logic for batch runs.

Important directories
---------------------
- `custom/src/dst_data_builder/` - main generator code and data modules
  - `gpt_generators/` - generator implementations (batch_gpt_generator.py, single_gpt_generator.py, base_gpt_generator.py)
  - `data_sources/` - dataset modules (manual_dst_dataset.py, proassist_dst_dataset.py, dst_data_module.py)
  - `tests/` - unit tests for generator behaviors

- `custom/config/` - Hydra configs (generator types and data sources)
- `custom/runner/run_dst_generator.sh` - recommended entrypoint that activates the project's venv and runs the Hydra module
- `custom/outputs/dst_generated/` - Hydra-managed run outputs (timestamped directories). Batch intermediate files are saved under `<run_dir>/batch/`.

How to run
----------
1. Ensure the project virtualenv exists (the runner expects `.venv` in repo root). If you don't have one, create it and install requirements:

```bash
# from repo root
python -m venv .venv
.venv/bin/pip install -r custom/src/dst_data_builder/simple_requirements.txt
```

2. Set your OpenAI API key in the environment (required for actual generation):

```bash
export OPENROUTER_API_KEY="sk-..."
```

3. Run the generator via the runner (recommended):

```bash
bash custom/runner/run_dst_generator.sh
```

What to expect
---------------
- Hydra creates a timestamped run directory under `custom/outputs/dst_generated/<date>/<time>_<data_source>/`.
- DST outputs are saved under `<run_dir>/dst_outputs/` as `dst_<input_basename>.json`.
- When using the batch generator, intermediate batch request and result files are saved under `<run_dir>/batch/` (e.g. `batch_dst_requests_<ts>_attempt0.jsonl`, `batch_dst_results_<ts>_attempt0.jsonl`).
- For failed rows, the batch generator writes diagnostics into the same `batch/` directory: `failed_error_<basename>.json` and `failed_raw_<basename>.txt`.

Implementation notes
--------------------
- DSTOutput: central dataclass to assemble/save DST outputs (see `datatypes/dst_output.py`).
- Generator factory (`gpt_generators/gpt_generator_factory.py`) reads `cfg.generator` and returns configured `BatchGPTGenerator` or `SingleGPTGenerator`.
- Batch behavior: creates JSONL, uploads via files.create(..., purpose='batch'), creates a batch job, polls until completion, downloads results, then parses each line and validates the DST JSON.
- Parser: robust against content wrapped in markdown code fences and variable batch envelope shapes.

Contact and next steps
-----------------------
- If you want additional observability, consider enabling `generator.save_intermediate: true` in your Hydra config or increasing logging level to DEBUG.
- For CI-friendly runs without a real API key, tests are written to mock the generator client. See `custom/src/dst_data_builder/tests/`.

This file was generated to provide a concise single-page reference for contributors and maintainers.
# ProAssist — Project Context

Purpose
- Concise, portable context to reproduce development tasks in this repository and to give to any LLM.
- Covers environment, common commands, important file locations, data-loading and generator conventions, testing, troubleshooting, and recommended LLM prompt snippets.

Repository root
- Path (repo root used throughout): /u/siddique-d1/adib/ProAssist

Python environment
- Project virtualenv (use this for running tests and scripts):
  - /u/siddique-d1/adib/ProAssist/.venv
- Use the venv's python for reliable results:
  - /u/siddique-d1/adib/ProAssist/.venv/bin/python

PYTHONPATH convention
- When running project modules directly (not installed as a package), set:
  - export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"

Dependencies
- See top-level `requirements.txt` and `custom/src/dst_data_builder/simple_requirements.txt` (generator-specific).
- Hydras and PyTorch are used (torch expected in venv).

Key file locations (most relevant)
- Generator (main orchestrator):
  - `custom/src/dst_data_builder/simple_dst_generator.py`
- Data sources / dataset modules:
  - `custom/src/dst_data_builder/data_sources/`
    - `base_dst_dataset.py` — BaseDSTDataset
    - `manual_dst_dataset.py` — ManualDSTDataset
    - `proassist_dst_dataset.py` — ProAssistDSTDataset
    - `data_source_factory.py` — returns dataset instances (dataset-first API)
    - `dst_data_module.py` — wraps dataset -> torch.DataLoader
- Generators:
  - `custom/src/dst_data_builder/gpt_generators/` (contains `base_gpt_generator.py`, `single_gpt_generator.py`, `batch_gpt_generator.py`, factory, etc.)
- Tests:
  - `custom/src/dst_data_builder/tests/test_dataloader.py`
- Configs & examples:
  - `custom/config/simple_dst_generator.yaml` — default generator config
  - `custom/config/data_source/proassist.yaml` and other data source configs
- Outputs (Hydra-managed):
  - `custom/outputs/dst_generated/<date>/<timestamp>_<data_source>/dst_outputs` — generated DST JSON files
  - The run log is in the Hydra run directory, e.g. `simple_dst_generator.log` (under the Hydra output directory)

How data loading now works (dataset-first API)
- DataSourceFactory returns lightweight dataset instances, not legacy "DataSource" objects.
  - Call: `DataSourceFactory.get_data_source(name, cfg)` yields a dataset instance:
    - `ManualDSTDataset(data_path, num_rows)`
    - `ProAssistDSTDataset(data_path, num_rows, datasets=[...])`
- Callers should wrap returned dataset in `torch.utils.data.DataLoader` for batching:
  - Use `collate_fn=lambda x: x` if you want to keep each batch as a list of dicts (avoids PyTorch trying to stack tensors).
- `num_rows` semantics:
  - `None` or `-1` => no truncation (all rows)
  - Positive integer => truncate to that many rows
- DSTDataModule builds dataloader by calling DataSourceFactory and wrapping result in a PyTorch DataLoader; it also attaches helper methods to the DataLoader:
  - `dataloader.get_dataset_size()`
  - `dataloader.get_file_paths()`

How to run the tests (recommended)
- From repo root, using the project venv:
  - export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  - /u/siddique-d1/adib/ProAssist/.venv/bin/python -m pytest -q custom/src/dst_data_builder/tests/test_dataloader.py

How to run the Simple DST generator (example)
- Standard run (uses Hydra config defaults):
  - export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  - /u/siddique-d1/adib/ProAssist/.venv/bin/python custom/src/dst_data_builder/simple_dst_generator.py
- Script uses Hydra — default settings come from `custom/config/...`. Example run in our session used `proassist` data source with `num_rows: 3`.
- Outputs (DST JSON files) are saved under the Hydra-managed directory printed by the run (example):
  - `custom/outputs/dst_generated/2025-10-22/00-26-25_proassist/dst_outputs`

Notes about running the generator
- It issues HTTP API calls to OpenAI (via `httpx` or OpenAI library) — ensure you have valid keys or mocks for offline/test runs.
- The generator supports:
  - `generator.type: single` — single-file processing with retry logic (default)
  - `generator.type: batch` — batch processing (uses batch API)
- Logs:
  - Uses standard Python logging (Hydra captures run logs)
  - Hydra emits an informational warning if defaults composition is missing `_self_` in the defaults list (this is a config composition cleanliness issue, not fatal).

Common commands (copyable)
- Run tests:
  - export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  - /u/siddique-d1/adib/ProAssist/.venv/bin/python -m pytest -q custom/src/dst_data_builder/tests/test_dataloader.py
- Run generator:
  - export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  - /u/siddique-d1/adib/ProAssist/.venv/bin/python custom/src/dst_data_builder/simple_dst_generator.py

Notes about tests and warnings seen
- The `test_dataloader.py` file earlier returned boolean values from tests, causing `PytestReturnNotNoneWarning`. Recommended fix: change tests to use `assert` (pytest expects functions to return None).
- Tests were run with the project's venv python to ensure correct dependency resolution (torch present, etc.).

OpenAI & testing without a key
- Tests have been refactored/mocked where needed to avoid requiring a real OpenAI key.
- For generator runs that call real OpenAI endpoints, use valid API keys in the environment, or
  - Mock the HTTP client or replace OpenAI calls with a fake response in tests.
  - Example mocking approach: monkeypatch or patch `httpx` or the OpenAI client in unit tests.

Hydra notes & config tips
- Hydra configs live under `custom/config/` (for the DST generator).
- Hydra-run directory is printed at start; use that to find generated results and logs.
- If Hydra warns about missing `_self_` in defaults lists, update the YAML defaults composition to include `_self_` as per Hydra docs.

File-level migration notes (what we changed in session)
- Migrated "data sources" to a dataset-first API:
  - Removed legacy classes and loader modules (we removed `base_data_source.py`, `manual_data_source.py`, `proassist_data_source.py`, and a legacy `dst_dataloader.py`).
  - Created `base_dst_dataset.py`, `manual_dst_dataset.py`, `proassist_dst_dataset.py`.
  - Updated `data_source_factory.py` to return dataset instances.
  - Updated `dst_data_module.py` to wrap those datasets in `torch.utils.data.DataLoader`.
- Re-export wrapper `dst_datasets.py` originally existed to provide compatibility; it was removed after callers were updated to import concrete modules.

Useful patterns / conventions to follow
- Use the venv python and set PYTHONPATH to `custom/src` for all direct script runs.
- Prefer the dataset-first API: DataSourceFactory returns a dataset instance to be wrapped by DataLoader; this makes tests and batching simpler.
- Use a lightweight collate function that returns the raw list (e.g., `collate_fn=lambda x: x`) when batches contain dicts with varying keys.

Troubleshooting checklist
- Import errors after moving files?
  - Check `PYTHONPATH` is set to `custom/src` or run the script via the venv where the package is installed.
- Missing `torch` during tests?
  - Activate the project venv or run the tests with the venv python (explicit path ensures correct interpreter).
- Hydra missing default composition `_self_`?
  - Add `_self_` to defaults or follow Hydra docs for default composition order.
- OpenAI failures:
  - Check API key in environment or switch to mocks for offline runs.

Suggested LLM prompt / context snippet (copy & paste to any LLM)
- Short context to provide to an LLM before asking it to modify code or run tasks:

  ```
  Repo root: /u/siddique-d1/adib/ProAssist
  Use venv python: /u/siddique-d1/adib/ProAssist/.venv/bin/python
  PYTHONPATH for direct runs: export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"

  Key modules:
  - Simple generator: custom/src/dst_data_builder/simple_dst_generator.py
  - Data source factory (dataset-first): custom/src/dst_data_builder/data_sources/data_source_factory.py
  - Datasets:
    - base_dst_dataset.py
    - manual_dst_dataset.py
    - proassist_dst_dataset.py
    (all under custom/src/dst_data_builder/data_sources/)
  - DSTDataModule: custom/src/dst_data_builder/data_sources/dst_data_module.py
  - Tests: custom/src/dst_data_builder/tests/test_dataloader.py

  How to run tests:
  export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  /u/siddique-d1/adib/ProAssist/.venv/bin/python -m pytest -q custom/src/dst_data_builder/tests/test_dataloader.py

  How to run generator:
  export PYTHONPATH="/u/siddique-d1/adib/ProAssist/custom/src:$PYTHONPATH"
  /u/siddique-d1/adib/ProAssist/.venv/bin/python custom/src/dst_data_builder/simple_dst_generator.py

  Note: DataSourceFactory returns torch-style Dataset instances (wrap with DataLoader). num_rows: None/-1 => all rows; number => truncation.
  ```

Optional additions I can make (pick any)
- Produce a one-file `CONTRIBUTING.md` with the above commands and conventions.
- Add a short shell helper script that automatically exports PYTHONPATH and uses the venv python to run common commands (tests, generator).
- Update tests to remove pytest warnings (convert returns to asserts).
- Add a short README in `custom/src/dst_data_builder/data_sources/` describing the dataset-first API with examples.

If you want any of the optional additions, tell me which one and I will create it now.