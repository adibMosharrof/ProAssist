Dataset helpers for DST generator

This package provides lightweight dataset helpers (dataset-first API). Use
`DataSourceFactory.get_data_source(...)` to obtain a dataset instance such as
`ManualDSTDataset` or `ProAssistDSTDataset`, then wrap it with
`torch.utils.data.DataLoader` if you need batching.

Available dataset helpers:
- manual: `ManualDSTDataset` - reads JSON files from a directory (default)
- proassist: `ProAssistDSTDataset` - scans ProAssist-style directory layout

Example (Python):

from dst_data_builder.data_sources.data_source_factory import DataSourceFactory
from torch.utils.data import DataLoader

cfg = {"data_path": "data/proassist_dst_manual_data", "num_rows": None}
dataset = DataSourceFactory.get_data_source("manual", cfg)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x)

for batch in dataloader:
    # `batch` is a list of JSON-loaded dicts (identity collate to avoid tensor stacking)
    process_batch(batch)

You can override directory roots and other options via the `cfg` dict passed to
the factory (e.g., `proassist_dir`, `datasets`, `num_rows`).