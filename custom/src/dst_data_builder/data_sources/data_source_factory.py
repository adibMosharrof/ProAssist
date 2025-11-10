"""Factory for creating dataset helpers for DST generation.

Historically this returned DataSource objects. After the migration to
self-contained datasets we return dataset instances (which are lightweight
and can be wrapped with torch.utils.data.DataLoader by callers).
"""

from typing import Dict, Any, Optional

from dst_data_builder.data_sources.manual_dst_dataset import ManualDSTDataset
from dst_data_builder.data_sources.proassist_dst_dataset import (
    ProAssistDSTDataset,
)


class DataSourceFactory:
    """Factory for creating dataset-like helpers by name.

    Returns:
        ManualDSTDataset or ProAssistDSTDataset depending on `name`.
    """

    @staticmethod
    def get_data_source(name: str, cfg: Optional[Dict[str, Any]] = None):
        """Get a dataset helper by name.

        Args:
            name: 'manual' or 'proassist'
            cfg: optional configuration dict that may include 'data_path',
                 'proassist_dir', 'num_rows', and 'datasets'.
        """
        name = (name or "manual").lower()

        cfg = cfg or {}

        if name == "manual":
            data_path = cfg.get("data_path", "data/proassist_dst_manual_data")
            num_rows = cfg.get("num_rows", None)
            return ManualDSTDataset(data_path=data_path, num_rows=num_rows)

        if name in ("proassist", "proassist_raw"):
            data_path = cfg.get("data_path")
            num_rows = cfg.get("num_rows", None)
            datasets = cfg.get("datasets", [])
            suffix = cfg.get("suffix", "")
            return ProAssistDSTDataset(
                data_path=data_path, num_rows=num_rows, datasets=datasets, suffix=suffix
            )

        raise ValueError(f"Unknown data source/dataset name: {name}")
