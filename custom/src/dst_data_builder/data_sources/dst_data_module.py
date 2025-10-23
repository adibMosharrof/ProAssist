"""DataModule for DST generation that integrates data sources and dataloaders"""

from typing import Dict, Any, Optional, Union
from pathlib import Path

from dst_data_builder.data_sources.data_source_factory import DataSourceFactory
from dst_data_builder.data_sources.manual_dst_dataset import ManualDSTDataset
from dst_data_builder.data_sources.proassist_dst_dataset import (
    ProAssistDSTDataset,
)
from dst_data_builder.data_sources.base_dst_dataset import BaseDSTDataset
from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader as TorchDataLoader


class DSTDataModule:
    """DataModule for DST generation that handles different data sources"""

    def __init__(
        self,
        data_source_name: str = "manual",
        data_path: Optional[Union[str, Path]] = None,
        data_source_cfg: Optional[Dict[str, Any]] = None,
        batch_size: int = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        drop_last: bool = False,
        pin_memory: bool = False,
    ):
        """
        Initialize DST DataModule

        Args:
            data_source_name: Name of the data source ('manual' or 'proassist')
            data_path: Path to the data directory (used when creating dataloader directly)
            data_source_cfg: Configuration for the data source
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle the data
            num_workers: Number of worker processes for data loading
            drop_last: Whether to drop the last incomplete batch
            pin_memory: Whether to pin memory for GPU transfer
        """
        self.data_source_name = data_source_name
        self.data_path = Path(data_path) if data_path else None
        self.data_source_cfg = data_source_cfg or {}
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.drop_last = drop_last
        self.pin_memory = pin_memory

        # Will be created when needed
        self._data_source = None
        self._dataloader = None

    @property
    def data_source(self):
        """Get or create the data source"""
        if self._data_source is None:
            # DataSourceFactory now returns dataset helpers (ManualDSTDataset / ProAssistDSTDataset)
            self._data_source = DataSourceFactory.get_data_source(
                self.data_source_name, self.data_source_cfg
            )
        return self._data_source

    @property
    def dataloader(self):
        """Get or create the dataloader"""
        # Early return when dataloader already exists (reduces nesting)
        if self._dataloader is not None:
            return self._dataloader

        # Merge data_path into a copy of data_source_cfg so the underlying
        # data source constructors receive the expected key names. All data
        # sources are expected to accept a data_path in their cfg and
        # provide a `get_dataloader` method.
        # Convert OmegaConf DictConfig to plain dict if necessary so data source
        # constructors receive expected keys like 'data_path' and 'num_rows'.
        if self.data_source_cfg is None:
            ds_cfg = {}
        elif isinstance(self.data_source_cfg, dict):
            ds_cfg = dict(self.data_source_cfg)
        else:
            # Attempt to convert OmegaConf/Dataclass-like configs to dict
            try:
                ds_cfg = OmegaConf.to_container(self.data_source_cfg, resolve=True)
                if not isinstance(ds_cfg, dict):
                    ds_cfg = dict(ds_cfg)
            except Exception:
                # Fallback to empty dict to avoid crashing here; data source
                # constructors will validate required keys.
                ds_cfg = {}

        if self.data_path:
            ds_cfg.setdefault("data_path", str(self.data_path))

        num_rows = ds_cfg.get("num_rows")

        # Build a torch Dataset from the data source (preferential path)
        data_path = ds_cfg.get("data_path") or (
            str(self.data_path) if self.data_path else None
        )
        if data_path is None:
            raise ValueError(
                "data_path must be provided in data_source_cfg or via DSTDataModule(data_path)"
            )

        # DataSourceFactory returns dataset instances now; use them directly
        dataset = DataSourceFactory.get_data_source(self.data_source_name, ds_cfg)

        # Ensure dataset respects requested num_rows if provided (factory already passed it,
        # but honor explicit override)
        if hasattr(dataset, "_num_rows") and num_rows is not None:
            dataset._num_rows = None if (num_rows == -1) else int(num_rows)

        # Wrap the dataset with PyTorch DataLoader (torch is expected to be installed)
        self._dataloader = TorchDataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory,
        )
        # Attach helper methods used elsewhere
        self._dataloader.get_dataset_size = dataset.get_dataset_size
        self._dataloader.get_file_paths = dataset.get_file_paths

        return self._dataloader

    def get_dataset_size(self) -> int:
        """Get total number of samples in the dataset"""
        return self.dataloader.get_dataset_size()

    def get_file_paths(self) -> list:
        """Get list of all file paths"""
        return self.dataloader.get_file_paths()

    def __iter__(self):
        """Return iterator for the dataloader"""
        return iter(self.dataloader)

    def __len__(self) -> int:
        """Return number of batches"""
        return len(self.dataloader)

    def reset(self):
        """Reset the dataloader iterator"""
        self.dataloader.reset()

    def __repr__(self) -> str:
        return (
            f"DSTDataModule(data_source={self.data_source_name}, "
            f"dataset_size={self.get_dataset_size()}, "
            f"batch_size={self.batch_size})"
        )
