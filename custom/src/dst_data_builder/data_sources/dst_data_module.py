"""DataModule for DST generation that integrates data sources and dataloaders"""

from typing import Dict, Any, Optional, Union, List
from pathlib import Path
import json
import logging

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
        
        # Split datasets (for proassist)
        self._split_datasets = {}
        self._split_dataloaders = {}
        self._logger = logging.getLogger(self.__class__.__name__)

    def setup(self, stage: str = None):
        """Setup the data module by loading data and creating splits.
        
        Args:
            stage: Optional stage hint ('fit', 'test', or None)
        """
        if self.data_source_name == "proassist":
            self._setup_proassist_splits()
        else:
            # For manual data, use original approach
            self._setup_manual_dataloader()

    def _setup_proassist_splits(self):
        """Load ProAssist data and create split datasets"""
        # Convert config to plain dict
        if isinstance(self.data_source_cfg, dict):
            cfg = dict(self.data_source_cfg)
        else:
            try:
                cfg = OmegaConf.to_container(self.data_source_cfg, resolve=True)
                if not isinstance(cfg, dict):
                    cfg = dict(cfg)
            except Exception:
                cfg = {}

        data_path = cfg.get("data_path") or (
            str(self.data_path) if self.data_path else None
        )
        if not data_path:
            raise ValueError("data_path must be provided for proassist data source")

        datasets = cfg.get("datasets", [])
        suffix = cfg.get("suffix", "")
        num_rows = cfg.get("num_rows")

        if not datasets:
            raise ValueError("ProAssistDataModule requires 'datasets' list in config")

        self._logger.info(f"Setting up ProAssist splits with suffix: {suffix}")
        self._logger.info(f"Datasets: {datasets}")
        if num_rows is not None and num_rows != -1:
            self._logger.info(f"Limiting to {num_rows} videos per split")

        # Load data for each split (train, val, test)
        for split in ["train", "val", "test"]:
            all_videos = []
            
            for dataset_name in datasets:
                json_file = Path(data_path) / dataset_name / "generated_dialogs" / f"{split}{suffix}.json"
                
                if json_file.exists():
                    self._logger.info(f"Loading {json_file}")
                    with open(json_file, 'r') as f:
                        video_list = json.load(f)
                    
                    # Add metadata to each video
                    for video in video_list:
                        video['_dataset_name'] = dataset_name
                        video['_split'] = split
                        video['_file_path'] = str(json_file)
                    
                    all_videos.extend(video_list)
                else:
                    self._logger.warning(f"File not found: {json_file}")

            if all_videos:
                # Apply num_rows truncation per split (for faster testing)
                if num_rows is not None and num_rows != -1:
                    all_videos = all_videos[:int(num_rows)]
                    self._logger.info(f"Limited {split} to {len(all_videos)} videos")
                
                # Create dataset (no need to pass num_rows again since we already truncated)
                dataset = ProAssistDSTDataset(data=all_videos, num_rows=None)
                self._split_datasets[split] = dataset
                
                # Save dataset to file for reuse
                self._save_dataset_to_file(dataset, split, data_path, datasets[0] if datasets else "proassist")
                
                # Create dataloader
                dataloader = TorchDataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    drop_last=self.drop_last,
                    pin_memory=self.pin_memory,
                )
                self._split_dataloaders[split] = dataloader
                
                self._logger.info(f"Created {split} dataset with {len(dataset)} videos")
            else:
                self._logger.info(f"No data found for split: {split}")

    def _save_dataset_to_file(self, dataset: BaseDSTDataset, split: str, data_path: str, dataset_name: str):
        """Save processed dataset to file for later reuse"""
        try:
            # Create save directory
            save_dir = Path(data_path) / "processed_datasets"
            save_dir.mkdir(exist_ok=True)
            
            # Get dataset data
            data = dataset.get_data()
            
            # Create metadata
            metadata = {
                "split": split,
                "dataset_name": dataset_name,
                "video_count": len(data),
                "timestamp": str(Path().resolve()),
                "source_files": list(set(item.get('_file_path', '') for item in data if item.get('_file_path'))),
                "datasets": list(set(item.get('_dataset_name', '') for item in data if item.get('_dataset_name')))
            }
            
            # Save data and metadata
            save_file = save_dir / f"proassist_{split}_{dataset_name}_dataset.json"
            save_metadata_file = save_dir / f"proassist_{split}_{dataset_name}_metadata.json"
            
            with open(save_file, 'w') as f:
                json.dump(data, f, indent=2)
            
            with open(save_metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self._logger.info(f"💾 Saved dataset: {save_file} ({len(data)} videos)")
            self._logger.info(f"💾 Saved metadata: {save_metadata_file}")
            
        except Exception as e:
            self._logger.error(f"Failed to save dataset: {e}")

    def _setup_manual_dataloader(self):
        """Setup for manual data source (original behavior)"""
        # Convert OmegaConf DictConfig to plain dict if necessary
        if self.data_source_cfg is None:
            ds_cfg = {}
        elif isinstance(self.data_source_cfg, dict):
            ds_cfg = dict(self.data_source_cfg)
        else:
            try:
                ds_cfg = OmegaConf.to_container(self.data_source_cfg, resolve=True)
                if not isinstance(ds_cfg, dict):
                    ds_cfg = dict(ds_cfg)
            except Exception:
                ds_cfg = {}

        if self.data_path:
            ds_cfg.setdefault("data_path", str(self.data_path))

        num_rows = ds_cfg.get("num_rows")

        # Build a torch Dataset from the data source
        data_path = ds_cfg.get("data_path") or (
            str(self.data_path) if self.data_path else None
        )
        if data_path is None:
            raise ValueError(
                "data_path must be provided in data_source_cfg or via DSTDataModule(data_path)"
            )

        # DataSourceFactory returns dataset instances now; use them directly
        dataset = DataSourceFactory.get_data_source(self.data_source_name, ds_cfg)

        # Ensure dataset respects requested num_rows if provided
        if hasattr(dataset, "_num_rows") and num_rows is not None:
            dataset._num_rows = None if (num_rows == -1) else int(num_rows)

        # Wrap the dataset with PyTorch DataLoader
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

    @property 
    def train_dataset(self) -> Optional[BaseDSTDataset]:
        """Get training dataset"""
        return self._split_datasets.get("train")

    @property
    def val_dataset(self) -> Optional[BaseDSTDataset]:
        """Get validation dataset"""
        return self._split_datasets.get("val")

    @property
    def test_dataset(self) -> Optional[BaseDSTDataset]:
        """Get test dataset"""
        return self._split_datasets.get("test")

    @property
    def train_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Get training dataloader"""
        return self._split_dataloaders.get("train")

    @property
    def val_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Get validation dataloader"""
        return self._split_dataloaders.get("val")

    @property
    def test_dataloader(self) -> Optional[torch.utils.data.DataLoader]:
        """Get test dataloader"""
        return self._split_dataloaders.get("test")

    def get_available_splits(self) -> List[str]:
        """Get list of available splits"""
        return [split for split in ["train", "val", "test"] if split in self._split_datasets]

    def get_file_paths(self) -> List[Path]:
        """Get list of all file paths (for backward compatibility)"""
        if self._split_datasets:
            # For split-based approach, extract file paths from train split data
            train_dataset = self._split_datasets.get("train")
            if train_dataset and hasattr(train_dataset, 'get_data'):
                train_data = train_dataset.get_data()
                file_paths = []
                for item in train_data:
                    if isinstance(item, dict) and '_file_path' in item:
                        file_paths.append(Path(item['_file_path']))
                return file_paths
            return []
        else:
            return self.dataloader.get_file_paths()

    def get_dataset_size(self) -> int:
        """Get total number of samples in the dataset (for backward compatibility)"""
        if self._split_datasets:
            # For split-based approach, return size from train split
            return len(self._split_datasets.get("train", BaseDSTDataset()))
        else:
            return self.dataloader.get_dataset_size()

    @property
    def dataloader(self):
        """Get or create the dataloader (for backward compatibility)"""
        if self._split_datasets:
            # For split-based approach, return train dataloader by default
            return self._split_dataloaders.get("train")
        
        # Original behavior for manual data
        if self._dataloader is not None:
            return self._dataloader

        self._setup_manual_dataloader()
        return self._dataloader

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
            f"batch_size={self.batch_size}, "
            f"splits={list(self._split_datasets.keys())})"
        )
