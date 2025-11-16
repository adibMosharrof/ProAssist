"""DST Training DataModule for multimodal DST training with video frames and context compression.

This datamodule handles loading DST data with dynamic video frame retrieval,
supporting multimodal training with conversation + video frames + DST data.
Implements DST-aware context compression during training for memory management.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import torch
from torch.utils.data import DataLoader as TorchDataLoader

from prospect.data_sources.dst_training_dataset import DSTTrainingDataset
from prospect.data_sources.dst_data_collator import DSTDataCollator

logger = logging.getLogger(__name__)


class DSTTrainingDataModule:
    """DataModule for multimodal DST training with video frames and context compression."""

    def __init__(
        self,
        dst_data_path: str,
        raw_data_path: str,
        datasets: List[str],
        split: str = "train",
        fps: int = 2,
        max_seq_len: int = 4096,
        reserved_seq_len: int = 128,
        num_frames_per_conversation: int = 5,
        tokenizer=None,
    ):
        """
        Initialize DST Training DataModule with stateless multimodal training.

        Args:
            dst_data_path: Path to DST generator output directory
            raw_data_path: Base path for raw data (contains frames directories)
            datasets: List of dataset names
            split: Data split
            fps: Frame rate for video data
            max_seq_len: Maximum sequence length for input
            reserved_seq_len: Reserved sequence length for special tokens
            num_frames_per_conversation: Number of frames to sample per conversation
            tokenizer: Tokenizer for real tokenization (None for fake tokenization)
        """
        self.dst_data_path = dst_data_path
        self.raw_data_path = raw_data_path
        self.datasets = datasets
        self.split = split
        self.fps = fps
        self.max_seq_len = max_seq_len
        self.reserved_seq_len = reserved_seq_len
        self.num_frames_per_conversation = num_frames_per_conversation
        self.tokenizer = tokenizer

        # Initialize data collator with stateless multimodal support (no KV cache)
        self.collator = DSTDataCollator(
            max_seq_len=max_seq_len,
            normalize_frames=True,
            tokenizer=tokenizer,
        )

        logger.info(f"Initialized DSTTrainingDataModule with {len(datasets)} datasets")
        logger.info("Training mode: STATELESS (no KV cache, each batch independent)")

    def setup(self, stage: Optional[str] = None):
        """Setup datamodule - create train and val datasets."""
        logger.info("Setting up DST training datasets...")

        # Create train dataset
        self.train_dataset = DSTTrainingDataset(
            dst_data_path=self.dst_data_path,
            raw_data_path=self.raw_data_path,
            datasets=self.datasets,
            split="train",
            fps=self.fps,
            max_seq_len=self.max_seq_len,
            num_frames_per_conversation=self.num_frames_per_conversation,
        )

        # Create validation dataset
        self.val_dataset = DSTTrainingDataset(
            dst_data_path=self.dst_data_path,
            raw_data_path=self.raw_data_path,
            datasets=self.datasets,
            split="val",
            fps=self.fps,
            max_seq_len=self.max_seq_len,
            num_frames_per_conversation=self.num_frames_per_conversation,
        )

        logger.info(
            f"Datasets loaded: train={len(self.train_dataset)}, val={len(self.val_dataset)}"
        )

    def get_train_dataset(self) -> DSTTrainingDataset:
        """Get training dataset."""
        if not hasattr(self, "train_dataset"):
            self.setup()
        return self.train_dataset

    def get_val_dataset(self) -> DSTTrainingDataset:
        """Get validation dataset."""
        if not hasattr(self, "val_dataset"):
            self.setup()
        return self.val_dataset

    def get_data_collator(self):
        """Get data collator for HF Trainer."""
        return self.collator

    def get_dataset_size(self) -> int:
        """Get total dataset size."""
        if not hasattr(self, "train_dataset"):
            self.setup()
        return len(self.train_dataset)

    def __len__(self) -> int:
        """Return total number of batches (for compatibility)."""
        import math

        if not hasattr(self, "train_dataset"):
            self.setup()
        return math.ceil(len(self.train_dataset) / 8)  # Assume batch_size=8

    def __repr__(self) -> str:
        return (
            f"DSTTrainingDataModule("
            f"datasets={self.datasets}, "
            f"split={self.split}, "
            f"frame_strategy=conversation_timestamps, "
            f"frames_per_conversation={self.num_frames_per_conversation}, "
            f"mode=stateless_training"
            f")"
        )
