#!/usr/bin/env python3
"""Test script for the new DataLoader implementation"""

import json
import pytest
from dst_data_builder.data_sources.data_source_factory import DataSourceFactory
from dst_data_builder.data_sources.dst_data_module import DSTDataModule
from torch.utils.data import DataLoader as TorchDataLoader


def test_manual_dataloader(populated_test_dir):
    """Test ManualDataLoader functionality with test data"""
    dataset = DataSourceFactory.get_data_source("manual", {"data_path": str(populated_test_dir)})
    dataloader = TorchDataLoader(dataset, batch_size=2, shuffle=False, collate_fn=lambda x: x)
    
    # Verify dataset properties
    assert dataset.get_dataset_size() == 3
    assert len(dataloader) == 2  # 3 items, batch_size=2 = 2 batches
    
    # Test first batch
    batch = next(iter(dataloader))
    assert len(batch) == 2  # batch_size=2
    assert isinstance(batch[0], dict)
    assert "video_uid" in batch[0]
    
    # Test file paths
    file_paths = dataset.get_file_paths()
    assert len(file_paths) == 3


def test_proassist_dataloader(tmp_path, sample_input_data):
    """Test ProAssistDataLoader functionality"""
    # Create test directory structure
    proassist_dir = tmp_path / "proassist_test"
    processed_dir = proassist_dir / "processed_data"
    processed_dir.mkdir(parents=True)
    
    # Create sample file
    test_file = processed_dir / "test_video.json"
    test_file.write_text(json.dumps(sample_input_data))
    
    # Create dataset and dataloader
    dataset = DataSourceFactory.get_data_source("proassist", {"data_path": str(proassist_dir)})
    dataloader = TorchDataLoader(dataset, batch_size=1, shuffle=False, collate_fn=lambda x: x)
    
    # Verify properties
    assert dataset.get_dataset_size() == 1
    batch = next(iter(dataloader))
    assert len(batch) == 1
    assert isinstance(batch[0], dict)
    assert len(dataset.get_file_paths()) == 1


def test_data_source_factory(tmp_path, sample_input_data):
    """Test DataSourceFactory functionality"""
    # Create manual dataset
    manual_dir = tmp_path / "manual"
    manual_dir.mkdir()
    (manual_dir / "test.json").write_text(json.dumps(sample_input_data))
    
    manual_ds = DataSourceFactory.get_data_source("manual", {"data_path": str(manual_dir)})
    assert manual_ds.get_dataset_size() == 1
    
    # Create proassist dataset
    proassist_dir = tmp_path / "proassist" / "processed_data"
    proassist_dir.mkdir(parents=True)
    (proassist_dir / "test.json").write_text(json.dumps(sample_input_data))
    
    proassist_ds = DataSourceFactory.get_data_source(
        "proassist", {"data_path": str(tmp_path / "proassist")}
    )
    assert proassist_ds.get_dataset_size() == 1
    
    # Verify dataloaders work
    manual_loader = TorchDataLoader(manual_ds, batch_size=1)
    proassist_loader = TorchDataLoader(proassist_ds, batch_size=1)
    assert manual_loader is not None
    assert proassist_loader is not None


def test_dst_data_module(populated_test_dir):
    """Test DSTDataModule functionality"""
    data_module = DSTDataModule(
        data_source_name="manual",
        data_path=str(populated_test_dir),
        batch_size=2,
        shuffle=False,
    )
    
    # Test properties
    assert data_module.get_dataset_size() == 3
    assert data_module.batch_size == 2
    assert len(data_module) == 2  # 3 items / batch_size 2 = 2 batches
    
    # Test dataset access
    dataset = data_module.data_source
    assert hasattr(dataset, 'get_file_paths')
    assert hasattr(dataset, '__getitem__')
    
    # Test iteration - DSTDataModule uses default torch collate (dict of lists)
    batch = next(iter(data_module))
    assert isinstance(batch, dict)
    assert "video_uid" in batch
    # Batch size 2, so should have 2 items in each field
    assert len(batch["video_uid"]) == 2


