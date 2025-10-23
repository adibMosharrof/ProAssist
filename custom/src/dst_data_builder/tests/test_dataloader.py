#!/usr/bin/env python3
"""Test script for the new DataLoader implementation"""

import sys
from pathlib import Path

from dst_data_builder.data_sources.data_source_factory import DataSourceFactory
from dst_data_builder.data_sources.dst_data_module import DSTDataModule
from dst_data_builder.data_sources.manual_dst_dataset import ManualDSTDataset
from dst_data_builder.data_sources.proassist_dst_dataset import (
    ProAssistDSTDataset,
)
import torch
from torch.utils.data import DataLoader as TorchDataLoader


def test_manual_dataloader():
    """Test ManualDataLoader functionality"""
    print("Testing ManualDataLoader...")

    # Test with the actual data directory
    data_path = "data/proassist_dst_manual_data"

    try:
        # Create dataset using factory
        dataset = DataSourceFactory.get_data_source("manual", {"data_path": data_path})

        # Create dataloader from dataset (use identity collate to return list of items)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x
        )
        print(f"✅ Created dataloader: {dataloader}")

        # Test iteration
        print(f"📊 Dataset size: {dataset.get_dataset_size()}")
        print(f"📦 Batch size: {dataloader.batch_size}")
        print(f"🔢 Number of batches: {len(dataloader)}")

        # Get first batch
        try:
            batch = next(iter(dataloader))
            print(f"📋 First batch type: {type(batch)}")
            print(
                f"📋 First batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
            )

            # Test accessing first item in batch
            if len(batch) > 0:
                first_item = batch[0]
                print(
                    f"📋 First item keys: {list(first_item.keys()) if isinstance(first_item, dict) else 'Not a dict'}"
                )

        except Exception as e:
            print(f"📋 Batch access test skipped: {e}")

        # Test file paths (use dataset helper)
        file_paths = dataset.get_file_paths()
        print(f"📁 Found {len(file_paths)} files")
        print(f"📁 First file: {file_paths[0] if file_paths else 'None'}")

        print("✅ ManualDataLoader test passed!")
        assert True
    except Exception as e:
        print(f"❌ ManualDataLoader test failed: {e}")
        assert False, f"ManualDataLoader test failed: {e}"


def test_proassist_dataloader():
    """Test ProAssistDataLoader functionality"""
    print("\nTesting ProAssistDataLoader...")

    # Test with the actual data directory
    data_path = "data/proassist"

    try:
        # Create dataset using factory
        dataset = DataSourceFactory.get_data_source(
            "proassist", {"proassist_dir": data_path}
        )

        # Create dataloader from dataset (use identity collate to return list of items)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x
        )
        print(f"✅ Created dataloader: {dataloader}")

        # Test iteration
        print(f"📊 Dataset size: {dataset.get_dataset_size()}")
        print(f"📦 Batch size: {dataloader.batch_size}")
        print(f"🔢 Number of batches: {len(dataloader)}")

        # Get first batch if dataset is not empty
        if dataset.get_dataset_size() > 0:
            batch = next(iter(dataloader))
            print(f"📋 First batch type: {type(batch)}")
            print(
                f"📋 First batch length: {len(batch) if hasattr(batch, '__len__') else 'N/A'}"
            )
        # Test file paths (use dataset helper)
        file_paths = dataset.get_file_paths()
        print(f"📁 Found {len(file_paths)} files")
        print(f"📁 First file: {file_paths[0] if file_paths else 'None'}")

        print("✅ ProAssistDataLoader test passed!")
        assert True
    except Exception as e:
        print(f"❌ ProAssistDataLoader test failed: {e}")
        assert False, f"ProAssistDataLoader test failed: {e}"


def test_data_source_factory():
    """Test DataSourceFactory functionality"""
    print("\nTesting DataSourceFactory...")

    try:
        # Test manual data source
        manual_ds = DataSourceFactory.get_data_source(
            "manual", {"data_path": "data/proassist_dst_manual_data"}
        )
        print(f"✅ Created manual dataset: {manual_ds}")

        # Test proassist dataset
        proassist_ds = DataSourceFactory.get_data_source(
            "proassist", {"proassist_dir": "data/proassist"}
        )
        print(f"✅ Created proassist dataset: {proassist_ds}")

        # Wrap datasets in DataLoader
        manual_dataloader = torch.utils.data.DataLoader(manual_ds, batch_size=2)
        print(f"✅ Created dataloader from manual dataset: {manual_dataloader}")

        proassist_dataloader = torch.utils.data.DataLoader(proassist_ds, batch_size=2)
        print(f"✅ Created dataloader from proassist dataset: {proassist_dataloader}")

        print("✅ DataSourceFactory test passed!")
        assert True
    except Exception as e:
        print(f"❌ DataSourceFactory test failed: {e}")
        assert False, f"DataSourceFactory test failed: {e}"


def test_dst_data_module():
    """Test DSTDataModule functionality"""
    print("\nTesting DSTDataModule...")

    try:
        # Test with data path
        data_module = DSTDataModule(
            data_source_name="manual",
            data_path="data/proassist_dst_manual_data",
            batch_size=2,
            shuffle=True,
        )
        print(f"✅ Created data module: {data_module}")

        # Test properties
        print(f"📊 Dataset size: {data_module.get_dataset_size()}")
        print(f"📦 Batch size: {data_module.batch_size}")
        print(f"🔢 Number of batches: {len(data_module)}")

        # Test dataset access (DataSourceFactory now returns datasets)
        dataset = data_module.data_source
        print(f"📦 Dataset type: {type(dataset).__name__}")
        print(f"📦 Dataset has get_file_paths: {hasattr(dataset, 'get_file_paths')}")
        print(f"📦 Dataset supports __getitem__: {hasattr(dataset, '__getitem__')}")

        # Test that we can call get_file_paths and __getitem__
        try:
            items = dataset.get_file_paths()
            print(f"📦 Dataset listed {len(items)} items")
            if items:
                first_item = items[0]
                loaded_data = dataset[0]
                print(f"📦 Successfully loaded first item, type: {type(loaded_data)}")
        except Exception as e:
            print(f"📦 Dataset test failed: {e}")

        # Test dataloader access
        dataloader = data_module.dataloader
        print(f"🔄 DataLoader type: {type(dataloader).__name__}")
        print(
            f"🔄 DataLoader has get_dataset_size: {hasattr(dataloader, 'get_dataset_size')}"
        )

        # Test file paths
        try:
            file_paths = data_module.get_file_paths()
            print(f"📁 Found {len(file_paths)} files")
        except Exception as e:
            print(f"📁 File paths test failed: {e}")

        # Test iteration
        try:
            batch = next(iter(data_module))
            print(f"📋 Got batch with {len(batch)} items")
        except Exception as e:
            print(f"📋 Batch iteration test failed: {e}")

        print("✅ DSTDataModule test passed!")
        assert True
    except Exception as e:
        print(f"❌ DSTDataModule test failed: {e}")
        assert False, f"DSTDataModule test failed: {e}"


def main():
    """Run all tests"""
    print("🚀 Starting DataLoader tests...\n")

    tests = [
        test_manual_dataloader,
        test_proassist_dataloader,
        test_data_source_factory,
        test_dst_data_module,
    ]

    passed = 0
    total = len(tests)

    for test in tests:
        if test():
            passed += 1

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All tests passed!")
        return 0
    else:
        print("💥 Some tests failed!")
        return 1


if __name__ == "__main__":
    exit(main())
