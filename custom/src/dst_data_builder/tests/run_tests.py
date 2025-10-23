#!/usr/bin/env python3
"""
Test runner for Simple DST Generator
"""
import subprocess
import sys
from pathlib import Path

def run_tests():
    """Run all tests"""
    print("🧪 Running Simple DST Generator Tests...")

    # Test 1: Check if manual data files exist
    manual_data_dir = Path("custom/src/dst_data_builder/manual_data")
    if not manual_data_dir.exists():
        print(f"❌ Manual data directory not found: {manual_data_dir}")
        return False

    json_files = list(manual_data_dir.glob("*.json"))
    print(f"✅ Found {len(json_files)} test files in manual_data/")

    for file in json_files:
        print(f"  - {file.name}")

    # Test 2: Run the specific test function
    try:
        from test_simple_dst_generator import run_specific_tests
        run_specific_tests()
    except ImportError as e:
        print(f"⚠️ Could not import test module: {e}")
        print("This is expected if pytest is not installed")

    # Test 3: Check configuration files
    config_file = Path("custom/config/simple_dst_generator.yaml")
    if config_file.exists():
        print(f"✅ Hydra config file exists: {config_file}")
    else:
        print(f"❌ Hydra config file not found: {config_file}")

    print("\n🎯 To run with pytest (if installed):")
    print("   pip install pytest")
    print("   pytest custom/src/dst_data_builder/test_simple_dst_generator.py -v")

    print("\n🚀 To run the main script with Hydra:")
    print("   python custom/src/dst_data_builder/simple_dst_generator.py")


if __name__ == "__main__":
    run_tests()
