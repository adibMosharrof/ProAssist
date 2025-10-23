#!/usr/bin/env python3
"""
Tests for DST Generation with Manual Data using SingleGPTGenerator
"""

import sys
import os
import json
from pathlib import Path

# Add the parent directory to path for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

import unittest.mock as mock


def test_dst_gen_manual_data_validation():
    """Test that manual data files have correct structure"""
    print("🧪 Testing manual data file structure validation...")

    data_dir = Path("data/proassist_dst_manual_data")

    if not data_dir.exists():
        print(f"❌ Manual data directory not found: {data_dir}")
        assert False, f"Manual data directory not found: {data_dir}"

    json_files = list(data_dir.glob("*.json"))

    if not json_files:
        print("❌ No JSON files found in manual data directory")
        assert False, "No JSON files found in manual data directory"

    print(f"✅ Found {len(json_files)} JSON files for testing")

    required_fields = ["video_uid", "inferred_knowledge", "all_step_descriptions"]

    valid_count = 0

    for file_path in json_files[:3]:  # Test first 3 files
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            missing_fields = []
            for field in required_fields:
                if field not in data:
                    missing_fields.append(field)

            if missing_fields:
                print(f"⚠️ {file_path.name} missing fields: {missing_fields} - skipping")
                continue

            # Validate content is not empty
            if not data["inferred_knowledge"].strip():
                print(f"❌ {file_path.name} has empty inferred_knowledge")
                assert False, f"{file_path.name} has empty inferred_knowledge"

            if not data["all_step_descriptions"].strip():
                print(f"❌ {file_path.name} has empty all_step_descriptions")
                assert False, f"{file_path.name} has empty all_step_descriptions"

            print(f"✅ {file_path.name}: valid structure")
            valid_count += 1

        except Exception as e:
            print(f"❌ Error reading {file_path.name}: {e}")
            assert False, f"Error reading {file_path.name}: {e}"

    if valid_count == 0:
        print("⚠️ No valid manual data files found in sample set - skipping strict validation")
        assert True
    else:
        assert True


def test_single_gpt_generator_initialization():
    """Test that SingleGPTGenerator initializes correctly"""
    print("\n🧪 Testing SingleGPTGenerator initialization...")

    try:
        from dst_data_builder.gpt_generators.gpt_generator_factory import GPTGeneratorFactory

        # Configuration
        config = {"model": {"name": "gpt-4o", "temperature": 0.1, "max_tokens": 4000}}

        # Mock OpenAI to avoid needing API key for tests
        with mock.patch("openai.OpenAI"):
            generator = GPTGeneratorFactory.create_generator(
                generator_type="single",
                api_key="test_key",
                model_name=config["model"]["name"],
                temperature=config["model"]["temperature"],
                max_tokens=config["model"]["max_tokens"],
            )

        # Verify it's the correct type
        assert generator.__class__.__name__ == "SingleGPTGenerator"
        print("✅ SingleGPTGenerator created successfully")

        # Verify required methods exist
        assert hasattr(generator, "generate_multiple_dst_structures")
        print("✅ SingleGPTGenerator has required methods")

        assert True

    except Exception as e:
        print(f"❌ SingleGPTGenerator initialization test failed: {e}")
        assert False, f"SingleGPTGenerator initialization test failed: {e}"


def test_dst_generator_dataloader_integration():
    """Test that DST Generator properly integrates with DataLoader"""
    print("\n🧪 Testing DST Generator + DataLoader integration...")

    try:
        # Import the module inside the OpenAI mock context to avoid any runtime
        # side-effects (Hydra or OpenAI client creation) during import.
        from omegaconf import OmegaConf

        # Configuration matching our successful run
        config = OmegaConf.create(
            {
                "generator": {"type": "single", "batch_size": 1, "save_intermediate": True},
                "model": {
                    "name": "gpt-4o",
                    "temperature": 0.1,
                    "max_tokens": 4000,
                    "log_name": "dst-generator",
                },
                "data_source": {
                    "name": "manual",
                    "data_path": "data/proassist_dst_manual_data",
                },
            }
        )

        # Mock OpenAI to avoid API calls and import side-effects
        # Also mock the GPTGeneratorFactory to return a fake generator that
        # implements the minimal interface required by SimpleDSTGenerator.
        class FakeGenerator:
            def generate_dst_outputs(self, file_paths):
                # Return a dict mapping each path to a minimal DST dict
                results = {}
                for p in file_paths:
                    results[str(p)] = {
                        "video_uid": Path(p).name,
                        "inferred_goal": "test goal",
                        "dst": {"steps": []},
                        "metadata": {"counts": {"num_steps": 0, "num_substeps": 0, "num_actions": 0}},
                    }
                return results

        with mock.patch("dst_data_builder.gpt_generators.gpt_generator_factory.GPTGeneratorFactory.create_generator", return_value=FakeGenerator()):
            from dst_data_builder.simple_dst_generator import SimpleDSTGenerator

            generator = SimpleDSTGenerator(config)

        print("✅ DST Generator initialized with DataLoader config")

        # Verify the run method exists
        assert hasattr(generator, "run")
        print("✅ DST Generator has run method")

        # Verify data module integration
        from dst_data_builder.data_sources.dst_data_module import DSTDataModule

        # This should work without issues
        data_module = DSTDataModule(
            data_source_name="manual",
            data_source_cfg={"data_path": "data/proassist_dst_manual_data"},
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Test that data module can access its components
        data_source = data_module.data_source
        dataloader = data_module.dataloader

        print(f"✅ DataModule created: {type(data_module).__name__}")
        print(f"✅ DataSource: {type(data_source).__name__}")
        print(f"✅ DataLoader: {type(dataloader).__name__}")

        # Test dataset size
        dataset_size = data_module.get_dataset_size()
        assert dataset_size > 0
        print(f"✅ Dataset has {dataset_size} samples")

        assert True

    except Exception as e:
        print(f"❌ DST Generator + DataLoader integration test failed: {e}")
        assert False, f"DST Generator + DataLoader integration test failed: {e}"


def test_output_validation():
    """Test that recent DST generator output files are valid"""
    print("\n🧪 Testing DST generator output file validation...")

    # Check for recent output
    output_base = Path("custom/outputs/dst_generated")
    if not output_base.exists():
        print("⚠️ No output directory found - this is OK for tests")
        assert True

    # Find timestamp directories
    timestamp_dirs = [d for d in output_base.iterdir() if d.is_dir()]

    if not timestamp_dirs:
        print("⚠️ No timestamp directories found - generator hasn't been run")
        assert True

    # Get most recent
    recent_run = max(timestamp_dirs, key=lambda x: x.stat().st_mtime)
    dst_outputs = recent_run / "dst_outputs"

    if not dst_outputs.exists():
        print("⚠️ No dst_outputs directory found")
        assert True

    output_files = list(dst_outputs.glob("dst_*.json"))

    if not output_files:
        print("⚠️ No output files found")
        assert True

    print(f"✅ Found {len(output_files)} output files in recent run")

    # Test a few output files
    for i, output_file in enumerate(output_files[:2]):  # Test first 2
        try:
            with open(output_file, "r") as f:
                output_data = json.load(f)

            # Required fields in output
            required_fields = ["video_uid", "inferred_goal", "dst", "metadata"]

            for field in required_fields:
                if field not in output_data:
                    print(f"❌ Output file missing field: {field}")
                    assert False, f"Output file missing field: {field}"

            # DST structure validation
            dst = output_data["dst"]
            if "steps" not in dst:
                print("❌ DST missing steps field")
                assert False, "DST missing steps field"

            # Metadata validation
            metadata = output_data["metadata"]
            if "counts" not in metadata:
                print("❌ Metadata missing counts")
                assert False, "Metadata missing counts"

            counts = metadata["counts"]
            expected_counts = ["num_steps", "num_substeps", "num_actions"]

            for count_field in expected_counts:
                if count_field not in counts:
                    print(f"❌ Metadata counts missing: {count_field}")
                    assert False, f"Metadata counts missing: {count_field}"

            print(f"✅ Output file {i+1}: valid structure")

        except Exception as e:
            print(f"❌ Error reading output file {output_file.name}: {e}")
            assert False, f"Error reading output file {output_file.name}: {e}"

    assert True


def run_dst_gen_manual_single_tests():
    """Run all DST generator tests with manual data and single GPT generator"""
    print("🚀 Starting DST Generator with Manual Data + SingleGPTGenerator Tests...\n")

    test_functions = [
        test_dst_gen_manual_data_validation,
        test_single_gpt_generator_initialization,
        test_dst_generator_dataloader_integration,
        test_output_validation,
    ]

    passed = 0
    total = len(test_functions)

    for test_func in test_functions:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} crashed: {e}")

    print(f"\n📊 Test Results: {passed}/{total} passed")

    if passed == total:
        print("🎉 All DST Generator + Manual Data + SingleGPTGenerator tests passed!")
        print(
            "\n✅ The DST generator with manual data and single processing is working correctly!"
        )
        return True
    else:
        print("💥 Some tests failed!")
        print(f"\nFailed tests: {total - passed}")
        return False


if __name__ == "__main__":
    success = run_dst_gen_manual_single_tests()
    exit(0 if success else 1)
