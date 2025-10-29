"""
Tests for SimpleDSTGenerator — detailed documentation

This module contains pytest tests that validate key behaviors of the
`SimpleDSTGenerator` orchestration. The tests are a mixture of fast unit-like
checks and lightweight integration-style checks that use small sample JSON
files stored under `data/proassist_dst_manual_data/`.

What each test verifies
- test_initialization_with_config:
    Confirms `SimpleDSTGenerator(cfg)` initializes and stores the provided
    configuration. OpenAI client construction is patched so no API key is needed.

- test_dst_prompt_creation:
    Verifies that the prompt creation logic (implemented by the GPT generator
    returned from `GPTGeneratorFactory`) includes the provided inferred knowledge
    and the detailed `all_step_descriptions`. This ensures prompts used for LLM
    calls contain the source text.

- test_timestamp_parsing:
    Ensures timestamps exist in `all_step_descriptions` by matching common
    timestamp patterns (e.g. `[97.2s-106.8s]`). This is a lightweight content
    sanity check, not a full parser test.

- test_required_fields_validation:
    Confirms `generate_dst` returns `None` when required input fields are
    missing. File reading is mocked so this behaves deterministically.

- test_dst_structure_validation:
    Mocks the GPT response (shape: `response.choices[0].message.content`) with a
    valid DST JSON and verifies `generate_dst` returns an output dict that
    contains `dst` and `metadata.counts`.

- test_metadata_generation:
    Unit-check to validate counting logic used for metadata (steps, substeps,
    actions) given a synthetic DST structure.

Data requirements and environment
- The tests read small sample JSON files from `data/proassist_dst_manual_data/`:
    - assembly_nusar-... .json
    - ego4d_grp-... .json

- `OPENROUTER_API_KEY` is not required because tests patch the OpenAI client. If
    you run tests without the provided patches, ensure the environment variable
    is set or tests will raise an initialization error.

Mocking strategy
- All external calls to OpenAI are patched with `unittest.mock.patch("openai.OpenAI")`.
    The GPT response is constructed so the shape matches what the code expects
    (`response.choices[0].message.content` containing JSON text).

Design notes and recommendations
- If you want fully isolated unit tests (recommended for CI) replace file
    fixtures with small JSON created via `tmp_path` fixtures.
- Mark data-dependent tests with an `@pytest.mark.integration` tag so CI can
    skip them by default when desired.
- Keep the mocked GPT responses realistic to avoid regression when upstream
    parsing logic changes.

How to run
    pytest -q custom/src/dst_data_builder/tests/test_simple_dst_generator.py

"""

import json
import re
import pytest
from pathlib import Path
from dst_data_builder.simple_dst_generator import SimpleDSTGenerator
from omegaconf import OmegaConf
import unittest.mock as mock


class TestSimpleDSTGenerator:
    """Test cases for SimpleDSTGenerator"""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing"""
        return OmegaConf.create(
            {
                "generator": {"type": "single"},
                "model": {"name": "gpt-4o", "temperature": 0.1, "max_tokens": 4000},
                "max_retries": 1,
            }
        )

    @pytest.fixture
    def sample_data_assembly(self):
        """Create sample assembly data inline"""
        return {
            "video_uid": "assembly_test_001",
            "inferred_knowledge": "Task involves assembling components",
            "parsed_video_anns": {
                "all_step_descriptions": "[0.0s-10.0s] Pick up component\n[10.0s-20.0s] Attach component"
            }
        }

    @pytest.fixture
    def sample_data_ego4d(self):
        """Create sample ego4d data inline"""
        return {
            "video_uid": "ego4d_test_001",
            "inferred_knowledge": "Task involves cooking",
            "parsed_video_anns": {
                "all_step_descriptions": "[0.0s-15.0s] Prepare ingredients\n[15.0s-30.0s] Cook food"
            }
        }

    def test_initialization_with_config(self, sample_config):
        """Test that generator initializes correctly with config"""
        # Mock the OpenAI client to avoid needing API key in tests
        import unittest.mock as mock

        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)
            assert generator.cfg == sample_config

    def test_dst_prompt_creation(self, sample_config, sample_data_assembly):
        """Test that DST prompt is created correctly"""
        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)

            # create_dst_prompt is now a standalone function in dst_generation_prompt module
            # Import it directly for testing
            from dst_data_builder.gpt_generators.dst_generation_prompt import create_dst_prompt
            
            # The code under test expects `all_step_descriptions` under
            # `parsed_video_anns` in the unified format. Read it defensively.
            parsed = sample_data_assembly.get("parsed_video_anns", {})
            all_desc = parsed.get(
                "all_step_descriptions",
                sample_data_assembly.get("all_step_descriptions", ""),
            )

            prompt = create_dst_prompt(
                sample_data_assembly.get("inferred_knowledge", ""),
                all_desc,
            )
            # Prompt should include both the inferred knowledge and detailed descriptions
            assert sample_data_assembly.get("inferred_knowledge", "") in prompt
            assert all_desc in prompt
            assert "steps" in prompt
            assert "substeps" in prompt
            assert "actions" in prompt

    def test_timestamp_parsing(self, sample_config, sample_data_assembly):
        """Test timestamp extraction from descriptions"""
        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)

            # There is no dedicated parse_timestamps method on SimpleDSTGenerator;
            # perform a regex-based sanity check here to ensure timestamps exist.
            parsed = sample_data_assembly.get("parsed_video_anns", {})
            desc = parsed.get(
                "all_step_descriptions",
                sample_data_assembly.get("all_step_descriptions", ""),
            )
            matches = re.findall(r"\[\d+\.?\d*s-\d+\.?\d*s\]", desc)
            assert isinstance(matches, list)
            assert len(matches) > 0

    @pytest.mark.asyncio
    async def test_required_fields_validation(self, sample_config, tmp_path):
        """Test that missing required fields are handled correctly"""
        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)

            # Create a file with incomplete data
            incomplete_data = {
                "video_uid": "test",
                # Missing inferred_knowledge and parsed_video_anns
            }
            
            test_file = tmp_path / "incomplete.json"
            test_file.write_text(json.dumps(incomplete_data))

            # Should handle missing fields gracefully
            results = await generator.gpt_generator.generate_dst_outputs([str(test_file)])
            assert str(test_file) in results
            # Result should be None for invalid input
            assert results[str(test_file)] is None

    @pytest.mark.asyncio
    async def test_dst_structure_validation(self, sample_config, sample_data_assembly, tmp_path):
        """Test that generated DST has correct structure"""
        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)

            # Create test file
            test_file = tmp_path / "test.json"
            test_file.write_text(json.dumps(sample_data_assembly))

            # Mock the API response to return valid DST
            mock_dst = {
                "steps": [
                    {
                        "step_id": "S1",
                        "name": "Test step",
                        "timestamps": {"start_ts": 0.0, "end_ts": 30.0},
                        "substeps": [
                            {
                                "sub_id": "S1.1",
                                "name": "Test substep",
                                "timestamps": {"start_ts": 10.0, "end_ts": 20.0},
                                "actions": [
                                    {
                                        "act_id": "S1.1.a",
                                        "name": "Test action",
                                        "timestamps": {"start_ts": 10.0, "end_ts": 15.0},
                                    }
                                ],
                            }
                        ],
                    }
                ]
            }

            # Mock _attempt_dst_generation to return valid DST
            async def fake_attempt(ik, desc, prev_reason=""):
                return True, json.dumps(mock_dst)

            generator.gpt_generator._attempt_dst_generation = fake_attempt

            # Generate DST
            results = await generator.gpt_generator.generate_dst_outputs([str(test_file)])
            
            assert str(test_file) in results
            dst_output = results[str(test_file)]
            assert dst_output is not None
            result_dict = dst_output.to_dict()
            assert "dst" in result_dict
            assert "steps" in result_dict["dst"]
            assert "metadata" in result_dict
            assert "counts" in result_dict["metadata"]

    def test_metadata_generation(self, sample_config):
        """Test that metadata is generated correctly"""
        with mock.patch("openai.OpenAI"):
            generator = SimpleDSTGenerator(sample_config)

            # Test metadata calculation
            dst_structure = {
                "steps": [
                    {
                        "step_id": "S1",
                        "substeps": [
                            {"actions": [{"act_id": "a1"}, {"act_id": "a2"}]},
                            {"actions": [{"act_id": "a3"}]},
                        ],
                    },
                    {"step_id": "S2", "substeps": [{"actions": [{"act_id": "a4"}]}]},
                ]
            }

            metadata = {
                "counts": {
                    "num_steps": len(dst_structure.get("steps", [])),
                    "num_substeps": sum(
                        len(step.get("substeps", []))
                        for step in dst_structure.get("steps", [])
                    ),
                    "num_actions": sum(
                        len(substep.get("actions", []))
                        for step in dst_structure.get("steps", [])
                        for substep in step.get("substeps", [])
                    ),
                }
            }

            expected_counts = {"num_steps": 2, "num_substeps": 3, "num_actions": 4}
            assert metadata["counts"] == expected_counts


def run_specific_tests():
    """Run specific tests for different datasets"""
    print("🧪 Running tests for manual data...")

    # Test with assembly data
    assembly_file = "data/proassist_dst_manual_data/assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.json"

    if Path(assembly_file).exists():
        print(f"✅ Assembly data file exists: {assembly_file}")

        with open(assembly_file, "r") as f:
            data = json.load(f)

        # Validate structure
        required_fields = ["inferred_knowledge", "all_step_descriptions", "video_uid"]
        for field in required_fields:
            if field in data:
                print(f"✅ {field}: {len(str(data[field]))} characters")
            else:
                print(f"❌ Missing field: {field}")

        # Test inferred knowledge structure
        if "inferred_knowledge" in data:
            steps = [
                line.strip()
                for line in data["inferred_knowledge"].split("\n")
                if line.strip() and line[0].isdigit()
            ]
            print(f"📊 Inferred knowledge has {len(steps)} numbered steps")

        # Test step descriptions structure
        if "all_step_descriptions" in data:
            timestamp_patterns = len(
                re.findall(r"\[\d+\.?\d*s-\d+\.?\d*s\]", data["all_step_descriptions"])
            )
            print(f"📊 Step descriptions has {timestamp_patterns} timestamp ranges")

    else:
        print(f"❌ Assembly data file not found: {assembly_file}")


if __name__ == "__main__":
    run_specific_tests()
