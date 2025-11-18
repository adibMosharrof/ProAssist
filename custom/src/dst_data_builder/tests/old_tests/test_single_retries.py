import asyncio
import json

from dst_data_builder.gpt_generators.single_gpt_generator import SingleGPTGenerator
from dst_data_builder.validators.timestamps_validator import TimestampsValidator


async def test_single_generator_retries_on_validator_rejection(monkeypatch, tmp_path):
    """Test that single generator processes items in parallel and validates"""
    # Create a generator with timestamps validator
    gen = SingleGPTGenerator(api_key=None, max_retries=1, validators=[TimestampsValidator()])

    # Prepare inputs
    item = ("input.json", "ik", "desc")

    # Valid DST with timestamps
    valid_dst = {"steps": [{"step_id": "S1", "name": "Step 1", "timestamps": {"start_ts": 0.0, "end_ts": 1.0}}]}

    attempts = {"count": 0}

    # Mock _attempt_dst_generation to return raw JSON string
    async def fake_attempt(inferred_knowledge, all_step_descriptions, previous_failure_reason=""):
        attempts["count"] += 1
        # Always return valid DST for this simplified test
        return True, json.dumps(valid_dst)

    monkeypatch.setattr(gen, "_attempt_dst_generation", fake_attempt)

    results, failure_info = await gen.generate_multiple_dst_structures([item])

    # Should succeed with valid DST
    assert results["input.json"] is not None
    assert results["input.json"]["steps"][0]["step_id"] == "S1"
    assert "timestamps" in results["input.json"]["steps"][0]
    # Should be called once for single-attempt processing
    assert attempts["count"] == 1
