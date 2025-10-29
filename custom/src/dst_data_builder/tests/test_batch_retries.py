import os
import json
from pathlib import Path

from dst_data_builder.gpt_generators.batch_gpt_generator import BatchGPTGenerator
from hydra.core.hydra_config import HydraConfig


class DummyHydraCfg:
    def __init__(self, output_dir: str):
        class Run:
            def __init__(self, dir):
                self.dir = dir

        self.runtime = Run(output_dir)


async def test_rebatch_retries(tmp_path, monkeypatch):
    """Test batch processing with single attempt (retries handled at higher level)"""
    # Prepare two items
    item1 = (str(tmp_path / "file1.json"), "ik1", "steps1")
    item2 = (str(tmp_path / "file2.json"), "ik2", "steps2")

    # Create dummy input files
    for p in [item1[0], item2[0]]:
        with open(p, "w") as f:
            json.dump(
                {
                    "inferred_knowledge": "ik",
                    "parsed_video_anns": {"all_step_descriptions": "s"},
                },
                f,
            )

    # Instantiate generator
    gen = BatchGPTGenerator(
        api_key=None,
        batch_size=10,
        check_interval=0,
        save_intermediate=False,
        max_retries=1,
    )

    # Patch HydraConfig to point to tmp_path
    monkeypatch.setattr(HydraConfig, "get", lambda: DummyHydraCfg(str(tmp_path)))

    # Mock to return successful results for both items
    def fake_create_batch_jsonl(items, batch_file):
        Path(batch_file).write_text("\n".join(["{}" for _ in items]))

    def fake_upload_batch(batch_file):
        return f"batch-{os.path.basename(batch_file)}"

    def fake_wait_for_batch_completion(batch_id):
        return {"id": batch_id}

    def fake_download_batch_results(batch_id, output_file):
        Path(output_file).write_text("{}\n")

    def fake_parse_batch_results(results_file):
        # Return success for both items
        return {
            item1[0]: {"ok": True, "steps": []},
            item2[0]: {"ok": True, "steps": []}
        }

    # Patch the generator internal methods
    monkeypatch.setattr(gen, "_create_batch_jsonl", fake_create_batch_jsonl)
    monkeypatch.setattr(gen, "_upload_batch", fake_upload_batch)
    monkeypatch.setattr(gen, "_wait_for_batch_completion", fake_wait_for_batch_completion)
    monkeypatch.setattr(gen, "_download_batch_results", fake_download_batch_results)
    monkeypatch.setattr(gen, "_parse_batch_results", fake_parse_batch_results)

    # Run generate_multiple_dst_structures
    results = await gen.generate_multiple_dst_structures([item1, item2])

    # Both items should succeed
    assert results[item1[0]] is not None
    assert results[item2[0]] is not None
    assert results[item1[0]]["ok"] == True
    assert results[item2[0]]["ok"] == True
