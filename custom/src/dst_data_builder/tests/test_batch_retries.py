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


def test_rebatch_retries(tmp_path, monkeypatch):
    # Prepare two items; one will fail first and succeed on second attempt
    item1 = (str(tmp_path / "file1.json"), "ik1", "steps1")
    item2 = (str(tmp_path / "file2.json"), "ik2", "steps2")

    # Create dummy input files (not strictly required for the generator internals)
    for p in [item1[0], item2[0]]:
        with open(p, "w") as f:
            json.dump(
                {
                    "inferred_knowledge": "ik",
                    "parsed_video_anns": {"all_step_descriptions": "s"},
                },
                f,
            )

    # Instantiate generator with a dummy API key and low wait interval
    gen = BatchGPTGenerator(
        api_key=None,
        batch_size=10,
        check_interval=0,
        save_intermediate=False,
        max_retries=1,
    )

    # Patch HydraConfig to point to tmp_path as the hydra output dir
    monkeypatch.setattr(HydraConfig, "get", lambda: DummyHydraCfg(str(tmp_path)))

    # Prepare stateful parse results to simulate failure then success
    # On first attempt, file1 fails (None), file2 succeeds; on second attempt both succeed.
    parse_attempts = {
        1: {item1[0]: None, item2[0]: {"ok": True}},
        2: {item1[0]: {"ok": True}},
    }

    # Counters to track attempts
    state = {"attempt": 0}

    def fake_create_batch_jsonl(items, batch_file):
        # create an empty file to satisfy existence
        Path(batch_file).write_text("\n".join(["{}" for _ in items]))

    def fake_upload_batch(batch_file):
        return f"batch-{os.path.basename(batch_file)}"

    def fake_wait_for_batch_completion(batch_id):
        # increment attempt counter each time wait is called
        state["attempt"] += 1
        return {"id": batch_id}

    def fake_download_batch_results(batch_id, output_file):
        # write a placeholder results file indicating which attempt it was
        # The parse function will not use contents but we'll write something
        Path(output_file).write_text("{}\n")

    def fake_parse_batch_results(results_file):
        # return simulated parse results based on the current attempt
        return parse_attempts.get(state["attempt"], {})

    # Patch the generator internal methods
    monkeypatch.setattr(gen, "_create_batch_jsonl", fake_create_batch_jsonl)
    monkeypatch.setattr(gen, "_upload_batch", fake_upload_batch)
    monkeypatch.setattr(
        gen, "_wait_for_batch_completion", fake_wait_for_batch_completion
    )
    monkeypatch.setattr(gen, "_download_batch_results", fake_download_batch_results)
    monkeypatch.setattr(gen, "_parse_batch_results", fake_parse_batch_results)

    # Run generate_multiple_dst_structures with 2 items
    results = gen.generate_multiple_dst_structures([item1, item2], max_retries=1)

    # Both items should be present and successful after retry
    assert results[item1[0]] == {"ok": True}
    assert results[item2[0]] == {"ok": True}
