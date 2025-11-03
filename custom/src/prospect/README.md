# PROSPECT: PROactive State tracking for ProCEdural Task assistance

**Hydra-based evaluation framework for proactive dialogue generation from egocentric videos.**

## Quick Start

```bash
cd /u/siddique-d1/adib/ProAssist

# Run sanity check baseline (validates pipeline)
./custom/runner/run_prospect.sh generator=sanity_check

# Run VLM baseline (SmolVLM2)
./custom/runner/run_prospect.sh generator=baseline

# Run on specific videos
./custom/runner/run_prospect.sh 'data_source.video_ids=[9011-c03f,9012-c14b]'

# Run with custom experiment name
./custom/runner/run_prospect.sh exp_name=my_experiment
```

## Project Structure

```
prospect/
├── prospect_evaluator.py          # Main entry point (Hydra)
├── data_sources/
│   ├── proassist_video_dataset.py # Load TSV + frames + dialogues
│   └── data_source_factory.py     # Factory pattern
├── runners/
│   ├── vlm_stream_runner.py       # SmolVLM2 inference
│   └── sanity_check_runner.py     # Ground truth oracle
├── generators/
│   ├── baseline_generator.py      # VLM baseline
│   ├── sanity_check_generator.py  # Sanity check
│   └── generator_factory.py       # Factory pattern
└── README.md                       # This file

Configuration (Hydra):
custom/config/prospect/
├── prospect.yaml                   # Main config
├── data_source/proassist_dst.yaml  # Data config
├── generator/
│   ├── baseline.yaml               # VLM baseline
│   └── sanity_check.yaml           # Sanity check
└── model/smolvlm2.yaml             # Model config
```

## Architecture

PROSPECT uses:
- **Hydra** for configuration management (matches dst_data_builder pattern)
- **Factory pattern** for extensibility (DataSourceFactory, GeneratorFactory)
- **ProAssist's StreamEvaluator** for evaluation (identical metrics, no code duplication)
- **Modular design** for easy extension (add new generators/runners/models)

## Available Generators

### 1. Sanity Check Baseline (`generator=sanity_check`)

**Purpose:** Validate pipeline by using ground truth dialogues as predictions.

**Expected metrics:**
- Precision (AP): ~0.97
- Recall (AR): ~0.97
- F1: ~0.97
- BLEU-4: ~0.93

**Usage:**
```bash
./custom/runner/run_prospect.sh generator=sanity_check
```

**When to use:**
- Debugging pipeline issues
- Validating new data sources
- Establishing upper performance bound

### 2. VLM Baseline (`generator=baseline`)

**Purpose:** Zero-shot dialogue generation using SmolVLM2-2.2B-Instruct.

**What it does:**
1. Loads video frames + DST annotations
2. Detects substep transitions (uses ground truth for baseline)
3. Generates helpful dialogue at transitions using VLM
4. Evaluates with ProAssist metrics

**Expected metrics:**
- Precision (AP): 0.10-0.20
- Recall (AR): 0.05-0.10
- F1: 0.07-0.13
- BLEU-4: 0.08-0.12

**Usage:**
```bash
./custom/runner/run_prospect.sh generator=baseline
```

**Example output:**
```
[106.8s] Excellent! You've assembled the chassis. Now let's attach the wheels.
[123.7s] Good! First wheel attached. Now screw the second wheel in the same way.
[130.7s] Great! Second wheel done. Continue with the third wheel.
```

### 3. DST-Enhanced (Coming Soon)

**What it adds:**
- Explicit DST state prediction (Step/Substep/Action)
- Context-aware prompts with completed/current/next steps
- Expected improvement: +3-5% F1, +0.02-0.04 BLEU

## Configuration with Hydra

PROSPECT uses Hydra for configuration management. All settings are in YAML files:

### Main Config (`custom/config/prospect/prospect.yaml`)
```yaml
defaults:
  - data_source: proassist_dst
  - generator: baseline
  - model: smolvlm2

fps: 2
not_talk_threshold: 0.5
match_window_time: [-15, 15]  # seconds
exp_name: baseline_run
```

### Override from Command Line
```bash
# Change model
./custom/runner/run_prospect.sh model=qwen2vl

# Change videos
./custom/runner/run_prospect.sh 'data_source.video_ids=[9011-c03f,9012-c14b]'

# Change experiment name
./custom/runner/run_prospect.sh exp_name=my_test

# Multiple overrides
./custom/runner/run_prospect.sh \
  generator=baseline \
  'data_source.video_ids=[9011-c03f]' \
  exp_name=single_video_test
```

## Output Structure

After running, outputs are saved to Hydra's timestamped directory:

```
custom/outputs/prospect/2025-11-03/14-30-22_smolvlm2-2.2b_baseline_my_test/
├── .hydra/                    # Hydra config snapshot
│   ├── config.yaml
│   ├── hydra.yaml
│   └── overrides.yaml
├── eval/                      # ProAssist evaluation results
│   └── prospect-proassist_dst/
│       └── stream/
│           └── notalk0.5-maxlen_4k-none/
│               ├── results/   # Per-video predictions
│               │   ├── 0.json
│               │   └── ...
│               ├── metrics.json      # Final metrics
│               └── all_results.json  # Aggregated results
└── prospect_evaluator.log    # Execution log
```

### Metrics Format (`metrics.json`)
```json
{
  "precision": 0.152,
  "recall": 0.089,
  "F1": 0.111,
  "jaccard_index": 0.267,
  "dialog_Bleu_1": 0.234,
  "dialog_Bleu_2": 0.156,
  "dialog_Bleu_3": 0.112,
  "dialog_Bleu_4": 0.094,
  "dialog_CIDEr": 0.187,
  "dialog_METEOR": 0.145,
  "num_matched": 5,
  "num_missed": 7,
  "num_redundant": 3,
  "semantic_score": 0.623,
  "time_diff": 3.45
}
```

### Predictions Format (`results/0.json`)
```json
{
  "predictions": [
    {
      "gen": "Great! You've completed the chassis assembly.",
      "ref": "Excellent work on the chassis!",
      "frame_idx_in_stream": 234,
      "timestamp_in_stream": 117.0
    },
    ...
  ],
  "video_id": "9011-c03f"
}
```

## Available Videos

**Assembly101 Videos (48 total):**
```
9011-c03f, 9012-c14b, 9013-a28, 9013-c09c, 9014-a23, 9014-b05a,
9015-c10c, 9016-c03c, 9021-c10a, 9022-a18, 9022-b06c, 9023-c09c,
9025-b08d, 9025-c06b, 9031-c04d, 9032-c06f, 9032-c07a, 9033-a30,
9034-c02b, 9036-c13b, 9042-c09c, 9043-b05a, 9043-c03c, 9045-a23,
9046-b06b, 9051-c12a, 9051-c13a, 9053-c12e, 9054-a18, 9054-c06a,
9054-c11a, 9055-c06e, 9056-a19, 9056-b08a, 9061-b08d, 9062-c07a,
9063-c02b, 9063-c13f, 9064-a30, 9065-b05a, 9065-c09c, 9073-a10,
9073-a18, 9074-a03, 9075-c10c, 9076-a20, 9081-a30, 9082-a10, 9082-c08c
```

**Note:** Ground truth dialogues available for all videos in `val_filtered.json`. DST annotations only available for `9011-c03f`.

## Requirements

All dependencies already installed in ProAssist environment:
- Python 3.10+
- PyTorch 2.3+
- transformers 4.57+
- sentence-transformers
- hydra-core
- omegaconf
- pandas
- pyarrow

## Troubleshooting

**"No ground truth dialogue file found"**
- Normal - dialogues loaded from `val_filtered.json` centrally
- Individual JSON files are fallback only

**"No DST TSV file found"**
- DST annotations optional for baseline (uses ground truth substeps)
- Only affects DST-enhanced generator (Day 2)

**"Model loading failed"**
- Check cache directory: `/u/siddique-d1/adib/.cache/huggingface`
- Ensure sufficient disk space (model ~5GB)
- Check GPU availability: `nvidia-smi`

**"Out of memory"**
- SmolVLM2 uses ~5GB VRAM
- Close other GPU processes
- Use smaller model or CPU inference

**"Hydra config error"**
- Check YAML syntax in config files
- Verify override syntax: `key=value` or `'key=[list]'`
- Check Hydra logs in `.hydra/` directory

## Performance Benchmarks

**Sanity Check Baseline:**
- F1: 0.97, BLEU-4: 0.93
- Runtime: ~1.6s per video
- Purpose: Validate pipeline

**VLM Baseline (Expected):**
- F1: 0.10-0.15, BLEU-4: 0.08-0.12
- Runtime: ~30-60s per video (model inference)
- Purpose: Zero-shot dialogue generation

**ProAssist (Trained Model):**
- F1: ~0.35, BLEU-4: ~0.25
- Purpose: Upper bound comparison

## Next Steps

1. **Validate Pipeline:** Run sanity check baseline
   ```bash
   ./custom/runner/run_prospect.sh generator=sanity_check
   ```

2. **Test VLM Baseline:** Run on single video
   ```bash
   ./custom/runner/run_prospect.sh generator=baseline 'data_source.video_ids=[9011-c03f]'
   ```

3. **Scale Up:** Run on multiple videos
   ```bash
   ./custom/runner/run_prospect.sh 'data_source.video_ids=[9011-c03f,9012-c14b,9013-a28]'
   ```

4. **Analyze Results:** Compare metrics to sanity check and ProAssist

5. **Iterate:** Tune prompts, try different models, add DST enhancement

## Documentation

- **Refactoring Plan:** `custom/docs/updates/8-prospect_refactoring_plan.md`
- **Sanity Check Results:** `custom/docs/updates/11-sanity_check_implementation_complete.md`
- **Original Plan:** `custom/docs/updates/7-prospect_plan.md`
