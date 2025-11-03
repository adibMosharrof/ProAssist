# PROSPECT: PROactive State tracking for ProCEdural Task assistance

Zero-shot VLM baseline for proactive dialogue generation from egocentric videos.

## Quick Start

```bash
cd /u/siddique-d1/adib/ProAssist

# Run Day 1 baseline
python custom/src/prospect/run_baseline.py --video_id 9011-c03f
```

## Project Structure

```
prospect/
├── run_baseline.py      # Main script: Run inference + evaluation
├── data_loader.py       # Load frames, DST annotations, dialogues
├── baseline.py          # Zero-shot dialogue generation
├── evaluate.py          # ProAssist metrics (AP/AR/F1/BLEU)
├── dst_tree.py          # DST hierarchy (Day 2)
└── dst_enhanced.py      # DST-enhanced generator (Day 2)
```

## Day 1: Baseline

**What it does:**
1. Loads video frames + DST annotations
2. Detects substep transitions using VLM
3. Generates helpful dialogue at transitions
4. Evaluates with ProAssist metrics

**Example output:**
```
[106.8s] Excellent! You've assembled the chassis. Now let's attach the wheels.
[123.7s] Good! First wheel attached. Now screw the second wheel in the same way.
[130.7s] Great! Second wheel done. Continue with the third wheel.
```

**Expected metrics:**
- Precision (AP): 0.10-0.20
- Recall (AR): 0.05-0.10
- F1: 0.07-0.13
- BLEU-4: 0.08-0.12

## Day 2: DST-Enhanced (Coming Soon)

**What it adds:**
- Explicit DST state prediction (Step/Substep/Action)
- Context-aware prompts with completed/current/next steps
- Expected improvement: +3-5% F1, +0.02-0.04 BLEU

## Usage

### Run Inference Only
```bash
python custom/src/prospect/run_baseline.py \
    --video_id 9011-c03f \
    --output_dir custom/outputs/prospect_baseline \
    --skip_eval
```

### Run Evaluation Only
```bash
python custom/src/prospect/run_baseline.py \
    --video_id 9011-c03f \
    --output_dir custom/outputs/prospect_baseline \
    --skip_inference
```

### Test Individual Modules
```bash
# Test data loader
python custom/src/prospect/data_loader.py

# Test baseline generator (first 20 frames)
python custom/src/prospect/baseline.py

# Test evaluator
python custom/src/prospect/evaluate.py \
    --predictions outputs/predictions.json \
    --ground_truth data/dialogues.json \
    --output outputs/metrics.json
```

## Requirements

- Python 3.10+
- PyTorch 2.3+
- transformers 4.57+
- sentence-transformers
- nltk
- pandas
- pyarrow

All already installed in the ProAssist environment.

## Output Files

After running, you'll get:

```
custom/outputs/prospect_baseline/
├── 9011-c03f_predictions.json    # Generated dialogues with timestamps
└── 9011-c03f_metrics.json        # Evaluation metrics
```

**predictions.json format:**
```json
{
  "video_id": "9011-c03f",
  "model": "SmolVLM2-2.2B-zero-shot",
  "predictions": [
    {
      "frame_idx": 234,
      "timestamp": 106.8,
      "dialogue": "Excellent! You've assembled the chassis. Now let's attach the wheels.",
      "transition": "Screwing chassis parts → Attaching wheel to chassis"
    },
    ...
  ]
}
```

**metrics.json format:**
```json
{
  "metrics": {
    "AP": 0.152,
    "AR": 0.089,
    "F1": 0.111,
    "BLEU-4": 0.094,
    "JI": 0.267,
    "num_predictions": 8,
    "num_ground_truth": 12,
    "num_matched": 5,
    "num_missed": 7,
    "num_redundant": 3
  }
}
```

## Data Requirements

For a video to work, you need:

1. **DST annotations (TSV)**
   - Path: `data/proassist_dst_manual_data/assembly_*{video_id}*.tsv`
   - Format: type, id, start_ts, end_ts, name
   - Provides hierarchical task structure

2. **Video frames (Arrow)**
   - Path: `data/proassist/processed_data/assembly101/frames/*{video_id}*.arrow`
   - Format: PyArrow table with 'image' column
   - 2 FPS, 384x384 resolution

3. **Ground truth dialogues (JSON)** (optional, for evaluation)
   - Path: `data/processed_data/assembly101/generated_dialogs/*{video_id}*.json`
   - Format: ProAssist conversation format
   - Contains assistant dialogues with timestamps

Currently available: `assembly_9011-c03f` has all 3 data types.

## Troubleshooting

**"No TSV file found"**
- Check available videos in `data/proassist_dst_manual_data/`
- Use correct video_id (e.g., "9011-c03f", not full filename)

**"No Arrow file found"**
- Only Assembly101 videos have frames extracted
- Check `data/proassist/processed_data/assembly101/frames/`

**"No ground truth dialogue file found"**
- Evaluation will be skipped
- You can still get predictions, just no metrics

**Out of memory**
- Model uses ~5GB VRAM
- Reduce batch size or use smaller model

## Next Steps

After Day 1 baseline:
1. Check F1 score - is it >0.10?
2. Inspect generated dialogues - are they helpful?
3. Compare to ProAssist paper numbers
4. If promising → proceed to Day 2 (DST enhancement)

## Documentation

See `custom/docs/updates/7-prospect_plan.md` for full implementation plan.
