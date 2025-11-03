# PROSPECT Sanity Check Baseline - Implementation Plan

**Date:** 2025-11-02  
**Purpose:** End-to-End Pipeline Validation  
**Status:** 📝 Planning Phase

---

## Executive Summary

Create a **sanity check baseline** that uses ground truth dialogues and timing as "predictions" to verify the PROSPECT evaluation pipeline works end-to-end. This acts as a perfect oracle that should achieve near-perfect metrics (F1 ≈ 1.0, BLEU ≈ 1.0, JI ≈ 1.0).

**Why This Matters:**
- ✅ Validates entire evaluation pipeline (data loading → inference → metrics)
- ✅ Confirms ProAssist StreamEvaluator integration works correctly
- ✅ Establishes upper bound for what's achievable
- ✅ Debugging baseline: any deviation from perfect scores indicates pipeline bugs
- ✅ Foundation for comparing real VLM baselines against

---

## Goals

### Primary Goal
Create a "model" that returns ground truth dialogues as predictions to verify the pipeline achieves perfect metrics.

### Success Criteria
After running on all 6 videos:
- **F1 Score**: ≥ 0.95 (expect ~1.0, allowing for timestamp matching tolerance)
- **Precision (AP)**: ≥ 0.95
- **Recall (AR)**: ≥ 0.95
- **BLEU-4**: ≥ 0.95 (exact text match)
- **Jaccard Index (JI)**: ≥ 0.95
- **Zero crashes**: All 6 videos complete successfully

---

## Data Structure Understanding

### Folder Structure
```
data/proassist/processed_data/assembly101/
├── frames/                          # Arrow files with video frames
├── generated_dialogs/
│   ├── val_filtered.json            # ← WE USE THIS
│   ├── test_filtered.json
│   ├── train_filtered.json
│   └── ...
└── prepared/
```

### val_filtered.json Structure
```json
[
  {
    "video_uid": "assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit",
    "conversations": [
      {
        "conversation": [
          {
            "role": "user",
            "time": 97.2,
            "content": "I want to assemble a toy roller...",
            "labels": ""
          },
          {
            "role": "assistant",
            "time": 97.2,
            "content": "Great! Let's get started...",
            "labels": "initiative|instruction"
          },
          {
            "role": "assistant",
            "time": 106.8,
            "content": "Now, let's screw the chassis together...",
            "labels": "initiative|instruction,info_sharing"
          }
        ]
      }
    ],
    "parsed_video_anns": { ... }
  }
]
```

**Key Observations:**
1. Each video has a `conversations` array (can have multiple conversation objects)
2. Each conversation has a `conversation` array with user/assistant turns
3. Each turn has:
   - `role`: "user" or "assistant"
   - `time`: timestamp in seconds
   - `content`: the actual dialogue text
   - `labels`: dialogue act labels
4. `video_uid` matches the video ID from frames folder (with prefix "assembly_")

### Video IDs to Test
From previous work, we have 6 videos:
1. `9011-c03f` (Assembly101)
2. `grp-cec778f9-9b54-4b67-b013-116378fd7a85` (Ego4D - need to check if in val_filtered.json)
3. `bee9d8dc-ac78-11ee-819f-80615f12b59e` (EgoExoLearn - need to check)
4. `P01_11` (EPIC-Kitchens - need to check)
5. `R0027-12` (Assembly NUSAR - need to check)
6. `T48` (WTaG - need to check)

**Note:** We may need to check which videos are in val_filtered.json for assembly101. For now, we'll focus on assembly101 videos and expand later if needed.

---

## Architecture Design

### Overview
Reuse the existing PROSPECT pipeline with minimal modifications:

```
prospect_evaluator.py (main entry)
    ↓
ProspectEvaluator.run()
    ↓ (1) Load Dataset
    ↓
SanityCheckDataset (NEW - reuses ProAssistVideoDataset)
    - Loads ground truth dialogues from val_filtered.json
    ↓ (2) Create Runner
    ↓
SanityCheckRunner (NEW)
    - Returns ground truth dialogues as "predictions"
    - No model loading, just data passthrough
    ↓ (3) Create Generator
    ↓
SanityCheckGenerator (NEW - reuses BaselineGenerator structure)
    - Orchestrates runner + StreamEvaluator
    ↓ (4) Evaluate
    ↓
StreamEvaluator (ProAssist - NO CHANGES)
    - Computes metrics
    - Saves results
```

### Reuse vs. New Code

**Reuse (No Changes):**
- ✅ `prospect_evaluator.py` (main entry point)
- ✅ `mmassist/eval/evaluators/stream_evaluator.py` (ProAssist evaluation)
- ✅ Shell script: `custom/runner/run_prospect.sh`
- ✅ Most of the data loading logic

**New Files to Create:**
1. **Runner**: `custom/src/prospect/runners/sanity_check_runner.py` (~150 lines)
2. **Generator**: `custom/src/prospect/generators/sanity_check_generator.py` (~120 lines)
3. **Config**: `custom/config/prospect/generator/sanity_check.yaml` (~30 lines)
4. **Data Source** (Optional): Extend existing or create new (~50 lines)

**Total New Code:** ~350 lines

---

## Implementation Plan

### Phase 1: Data Loading Enhancement (30 minutes)

**Goal:** Ensure `ProAssistVideoDataset` can load ground truth dialogues from val_filtered.json

**Files to Modify:**
- `custom/src/prospect/data_sources/proassist_video_dataset.py`

**Changes Needed:**
1. Update `_load_dialogue_data()` method to load from val_filtered.json
2. Parse the JSON structure correctly
3. Extract assistant dialogues (role="assistant") with timestamps
4. Store in `VideoSample.conversation` field

**Implementation Details:**
```python
def _load_dialogue_data(self, video_id: str) -> List[Dict[str, Any]]:
    """
    Load ground truth dialogues from val_filtered.json.
    
    Returns:
        List of dialogue dicts with keys: time, content, labels
    """
    dialogue_file = Path(self.dialogue_path) / "val_filtered.json"
    
    if not dialogue_file.exists():
        return []
    
    # Load JSON
    with open(dialogue_file) as f:
        data = json.load(f)
    
    # Find entry matching video_id
    for entry in data:
        # video_uid format: "assembly_nusar-2021_action_both_9011-c03f_..."
        # video_id format: "9011-c03f"
        if video_id in entry["video_uid"]:
            # Extract assistant dialogues
            dialogues = []
            for conv_obj in entry.get("conversations", []):
                for turn in conv_obj.get("conversation", []):
                    if turn["role"] == "assistant":
                        dialogues.append({
                            "time": turn["time"],
                            "content": turn["content"],
                            "labels": turn.get("labels", "")
                        })
            return dialogues
    
    return []
```

**Testing:**
```bash
# Test data loading
python -c "
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
ds = ProAssistVideoDataset(
    data_path='data/proassist/processed_data/assembly101',
    dst_annotation_path='data/proassist_dst_manual_data/assembly101',
    dialogue_path='data/proassist/processed_data/assembly101/generated_dialogs',
    video_ids=['9011-c03f']
)
video = ds[0]
print(f'Video ID: {video.video_id}')
print(f'Num frames: {len(video.frames)}')
print(f'Num dialogues: {len(video.conversation)}')
print(f'First dialogue: {video.conversation[0]}')
"
```

---

### Phase 2: Sanity Check Runner (45 minutes)

**Goal:** Create a runner that returns ground truth dialogues as predictions

**File to Create:**
- `custom/src/prospect/runners/sanity_check_runner.py`

**Class Design:**
```python
class SanityCheckRunner:
    """
    Sanity check runner that returns ground truth dialogues as predictions.
    
    This is a perfect oracle that should achieve near-perfect metrics.
    Used to validate the evaluation pipeline works correctly.
    """
    
    def __init__(self, fps: float = 2.0, **kwargs):
        """
        Args:
            fps: Frames per second (for frame index calculation)
            **kwargs: Ignored (for compatibility with other runners)
        """
        self.fps = fps
        self.eval_name = "sanity_check"
        logger.info("Initialized SanityCheckRunner (Perfect Oracle)")
    
    def run_inference_on_video(
        self, 
        video: Dict[str, Any], 
        output_dir: str = "", 
        **kwargs
    ) -> Dict[str, Any]:
        """
        Return ground truth dialogues as predictions.
        
        Args:
            video: Dict with keys:
                - video_id: str
                - frames: List[PIL.Image]
                - conversation: List[Dict] with keys: time, content, labels
                - dst_annotations: pd.DataFrame (optional)
                - fps: float
        
        Returns:
            Dict with:
                - predictions: List[FrameOutput]
                - video_id: str
        """
        video_id = video["video_id"]
        frames = video["frames"]
        ground_truth_conv = video.get("conversation", [])
        
        logger.info(f"Running sanity check on video {video_id}: {len(frames)} frames")
        logger.info(f"Ground truth dialogues: {len(ground_truth_conv)}")
        
        # Create FrameOutput for each frame
        outputs = []
        dialogue_map = {d["time"]: d["content"] for d in ground_truth_conv}
        
        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / self.fps
            
            # Check if there's a dialogue at this timestamp
            # Allow small tolerance for floating point comparison
            dialogue = ""
            ref_dialogue = ""
            
            for gt_time, gt_content in dialogue_map.items():
                if abs(timestamp - gt_time) < 0.5:  # 0.5s tolerance
                    dialogue = gt_content
                    ref_dialogue = gt_content
                    break
            
            # Create FrameOutput
            outputs.append(
                FrameOutput(
                    gen=dialogue,  # Prediction = ground truth
                    ref=ref_dialogue,  # Reference = ground truth
                    image=frame,
                    frame_idx_in_stream=frame_idx,
                    timestamp_in_stream=timestamp,
                )
            )
        
        # Count dialogues
        num_dialogues = sum(1 for o in outputs if o.gen != "")
        logger.info(f"Returned {num_dialogues} dialogues as predictions")
        
        return {
            "predictions": outputs,
            "video_id": video_id,
        }
```

**Key Features:**
1. No model loading (instant startup)
2. Returns ground truth dialogues as both `gen` and `ref`
3. Timestamp matching with tolerance for FPS conversion
4. Compatible with ProAssist's StreamEvaluator interface

**Testing:**
```bash
# Test runner
python -c "
from prospect.runners.sanity_check_runner import SanityCheckRunner
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset

# Load video
ds = ProAssistVideoDataset(
    data_path='data/proassist/processed_data/assembly101',
    dst_annotation_path='data/proassist_dst_manual_data/assembly101',
    dialogue_path='data/proassist/processed_data/assembly101/generated_dialogs',
    video_ids=['9011-c03f']
)
video = ds[0]

# Run inference
runner = SanityCheckRunner(fps=2.0)
result = runner.run_inference_on_video(video.__dict__)
print(f'Predictions: {len(result[\"predictions\"])}')
print(f'Dialogues: {sum(1 for p in result[\"predictions\"] if p.gen)}')
"
```

---

### Phase 3: Sanity Check Generator (45 minutes)

**Goal:** Create generator that orchestrates sanity check runner + evaluation

**File to Create:**
- `custom/src/prospect/generators/sanity_check_generator.py`

**Class Design:**
```python
class SanityCheckGenerator:
    """
    Sanity check generator for pipeline validation.
    
    Uses ground truth dialogues as predictions to verify the evaluation
    pipeline achieves perfect metrics.
    """
    
    def __init__(
        self,
        dataset: Dataset,
        runner: SanityCheckRunner,
        output_dir: str,
        match_window_time: Tuple[int, int] = (-15, 15),
        match_semantic_score_threshold: float = 0.5,
        nlg_metrics: List[str] = ["Bleu", "CIDEr", "METEOR"],
        fps: int = 2,
        not_talk_threshold: float = 0.5,
        eval_max_seq_len_str: str = "4k",
        **kwargs,
    ):
        self.dataset = dataset
        self.runner = runner
        self.output_dir = output_dir
        
        logger.info("Initializing SanityCheckGenerator")
        logger.info(f"Dataset: {len(dataset)} videos")
        logger.info(f"Output: {output_dir}")
        
        # Create StreamEvaluator (reuse ProAssist evaluation)
        logger.info("Creating StreamEvaluator (ProAssist evaluation framework)")
        logger.info(f"match_window_time type: {type(match_window_time)}, value: {match_window_time}")
        
        self.evaluator = StreamEvaluator(
            eval_name=self.runner.eval_name,
            output_dir=self.output_dir,
            runner=self.runner,
            match_window_time=match_window_time,
            match_semantic_score_threshold=match_semantic_score_threshold,
            nlg_metrics=nlg_metrics,
            fps=fps,
            not_talk_threshold=not_talk_threshold,
            eval_max_seq_len_str=eval_max_seq_len_str,
        )
        logger.info("✅ StreamEvaluator created")
    
    def run(self) -> Dict[str, Any]:
        """
        Run sanity check evaluation.
        
        Returns:
            Dict with evaluation metrics
        """
        logger.info("=" * 60)
        logger.info("Starting PROSPECT Sanity Check Evaluation")
        logger.info("=" * 60)
        logger.info(f"Videos to evaluate: {len(self.dataset)}")
        
        # Run inference on all videos
        logger.info("Running inference on all videos...")
        video_samples = [
            {
                "video_id": sample.video_id,
                "frames": sample.frames,
                "conversation": sample.conversation,
                "dst_annotations": sample.dst_annotations,
                "fps": 2.0,
            }
            for sample in self.dataset
        ]
        
        self.evaluator.run_predictions(video_samples, must_complete=True)
        
        # Compute metrics
        logger.info("Computing metrics...")
        metrics = self.evaluator.compute_metrics(must_complete=True)
        
        logger.info("=" * 60)
        logger.info("Sanity Check Evaluation Complete!")
        logger.info("=" * 60)
        
        return metrics
```

**Factory Integration:**
Update `custom/src/prospect/generators/generator_factory.py`:
```python
def create_generator(cfg, dataset, runner, output_dir):
    generator_type = cfg.generator.type
    
    if generator_type == "baseline":
        from prospect.generators.baseline_generator import BaselineGenerator
        return BaselineGenerator(...)
    
    elif generator_type == "sanity_check":
        from prospect.generators.sanity_check_generator import SanityCheckGenerator
        return SanityCheckGenerator(
            dataset=dataset,
            runner=runner,
            output_dir=output_dir,
            match_window_time=tuple(cfg.match_window_time),
            match_semantic_score_threshold=cfg.match_semantic_score_threshold,
            nlg_metrics=cfg.nlg_metrics,
            fps=cfg.fps,
            not_talk_threshold=cfg.not_talk_threshold,
            eval_max_seq_len_str=cfg.eval_max_seq_len_str,
        )
    
    else:
        raise ValueError(f"Unknown generator type: {generator_type}")
```

---

### Phase 4: Configuration (15 minutes)

**Goal:** Create Hydra config for sanity check

**File to Create:**
- `custom/config/prospect/generator/sanity_check.yaml`

**Config Content:**
```yaml
# Sanity Check Generator Configuration
# Uses ground truth dialogues as predictions for pipeline validation

type: sanity_check
runner_type: sanity_check

# No model needed (ground truth passthrough)
# No prompts needed

# Optional: Context handling (not used in sanity check)
context_handling_method: none
use_gt_context: false
```

**Update Main Config:**
Modify `custom/config/prospect/prospect.yaml` to allow sanity_check as default:
```yaml
defaults:
  - data_source: proassist_dst
  - generator: baseline  # Can override with: generator=sanity_check
  - model: smolvlm2
  - _self_

# Rest of config unchanged...
```

---

### Phase 5: Entry Point Updates (15 minutes)

**Goal:** Update prospect_evaluator.py to handle sanity check runner

**File to Modify:**
- `custom/src/prospect/prospect_evaluator.py`

**Changes in `run()` method:**
```python
def run(self):
    """Run PROSPECT evaluation"""
    logger.info("=" * 60)
    logger.info("🚀 Starting PROSPECT Evaluation")
    logger.info("=" * 60)
    
    # ... existing dataset loading code ...
    
    # Step 2: Create inference runner
    logger.info("=" * 60)
    logger.info("🔧 Step 2: Creating Inference Runner")
    logger.info("=" * 60)
    
    runner_type = self.cfg.generator.runner_type
    
    if runner_type == "vlm_stream":
        from prospect.runners.vlm_stream_runner import VLMStreamRunner
        logger.info(f"Model: {self.cfg.model.name}")
        logger.info(f"Generator type: {self.cfg.generator.type}")
        runner = VLMStreamRunner(...)
        
    elif runner_type == "sanity_check":
        from prospect.runners.sanity_check_runner import SanityCheckRunner
        logger.info("Runner type: sanity_check (perfect oracle)")
        logger.info("Generator type: sanity_check")
        runner = SanityCheckRunner(
            fps=self.cfg.fps,
        )
        
    else:
        raise ValueError(f"Unknown runner type: {runner_type}")
    
    logger.info("✅ Runner created")
    
    # ... rest unchanged ...
```

---

### Phase 6: Testing & Validation (1 hour)

**Goal:** Test end-to-end on 1 video, then all 6 videos

#### Test 1: Single Video Smoke Test
```bash
cd /u/siddique-d1/adib/ProAssist

# Run sanity check on 1 video
./custom/runner/run_prospect.sh \
    generator=sanity_check \
    data_source.video_ids=[9011-c03f] \
    exp_name=sanity_check_smoke_test
```

**Expected Output:**
```
🚀 Starting PROSPECT Evaluation
📦 Step 1: Loading Dataset
✅ Loaded 1 videos
  - Video 1: 9011-c03f (461 frames)
🔧 Step 2: Creating Inference Runner
✅ Runner type: sanity_check (perfect oracle)
✅ Runner created
🎯 Step 3: Creating Generator
✅ Generator created: sanity_check
▶️  Step 4: Running Evaluation
============================================================
Starting PROSPECT Sanity Check Evaluation
============================================================
Videos to evaluate: 1
Running inference on all videos...
Run predictions: 100%|████████| 1/1 [XX:XX<00:00]
Returned N dialogues as predictions
Computing metrics...
============================================================
📊 PROSPECT Evaluation Results
============================================================
🎯 Dialogue Generation Metrics:
  Precision (AP):     0.9XXX  (expect ≥ 0.95)
  Recall (AR):        0.9XXX  (expect ≥ 0.95)
  F1 Score:           0.9XXX  (expect ≥ 0.95)
  Jaccard Index (JI): 0.9XXX  (expect ≥ 0.95)
  BLEU-4:             0.9XXX  (expect ≥ 0.95)
============================================================
✅ PROSPECT Evaluation Complete!
```

#### Test 2: Verify Output Structure
```bash
cd custom/outputs/prospect/
ls -la | tail -5

# Navigate to latest run
cd {latest_timestamp}_sanity_check/

# Check structure
ls -la
# Expected: eval/, .hydra/, prospect_evaluator.log

# Check metrics
cat eval/prospect-proassist_dst/stream/notalk0.5-maxlen_4k/metrics.json | jq '.'

# Expected JSON with high metrics:
{
  "precision": 0.9XX,
  "recall": 0.9XX,
  "F1": 0.9XX,
  "jaccard_index": 0.9XX,
  "Bleu_4": 0.9XX,
  ...
}

# Check predictions
cat eval/prospect-proassist_dst/stream/notalk0.5-maxlen_4k/results/0.json | \
    jq '.predictions[] | select(.gen != "")' | head -20

# Expected: gen == ref (identical)
```

#### Test 3: All 6 Videos (if available)
```bash
# Check which videos are in val_filtered.json for assembly101
cat data/proassist/processed_data/assembly101/generated_dialogs/val_filtered.json | \
    jq -r '.[].video_uid' | grep -o '[0-9][0-9][0-9][0-9]-[a-z0-9]*' | sort -u | head -10

# Run on all available videos (adjust list based on above)
./custom/runner/run_prospect.sh \
    generator=sanity_check \
    data_source.video_ids=[9011-c03f,P01_11,R0027-12,...] \
    exp_name=sanity_check_all_videos
```

#### Test 4: Metrics Validation
```python
# Analyze results programmatically
import json
from pathlib import Path

# Find latest run
output_dir = Path("custom/outputs/prospect")
latest_run = max(output_dir.glob("*_sanity_check/"))

# Load metrics
metrics_file = latest_run / "eval/prospect-proassist_dst/stream/notalk0.5-maxlen_4k/metrics.json"
with open(metrics_file) as f:
    metrics = json.load(f)

# Validate
print("Sanity Check Results:")
print(f"  Precision:     {metrics['precision']:.4f} (expect ≥ 0.95)")
print(f"  Recall:        {metrics['recall']:.4f} (expect ≥ 0.95)")
print(f"  F1:            {metrics['F1']:.4f} (expect ≥ 0.95)")
print(f"  Jaccard Index: {metrics['jaccard_index']:.4f} (expect ≥ 0.95)")
print(f"  BLEU-4:        {metrics['Bleu_4']:.4f} (expect ≥ 0.95)")

# Check if passed
passed = all([
    metrics['precision'] >= 0.95,
    metrics['recall'] >= 0.95,
    metrics['F1'] >= 0.95,
    metrics['jaccard_index'] >= 0.95,
    metrics['Bleu_4'] >= 0.95,
])

if passed:
    print("\n✅ SANITY CHECK PASSED - Pipeline works correctly!")
else:
    print("\n⚠️ SANITY CHECK FAILED - Pipeline needs debugging")
    print("This indicates a bug in the evaluation pipeline.")
```

---

## Expected Results

### Perfect Metrics Scenario
If the pipeline works correctly, we expect:

| Metric | Expected Value | Interpretation |
|--------|----------------|----------------|
| **Precision (AP)** | ≥ 0.95 | Generated dialogues match ground truth timing |
| **Recall (AR)** | ≥ 0.95 | All ground truth dialogues are generated |
| **F1 Score** | ≥ 0.95 | Harmonic mean of precision and recall |
| **Jaccard Index (JI)** | ≥ 0.95 | Set overlap between generated and ground truth |
| **BLEU-4** | ≥ 0.95 | Exact text match (4-gram precision) |
| **CIDEr** | High (>3.0) | Consensus-based image description metric |
| **METEOR** | ≥ 0.95 | Alignment-based metric with synonyms |

**Why not 1.0?**
- Timestamp matching uses windows (±15s by default)
- Floating point precision in FPS conversion
- Potential minor preprocessing differences

### Debugging Lower Metrics

If metrics are significantly lower (<0.90), possible causes:

1. **Low Precision (<0.90)**
   - Issue: Generated dialogues at wrong timestamps
   - Debug: Check timestamp alignment in SanityCheckRunner
   - Fix: Adjust tolerance in timestamp matching

2. **Low Recall (<0.90)**
   - Issue: Missing dialogues from ground truth
   - Debug: Check dialogue loading in ProAssistVideoDataset
   - Fix: Verify JSON parsing extracts all assistant turns

3. **Low BLEU (<0.90)**
   - Issue: Text doesn't match exactly
   - Debug: Compare gen vs ref in predictions JSON
   - Fix: Check text preprocessing/normalization

4. **Low F1 but High P/R**
   - Issue: Math error in metric computation
   - Debug: Check StreamEvaluator code
   - Fix: Verify ProAssist integration

---

## Timeline

### Estimated Duration: 3-4 hours

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| 1 | Data Loading Enhancement | 30 min | None |
| 2 | Sanity Check Runner | 45 min | Phase 1 |
| 3 | Sanity Check Generator | 45 min | Phase 2 |
| 4 | Configuration | 15 min | Phase 3 |
| 5 | Entry Point Updates | 15 min | Phase 4 |
| 6 | Testing & Validation | 1 hour | Phase 5 |
| - | **TOTAL** | **3.5 hours** | - |

### Breakdown by Day (Optional)
- **Day 1 Morning (2 hours)**: Phases 1-3 (Core implementation)
- **Day 1 Afternoon (1.5 hours)**: Phases 4-6 (Config, integration, testing)

---

## Files to Create/Modify

### New Files (4 files, ~350 lines total)
1. `custom/src/prospect/runners/sanity_check_runner.py` (~150 lines)
2. `custom/src/prospect/generators/sanity_check_generator.py` (~120 lines)
3. `custom/config/prospect/generator/sanity_check.yaml` (~30 lines)
4. This plan document (~50 lines of actual code snippets)

### Files to Modify (3 files, ~50 lines of changes)
1. `custom/src/prospect/data_sources/proassist_video_dataset.py`
   - Update `_load_dialogue_data()` method (~30 lines)

2. `custom/src/prospect/generators/generator_factory.py`
   - Add sanity_check case to factory (~10 lines)

3. `custom/src/prospect/prospect_evaluator.py`
   - Add sanity_check_runner case to run() method (~10 lines)

### No Changes Needed
- ✅ `mmassist/eval/evaluators/stream_evaluator.py` (ProAssist code)
- ✅ `custom/runner/run_prospect.sh` (shell script)
- ✅ Most configs

---

## Risk Assessment

### Low Risk ✅
- **Reusing existing pipeline**: Minimal new code, mostly data passthrough
- **No model loading**: Instant startup, no GPU issues
- **Ground truth source**: Data already exists and is validated

### Medium Risk ⚠️
- **Timestamp alignment**: FPS conversion might cause off-by-one errors
- **Multiple conversations per video**: Need to handle multiple conversation objects
- **Video ID matching**: Format differences between JSON and frame files

### Mitigation Strategies
1. **Timestamp tolerance**: Use 0.5s tolerance in matching
2. **Debug logging**: Add extensive logging at each step
3. **Incremental testing**: Test 1 video first, then scale to 6
4. **Manual inspection**: Verify first few predictions match ground truth

---

## Success Validation Checklist

- [ ] **Phase 1**: Data loading returns non-empty dialogue list
- [ ] **Phase 2**: Runner creates FrameOutput with gen==ref
- [ ] **Phase 3**: Generator runs without crashes
- [ ] **Phase 4**: Config loads via Hydra
- [ ] **Phase 5**: Entry point recognizes sanity_check runner
- [ ] **Phase 6**: Smoke test completes successfully
- [ ] **Metrics**: F1 ≥ 0.95, BLEU ≥ 0.95, JI ≥ 0.95
- [ ] **All videos**: Runs on all 6 videos without errors
- [ ] **Output**: Predictions JSON has gen==ref for all dialogues

---

## Future Extensions

Once sanity check passes, we can:

1. **Compare VLM Baselines**
   - Run SmolVLM2 baseline
   - Compare against sanity check upper bound
   - Quantify VLM performance gap

2. **Ablation Studies**
   - Remove timestamp oracle (predict timing too)
   - Add noise to text (test robustness)
   - Test with partial dialogues (simulate errors)

3. **Other Datasets**
   - Extend to Ego4D (if dialogues available)
   - Extend to EPIC-Kitchens
   - Cross-dataset validation

4. **Error Analysis**
   - Identify which types of dialogues are hardest
   - Analyze timing sensitivity
   - Study semantic matching behavior

---

## Questions to Answer During Implementation

1. **Data Format:**
   - ✅ Are dialogues stored in val_filtered.json? (YES, confirmed)
   - ✅ Structure: video_uid, conversations array, conversation turns (YES, confirmed)
   - ⚠️ Do all 6 videos have dialogues in assembly101? (NEED TO CHECK)

2. **Timestamp Handling:**
   - What tolerance should we use for timestamp matching? (Proposed: 0.5s)
   - Should we round timestamps to nearest frame? (No, use continuous)

3. **Multiple Conversations:**
   - How to handle multiple conversation objects per video? (Use first one for now)
   - Should we concatenate them? (Not needed for sanity check)

4. **Metrics Interpretation:**
   - What is acceptable deviation from 1.0? (0.95+ is good)
   - How to debug if metrics are low? (See debugging section above)

---

## Summary

This plan creates a **sanity check baseline** that validates the PROSPECT evaluation pipeline by using ground truth dialogues as predictions. The implementation reuses ~90% of existing code and adds ~350 lines of new code across 4 files.

**Key Benefits:**
1. ✅ Validates entire pipeline end-to-end
2. ✅ Establishes performance upper bound
3. ✅ Debugging baseline for future experiments
4. ✅ Minimal implementation effort (3-4 hours)
5. ✅ No GPU/model dependencies

**Next Steps:**
1. Review this plan
2. Ask clarifying questions
3. Begin implementation (Phase 1 → Phase 6)
4. Validate results meet success criteria
5. Document findings

---

**Status:** 📝 Ready for Review  
**Approval Needed:** Yes - Please confirm before implementation  
**Questions:** See "Questions to Answer During Implementation" section above
