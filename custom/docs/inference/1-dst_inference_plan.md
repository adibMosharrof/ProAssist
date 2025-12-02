# DST Inference Plan for PROSPECT

## Overview

This document outlines the inference pipeline for DST-enhanced PROSPECT, designed to evaluate trained models and compare against ProAssist baselines.

**Key Insight**: ProAssist uses the **same dataset class** for training and evaluation. We follow this pattern - no separate inference dataset needed.

---

## 1. Architecture

### 1.1 Reusing ProAssist Components

ProAssist's evaluation code is modular and can be reused directly:

```python
# What we CAN reuse directly from mmassist:
from mmassist.eval.evaluators.pred_match import find_match, MatchResult
from mmassist.eval.metrics.nlg_scorer import NLGEval
from mmassist.eval.runners.stream_inference import FrameOutput, StreamProcessor
from mmassist.eval.eval_utils import get_match_time_window, save_json

# What we need to adapt:
# - DSTInferenceRunner (our model's forward pass differs from ProAssist)
# - DSTEvaluator (thin wrapper that uses ProAssist's metrics)
```

### 1.2 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           DST Inference Pipeline                             │
└─────────────────────────────────────────────────────────────────────────────┘

     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
     │  Trained Model   │     │ DSTTrainingDataset│     │  ProAssist       │
     │  Checkpoint      │     │ (same as training)│     │  Metrics Code    │
     └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
                            ┌──────────────────────┐
                            │  DSTInferenceRunner  │
                            │  (our forward pass)  │
                            └──────────┬───────────┘
                                       │
                                       ▼
                            ┌──────────────────────┐
                            │  List[FrameOutput]   │  ← ProAssist format!
                            │  (compatible format) │
                            └──────────┬───────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
     ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
     │  find_match()    │    │  NLGEval         │    │  LLM Eval        │
     │  (from mmassist) │    │  (from mmassist) │    │  (from mmassist) │
     └──────────────────┘    └──────────────────┘    └──────────────────┘
              │                        │                        │
              └────────────────────────┼────────────────────────┘
                                       ▼
                            ┌──────────────────────┐
                            │  metrics.json        │
                            │  (ProAssist format)  │
                            └──────────────────────┘
```

---

## 2. Why Same Dataset Works

### ProAssist Pattern
```python
# mmassist/data/build.py
def build_train_dataset(train_datasets, **kwargs) -> ConcatDataset:
    return ConcatDataset([build_dataset(name, **kwargs) for name in train_datasets])

def build_eval_datasets(eval_datasets, **kwargs) -> dict[str, Dataset]:
    return {name: build_dataset(name, **kwargs) for name in eval_datasets}

# SAME build_dataset() and BaseDataset for both!
```

### Our Pattern
```python
# We use DSTTrainingDataset for BOTH training and evaluation
# Just change step_name: "train" → "test"

train_ds = DSTTrainingDataset(data_path=path, step_name="train", ...)
eval_ds = DSTTrainingDataset(data_path=path, step_name="test", ...)  # Same class!
```

---

## 3. Reusing ProAssist Metrics Code

### 3.1 Direct Imports

```python
# custom/src/prospect/eval/evaluators/dst_evaluator.py

# Import ProAssist metrics directly!
from mmassist.eval.evaluators.pred_match import find_match, MatchResult
from mmassist.eval.metrics.nlg_scorer import NLGEval
from mmassist.eval.runners.stream_inference import FrameOutput
from mmassist.eval.eval_utils import get_match_time_window, save_json, get_file_path
```

### 3.2 The Key: FrameOutput Compatibility

ProAssist's metrics require `List[FrameOutput]`. We just need to output in this format:

```python
from dataclasses import dataclass

@dataclass
class FrameOutput:
    gen: str                          # Generated text (or "" for silence)
    ref: str | None = None            # Reference text (or "" for silence)
    image: Image.Image | None = None  # Optional frame image
    text_inputs: list[tuple[str, str]] | None = None  # [(role, text), ...]
    frame_idx_in_stream: int | None = None
    frame_idx_in_original_video: int | None = None
    timestamp_in_stream: float | None = None

    def to_dict(self, ignore_keys="image") -> dict:
        # ProAssist's serialization method
        ...
```

### 3.3 Using find_match() Directly

```python
from mmassist.eval.evaluators.pred_match import find_match
import sentence_transformers as sbert

# Load sentence transformer (same as ProAssist)
sts_model = sbert.SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Our predictions in FrameOutput format
predictions: List[FrameOutput] = run_inference_on_video(sample)

# Call ProAssist's matching directly!
match_result = find_match(
    predictions,
    sts_model=sts_model,
    match_window=(-4, 8),  # frames at 2fps → (-2s, +4s)
    dist_func_factor=0.3,
    dist_func_power=1.5,
)

# Returns MatchResult with:
# - matched: List[Tuple[FrameOutput, FrameOutput]]
# - missed: List[FrameOutput]
# - redundant: List[FrameOutput]
# - match_costs: List[float]
# - semantic_scores: List[float]
```

### 3.4 Using NLGEval Directly

```python
from mmassist.eval.metrics.nlg_scorer import NLGEval

nlg_scorer = NLGEval()

# Gather matched texts
hyps = {f"{idx}": pred.gen for idx, (pred, ref) in enumerate(match_result.matched)}
refs = {f"{idx}": [ref.ref] for idx, (pred, ref) in enumerate(match_result.matched)}

# Call ProAssist's NLG scorer directly!
nlg_scores = nlg_scorer.compute_metrics(refs, hyps, metrics_to_eval=["Bleu", "CIDEr", "METEOR"])
# Returns: {"Bleu_1": 0.45, "Bleu_2": 0.32, ..., "CIDEr": 0.67, "METEOR": 0.41}
```

---

## 4. Hydra Configuration

### 4.1 Main Config: `dst_inference.yaml`

```yaml
# custom/config/prospect/dst_inference.yaml

defaults:
  - model: dst_smolvlm2
  - data_source: dst_training
  - _self_

# Project paths
project_root: /u/siddique-d1/adib/ProAssist
raw_data_root: ${project_root}/data/proassist/processed_data

# Checkpoint path (required - set via command line or override)
checkpoint_path: null  # e.g., custom/outputs/prospect/dst_training/2025-12-01/checkpoint-100

# Output directory
output_dir: custom/outputs/prospect/dst_inference/${now:%Y-%m-%d}/${now:%H-%M-%S}

# Experiment name
exp_name: "dst_inference_assembly101"

# Inference parameters
inference:
  # Speaking decision threshold (sigmoid > threshold means speak)
  speaking_threshold: 0.5
  
  # DST update decision threshold
  dst_update_threshold: 0.5
  
  # Generation parameters
  max_new_tokens: 256
  temperature: 0.7
  top_p: 0.9
  do_sample: false  # Greedy decoding for reproducibility
  
  # Batch size for inference
  batch_size: 1
  
  # Device
  device: "cuda"
  
  # FPS for frame timing
  fps: 2

# Evaluation parameters
eval:
  # Sentence transformer model for semantic matching
  sts_model: "sentence-transformers/all-mpnet-base-v2"
  
  # Match window in seconds (converted to frames based on fps)
  match_window_seconds: [-2.0, 4.0]  # ProAssist default: 2s before to 4s after
  
  # Bipartite matching parameters
  dist_func_factor: 0.3
  dist_func_power: 1.5
  
  # Semantic similarity threshold for NLG evaluation
  semantic_threshold: 0.5
  
  # NLG metrics to compute
  nlg_metrics:
    - "Bleu"
    - "METEOR"
    - "CIDEr"

# Hydra configuration
hydra:
  run:
    dir: ${output_dir}
  job:
    chdir: true
```

### 4.2 Data Source Override for Evaluation: `dst_eval.yaml`

```yaml
# custom/config/prospect/data_source/dst_eval.yaml

# Same as dst_training.yaml but with step_name: "test"
name: "dst_eval"

# Data path (same as training)
data_path: ${project_root}/custom/outputs/dst_generated/hybrid_dst/2025-11-30/00-56-27_gpt-4o_proassist_10rows

# Dataset name
datasets:
  - assembly101

# Data parameters
max_seq_len: 4096
neg_frame_sampling_rate: 0.0  # No negative sampling during evaluation - evaluate ALL frames

# Evaluation split
step_name: "test"  # Use test split instead of train
```

### 4.3 Main Entry Point: `run_inference.py` 

```python
# custom/src/prospect/eval/run_inference.py

"""
DST Inference Entry Point - Hydra runner.

Usage:
    python -m custom.src.prospect.eval.run_inference \
        checkpoint_path=custom/outputs/prospect/dst_training/2025-12-01/checkpoint-100
"""

import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from transformers import AutoTokenizer

from custom.src.prospect.models.dst_smolvlm_with_strategies import DSTSmolVLMWithStrategies
from custom.src.prospect.data_sources.dst_training_dataset import DSTTrainingDataset
from custom.src.prospect.eval.evaluators.dst_evaluator import DSTEvaluator

logger = logging.getLogger(__name__)


class InferenceManager:
    """Manages the entire inference pipeline with Hydra configuration."""
    
    def __init__(self, cfg: DictConfig):
        """Initialize with Hydra config."""
        self.cfg = cfg
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.evaluator = None
    
    def load_model(self) -> None:
        """Load model and tokenizer from checkpoint."""
        checkpoint_path = Path(self.cfg.checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        logger.info(f"Loading model from: {checkpoint_path}")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        
        # Load model
        device = self.cfg.inference.device
        dtype = torch.bfloat16 if self.cfg.model.bf16 else torch.float32
        
        self.model = DSTSmolVLMWithStrategies.from_pretrained(
            checkpoint_path,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        
        logger.info(f"✓ Model loaded: {self.model.__class__.__name__}")
        logger.info(f"✓ Device: {next(self.model.parameters()).device}")
    
    def load_dataset(self) -> None:
        """Load evaluation dataset."""
        data_cfg = self.cfg.data_source
        
        self.dataset = DSTTrainingDataset(
            data_path=data_cfg.data_path,
            step_name=data_cfg.get("step_name", "test"),
            max_seq_len=data_cfg.max_seq_len,
            neg_frame_sampling_rate=data_cfg.get("neg_frame_sampling_rate", 0.0),
            raw_data_root=self.cfg.raw_data_root,
        )
        
        logger.info(f"✓ Loaded dataset: {len(self.dataset)} samples")
    
    def create_evaluator(self) -> None:
        """Create DSTEvaluator with configuration."""
        self.evaluator = DSTEvaluator(
            dataset=self.dataset,
            model=self.model,
            tokenizer=self.tokenizer,
            fps=self.cfg.inference.fps,
            speaking_threshold=self.cfg.inference.speaking_threshold,
            dst_update_threshold=self.cfg.inference.dst_update_threshold,
            sts_model_type=self.cfg.eval.sts_model,
            match_window_seconds=tuple(self.cfg.eval.match_window_seconds),
            dist_func_factor=self.cfg.eval.dist_func_factor,
            dist_func_power=self.cfg.eval.dist_func_power,
            nlg_metrics=list(self.cfg.eval.nlg_metrics),
        )
    
    def run(self) -> dict:
        """Run full inference and evaluation pipeline."""
        logger.info("=" * 60)
        logger.info("Starting DST Inference Pipeline")
        logger.info("=" * 60)
        
        # Load components
        self.load_model()
        self.load_dataset()
        self.create_evaluator()
        
        # Run evaluation
        output_dir = Path(self.cfg.output_dir)
        metrics = self.evaluator.run_and_evaluate(str(output_dir))
        
        # Log results
        self._log_results(metrics, output_dir)
        
        return metrics
    
    def _log_results(self, metrics: dict, output_dir: Path) -> None:
        """Log evaluation results."""
        logger.info("=" * 60)
        logger.info("✅ Evaluation Complete!")
        logger.info("=" * 60)
        logger.info(f"Results saved to: {output_dir}")
        
        logger.info("\n📊 Turn-Taking Metrics:")
        logger.info(f"  F1:            {metrics.get('F1', 0):.4f}")
        logger.info(f"  Jaccard:       {metrics.get('jaccard_index', 0):.4f}")
        logger.info(f"  Precision:     {metrics.get('precision', 0):.4f}")
        logger.info(f"  Recall:        {metrics.get('recall', 0):.4f}")
        
        logger.info("\n📝 NLG Metrics:")
        logger.info(f"  BLEU-4:        {metrics.get('Bleu_4', 0):.4f}")
        logger.info(f"  METEOR:        {metrics.get('METEOR', 0):.4f}")
        logger.info(f"  CIDEr:         {metrics.get('CIDEr', 0):.4f}")
        
        if "dst" in metrics:
            logger.info("\n🎯 DST Metrics:")
            dst = metrics["dst"]
            logger.info(f"  Speaking F1:   {dst.get('speaking_f1', 0):.4f}")
            logger.info(f"  DST Update F1: {dst.get('dst_update_f1', 0):.4f}")
            logger.info(f"  Slot Accuracy: {dst.get('dst_state_slot_accuracy', 0):.4f}")


@hydra.main(
    config_path="../../../config/prospect",
    config_name="dst_inference",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra."""
    
    # Log config
    logger.info(OmegaConf.to_yaml(cfg))
    
    # Validate checkpoint
    if cfg.checkpoint_path is None:
        raise ValueError(
            "checkpoint_path is required. "
            "Set via: checkpoint_path=path/to/checkpoint"
        )
    
    # Run pipeline
    manager = InferenceManager(cfg)
    metrics = manager.run()


if __name__ == "__main__":
    main()
```

### 4.4 Bash Runner: `run_dst_inference.sh`

A convenient shell script wrapper similar to training:

```bash
# custom/runner/run_dst_inference.sh

#!/bin/bash
# DST Inference Runner with environment setup and validation

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ... (environment setup, HF_HOME, CUDA_VISIBLE_DEVICES, etc.)

# Validate checkpoint provided
if [ -z "$1" ]; then
    echo "Usage: ./run_dst_inference.sh <checkpoint_path> [additional_hydra_args]"
    exit 1
fi

CHECKPOINT_PATH="$1"

# Build command with Python module runner
CMD="$PYTHON_CMD -m custom.src.prospect.eval.run_inference checkpoint_path=$CHECKPOINT_PATH"

# Add any additional arguments
if [ $# -gt 1 ]; then
    shift
    CMD="$CMD $@"
fi

eval "$CMD"
```

**Usage:**

```bash
# Basic inference with checkpoint
./custom/runner/run_dst_inference.sh custom/outputs/prospect/dst_training/2025-12-01/checkpoint-100

# With parameter overrides
./custom/runner/run_dst_inference.sh \
    custom/outputs/prospect/dst_training/2025-12-01/checkpoint-100 \
    inference.speaking_threshold=0.6 \
    inference.batch_size=4

# Use different data source
./custom/runner/run_dst_inference.sh \
    path/to/checkpoint \
    data_source=dst_eval
```

**Features:**
- ✅ Automatic environment setup (venv, PYTHONPATH, HF_HOME)
- ✅ Checkpoint validation
- ✅ GPU device selection (CUDA_VISIBLE_DEVICES=0)
- ✅ Colored output with progress indicators
- ✅ Error checking and helpful error messages
- ✅ Support for Hydra parameter overrides

### 4.5 Command Line Usage (Direct Python)

For fine-grained control, you can also call Python directly:

```bash
# Basic inference with checkpoint
python -m custom.src.prospect.eval.run_inference \
    checkpoint_path=custom/outputs/prospect/dst_training/2025-12-01/checkpoint-100

# Override inference parameters
python -m custom.src.prospect.eval.run_inference \
    checkpoint_path=path/to/checkpoint \
    inference.speaking_threshold=0.6 \
    inference.do_sample=true \
    inference.temperature=0.8

# Use different data source
python -m custom.src.prospect.eval.run_inference \
    checkpoint_path=path/to/checkpoint \
    data_source=dst_eval

# Override output directory
python -m custom.src.prospect.eval.run_inference \
    checkpoint_path=path/to/checkpoint \
    output_dir=custom/outputs/my_experiment

# Full override example
python -m custom.src.prospect.eval.run_inference \
    checkpoint_path=path/to/checkpoint \
    data_source.data_path=/path/to/different/data \
    data_source.step_name=val \
    inference.batch_size=4 \
    eval.semantic_threshold=0.6
```

---

## 5. Implementation


### 5.1 DSTInferenceRunner (Only New Component)

```python
# custom/src/prospect/eval/runners/dst_inference_runner.py

from mmassist.eval.runners.stream_inference import FrameOutput
from custom.src.prospect.data_sources.dst_training_dataset import DSTTrainingDataset

class DSTInferenceRunner:
    """Minimal runner - just converts our model output to FrameOutput format."""
    
    def __init__(
        self,
        model,
        tokenizer,
        fps: int = 2,
        not_talk_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.fps = fps
        self.not_talk_threshold = not_talk_threshold
    
    def run_inference_on_video(self, sample: dict) -> List[FrameOutput]:
        """Run inference and return ProAssist-compatible FrameOutput list."""
        
        embeddings = sample["embeddings"]  # [T, embed_dim]
        conversation = sample["conversation"]
        
        outputs = []
        
        for frame_idx in range(len(embeddings)):
            frame_embed = embeddings[frame_idx:frame_idx+1]
            timestamp = frame_idx / self.fps
            
            # Our model's forward pass
            with torch.no_grad():
                model_out = self.model.forward_for_inference(
                    image_embeds=frame_embed,
                    # ... other inputs
                )
            
            # Speaking decision from binary head
            should_speak = model_out.speaking_logits.sigmoid() > self.not_talk_threshold
            
            # Generate text if speaking
            gen_text = ""
            if should_speak:
                gen_text = self._generate_text(model_out)
            
            # Get reference for this frame
            ref_text = self._get_ref_at_frame(conversation, frame_idx)
            
            # Output in ProAssist format!
            outputs.append(FrameOutput(
                gen=gen_text,
                ref=ref_text,
                frame_idx_in_stream=frame_idx,
                timestamp_in_stream=timestamp,
            ))
        
        return outputs
```

### 4.2 DSTEvaluator (Thin Wrapper)

```python
# custom/src/prospect/eval/evaluators/dst_evaluator.py

from mmassist.eval.evaluators.pred_match import find_match
from mmassist.eval.metrics.nlg_scorer import NLGEval
from mmassist.eval.eval_utils import get_match_time_window, save_json
import sentence_transformers as sbert

class DSTEvaluator:
    """Thin wrapper that delegates to ProAssist's metrics code."""
    
    def __init__(
        self,
        dataset: DSTTrainingDataset,
        model,
        tokenizer,
        fps: int = 2,
        not_talk_threshold: float = 0.5,
        sts_model_type: str = "sentence-transformers/all-mpnet-base-v2",
    ):
        self.dataset = dataset
        self.runner = DSTInferenceRunner(model, tokenizer, fps, not_talk_threshold)
        self.fps = fps
        
        # ProAssist components
        self.sts_model = sbert.SentenceTransformer(sts_model_type)
        self.nlg_scorer = NLGEval()
    
    def run_and_evaluate(self, output_dir: str) -> dict:
        """Run inference and compute all metrics using ProAssist code."""
        
        all_predictions = {}
        
        for idx in tqdm(range(len(self.dataset))):
            sample = self.dataset[idx]
            
            # Run inference → List[FrameOutput]
            predictions = self.runner.run_inference_on_video(sample)
            
            # Use ProAssist's find_match directly!
            match_window = get_match_time_window(self.dataset.dataset_name)
            match_window_frames = tuple(int(t * self.fps) for t in match_window)
            
            match_result = find_match(
                predictions,
                sts_model=self.sts_model,
                match_window=match_window_frames,
            )
            
            all_predictions[idx] = {
                "predictions": [p.to_dict() for p in predictions],
                "match_result": match_result.to_json(),
            }
            
            # Save per-sample (ProAssist pattern)
            save_json(all_predictions[idx], f"{output_dir}/results/{idx}.json")
        
        # Compute aggregate metrics
        metrics = self._compute_aggregate_metrics(all_predictions)
        save_json(metrics, f"{output_dir}/metrics.json")
        
        return metrics
    
    def _compute_aggregate_metrics(self, all_predictions: dict) -> dict:
        """Compute metrics exactly like ProAssist's StreamEvaluator.compute_metrics()."""
        
        # Aggregate matching results
        results = {"matched": [], "missed": [], "redundant": [], "semantic_scores": []}
        hyps, refs = {}, {}
        
        for idx, pred in all_predictions.items():
            match_result = pred["match_result"]
            results["matched"].extend(match_result["matched"])
            results["missed"].extend(match_result["missed"])
            results["redundant"].extend(match_result["redundant"])
            results["semantic_scores"].extend(match_result["semantic_scores"])
            
            # Gather texts for NLG
            for i, ((g, r), s) in enumerate(zip(match_result["matched"], match_result["semantic_scores"])):
                if s > 0.5:  # semantic threshold
                    uid = f"{idx}_{i}"
                    hyps[uid] = g["gen"]
                    refs[uid] = [r["ref"]]
        
        # Turn-taking metrics (same formulas as ProAssist)
        num_matched = len(hyps)
        num_missed = len(results["missed"])
        num_redundant = len(results["redundant"])
        total = num_matched + num_missed + num_redundant
        
        metrics = {
            "jaccard_index": num_matched / total if total > 0 else 0,
            "missing_rate": num_missed / (num_matched + num_missed) if (num_matched + num_missed) > 0 else 0,
            "redundant_rate": num_redundant / (num_matched + num_redundant) if (num_matched + num_redundant) > 0 else 0,
            "precision": num_matched / (num_matched + num_redundant) if (num_matched + num_redundant) > 0 else 0,
            "recall": num_matched / (num_matched + num_missed) if (num_matched + num_missed) > 0 else 0,
            "semantic_score": sum(results["semantic_scores"]) / len(results["semantic_scores"]) if results["semantic_scores"] else 0,
            "num_matched": num_matched,
            "num_missed": num_missed,
            "num_redundant": num_redundant,
        }
        
        p, r = metrics["precision"], metrics["recall"]
        metrics["F1"] = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        # NLG metrics using ProAssist's NLGEval
        if hyps:
            nlg_scores = self.nlg_scorer.compute_metrics(refs, hyps)
            metrics.update(nlg_scores)
        
        return metrics
```

---

## 6. DST-Specific Metrics

The model outputs DST predictions that need their own metrics:

### 6.1 Model DST Outputs

```python
# From DSTSmolVLMWithStrategies.forward():
outputs = {
    "speaking_logits": torch.Tensor,      # [batch, seq_len, 1] - binary: should speak?
    "dst_update_logits": torch.Tensor,    # [batch, seq_len, 1] - binary: should update DST?
    "logits": torch.Tensor,               # LM logits for generated text (response OR DST JSON)
}
```

### 6.2 DST Metrics Class

```python
# custom/src/prospect/eval/metrics/dst_metrics.py

import torch
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import json


@dataclass
class DSTMetrics:
    """Metrics for Dialog State Tracking evaluation."""
    
    # Speaking decision metrics
    speaking_accuracy: float
    speaking_precision: float
    speaking_recall: float
    speaking_f1: float
    
    # DST update decision metrics
    dst_update_accuracy: float
    dst_update_precision: float
    dst_update_recall: float
    dst_update_f1: float
    
    # DST state accuracy (if applicable)
    dst_state_exact_match: Optional[float] = None   # Exact match of DST JSON
    dst_state_slot_accuracy: Optional[float] = None  # Slot-level accuracy
    dst_state_joint_goal_accuracy: Optional[float] = None  # All slots correct
    
    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


class DSTMetricsCalculator:
    """Calculator for DST-specific metrics."""
    
    def __init__(self, threshold: float = 0.5):
        """
        Args:
            threshold: Sigmoid threshold for binary decisions (default 0.5 means logit > 0)
        """
        self.threshold = threshold
        
        # Accumulators for per-sample metrics
        self.speaking_preds = []
```
### 6.3 Metrics Summary

| Metric | Description |
|--------|-------------|
| `speaking_accuracy` | Accuracy of "should I speak?" binary decision |
| `speaking_precision` | Precision of speaking predictions |
| `speaking_recall` | Recall of speaking predictions |
| `speaking_f1` | F1 score for speaking decision |
| `dst_update_accuracy` | Accuracy of "should I update DST?" binary decision |
| `dst_update_precision` | Precision of DST update predictions |
| `dst_update_recall` | Recall of DST update predictions |
| `dst_update_f1` | F1 score for DST update decision |
| `dst_state_exact_match` | % of samples where DST JSON matches exactly |
| `dst_state_slot_accuracy` | % of (step_id, transition) pairs that match |
| `dst_state_joint_goal_accuracy` | % of samples where ALL slots are correct |

---


## 7. File Structure (Updated)

```
custom/
├── config/prospect/
│   ├── dst_inference.yaml            # Main Hydra config for inference
│   ├── model/
│   │   └── dst_smolvlm2.yaml         # Model config (shared with training)
│   └── data_source/
│       ├── dst_training.yaml         # Training data source
│       └── dst_eval.yaml             # Evaluation data source (step_name=test)
│
└── src/prospect/eval/
    ├── __init__.py
    ├── run_inference.py              # Main entry point (Hydra)
    ├── runners/
    │   ├── __init__.py
    │   └── dst_inference_runner.py   # Model forward pass wrapper
    ├── evaluators/
    │   ├── __init__.py
    │   └── dst_evaluator.py          # Wrapper using ProAssist + DST metrics
    └── metrics/
        ├── __init__.py
        └── dst_metrics.py            # DST-specific metrics calculator

# We directly import from mmassist:
# - mmassist.eval.evaluators.pred_match.find_match
# - mmassist.eval.metrics.nlg_scorer.NLGEval
# - mmassist.eval.runners.stream_inference.FrameOutput
# - mmassist.eval.eval_utils.*
```

---

## 8. Summary

| Component | Source | Notes |
|-----------|--------|-------|
| Dataset | `DSTTrainingDataset` | **Same class for train/eval** (just change `step_name`) |
| `FrameOutput` | `mmassist.eval.runners.stream_inference` | **Reuse directly** |
| `find_match()` | `mmassist.eval.evaluators.pred_match` | **Reuse directly** |
| `NLGEval` | `mmassist.eval.metrics.nlg_scorer` | **Reuse directly** |
| `get_match_time_window()` | `mmassist.eval.eval_utils` | **Reuse directly** |
| `DSTInferenceRunner` | New | Thin wrapper for our model's forward pass |
| `DSTEvaluator` | New | Wrapper that calls ProAssist metrics + DST metrics |
| `DSTMetricsCalculator` | New | **DST-specific metrics** (speaking/update decisions, state accuracy) |

**Metrics Categories**:

1. **Turn-Taking Metrics** (from ProAssist):
   - Jaccard Index, Precision, Recall, F1
   - Missing Rate, Redundant Rate

2. **NLG Metrics** (from ProAssist):
   - BLEU-1/2/3/4, METEOR, CIDEr

3. **DST Binary Decision Metrics** (NEW):
   - Speaking: Accuracy, Precision, Recall, F1
   - DST Update: Accuracy, Precision, Recall, F1

4. **DST State Metrics** (NEW):
   - Exact Match, Slot Accuracy, Joint Goal Accuracy

**Benefits**:
1. ✅ Guaranteed metric compatibility with ProAssist
2. ✅ DST-specific metrics for our multi-task model
3. ✅ Same dataset class for training and evaluation
4. ✅ Direct comparison with ProAssist baseline numbers
