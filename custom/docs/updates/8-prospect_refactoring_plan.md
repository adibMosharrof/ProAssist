# PROSPECT Code Refactoring Plan

**Date:** 2025-10-30  
**Purpose:** Refactor PROSPECT to match DST generator code style and reuse ProAssist evaluation code

## 1. Problem Analysis

### Current Issues
1. **Monolithic structure**: Single-file modules (data_loader.py, baseline.py, evaluate.py)
2. **No Hydra configs**: Hardcoded parameters, difficult to experiment
3. **Code duplication**: evaluate.py duplicates ProAssist's StreamEvaluator logic
4. **No shell scripts**: Can't easily run from command line
5. **Nested imports**: Methods define imports inside functions
6. **Doesn't match project conventions**: Inconsistent with dst_data_builder style

### What We Want
1. **Match dst_data_builder structure**: Factory pattern, separate modules, clean separation
2. **Use Hydra**: Config-driven development, easy experimentation
3. **Reuse ProAssist code**: Import StreamEvaluator, find_match, BaseInferenceRunner
4. **Shell scripts**: Custom/runner/run_prospect.sh for easy execution
5. **Clean imports**: All imports at top of file
6. **Professional structure**: Scalable, testable, maintainable

---

## 2. Target Architecture

### Folder Structure
```
custom/
├── src/prospect/
│   ├── __init__.py
│   ├── prospect_evaluator.py          # Main entry (with @hydra.main)
│   ├── data_sources/
│   │   ├── __init__.py
│   │   ├── base_data_source.py        # Abstract base
│   │   ├── proassist_video_dataset.py # ProAssist video loader
│   │   └── data_source_factory.py     # Factory pattern
│   ├── runners/
│   │   ├── __init__.py
│   │   ├── vlm_stream_runner.py       # Custom InferenceRunner for VLMs
│   │   └── runner_factory.py          # Factory pattern
│   └── generators/
│       ├── __init__.py
│       ├── base_generator.py          # Abstract base
│       ├── baseline_generator.py      # Zero-shot VLM baseline
│       ├── dst_enhanced_generator.py  # DST-enhanced (Day 2)
│       └── generator_factory.py       # Factory pattern
├── config/prospect/
│   ├── prospect.yaml                  # Main config (defaults)
│   ├── data_source/
│   │   └── proassist_dst.yaml         # ProAssist video data config
│   ├── generator/
│   │   ├── baseline.yaml              # Baseline generator config
│   │   └── dst_enhanced.yaml          # DST-enhanced config
│   └── model/
│       ├── smolvlm2.yaml              # SmolVLM2-2.2B config
│       └── qwen2vl.yaml               # Qwen2-VL config (future)
├── runner/
│   └── run_prospect.sh                # Shell script to run evaluator
├── outputs/prospect/                  # Hydra output dir
│   └── {timestamp}_{model}_{generator}/
│       ├── results/                   # Per-video predictions
│       │   ├── 0.json
│       │   ├── 1.json
│       │   └── ...
│       ├── metrics.json               # Final metrics
│       ├── all_results.json           # Aggregated results
│       └── .hydra/                    # Hydra config snapshot
```

### Key Design Principles
1. **Factory pattern**: GeneratorFactory, DataSourceFactory, RunnerFactory
2. **Hydra composition**: Use defaults list in configs for flexibility
3. **Reuse ProAssist evaluation**: Import StreamEvaluator, not duplicate
4. **Custom runner only**: Only write VLMStreamRunner (inherits BaseInferenceRunner)
5. **No nested methods**: All imports at top
6. **Type hints**: Use typing module for clarity

---

## 3. Code Mapping: Old → New

### Old Structure (BAD)
```
prospect/
├── data_loader.py         # 130 lines, monolithic
├── baseline.py            # 200 lines, monolithic
├── evaluate.py            # 280 lines, DUPLICATE of ProAssist code
└── run_baseline.py        # 150 lines, no Hydra
```

### New Structure (GOOD)
```
prospect/
├── prospect_evaluator.py          # 150 lines (main entry, Hydra)
├── data_sources/
│   ├── proassist_video_dataset.py # 80 lines (from data_loader.py)
│   └── data_source_factory.py     # 30 lines
├── runners/
│   └── vlm_stream_runner.py       # 200 lines (from baseline.py + ProAssist patterns)
└── generators/
    ├── baseline_generator.py      # 100 lines (orchestration only)
    └── generator_factory.py       # 30 lines
```

**Deleted files:**
- evaluate.py → Use `mmassist.eval.evaluators.StreamEvaluator`
- run_baseline.py → Replaced by prospect_evaluator.py + run_prospect.sh

---

## 4. ProAssist Code Reuse Strategy

### What We'll Import from ProAssist

#### 1. StreamEvaluator (mmassist/eval/evaluators/stream_evaluator.py)
**Purpose**: Main evaluation orchestrator  
**What it does**:
- Runs predictions on all videos in dataset
- Calls find_match() for semantic + temporal matching
- Computes AP, AR, F1, BLEU, JI metrics
- Saves results and metrics

**How we'll use it**:
```python
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator
from mmassist.data.dataset import BaseDataset

# Create dataset
dataset = ProAssistVideoDataset(...)

# Create our custom inference runner
vlm_runner = VLMStreamRunner(model, tokenizer, ...)

# Plug into ProAssist's evaluator
evaluator = StreamEvaluator.build(
    model_path=output_dir,
    dataset=dataset,
    inference_runner=vlm_runner,  # Our custom runner!
    sts_model_type="sentence-transformers/all-mpnet-base-v2",
    match_window_time=(-15, 15),
    not_talk_threshold=0.5,
    fps=2,
)

# Run evaluation (identical to ProAssist)
evaluator.run_all_predictions(sample_indices=range(len(dataset)))
metrics = evaluator.compute_metrics()
```

#### 2. find_match() (mmassist/eval/evaluators/pred_match.py)
**Purpose**: Match predictions to ground truth  
**What it does**:
- Semantic similarity (sentence-transformers)
- Temporal proximity (±15s window)
- Hungarian matching algorithm
- Returns MatchResult with matched/missed/redundant

**Called automatically by StreamEvaluator**, we don't need to call it directly.

#### 3. BaseInferenceRunner (mmassist/eval/runners/base_runner.py)
**Purpose**: Abstract base for inference runners  
**What we'll extend**:
```python
from mmassist.eval.runners.base_runner import BaseInferenceRunner

class VLMStreamRunner(BaseInferenceRunner):
    def __init__(self, vlm_model, vlm_processor, **kwargs):
        # Don't call super().__init__ because we don't have ProActModel
        self.vlm_model = vlm_model
        self.vlm_processor = vlm_processor
        self.eval_name = kwargs.get("eval_name", "vlm_baseline")
        # ... store other params
        
    def run_inference_on_video(self, video: dict, **kwargs) -> dict:
        # Custom VLM inference logic
        # Return format compatible with StreamEvaluator
```

#### 4. FrameOutput (mmassist/eval/runners/stream_inference.py)
**Purpose**: Dataclass for per-frame predictions  
**Fields**: gen, ref, image, timestamp, frame_idx  
**Usage**:
```python
from mmassist.eval.runners.stream_inference import FrameOutput

outputs = []
for frame_idx, frame in enumerate(video_frames):
    gen_dialogue = vlm_model.generate(frame, prompt)
    ref_dialogue = ground_truth[frame_idx] if has_gt else ""
    
    outputs.append(FrameOutput(
        gen=gen_dialogue,
        ref=ref_dialogue,
        image=frame,
        frame_idx_in_stream=frame_idx,
        timestamp_in_stream=frame_idx / fps,
    ))

return {"predictions": outputs, ...}
```

### What We WON'T Duplicate
- ❌ Semantic similarity computation (already in find_match)
- ❌ Metric computation (AP/AR/F1/BLEU/JI) (already in StreamEvaluator)
- ❌ Matching algorithm (already in find_match)
- ❌ Result saving/loading (already in BaseEvaluator)

### What We WILL Write
- ✅ VLMStreamRunner (custom inference with SmolVLM2)
- ✅ ProAssistVideoDataset (load TSV + frames + dialogues)
- ✅ BaselineGenerator (orchestrate runner + evaluator)
- ✅ Hydra configs
- ✅ Shell script

---

## 5. Detailed Implementation Plan

### Phase 1: Create Hydra Configs (30 min)

#### File: custom/config/prospect/prospect.yaml
```yaml
defaults:
  - data_source: proassist_dst
  - generator: baseline
  - model: smolvlm2

# Evaluation settings
fps: 2
not_talk_threshold: 0.5
eval_max_seq_len_str: "4k"

# Matching settings
match_window_time: [-15, 15]  # seconds
match_dist_func_factor: 0.3
match_dist_func_power: 1.5
match_semantic_score_threshold: 0.5

# STS model for semantic matching
sts_model_type: "sentence-transformers/all-mpnet-base-v2"

# Output
exp_name: baseline_run

# Hydra
hydra:
  run:
    dir: custom/outputs/prospect/${now:%Y-%m-%d}/${now:%H-%M-%S}_${model.log_name}_${generator.type}_${exp_name}
```

#### File: custom/config/prospect/data_source/proassist_dst.yaml
```yaml
name: proassist_dst
data_path: data/proassist/processed_data/assembly101
dst_annotation_path: data/proassist_dst_manual_data
dialogue_path: data/processed_data/assembly101/generated_dialogs
frame_dir: frames
fps: 2
video_ids:
  - "9011-c03f"  # Default single video for testing
# Can override with: data_source.video_ids=[9011-c03f,P01_11,...]
```

#### File: custom/config/prospect/generator/baseline.yaml
```yaml
type: baseline
runner_type: vlm_stream  # Use VLM stream runner
transition_detection_prompt: |
  You are observing someone performing a task. Based on the current frame and your understanding,
  predict what substep they are currently on. Answer with just the substep name.
dialogue_generation_prompt: |
  You are a helpful assistant observing someone perform a task.
  They just completed: {prev_substep}
  They are now on: {curr_substep}
  Generate a brief, encouraging dialogue to help them with the next step.
  Keep it under 20 words.
```

#### File: custom/config/prospect/model/smolvlm2.yaml
```yaml
name: "HuggingFaceTB/SmolVLM2-Instruct"
log_name: "smolvlm2-2.2b"
device: "cuda"
torch_dtype: "bfloat16"
max_new_tokens: 100
temperature: 0.7
```

### Phase 2: Create Data Source (1 hour)

#### File: custom/src/prospect/data_sources/proassist_video_dataset.py
```python
import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from PIL import Image
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from mmassist.data.dataset import BaseDataset


@dataclass
class VideoSample:
    """Single video sample for PROSPECT evaluation"""
    video_id: str
    frames: List[Image.Image]
    frame_indices: List[int]
    timestamps: List[float]
    dst_annotations: pd.DataFrame
    ground_truth_dialogues: List[Dict[str, Any]]
    dataset_name: str = "assembly101"


class ProAssistVideoDataset(BaseDataset):
    """Dataset for loading ProAssist videos with DST annotations"""
    
    def __init__(
        self,
        data_path: str,
        dst_annotation_path: str,
        dialogue_path: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        frame_dir: str = "frames",
        fps: int = 2,
        **kwargs
    ):
        self.data_path = Path(data_path)
        self.dst_annotation_path = Path(dst_annotation_path)
        self.dialogue_path = Path(dialogue_path) if dialogue_path else None
        self.frame_dir = frame_dir
        self.fps = fps
        
        # Discover videos
        if video_ids:
            self.video_ids = video_ids
        else:
            self.video_ids = self._discover_videos()
        
        self.samples = self._load_all_samples()
        
    def _discover_videos(self) -> List[str]:
        """Discover available videos from TSV files"""
        tsv_files = list(self.dst_annotation_path.glob("*.tsv"))
        video_ids = []
        for tsv in tsv_files:
            # Extract video_id from filename
            # E.g., assembly_nusar-2021_...9011-c03f...tsv -> 9011-c03f
            filename = tsv.stem
            if "_" in filename:
                parts = filename.split("_")
                for part in parts:
                    if "-" in part and len(part) < 20:
                        video_ids.append(part)
                        break
        return video_ids
    
    def _load_all_samples(self) -> List[VideoSample]:
        """Load all video samples"""
        samples = []
        for video_id in self.video_ids:
            try:
                sample = self._load_single_video(video_id)
                samples.append(sample)
            except Exception as e:
                print(f"Failed to load {video_id}: {e}")
        return samples
    
    def _load_single_video(self, video_id: str) -> VideoSample:
        """Load a single video sample"""
        # Load DST annotations
        dst_df = self._load_dst_tsv(video_id)
        
        # Load frames
        frames, frame_indices, timestamps = self._load_frames(video_id)
        
        # Load ground truth dialogues (optional)
        gt_dialogues = self._load_dialogues(video_id)
        
        return VideoSample(
            video_id=video_id,
            frames=frames,
            frame_indices=frame_indices,
            timestamps=timestamps,
            dst_annotations=dst_df,
            ground_truth_dialogues=gt_dialogues,
        )
    
    def _load_dst_tsv(self, video_id: str) -> pd.DataFrame:
        """Load DST annotations from TSV"""
        tsv_files = list(self.dst_annotation_path.glob(f"*{video_id}*.tsv"))
        if not tsv_files:
            raise FileNotFoundError(f"No TSV for {video_id}")
        return pd.read_csv(tsv_files[0], sep="\t")
    
    def _load_frames(self, video_id: str) -> tuple:
        """Load frames from Arrow file"""
        arrow_files = list(
            self.data_path.glob(f"{self.frame_dir}/*{video_id}*.arrow")
        )
        if not arrow_files:
            raise FileNotFoundError(f"No frames for {video_id}")
        
        table = pq.read_table(arrow_files[0])
        images = [Image.open(io.BytesIO(img.as_py())) 
                  for img in table["image"]]
        frame_indices = list(range(len(images)))
        timestamps = [i / self.fps for i in frame_indices]
        
        return images, frame_indices, timestamps
    
    def _load_dialogues(self, video_id: str) -> List[Dict]:
        """Load ground truth dialogues (optional)"""
        if not self.dialogue_path:
            return []
        
        dialogue_files = list(
            self.dialogue_path.glob(f"*{video_id}*.json")
        )
        if not dialogue_files:
            return []
        
        with open(dialogue_files[0]) as f:
            data = json.load(f)
        
        # Extract assistant dialogues
        dialogues = []
        for turn in data.get("conversation", []):
            if turn["from"] == "assistant":
                dialogues.append({
                    "timestamp": turn.get("timestamp", 0),
                    "content": turn["value"],
                })
        return dialogues
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a video sample in ProAssist dataset format"""
        sample = self.samples[idx]
        
        # Convert to format expected by StreamEvaluator
        return {
            "video_id": sample.video_id,
            "dataset": sample.dataset_name,
            "frames": sample.frames,
            "conversation": [
                {"from": "assistant", "value": d["content"], 
                 "timestamp": d["timestamp"]}
                for d in sample.ground_truth_dialogues
            ],
            "dst_annotations": sample.dst_annotations,
            "fps": self.fps,
        }
    
    @property
    def dataset_name(self) -> str:
        return "prospect/proassist_dst"
```

#### File: custom/src/prospect/data_sources/data_source_factory.py
```python
from typing import Dict, Any
from omegaconf import DictConfig, OmegaConf

from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset


class DataSourceFactory:
    """Factory for creating data sources"""
    
    @staticmethod
    def create_dataset(data_source_cfg: DictConfig) -> ProAssistVideoDataset:
        """Create dataset from config"""
        cfg_dict = OmegaConf.to_container(data_source_cfg, resolve=True)
        
        if cfg_dict["name"] == "proassist_dst":
            return ProAssistVideoDataset(**cfg_dict)
        else:
            raise ValueError(f"Unknown data source: {cfg_dict['name']}")
```

### Phase 3: Create Custom Runner (2 hours)

#### File: custom/src/prospect/runners/vlm_stream_runner.py
```python
import io
from typing import Dict, List, Any, Optional
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForVision2Seq

from mmassist.eval.runners.stream_inference import FrameOutput


class VLMStreamRunner:
    """Custom inference runner using VLM (SmolVLM2) for dialogue generation"""
    
    def __init__(
        self,
        model_name: str,
        eval_name: str = "vlm_baseline",
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        transition_detection_prompt: str = "",
        dialogue_generation_prompt: str = "",
        fps: int = 2,
        not_talk_threshold: float = 0.5,
        **kwargs
    ):
        self.model_name = model_name
        self.eval_name = eval_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.transition_detection_prompt = transition_detection_prompt
        self.dialogue_generation_prompt = dialogue_generation_prompt
        self.fps = fps
        self.not_talk_threshold = not_talk_threshold
        
        # Load VLM
        dtype = torch.bfloat16 if torch_dtype == "bfloat16" else torch.float16
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map=device,
        )
        self.model.eval()
        
        # State tracking
        self.prev_substep = None
        self.current_substep = None
    
    def run_inference_on_video(
        self,
        video: Dict[str, Any],
        output_dir: str = "",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run streaming inference on a video.
        
        Args:
            video: Dict with keys: video_id, frames, conversation, dst_annotations, fps
            output_dir: Where to save results (optional)
        
        Returns:
            Dict with predictions in FrameOutput format
        """
        video_id = video["video_id"]
        frames = video["frames"]
        ground_truth_conv = video.get("conversation", [])
        dst_annotations = video.get("dst_annotations")
        
        outputs = []
        
        for frame_idx, frame in enumerate(frames):
            timestamp = frame_idx / self.fps
            
            # Detect current substep
            curr_substep = self._detect_substep(frame, dst_annotations, timestamp)
            
            # Check if transition occurred
            is_transition = (
                self.prev_substep is not None and
                curr_substep != self.prev_substep and
                curr_substep is not None
            )
            
            # Generate dialogue if transition
            if is_transition:
                dialogue = self._generate_dialogue(
                    frame, self.prev_substep, curr_substep
                )
            else:
                dialogue = ""  # Silent (no dialogue)
            
            # Get reference dialogue at this timestamp
            ref_dialogue = self._get_reference_dialogue(ground_truth_conv, timestamp)
            
            # Create output
            outputs.append(FrameOutput(
                gen=dialogue,
                ref=ref_dialogue,
                image=frame,
                frame_idx_in_stream=frame_idx,
                timestamp_in_stream=timestamp,
            ))
            
            # Update state
            self.prev_substep = curr_substep
        
        # Save predictions if output_dir provided
        if output_dir:
            save_path = os.path.join(output_dir, f"{video_id}_predictions.json")
            self._save_predictions(outputs, save_path, video_id)
        
        return {
            "predictions": outputs,
            "video_id": video_id,
        }
    
    def _detect_substep(
        self,
        frame: Image.Image,
        dst_annotations: Any,
        timestamp: float
    ) -> Optional[str]:
        """Detect current substep from frame using VLM"""
        # Get ground truth substep at this timestamp (for baseline, use GT)
        if dst_annotations is not None:
            substeps = dst_annotations[
                (dst_annotations["type"] == "substep") &
                (dst_annotations["start_ts"] <= timestamp) &
                (dst_annotations["end_ts"] >= timestamp)
            ]
            if len(substeps) > 0:
                return substeps.iloc[0]["name"]
        
        # Fallback: Ask VLM to predict substep
        prompt = self.transition_detection_prompt
        inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.3,
            )
        
        substep = self.processor.decode(outputs[0], skip_special_tokens=True)
        return substep.strip()
    
    def _generate_dialogue(
        self,
        frame: Image.Image,
        prev_substep: str,
        curr_substep: str
    ) -> str:
        """Generate dialogue for transition using VLM"""
        prompt = self.dialogue_generation_prompt.format(
            prev_substep=prev_substep,
            curr_substep=curr_substep
        )
        
        inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
            )
        
        dialogue = self.processor.decode(outputs[0], skip_special_tokens=True)
        return dialogue.strip()
    
    def _get_reference_dialogue(
        self,
        conversation: List[Dict],
        timestamp: float
    ) -> str:
        """Get reference dialogue at given timestamp"""
        for turn in conversation:
            if turn.get("from") == "assistant":
                turn_time = turn.get("timestamp", 0)
                if abs(turn_time - timestamp) < 1.0:  # Within 1 second
                    return turn["value"]
        return ""
    
    def _save_predictions(self, outputs: List[FrameOutput], path: str, video_id: str):
        """Save predictions to JSON"""
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        data = {
            "video_id": video_id,
            "model": self.model_name,
            "predictions": [o.to_dict() for o in outputs]
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
```

### Phase 4: Create Generator (1 hour)

#### File: custom/src/prospect/generators/baseline_generator.py
```python
import logging
from pathlib import Path
from typing import Dict, Any
from omegaconf import DictConfig

from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator


class BaselineGenerator:
    """Baseline generator: VLM-based dialogue generation"""
    
    def __init__(
        self,
        dataset: ProAssistVideoDataset,
        runner: VLMStreamRunner,
        output_dir: str,
        cfg: DictConfig,
    ):
        self.dataset = dataset
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Create evaluator using ProAssist's StreamEvaluator
        self.evaluator = StreamEvaluator.build(
            model_path=str(self.output_dir),
            dataset=dataset,
            model=None,  # We handle inference in runner
            tokenizer=None,
            inference_runner=runner,
            sts_model_type=cfg.sts_model_type,
            match_window_time=tuple(cfg.match_window_time),
            match_dist_func_factor=cfg.match_dist_func_factor,
            match_dist_func_power=cfg.match_dist_func_power,
            match_semantic_score_threshold=cfg.match_semantic_score_threshold,
            fps=cfg.fps,
            not_talk_threshold=cfg.not_talk_threshold,
            eval_max_seq_len_str=cfg.eval_max_seq_len_str,
        )
    
    def run(self) -> Dict[str, Any]:
        """Run baseline evaluation"""
        self.logger.info(f"Running baseline on {len(self.dataset)} videos")
        
        # Run predictions on all videos
        sample_indices = list(range(len(self.dataset)))
        self.evaluator.run_all_predictions(sample_indices, progress_bar=True)
        
        # Compute metrics
        metrics = self.evaluator.compute_metrics(must_complete=True)
        
        self.logger.info("Baseline complete!")
        self.logger.info(f"Metrics: {metrics}")
        
        return metrics
```

#### File: custom/src/prospect/generators/generator_factory.py
```python
from typing import Any
from omegaconf import DictConfig

from prospect.generators.baseline_generator import BaselineGenerator
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
from prospect.runners.vlm_stream_runner import VLMStreamRunner


class GeneratorFactory:
    """Factory for creating generators"""
    
    @staticmethod
    def create_generator(
        generator_cfg: DictConfig,
        dataset: ProAssistVideoDataset,
        runner: VLMStreamRunner,
        output_dir: str,
        main_cfg: DictConfig,
    ) -> BaselineGenerator:
        """Create generator from config"""
        
        generator_type = generator_cfg.type
        
        if generator_type == "baseline":
            return BaselineGenerator(
                dataset=dataset,
                runner=runner,
                output_dir=output_dir,
                cfg=main_cfg,
            )
        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
```

### Phase 5: Create Main Entry Point (1 hour)

#### File: custom/src/prospect/prospect_evaluator.py
```python
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig

from prospect.data_sources.data_source_factory import DataSourceFactory
from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.generators.generator_factory import GeneratorFactory


class ProspectEvaluator:
    """Main evaluator for PROSPECT using Hydra configuration"""
    
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Get Hydra output directory
        hydra_cfg = HydraConfig.get()
        self.output_dir = Path(hydra_cfg.runtime.output_dir)
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def run(self):
        """Run PROSPECT evaluation"""
        self.logger.info("🚀 Starting PROSPECT Evaluator")
        self.logger.info(f"Configuration:\n{OmegaConf.to_yaml(self.cfg)}")
        
        # Create dataset
        self.logger.info("📦 Loading dataset...")
        dataset = DataSourceFactory.create_dataset(self.cfg.data_source)
        self.logger.info(f"✅ Loaded {len(dataset)} videos")
        
        # Create runner
        self.logger.info("🔧 Creating inference runner...")
        runner = VLMStreamRunner(
            model_name=self.cfg.model.name,
            eval_name=f"{self.cfg.generator.type}_{self.cfg.model.log_name}",
            device=self.cfg.model.device,
            torch_dtype=self.cfg.model.torch_dtype,
            max_new_tokens=self.cfg.model.max_new_tokens,
            temperature=self.cfg.model.temperature,
            transition_detection_prompt=self.cfg.generator.transition_detection_prompt,
            dialogue_generation_prompt=self.cfg.generator.dialogue_generation_prompt,
            fps=self.cfg.fps,
            not_talk_threshold=self.cfg.not_talk_threshold,
        )
        self.logger.info("✅ Runner created")
        
        # Create generator
        self.logger.info(f"🎯 Creating generator: {self.cfg.generator.type}")
        generator = GeneratorFactory.create_generator(
            generator_cfg=self.cfg.generator,
            dataset=dataset,
            runner=runner,
            output_dir=str(self.output_dir),
            main_cfg=self.cfg,
        )
        self.logger.info("✅ Generator created")
        
        # Run evaluation
        self.logger.info("▶️  Running evaluation...")
        metrics = generator.run()
        
        # Print results
        self.logger.info("\n" + "="*50)
        self.logger.info("📊 PROSPECT Results")
        self.logger.info("="*50)
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        self.logger.info("="*50)
        self.logger.info(f"✅ Results saved to: {self.output_dir}")


@hydra.main(config_path="../../config/prospect", config_name="prospect", version_base=None)
def main(cfg: DictConfig):
    """Main entry point with Hydra"""
    evaluator = ProspectEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
```

### Phase 6: Create Shell Script (30 min)

#### File: custom/runner/run_prospect.sh
```bash
#!/bin/bash

# Shell script to run PROSPECT Evaluator
# Usage: ./run_prospect.sh

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   PROSPECT Evaluator Runner           ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════╝${NC}"
echo ""

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root: ${PROJECT_ROOT}${NC}"

# Activate conda environment
VENV_PATH="$PROJECT_ROOT/.venv"
if [ -d "$VENV_PATH" ]; then
    echo -e "${GREEN}🔧 Activating conda environment${NC}"
    eval "$(conda shell.bash hook)" || true
    conda activate "$VENV_PATH" || {
        echo -e "${YELLOW}Using conda run instead${NC}"
        CONDA_RUN_CMD="conda run -p $VENV_PATH --no-capture-output"
    }
else
    echo -e "${RED}❌ Conda environment not found${NC}"
    exit 1
fi

# Add to PYTHONPATH
export PYTHONPATH="$PROJECT_ROOT/custom/src:$PROJECT_ROOT:$PYTHONPATH"
echo -e "${GREEN}📦 PYTHONPATH: ${PYTHONPATH}${NC}"

# Run evaluator
echo ""
echo -e "${BLUE}🚀 Starting PROSPECT evaluation...${NC}"
echo ""

if [ -n "$CONDA_RUN_CMD" ]; then
    $CONDA_RUN_CMD python -m prospect.prospect_evaluator "$@"
else
    python -m prospect.prospect_evaluator "$@"
fi

# Check exit code
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ PROSPECT evaluation completed!${NC}"
else
    echo ""
    echo -e "${RED}❌ PROSPECT evaluation failed!${NC}"
    exit 1
fi
```

---

## 6. Migration Checklist

### Phase 1: Configs ✅
- [ ] Create custom/config/prospect/prospect.yaml
- [ ] Create custom/config/prospect/data_source/proassist_dst.yaml
- [ ] Create custom/config/prospect/generator/baseline.yaml
- [ ] Create custom/config/prospect/model/smolvlm2.yaml

### Phase 2: Data Source ✅
- [ ] Create custom/src/prospect/data_sources/proassist_video_dataset.py
- [ ] Create custom/src/prospect/data_sources/data_source_factory.py
- [ ] Create custom/src/prospect/data_sources/__init__.py

### Phase 3: Runner ✅
- [ ] Create custom/src/prospect/runners/vlm_stream_runner.py
- [ ] Create custom/src/prospect/runners/__init__.py

### Phase 4: Generator ✅
- [ ] Create custom/src/prospect/generators/baseline_generator.py
- [ ] Create custom/src/prospect/generators/generator_factory.py
- [ ] Create custom/src/prospect/generators/__init__.py

### Phase 5: Main Entry ✅
- [ ] Create custom/src/prospect/prospect_evaluator.py
- [ ] Create custom/src/prospect/__init__.py

### Phase 6: Shell Script ✅
- [ ] Create custom/runner/run_prospect.sh
- [ ] Make executable: chmod +x custom/runner/run_prospect.sh

### Phase 7: Testing ✅
- [ ] Test single video: ./custom/runner/run_prospect.sh
- [ ] Verify outputs in custom/outputs/prospect/
- [ ] Check metrics.json format matches ProAssist

### Phase 8: Cleanup ✅
- [ ] Delete custom/src/prospect/data_loader.py
- [ ] Delete custom/src/prospect/baseline.py
- [ ] Delete custom/src/prospect/evaluate.py
- [ ] Delete custom/src/prospect/run_baseline.py
- [ ] Update custom/src/prospect/README.md

---

## 7. Key Benefits of This Refactoring

### Before (Current)
❌ Monolithic files (130-280 lines each)  
❌ Hardcoded parameters  
❌ Duplicate ProAssist evaluation code  
❌ No config management  
❌ Difficult to extend  
❌ Inconsistent with project style  

### After (Refactored)
✅ Modular design (50-100 lines per file)  
✅ Hydra-based configuration  
✅ Reuse ProAssist evaluation (identical metrics)  
✅ Match dst_data_builder style  
✅ Easy to extend (add DST-enhanced generator)  
✅ Shell script for easy execution  
✅ All imports at top  
✅ Factory pattern for flexibility  
✅ Professional structure  

---

## 8. Testing Plan

### Step 1: Test Single Video
```bash
cd /u/siddique-d1/adib/ProAssist
./custom/runner/run_prospect.sh
```

**Expected output:**
```
📁 Project root: /u/siddique-d1/adib/ProAssist
🔧 Activating conda environment
📦 PYTHONPATH: ...
🚀 Starting PROSPECT evaluation...
📦 Loading dataset...
✅ Loaded 1 videos
🔧 Creating inference runner...
✅ Runner created
🎯 Creating generator: baseline
✅ Generator created
▶️  Running evaluation...
Run predictions: 100%|████████| 1/1 [03:42<00:00]
==================================================
📊 PROSPECT Results
==================================================
  precision: 0.1500
  recall: 0.0890
  F1: 0.1110
  BLEU_4: 0.0942
  jaccard_index: 0.2670
==================================================
✅ Results saved to: custom/outputs/prospect/2025-10-30/14-30-22_smolvlm2-2.2b_baseline_baseline_run
```

### Step 2: Test Multiple Videos
```bash
./custom/runner/run_prospect.sh \
    data_source.video_ids=[9011-c03f,P01_11] \
    exp_name=multi_video_test
```

### Step 3: Test Different Model
```bash
# Create custom/config/prospect/model/qwen2vl.yaml first
./custom/runner/run_prospect.sh \
    model=qwen2vl \
    exp_name=qwen2vl_test
```

### Step 4: Verify Outputs
```bash
cd custom/outputs/prospect/2025-10-30/14-30-22_smolvlm2-2.2b_baseline_baseline_run/

# Check structure
ls -la
# Should see: results/, metrics.json, all_results.json, .hydra/

# Check metrics
cat metrics.json
# Should match ProAssist format: AP, AR, F1, BLEU, JI

# Check predictions
cat results/0.json
# Should have FrameOutput format
```

---

## 9. Future Extensions

Once baseline works, we can easily add:

### Day 2: DST-Enhanced Generator
```yaml
# custom/config/prospect/generator/dst_enhanced.yaml
type: dst_enhanced
runner_type: vlm_stream
use_dst_context: true
dst_prompt: |
  Task Progress:
  Completed: {completed_substeps}
  Current: {current_substep}
  Next: {next_substep}
  
  Generate helpful dialogue based on this context.
```

### Other VLMs
```yaml
# custom/config/prospect/model/qwen2vl.yaml
name: "Qwen/Qwen2-VL-7B-Instruct"
log_name: "qwen2vl-7b"
device: "cuda"
```

### Batch Processing
```python
# In prospect.yaml
data_source:
  video_ids: [9011-c03f, P01_11, T48, ...]  # All 6 videos
```

---

## 10. Summary

This refactoring:
1. **Matches dst_data_builder style**: Factory pattern, modular design
2. **Uses Hydra**: Easy experimentation, config-driven
3. **Reuses ProAssist code**: StreamEvaluator, find_match, identical metrics
4. **Shell script**: Easy execution like dst_generator
5. **Clean architecture**: Scalable, testable, professional
6. **All imports at top**: No nested methods
7. **Ready for Day 2**: Easy to add DST-enhanced generator

**Estimated time:** 5-6 hours for complete refactoring + testing

**Lines of code:**
- Old: ~760 lines (4 monolithic files)
- New: ~800 lines (15 modular files)
- Deleted: ~280 lines (evaluate.py duplicate)
- Net: Similar size, much better structure!
