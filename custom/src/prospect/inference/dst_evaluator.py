import torch
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import sys
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from transformers import AutoTokenizer

from custom.src.prospect.inference.dst_stream_runner import DSTStreamRunner
from custom.src.prospect.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

# Global variables for worker processes
worker_model = None
worker_processor = None
worker_dataset = None
worker_fps = None
worker_runner = None
worker_id = None

def _init_worker(model, tokenizer, dataset, fps, speaking_threshold, dst_threshold):
    """Initialize worker process with model and dataset."""
    global worker_model, worker_processor, worker_dataset, worker_fps, worker_runner, worker_id
    
    # Determine worker ID from process name or identity
    ident = mp.current_process()._identity
    rank = ident[0] if ident else 1
    worker_id = rank - 1  # 0-based
    
    device = f"cuda:{worker_id % torch.cuda.device_count()}"
    
    worker_model = model
    worker_model.to(device)
    worker_processor = tokenizer
    worker_dataset = dataset
    worker_fps = fps
    
    # Initialize runner
    worker_runner = DSTStreamRunner(
        model=worker_model,
        processor=worker_processor,
        fps=fps,
        speaking_threshold=speaking_threshold,
        dst_threshold=dst_threshold,
        device=device,
        worker_id=worker_id + 1 
    )

def _process_single_sample(idx):
    """Process a single sample using the initialized runner."""
    global worker_runner, worker_dataset
    sample = worker_dataset[idx]
    outputs = worker_runner.run_inference_on_video(sample)
    return idx, outputs

class DSTEvaluator:
    """
    Evaluator for DST inference.
    Handles parallel inference execution and metric computation.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        metrics: List[BaseMetric],
        output_dir: str,
        num_gpus: int = 1,
        fps: float = 2.0,
        speaking_threshold: float = 0.5,
        dst_threshold: float = 0.5,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.metrics = metrics
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.fps = fps
        self.speaking_threshold = speaking_threshold
        self.dst_threshold = dst_threshold
        
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the entire dataset.
        """
        logger.info(f"Starting evaluation on {len(self.dataset)} samples using {self.num_gpus} GPUs.")
        
        # Create frame-level logs directory
        frame_logs_dir = Path(self.output_dir) / "frame_level_logs"
        frame_logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Frame-level logs will be saved to: {frame_logs_dir}")
        
        # Run inference
        if self.num_gpus > 1:
            results = self._run_parallel_inference()
        else:
            results = self._run_sequential_inference()
            
        # Compute metrics and save frame-level logs
        logger.info("Computing metrics...")
        for sample_idx, outputs in results.items():
            # Get reference sample
            reference = self.dataset[sample_idx]
            
            # Create per-video folder and save frame-level predictions
            self._save_frame_level_log(sample_idx, outputs, reference, frame_logs_dir)
            
            # Update each metric
            for metric in self.metrics:
                metric.update(outputs, reference)
                
        # Aggregate results
        final_metrics = {}
        for metric in self.metrics:
            final_metrics.update(metric.compute())
            
        logger.info(f"Evaluation complete. Metrics: {final_metrics}")
        logger.info(f"Frame-level logs saved to: {frame_logs_dir}")
        return final_metrics
        
    def _run_sequential_inference(self) -> Dict[int, List[Any]]:
        """Run inference sequentially on a single GPU."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        runner = DSTStreamRunner(
            model=self.model,
            processor=self.tokenizer,
            fps=self.fps,
            speaking_threshold=self.speaking_threshold,
            dst_threshold=self.dst_threshold,
            device=device,
            worker_id=1
        )
        
        results = {}
        for i in tqdm(range(len(self.dataset)), desc="Inference Progress", position=0):
            sample = self.dataset[i]
            outputs = runner.run_inference_on_video(sample)
            results[i] = outputs
                
        return results
        
    def _save_frame_level_log(
        self, 
        sample_idx: int, 
        outputs: List[Any], 
        reference: Dict[str, Any], 
        output_dir: Path
    ):
        """
        Save frame-level predictions to CSV and generate timeline visualization.
        
        Creates a folder per video with:
        - frame_predictions.csv: Detailed frame-level data
        - speaking_timeline.png: Visualization of speaking decisions
        - dst_update_timeline.png: Visualization of DST update decisions
        """
        try:
            # Create per-video folder
            # Use clip ID (id) if available to distinguish clips from same video
            clip_id = reference.get("id")
            video_uid = reference.get("video_uid")
            video_folder = output_dir / clip_id
            video_folder.mkdir(parents=True, exist_ok=True)
            
            # Build ground truth frame labels
            num_frames = len(outputs)
            gt_speaking = [0] * num_frames
            gt_dst_update = [0] * num_frames
            gt_response = [""] * num_frames
            gt_dst_updates = [""] * num_frames
            
            start_frame = reference.get("start_frame", reference.get("start_frame_idx", 0))
            
            # Log reference structure for debugging
            logger.debug(f"Video {video_uid}: start_frame={start_frame}, num_frames={num_frames}")
            logger.debug(f"  Reference keys: {reference.keys()}")
            
            # Extract from conversation format (same as training)
            conversation = reference.get("conversation", [])
            logger.debug(f"  Conversation: {len(conversation)} turns")
            
            for turn_idx, turn in enumerate(conversation):
                turn_role = turn.get("role", "")
                turn_start = turn.get("start_frame", turn.get("start", None))
                
                logger.debug(f"    Turn {turn_idx}: role={turn_role}, start_frame={turn_start}")
                
                if turn_start is not None:
                    idx = turn_start - start_frame
                    if 0 <= idx < num_frames:
                        # Extract speaking flag and content
                        speaking_flag = turn.get("speaking", 0)
                        dst_update_flag = turn.get("dst_update", 0)
                        
                        if speaking_flag == 1:
                            gt_speaking[idx] = 1
                            gt_response[idx] = turn.get("content", "")
                            logger.debug(f"      Marked frame {idx} as speaking: {gt_response[idx][:50]}...")
                        
                        if dst_update_flag == 1:
                            gt_dst_update[idx] = 1
                            content = turn.get("content", [])
                            if isinstance(content, list):
                                dst_list = [f"{item.get('id')}->{item.get('transition')}" for item in content]
                                gt_dst_updates[idx] = ";".join(dst_list)
                            logger.debug(f"      Marked frame {idx} as DST update: {gt_dst_updates[idx]}")
            
            # Log summary
            num_gt_speaking = sum(gt_speaking)
            num_gt_dst = sum(gt_dst_update)
            logger.debug(f"  GT summary: {num_gt_speaking} speaking frames, {num_gt_dst} DST update frames")
            
            # Extract predictions
            pred_speaking = []
            pred_speaking_prob = []
            pred_dst_update = []
            pred_dst_prob = []
            
            for output in outputs:
                pred_speaking.append(getattr(output, "speaking", 0))
                pred_speaking_prob.append(getattr(output, "speaking_prob", 0.0))
                pred_dst_update.append(getattr(output, "dst_update_binary", 0))
                pred_dst_prob.append(getattr(output, "dst_prob", 0.0))
            
            # Write CSV
            csv_path = video_folder / "frame_predictions.csv"
            
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "frame_idx",
                    "gt_speaking",
                    "pred_speaking",
                    "pred_speaking_prob",
                    "pred_response",
                    "gt_response",
                    "gt_dst_update",
                    "pred_dst_update",
                    "pred_dst_prob",
                    "pred_dst_updates",
                    "gt_dst_updates",
                ])
                
                for i, output in enumerate(outputs):
                    pred_response = getattr(output, "gen", "")
                    pred_dst_updates = getattr(output, "dst_update", "")
                    
                    writer.writerow([
                        i,
                        gt_speaking[i],
                        pred_speaking[i],
                        f"{pred_speaking_prob[i]:.4f}",
                        pred_response,
                        gt_response[i],
                        gt_dst_update[i],
                        pred_dst_update[i],
                        f"{pred_dst_prob[i]:.4f}",
                        pred_dst_updates,
                        gt_dst_updates[i],
                    ])
            
            # Generate visualizations
            self._generate_timeline_visualizations(
                video_uid, video_folder, num_frames,
                gt_speaking, pred_speaking, pred_speaking_prob,
                gt_dst_update, pred_dst_update, pred_dst_prob,
                self.speaking_threshold, self.dst_threshold
            )
            
            logger.info(f"Saved frame-level log for {video_uid} ({num_frames} frames) to {video_folder}")
            
        except Exception as e:
            logger.warning(f"Failed to save frame-level log for sample {sample_idx}: {e}")
    
    def _generate_timeline_visualizations(
        self,
        video_uid: str,
        video_folder: Path,
        num_frames: int,
        gt_speaking: List[int],
        pred_speaking: List[int],
        pred_speaking_prob: List[float],
        gt_dst_update: List[int],
        pred_dst_update: List[int],
        pred_dst_prob: List[float],
        speaking_threshold: float = 0.5,
        dst_threshold: float = 0.5,
    ):
        """Generate probability timeline visualization with confidence curves and decision points."""
        try:
            frames = np.arange(num_frames)
            
            # Create figure with 2 subplots
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8))
            
            # ===== TOP PLOT: Speaking Timeline =====
            ax1.set_title(f"{video_uid} - Speaking Probability Timeline", fontsize=14, fontweight="bold")
            ax1.set_ylabel("Confidence (Probability)", fontsize=12)
            ax1.set_ylim(-0.05, 1.05)
            
            # Blue line: Model's confidence over time
            ax1.plot(frames, pred_speaking_prob, color="blue", linewidth=2.5, 
                    label="Model Confidence", alpha=0.8, zorder=2)
            
            # Green vertical lines: Ground truth moments where assistant should speak
            gt_speak_frames = [f for f, s in enumerate(gt_speaking) if s == 1]
            for frame_idx in gt_speak_frames:
                ax1.axvline(x=frame_idx, color="green", linewidth=2, linestyle="--", 
                           alpha=0.6, zorder=1)
            if gt_speak_frames:
                ax1.axvline(x=gt_speak_frames[0], color="green", linewidth=2, linestyle="--", 
                           alpha=0.6, label="GT Speaking Moments", zorder=1)
            
            # Red dots: Moments where model decided to speak (probability > 0.5)
            pred_speak_frames = [f for f, s in enumerate(pred_speaking) if s == 1]
            ax1.scatter(pred_speak_frames, [pred_speaking_prob[f] for f in pred_speak_frames], 
                       color="red", s=70, marker="o", label="Predicted Speak Decision", 
                       zorder=3, edgecolors="darkred", linewidths=1.5)
            
            # Add threshold line
            ax1.axhline(y=speaking_threshold, color="gray", linewidth=1.5, linestyle=":", alpha=0.5, 
                       label=f"Decision Threshold ({speaking_threshold})")
            
            ax1.set_xlim(-5, num_frames + 5)
            ax1.grid(True, alpha=0.3, axis="both")
            ax1.legend(loc="upper right", fontsize=10, framealpha=0.95)
            ax1.set_xlabel("Frame Index", fontsize=12)
            
            # ===== BOTTOM PLOT: DST Update Timeline =====
            ax2.set_title(f"{video_uid} - DST Update Probability Timeline", fontsize=14, fontweight="bold")
            ax2.set_ylabel("Confidence (Probability)", fontsize=12)
            ax2.set_ylim(-0.05, 1.05)
            
            # Orange line: Model's confidence for DST updates
            ax2.plot(frames, pred_dst_prob, color="orange", linewidth=2.5, 
                    label="Model Confidence", alpha=0.8, zorder=2)
            
            # Blue vertical lines: Ground truth state changes
            gt_dst_frames = [f for f, s in enumerate(gt_dst_update) if s == 1]
            for frame_idx in gt_dst_frames:
                ax2.axvline(x=frame_idx, color="blue", linewidth=2, linestyle="--", 
                           alpha=0.6, zorder=1)
            if gt_dst_frames:
                ax2.axvline(x=gt_dst_frames[0], color="blue", linewidth=2, linestyle="--", 
                           alpha=0.6, label="GT State Changes", zorder=1)
            
            # Purple dots: Predicted updates (probability > 0.5)
            pred_dst_frames = [f for f, s in enumerate(pred_dst_update) if s == 1]
            ax2.scatter(pred_dst_frames, [pred_dst_prob[f] for f in pred_dst_frames], 
                       color="purple", s=70, marker="o", label="Predicted Update Decision", 
                       zorder=3, edgecolors="indigo", linewidths=1.5)
            
            # Add threshold line
            ax2.axhline(y=dst_threshold, color="gray", linewidth=1.5, linestyle=":", alpha=0.5, 
                       label=f"Decision Threshold ({dst_threshold})")
            
            ax2.set_xlim(-5, num_frames + 5)
            ax2.set_xlabel("Frame Index", fontsize=12)
            ax2.grid(True, alpha=0.3, axis="both")
            ax2.legend(loc="upper right", fontsize=10, framealpha=0.95)
            
            plt.tight_layout()
            timeline_path = video_folder / "binary_decisions_visualization.png"
            plt.savefig(timeline_path, dpi=150, bbox_inches="tight")
            plt.close()
            
            logger.debug(f"Generated probability timeline visualization: {timeline_path}")
            
        except Exception as e:
            logger.warning(f"Failed to generate timeline visualization: {e}")
    
    def _run_parallel_inference(self) -> Dict[int, List[Any]]:
        """Run inference in parallel across multiple GPUs."""
        indices = list(range(len(self.dataset)))
        
        ctx = mp.get_context('spawn')
        
        # Pass pre-loaded model and tokenizer to workers
        with ctx.Pool(
            processes=self.num_gpus, 
            initializer=_init_worker, 
            initargs=(self.model, self.tokenizer, self.dataset, self.fps, self.speaking_threshold, self.dst_threshold)
        ) as pool:
            
            # Use imap to get results as they complete for the global progress bar
            results_iterator = pool.imap_unordered(_process_single_sample, indices)
            
            results = {}
            # Global progress bar for videos
            # Position 0 is reserved for this bar
            for idx, output in tqdm(results_iterator, total=len(indices), desc="Inference Progress", position=0):
                results[idx] = output
                
        return results
