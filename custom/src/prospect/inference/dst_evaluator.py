import torch
import logging
from typing import List, Dict, Any
from tqdm import tqdm
import torch.multiprocessing as mp
import numpy as np
import sys

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

def _init_worker(model, tokenizer, dataset, fps):
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
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.metrics = metrics
        self.output_dir = output_dir
        self.num_gpus = num_gpus
        self.fps = fps
        
    def evaluate(self) -> Dict[str, float]:
        """
        Run evaluation on the entire dataset.
        """
        logger.info(f"Starting evaluation on {len(self.dataset)} samples using {self.num_gpus} GPUs.")
        
        # Run inference
        if self.num_gpus > 1:
            results = self._run_parallel_inference()
        else:
            results = self._run_sequential_inference()
            
        # Compute metrics
        logger.info("Computing metrics...")
        for sample_idx, outputs in results.items():
            # Get reference sample
            reference = self.dataset[sample_idx]
            
            # Update each metric
            for metric in self.metrics:
                metric.update(outputs, reference)
                
        # Aggregate results
        final_metrics = {}
        for metric in self.metrics:
            final_metrics.update(metric.compute())
            
        logger.info(f"Evaluation complete. Metrics: {final_metrics}")
        return final_metrics
        
    def _run_sequential_inference(self) -> Dict[int, List[Any]]:
        """Run inference sequentially on a single GPU."""
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        runner = DSTStreamRunner(
            model=self.model,
            processor=self.tokenizer,
            fps=self.fps,
            device=device,
            worker_id=1
        )
        
        results = {}
        for i in tqdm(range(len(self.dataset)), desc="Inference Progress", position=0):
            sample = self.dataset[i]
            outputs = runner.run_inference_on_video(sample)
            results[i] = outputs
                
        return results
        
    def _run_parallel_inference(self) -> Dict[int, List[Any]]:
        """Run inference in parallel across multiple GPUs."""
        indices = list(range(len(self.dataset)))
        
        ctx = mp.get_context('spawn')
        
        # Pass pre-loaded model and tokenizer to workers
        with ctx.Pool(
            processes=self.num_gpus, 
            initializer=_init_worker, 
            initargs=(self.model, self.tokenizer, self.dataset, self.fps)
        ) as pool:
            
            # Use imap to get results as they complete for the global progress bar
            results_iterator = pool.imap_unordered(_process_single_sample, indices)
            
            results = {}
            # Global progress bar for videos
            # Position 0 is reserved for this bar
            for idx, output in tqdm(results_iterator, total=len(indices), desc="Inference Progress", position=0):
                results[idx] = output
                
        return results
