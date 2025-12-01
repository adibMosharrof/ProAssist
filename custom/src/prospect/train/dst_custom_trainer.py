"""
DST Custom Trainer for Multi-Task Learning

Custom HuggingFace trainer for DST (Dialog State Tracking) multi-task training.
Supports 4 losses:
  1. speaking_gen_loss - LM loss for assistant utterances
  2. speaking_binary_loss - BCE loss for when to speak
  3. dst_gen_loss - LM loss for DST state generation
  4. dst_binary_loss - BCE loss for when to update DST
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import Trainer

logger = logging.getLogger(__name__)


class DSTCustomTrainer(Trainer):
    """Custom trainer for DST multi-task training with 4 losses."""

    def __init__(self, *args, **kwargs):
        # Extract training config and processor before passing to base Trainer
        self.train_config = kwargs.pop("train_config", None)
        self.processor = kwargs.pop("processor", None)

        super().__init__(*args, **kwargs)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized DSTCustomTrainer with 4-loss multi-task setup")
        
        # Buffer for accumulating metrics between logging steps
        self.training_metrics = {}
        self.eval_metrics = {}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute multi-task loss for DST training.
        
        The model computes all losses internally in the forward pass and returns
        the combined loss in outputs["loss"].
        """
        # Forward pass through model - model computes all losses
        outputs = model(**inputs)

        # Get the combined loss from model
        loss = outputs.get("loss", None)

        # Extract and buffer accuracies if present
        if isinstance(outputs, dict):
            # Determine which buffer to use
            metrics_buffer = self.training_metrics if model.training else self.eval_metrics
            
            # List of metrics to track
            metric_names = [
                "speaking_accuracy", "speaking_precision", "speaking_recall", "speaking_f1",
                "dst_accuracy", "dst_precision", "dst_recall", "dst_f1"
            ]
            
            for metric in metric_names:
                if metric in outputs and outputs[metric] is not None:
                    if metric not in metrics_buffer:
                        metrics_buffer[metric] = []
                    metrics_buffer[metric].append(outputs[metric].item())

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log metrics to stdout/stderr and trackers.
        Injects buffered accuracy metrics into logs.
        """
        # Determine log type
        is_eval_log = any(k.startswith("eval_") for k in logs.keys())
        
        # Inject averaged metrics from buffer (ONLY for training logs)
        if self.training_metrics and not is_eval_log:
            for metric_name, values in self.training_metrics.items():
                if values:
                    # Prefix with train_
                    logs[f"train_{metric_name}"] = sum(values) / len(values)
            
            # Clear buffer after logging
            self.training_metrics = {}

        # Inject averaged metrics from eval buffer (ONLY for eval logs)
        if self.eval_metrics and is_eval_log:
            # Infer prefix from existing logs (e.g. 'eval_loss' -> 'eval')
            prefix = "eval"
            for key in logs.keys():
                if key.endswith("_loss") and key != "loss" and key != "total_loss":
                    prefix = key.rsplit("_loss", 1)[0]
                    break
            
            for metric_name, values in self.eval_metrics.items():
                if values:
                    logs[f"{prefix}_{metric_name}"] = sum(values) / len(values)
            
            # Clear buffer after logging
            self.eval_metrics = {}
            
        super().log(logs, *args, **kwargs)


