"""
DST ProAssist Trainer for Binary Heads + Early Exit Training

Custom HuggingFace trainer for DST ProAssist training with:
  1. speaking_gen_loss - LM loss for assistant responses  
  2. dst_gen_loss - LM loss for DST updates
  3. speaking_binary_loss - BCE loss for speaking decision head
  4. dst_binary_loss - BCE loss for DST update decision head

This trainer handles the new continuous sequence format with role tokens
and separate loss calculation for DST and assistant responses.
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import Trainer

logger = logging.getLogger(__name__)


class DSTProAssistTrainer(Trainer):
    """Custom trainer for DST ProAssist with binary heads and separate losses."""

    def __init__(self, *args, **kwargs):
        # Extract custom config before passing to base Trainer
        self.train_config = kwargs.pop("train_config", None)
        self.processor = kwargs.pop("processor", None)

        super().__init__(*args, **kwargs)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info("Initialized DSTProAssistTrainer with binary heads + separate losses")
        
        # Buffer for accumulating metrics between logging steps
        self.training_metrics = {}
        self.eval_metrics = {}

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute multi-task loss for DST ProAssist training.
        
        Expected inputs:
            - input_ids: [batch_size, seq_len]
            - speaking_gen_labels: [batch_size, seq_len]  # LM labels for assistant
            - dst_gen_labels: [batch_size, seq_len]  # LM labels for DST
            - speaking_labels: [batch_size, seq_len]  # Binary labels at <image>
            - dst_update_labels: [batch_size, seq_len]  # Binary labels at <image>
            - image_embeds: List[Tensor]  # List of [num_frames, 1152] tensors
        
        The model computes all losses internally and returns the combined loss.
        """
        # Forward pass through model - model computes all losses
        outputs = model(**inputs)

        # Get the combined loss from model (outputs is a ModelOutput object, not a dict)
        loss = outputs.loss if hasattr(outputs, 'loss') else None

        # Extract and buffer metrics if present
        if hasattr(outputs, 'log_dict') and outputs.log_dict:
            # Determine which buffer to use
            metrics_buffer = self.training_metrics if model.training else self.eval_metrics
            
            # List of metrics to track
            metric_names = [
                "speaking_gen_loss", "dst_gen_loss",
                "speaking_binary_loss", "dst_binary_loss",
                "speaking_accuracy", "speaking_precision", "speaking_recall", "speaking_f1",
                "dst_accuracy", "dst_precision", "dst_recall", "dst_f1"
            ]
            
            for metric in metric_names:
                if metric in outputs.log_dict and outputs.log_dict[metric] is not None:
                    if metric not in metrics_buffer:
                        metrics_buffer[metric] = []
                    # Handle both tensor and scalar values
                    value = outputs.log_dict[metric]
                    if isinstance(value, torch.Tensor):
                        value = value.item()
                    metrics_buffer[metric].append(value)
            
            # Debug: Print what metrics were found
            if outputs.log_dict:
                self.logger.info(f"DEBUG: log_dict keys: {list(outputs.log_dict.keys())}")
                self.logger.info(f"DEBUG: metrics_buffer keys: {list(metrics_buffer.keys())}")

        return (loss, outputs) if return_outputs else loss

    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """
        Log metrics to stdout/stderr and trackers.
        Injects buffered metrics into logs.
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
