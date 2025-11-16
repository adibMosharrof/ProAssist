"""
DST Custom Trainer for Multi-Task Learning

Custom HuggingFace trainer for DST (Dialog State Tracking) multi-task training
with focal loss for class imbalance handling.
"""

import logging
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
from transformers import Trainer

logger = logging.getLogger(__name__)


class DSTCustomTrainer(Trainer):
    """Custom trainer for DST multi-task training with focal loss"""

    def __init__(self, *args, **kwargs):
        # Extract training config and processor before passing to base Trainer
        self.train_config = kwargs.pop("train_config", {})
        self.processor = kwargs.pop("processor", None)

        super().__init__(*args, **kwargs)

        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize focal loss criterion for class imbalance
        # Using kornia implementation: alpha=0.25, gamma=2.0
        from kornia.losses import FocalLoss as KorniaFocalLoss

        self.focal_criterion = KorniaFocalLoss(alpha=0.25, gamma=2.0, reduction="mean")
        self.logger.info(
            "Using Kornia focal loss for class imbalance: alpha=0.25, gamma=2.0"
        )

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """Compute multi-task loss for DST training with focal loss"""

        # Forward pass through model
        outputs = model(**inputs)

        # Base language modeling loss (for response generation)
        language_loss = 0.0
        if "logits" in outputs and "labels" in inputs:
            shift_logits = outputs["logits"][..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            language_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        total_loss = language_loss

        # DST-specific losses with focal loss for class imbalance
        if inputs.get("temporal_speaking_labels") is not None:
            # Temporal speaking decision loss (binary classification BCE)
            # Model should predict: speak (1) vs stay silent (0) at each timestep
            temporal_labels = inputs["temporal_speaking_labels"].view(-1)
            
            # Ensure logits and labels have matching dimensions
            speaking_logits = outputs["speaking_logits"].view(-1)
            if len(speaking_logits) != len(temporal_labels):
                # Handle dimension mismatch by truncating to minimum length
                min_len = min(len(speaking_logits), len(temporal_labels))
                speaking_logits = speaking_logits[:min_len]
                temporal_labels = temporal_labels[:min_len]
            
            # Compute BCE loss for temporal speaking decisions
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                speaking_logits, temporal_labels.float()
            )
            total_loss += bce_loss * self.train_config.model.speaking_loss_weight

        # DST update losses with temporal structure
        if inputs.get("temporal_dst_update_labels") is not None:
            # Temporal DST update decision loss (binary classification BCE)
            # Model should predict: update state (1) vs no change (0) at each timestep
            temporal_dst_labels = inputs["temporal_dst_update_labels"].view(-1)
            
            # Ensure logits and labels have matching dimensions
            dst_update_logits = outputs["dst_update_logits"].view(-1)
            if len(dst_update_logits) != len(temporal_dst_labels):
                # Handle dimension mismatch by truncating to minimum length
                min_len = min(len(dst_update_logits), len(temporal_dst_labels))
                dst_update_logits = dst_update_logits[:min_len]
                temporal_dst_labels = temporal_dst_labels[:min_len]
            
            # Compute BCE loss for temporal DST update decisions
            dst_bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                dst_update_logits, temporal_dst_labels.float()
            )
            total_loss += dst_bce_loss * self.train_config.model.dst_update_loss_weight
            
        elif inputs.get("dst_update_labels") is not None:
            # Legacy DST update loss for backward compatibility
            dst_update_probs = torch.softmax(
                outputs["dst_update_logits"].view(-1, 2), dim=-1
            )
            dst_update_loss = self.focal_criterion(
                dst_update_probs, inputs["dst_update_labels"].view(-1)
            )
            total_loss += (
                dst_update_loss * self.train_config.model.dst_update_loss_weight
            )

        if inputs.get("dst_state_labels") is not None:
            # DST state update loss (multi-class classification)
            num_dst_states = self.train_config.model.num_dst_states
            dst_state_probs = torch.softmax(
                outputs["dst_state_logits"].view(-1, num_dst_states), dim=-1
            )
            dst_state_loss = self.focal_criterion(
                dst_state_probs, inputs["dst_state_labels"].view(-1)
            )
            total_loss += dst_state_loss * self.train_config.model.dst_state_loss_weight

        # Log losses periodically for monitoring
        if (
            hasattr(self, "state")
            and self.state.global_step % self.train_config.training.logging_steps == 0
        ):
            log_msg = f"Step {self.state.global_step}: "
            log_msg += f"Total Loss: {total_loss.item():.4f}, "
            log_msg += f"Language Loss: {language_loss.item():.4f}"

            if inputs.get("temporal_speaking_labels") is not None:
                log_msg += f", Temporal Speaking Loss: {bce_loss.item():.4f}"
            if inputs.get("temporal_dst_update_labels") is not None:
                log_msg += f", Temporal DST Update Loss: {dst_bce_loss.item():.4f}"
            elif inputs.get("dst_update_labels") is not None:
                log_msg += f", DST Update Loss: {dst_update_loss.item():.4f}"
            if inputs.get("dst_state_labels") is not None:
                log_msg += f", DST State Loss: {dst_state_loss.item():.4f}"

            self.logger.info(log_msg)

        return (total_loss, outputs) if return_outputs else total_loss
