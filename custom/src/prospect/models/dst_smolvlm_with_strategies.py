"""
DST SmolVLM with Multi-Task Learning

Extends SmolVLMWithStrategies (not SmolVLMForConditionalGeneration directly)
to include Dialog State Tracking (DST) prediction heads for multi-task learning
with speaking decisions, DST update decisions, and DST state updates.

This inheritance approach avoids code duplication and leverages the proven
functionality from SmolVLMWithStrategies (tested with VLM stream runner).

VISION ENCODING STRATEGY (ProAssist Pattern):
- SmolVLM2 vision encoder produces 729 spatial patches + 1 [CLS] token = 730 total per image
- Instead of using all 730 tokens, we extract only the [CLS] token (global representation)
- This reduces image tokens from 730 → 1 per image (99.9% memory reduction!)
- The [CLS] token is then projected to LLM input space via a trainable MLP projector
- Vision encoder weights are frozen, only projector + LLM LoRA are trained
"""

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support

from kornia.losses import FocalLoss
from prospect.context_strategies import BaseContextStrategy
from prospect.models.smolvlm_with_strategies import SmolVLMWithStrategies

logger = logging.getLogger(__name__)

# Type alias for KV cache (legacy tuple format)
KV_CACHE = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]


class DSTSmolVLMWithStrategies(SmolVLMWithStrategies):
    """
    DST SmolVLM with Multi-Task Learning Heads.

    Extends SmolVLMWithStrategies (which extends SmolVLMForConditionalGeneration)
    to include DST-aware multi-task learning with 3 objectives:
    1. Speaking Decision (Binary: should I speak?)
    2. DST Update Decision (Binary: should I update DST?)
    3. Text Generation (Response OR DST state update as JSON text)

    **Architecture:**
    ```
    Video Frames + Dialog History + Current DST → SmolVLM2 (VLM-based)
                                                  ↓
                                          [Multi-Task Heads]
                                                  ↓
        ┌─────────────────┬─────────────────┬─────────────────────┐
        │                 │                 │                     │
        │  Speaking       │    DST Update   │  Text Generation    │
        │  Decision       │    Decision     │  (Response OR       │
        │  (Binary Head)  │  (Binary Head)  │   DST JSON)         │
        │                 │                 │  (LM Head)          │
        └─────────────────┴─────────────────┴─────────────────────┘
    ```

    **Training**: 3 losses computed simultaneously:
    - LM loss: Language modeling for response/DST generation
    - DST binary loss: Should update DST? (focal loss on image tokens)
    - DST gen loss: Language modeling only on DST_UPDATE content

    **DST Representation**: DST states are generated as JSON text:
    ```json
    [{"id": "S1", "transition": "start"}, {"id": "S2", "transition": "complete"}]
    ```

    **Inheritance Benefits**:
    - Inherits proven joint_embed(), fast_greedy_generate() from SmolVLMWithStrategies
    - Avoids code duplication and leverages tested functionality
    - Only adds DST-specific binary decision heads
    """

    def __init__(self, config):
        """Initialize DST SmolVLM with multi-task heads and [CLS]-based vision projector"""
        # Initialize parent model (SmolVLMWithStrategies) - this handles all base functionality
        super().__init__(config)

        # Initialize focal loss criterion (using 3rd party implementation)
        self.focal_criterion = FocalLoss(alpha=0.25, gamma=2, reduction="mean")

        # Get hidden size from config - this should be known from the base model config
        # SmolVLM2 has hidden_size = 1280 for the 2.2B model
        if hasattr(config, "text_config") and hasattr(
            config.text_config, "hidden_size"
        ):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            # Fallback to common SmolVLM2 hidden size
            hidden_size = 1280
            logger.warning(
                f"Could not determine hidden_size from config, using fallback: {hidden_size}"
            )

        # Store config for [CLS] token usage
        self.use_img_cls_token = getattr(config, 'use_img_cls_token', True)
        logger.info(f"Using [CLS] token strategy: {self.use_img_cls_token}")
        
        # SmolVLM2 vision encoder output dimension (from SigLIP vision model)
        vision_hidden_size = getattr(config, 'vision_hidden_size', 1152)
        
        # Create vision projector MLP (ProAssist pattern)
        # Projects [CLS] token from vision encoder to LLM input space
        # Following ProAssist's projector: Linear(vision_dim) -> GELU -> Linear(lm_dim)
        self.vision_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        
        logger.info(f"Vision projector: {vision_hidden_size} → {hidden_size} (GELU) → {hidden_size}")

        # Create DST prediction heads in __init__ (proper PyTorch design)
        # Note: DST state updates are generated as text (JSON), not classified
        # So we only need binary decision heads here
        # These heads operate on frame embeddings (one per turn) not token embeddings
        self.speaking_decision_head = nn.Linear(hidden_size, 1)  # 1 = binary decision per frame
        self.dst_update_head = nn.Linear(hidden_size, 1)  # 1 = binary decision per frame

        # Initialize weights properly
        nn.init.xavier_uniform_(self.speaking_decision_head.weight)
        nn.init.xavier_uniform_(self.dst_update_head.weight)

        # Initialize biases
        nn.init.zeros_(self.speaking_decision_head.bias)
        nn.init.zeros_(self.dst_update_head.bias)
        
        # Freeze vision encoder (not trainable, only projector is trainable)
        for param in self.model.vision_model.parameters():
            param.requires_grad = False
        
        # Set vision encoder to eval mode for inference-only
        self.model.vision_model.eval()

    def _extract_and_project_vision_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Extract [CLS] token from vision encoder and project to LLM space (ProAssist pattern).
        
        Args:
            pixel_values: Image tensor [batch, num_tiles, 3, 384, 384] from processor
                         SmolVLM2 processor creates 17 tiles per image
        
        Returns:
            vision_embeds: Projected embeddings [batch, hidden_size]
                          One embedding per image (averaged across tiles)
        """
        batch_size = pixel_values.shape[0]
        num_tiles = pixel_values.shape[1]
        
        # Reshape: [batch, num_tiles, 3, h, w] → [batch*num_tiles, 3, h, w]
        pixel_values_flat = pixel_values.view(
            batch_size * num_tiles, *pixel_values.shape[2:]
        )
        
        # Forward through vision encoder without gradients (frozen)
        with torch.no_grad():
            vision_outputs = self.model.vision_model(
                pixel_values=pixel_values_flat,
                output_hidden_states=False
            )
        
        # Get the last hidden state [batch*num_tiles, vision_hidden_size]
        # For SigLIP-based vision encoder, use last_hidden_state (includes [CLS] token at position 0)
        if hasattr(vision_outputs, 'last_hidden_state'):
            # Extract [CLS] token (first position in each sequence)
            cls_tokens = vision_outputs.last_hidden_state[:, 0, :]  # [batch*num_tiles, vision_hidden_size]
        else:
            raise ValueError(f"Unexpected vision output type: {type(vision_outputs)}")
        
        # Reshape back: [batch*num_tiles, vision_hidden_size] → [batch, num_tiles, vision_hidden_size]
        cls_tokens = cls_tokens.view(batch_size, num_tiles, -1)
        
        # Average pool across tiles to get one representation per image
        vision_embeds = cls_tokens.mean(dim=1)  # [batch, vision_hidden_size]
        
        # Move to vision_projector device before projection (in case vision model is on different device)
        projector_device = next(self.vision_projector.parameters()).device
        vision_embeds = vision_embeds.to(projector_device)
        
        # Project vision features to LLM input space via trainable MLP
        projected = self.vision_projector(vision_embeds)  # [batch, hidden_size]
        
        return projected


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,  # Pre-computed vision features (ProAssist pattern)
        pixel_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[KV_CACHE] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with [CLS] token vision extraction and DST predictions.

        Uses ProAssist pattern: Extract [CLS] token and fuse with text representations.
        
        Args:
            image_embeds: Pre-computed vision features [batch, num_images, hidden_size]
                         If provided, skips vision encoding (following ProAssist pre-extraction)
        """
        # Handle vision features (either pre-computed or extract on-the-fly)
        vision_embeds = None
        
        if image_embeds is not None:
            # Use pre-computed vision features from pickle file
            # Format: [total_frames_across_conversation, 2048]
            # Frames are in the same order as image tokens in the text
            vision_embeds = image_embeds.to(dtype=torch.float32)
            
            # Get LLM hidden size for projection
            text_model = self.model.text_model
            lm_hidden_size = text_model.config.hidden_size
            
            # Project vision embeddings from 2048 to LLM hidden size
            # Move to projector device (in case it's different)
            projector_device = next(self.vision_projector.parameters()).device
            vision_embeds = vision_embeds.to(projector_device)
            
            # Ensure gradients flow through the projector
            # Even though input embeddings are frozen, the projector is trainable
            vision_embeds = self.vision_projector(vision_embeds)  # [total_frames, hidden_size]
            vision_embeds = vision_embeds.to(input_ids.device)
            
            logger.debug(f"✓ Using precomputed image_embeds, projected shape: {vision_embeds.shape}")
        elif pixel_values is not None and self.use_img_cls_token:
            logger.debug(f"Using on-the-fly vision encoding from pixel_values (shape: {pixel_values.shape})")
            vision_embeds = self._extract_and_project_vision_features(pixel_values)
            # vision_embeds shape: [batch, hidden_size]
        else:
            logger.debug("No vision input provided (image_embeds=None, pixel_values=None)")
        
        # Get text embeddings
        text_model = self.model.text_model
        input_embeds = text_model.embed_tokens(input_ids)  # [batch, seq_len, hidden_size]
        
        # Replace image token embeddings with vision features
        if vision_embeds is not None:
            image_token_id = getattr(self.config, 'img_token_id', None)
            if image_token_id is not None:
                # Find image token positions
                image_mask = input_ids == image_token_id
                
                # Ensure vision_embeds is on the same device as input_embeds
                vision_embeds = vision_embeds.to(input_embeds.device)
                
                # Replace image tokens with precomputed vision embeddings sequentially
                # vision_embeds: [total_frames, hidden_size] (in order of appearance in conversation)
                frame_idx = 0
                for batch_idx in range(input_ids.shape[0]):
                    image_positions = torch.where(image_mask[batch_idx])[0]
                    for pos in image_positions:
                        if frame_idx < vision_embeds.shape[0]:
                            input_embeds[batch_idx, pos] = vision_embeds[frame_idx]
                            frame_idx += 1
        
        # Process through text model with fused embeddings
        text_outputs = text_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_hidden_states=True,
            return_dict=True,
            use_cache=kwargs.get('use_cache', False),
        )
        
        # Get logits from LM head
        last_hidden_state = text_outputs.hidden_states[-1] if text_outputs.hidden_states else text_outputs.last_hidden_state
        lm_logits = self.lm_head(last_hidden_state)
        
        # Prepare output dictionary
        outputs = {
            "logits": lm_logits,
            "hidden_states": text_outputs.hidden_states,
            "past_key_values": text_outputs.past_key_values,
        }
        
        # Compute language modeling loss if labels provided
        speaking_gen_loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(lm_logits.device)
            # Mask out ignore_id (-100) labels
            valid_mask = shift_labels != -100
            if valid_mask.any():
                shift_logits_valid = shift_logits[valid_mask.unsqueeze(-1).expand_as(shift_logits)].view(-1, shift_logits.size(-1))
                shift_labels_valid = shift_labels[valid_mask]
                speaking_gen_loss = loss_fct(shift_logits_valid, shift_labels_valid)
        
        # Compute DST generation loss if dst_gen_labels provided
        dst_gen_loss = None
        if kwargs.get('dst_gen_labels') is not None:
            loss_fct = nn.CrossEntropyLoss()
            dst_gen_labels = kwargs.get('dst_gen_labels').to(lm_logits.device)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = dst_gen_labels[..., 1:].contiguous().to(lm_logits.device)
            # Mask out ignore_id (-100) labels
            valid_mask = shift_labels != -100
            if valid_mask.any():
                shift_logits_valid = shift_logits[valid_mask.unsqueeze(-1).expand_as(shift_logits)].view(-1, shift_logits.size(-1))
                shift_labels_valid = shift_labels[valid_mask]
                dst_gen_loss = loss_fct(shift_logits_valid, shift_labels_valid)
        
        outputs["speaking_gen_loss"] = speaking_gen_loss
        outputs["dst_gen_loss"] = dst_gen_loss
        
        # Get token-level DST predictions (operating on fused text+vision hidden states)
        speaking_logits = self.speaking_decision_head(last_hidden_state)  # [batch, seq_len, 1]
        dst_update_logits = self.dst_update_head(last_hidden_state)  # [batch, seq_len, 1]

        # Compute additional losses if labels are provided
        speaking_binary_loss = None
        dst_binary_loss = None
        
        if kwargs.get('speaking_labels') is not None or kwargs.get('dst_update_labels') is not None:
            bce_loss_fct = nn.BCEWithLogitsLoss(reduction='mean')
            
            # Speaking decision loss (binary: should assistant speak?)
            if kwargs.get('speaking_labels') is not None:
                speaking_labels = kwargs.get('speaking_labels').to(speaking_logits.device)
                speaking_logits_flat = speaking_logits.squeeze(-1).reshape(-1)
                speaking_labels_flat = speaking_labels.reshape(-1).float()
                valid_mask = speaking_labels_flat != -100
                
                if valid_mask.any():
                    speaking_binary_loss = bce_loss_fct(
                        speaking_logits_flat[valid_mask],
                        speaking_labels_flat[valid_mask]
                    )
            
            # DST binary loss (binary: should DST be updated?)
            if kwargs.get('dst_update_labels') is not None:
                dst_update_labels = kwargs.get('dst_update_labels').to(dst_update_logits.device)
                dst_binary_logits_flat = dst_update_logits.squeeze(-1).reshape(-1)
                dst_binary_labels_flat = dst_update_labels.reshape(-1).float()
                valid_mask = dst_binary_labels_flat != -100
                
                if valid_mask.any():
                    dst_binary_loss = bce_loss_fct(
                        dst_binary_logits_flat[valid_mask],
                        dst_binary_labels_flat[valid_mask]
                    )
        
        # Compute accuracies and F1 scores (for logging)
        speaking_accuracy = None
        speaking_precision = None
        speaking_recall = None
        speaking_f1 = None
        
        dst_accuracy = None
        dst_precision = None
        dst_recall = None
        dst_f1 = None
        
        # Speaking metrics
        if kwargs.get('speaking_labels') is not None:
            speaking_labels = kwargs.get('speaking_labels').to(speaking_logits.device)
            speaking_logits_flat = speaking_logits.squeeze(-1).reshape(-1)
            speaking_labels_flat = speaking_labels.reshape(-1).float()
            valid_mask = speaking_labels_flat != -100
            
            if valid_mask.any():
                # Predictions: > 0 means sigmoid > 0.5
                preds = (speaking_logits_flat[valid_mask] > 0).float()
                targets = speaking_labels_flat[valid_mask]
                speaking_accuracy = (preds == targets).float().mean()
                
                # Calculate Precision, Recall, F1 using sklearn
                # Move to CPU for sklearn
                preds_cpu = preds.cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                
                # average='binary' for binary classification
                # zero_division=0 to handle cases with no positive predictions
                p, r, f1, _ = precision_recall_fscore_support(
                    targets_cpu, preds_cpu, average='binary', zero_division=0
                )
                speaking_precision = torch.tensor(p)
                speaking_recall = torch.tensor(r)
                speaking_f1 = torch.tensor(f1)

        # DST metrics
        if kwargs.get('dst_update_labels') is not None:
            dst_update_labels = kwargs.get('dst_update_labels').to(dst_update_logits.device)
            dst_binary_logits_flat = dst_update_logits.squeeze(-1).reshape(-1)
            dst_binary_labels_flat = dst_update_labels.reshape(-1).float()
            valid_mask = dst_binary_labels_flat != -100
            
            if valid_mask.any():
                # Predictions: > 0 means sigmoid > 0.5
                preds = (dst_binary_logits_flat[valid_mask] > 0).float()
                targets = dst_binary_labels_flat[valid_mask]
                dst_accuracy = (preds == targets).float().mean()
                
                # Calculate Precision, Recall, F1 using sklearn
                preds_cpu = preds.cpu().numpy()
                targets_cpu = targets.cpu().numpy()
                
                p, r, f1, _ = precision_recall_fscore_support(
                    targets_cpu, preds_cpu, average='binary', zero_division=0
                )
                dst_precision = torch.tensor(p)
                dst_recall = torch.tensor(r)
                dst_f1 = torch.tensor(f1)
        
        # Combine all 4 losses with configured weights
        loss_components = []
        
        # Speaking generation loss (LM loss for assistant turns)
        if outputs["speaking_gen_loss"] is not None:
            speaking_gen_weight = getattr(self.config, 'speaking_gen_weight', 1.0)
            loss_components.append(speaking_gen_weight * outputs["speaking_gen_loss"])
            logger.debug(f"✓ Added speaking_gen_loss: {outputs['speaking_gen_loss'].item():.4f}")
        
        # Speaking binary loss (decision: should speak?)
        if speaking_binary_loss is not None:
            speaking_binary_weight = getattr(self.config, 'speaking_binary_weight', 1.0)
            loss_components.append(speaking_binary_weight * speaking_binary_loss)
            logger.debug(f"✓ Added speaking_binary_loss: {speaking_binary_loss.item():.4f}")
        
        # DST generation loss (LM loss for DST_UPDATE turns)
        if outputs["dst_gen_loss"] is not None:
            dst_gen_weight = getattr(self.config, 'dst_gen_weight', 1.0)
            loss_components.append(dst_gen_weight * outputs["dst_gen_loss"])
            logger.debug(f"✓ Added dst_gen_loss: {outputs['dst_gen_loss'].item():.4f}")
        
        # DST binary loss (decision: should update DST?)
        if dst_binary_loss is not None:
            dst_binary_weight = getattr(self.config, 'dst_binary_weight', 1.0)
            loss_components.append(dst_binary_weight * dst_binary_loss)
            logger.debug(f"✓ Added dst_binary_loss: {dst_binary_loss.item():.4f}")
        
        # Combine all losses
        if loss_components:
            outputs["loss"] = sum(loss_components)
            logger.debug(f"✓ Combined loss from {len(loss_components)} components: {outputs['loss'].item():.4f}")
        else:
            # If no losses were computed, create a zero loss that maintains gradients
            # This is important for DDP to work correctly
            logger.warning("⚠️ No valid losses computed! All label tensors may be all -100")
            logger.warning(f"  speaking_gen_loss: {outputs['speaking_gen_loss']}")
            logger.warning(f"  speaking_binary_loss: {speaking_binary_loss}")
            logger.warning(f"  dst_gen_loss: {outputs['dst_gen_loss']}")
            logger.warning(f"  dst_binary_loss: {dst_binary_loss}")
            # Use the logits to create a zero loss that has requires_grad=True
            outputs["loss"] = (lm_logits * 0.0).sum()

        return {
            **outputs,
            "vision_embeds": vision_embeds,
            "speaking_logits": speaking_logits,
            "dst_update_logits": dst_update_logits,
            "speaking_gen_loss": outputs.get("speaking_gen_loss"),
            "speaking_binary_loss": speaking_binary_loss,
            "dst_gen_loss": outputs.get("dst_gen_loss"),
            "dst_binary_loss": dst_binary_loss,
            "speaking_accuracy": speaking_accuracy,
            "speaking_precision": speaking_precision,
            "speaking_recall": speaking_recall,
            "speaking_f1": speaking_f1,
            "dst_accuracy": dst_accuracy,
            "dst_precision": dst_precision,
            "dst_recall": dst_recall,
            "dst_f1": dst_f1,
        }
