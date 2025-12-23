"""
DST ProAct Modeling

Model classes for ProAssist-style DST architecture.
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from prospect.models.dst_proact.configuration import DSTProActLlamaConfig

logger = logging.getLogger(__name__)

KV_CACHE = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]
ce_loss = nn.functional.cross_entropy
bce_loss = nn.functional.binary_cross_entropy_with_logits


class DSTProActModelMixin(AutoModelForCausalLM):
    """Mixin adding multimodal and DST capabilities to base LLM."""

    def _init_multimodal_modules(
        self, mm_feature_size: int, lm_input_size: int
    ) -> None:
        """Initialize projector and decision heads."""
        self.mm_projector = nn.Sequential(
            nn.Linear(mm_feature_size, lm_input_size, bias=True),
            nn.GELU(),
            nn.Linear(lm_input_size, lm_input_size, bias=True),
        )
        self.mm_projector.to(self.device, self.dtype)
<<<<<<< HEAD

        # Speaking decision head
        self.speaking_decision_head = None
        if self.config.use_speaking_decision_head:
=======
        
        # Binary decision heads and separate generation heads (controlled by use_separate_generation_heads)
        if self.config.use_separate_generation_heads:
            # Speaking decision head
>>>>>>> 1b884d44130d507cba92db0474451da5ab992235
            if "linear" in self.config.binary_decision_head_type:
                self.speaking_decision_head = nn.Linear(lm_input_size, 1)
            else:
                self.speaking_decision_head = nn.Sequential(
                    nn.Linear(lm_input_size, lm_input_size // 2),
                    nn.GELU(),
                    nn.Linear(lm_input_size // 2, 1),
                )
            self.speaking_decision_head.to(self.device, self.dtype)
<<<<<<< HEAD

        # DST update decision head
        self.dst_update_head = None
        if self.config.use_dst_update_head:
=======
            
            # DST update decision head
>>>>>>> 1b884d44130d507cba92db0474451da5ab992235
            if "linear" in self.config.binary_decision_head_type:
                self.dst_update_head = nn.Linear(lm_input_size, 1)
            else:
                self.dst_update_head = nn.Sequential(
                    nn.Linear(lm_input_size, lm_input_size // 2),
                    nn.GELU(),
                    nn.Linear(lm_input_size // 2, 1),
                )
            self.dst_update_head.to(self.device, self.dtype)
<<<<<<< HEAD

=======
        else:
            # Single head mode - no binary decision heads
            self.speaking_decision_head = None
            self.dst_update_head = None
        
>>>>>>> 1b884d44130d507cba92db0474451da5ab992235
        # Separate generation heads for speaking and DST (optional - controlled by config)
        if self.config.use_separate_generation_heads:
            vocab_size = self.config.vocab_size

            self.speaking_generation_head = nn.Linear(
                lm_input_size, vocab_size, bias=False
            )
            self.dst_generation_head = nn.Linear(lm_input_size, vocab_size, bias=False)

            # Copy weights from original lm_head if it exists
            if hasattr(self, "lm_head") and self.lm_head is not None:
                logger.info("Initializing separate generation heads from lm_head")
                self.speaking_generation_head.weight.data = (
                    self.lm_head.weight.data.clone()
                )
                self.dst_generation_head.weight.data = self.lm_head.weight.data.clone()
            else:
                logger.info(
                    "Initializing separate generation heads with random weights"
                )

            self.speaking_generation_head.to(self.device, self.dtype)
            self.dst_generation_head.to(self.device, self.dtype)

            logger.info(
                f"✓ Initialized separate generation heads: speaking_generation_head, dst_generation_head"
            )
        else:
            # Single head mode - use existing lm_head
            self.speaking_generation_head = None
            self.dst_generation_head = None
            logger.info(f"✓ Using single lm_head for generation")

        logger.info(
            f"✓ Initialized multimodal modules ({mm_feature_size} -> {lm_input_size})"
        )

    def mm_feature_proj(self, features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space."""
        return self.mm_projector(features)

    def joint_embed(
        self,
        input_ids: torch.Tensor,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Create joint embeddings from text and image.

        Args:
            input_ids: [batch_size, seq_len]
            image_embeds: Either:
                - Tensor of shape [batch_size, num_frames, hidden_size]
                - List of tensors, each [num_frames_i, hidden_size]
        """
        clamped_input_ids = input_ids.clamp(max=self.config.vocab_size - 1)
        inputs_embeds = self.get_input_embeddings()(clamped_input_ids).clone()

        # Convert to model dtype (bfloat16) for mixed precision training
        # Embedding lookup is done in float32, but all subsequent ops use bfloat16
        inputs_embeds = inputs_embeds.to(torch.bfloat16)

        if image_embeds is None:
            return inputs_embeds

        # Handle list of tensors (variable-length embeddings per sample)
        if isinstance(image_embeds, list):
            # Process each sample in the batch
            img_token_id = self.config.img_token_id
            if img_token_id is None:
                raise ValueError("img_token_id not set in config")

            for batch_idx, sample_embeds in enumerate(image_embeds):
                # Project embeddings for this sample (already bfloat16 from collator)
                projected = self.mm_feature_proj(
                    sample_embeds.to(self.dtype)
                )  # [num_frames, hidden_size]

                # Find image token positions for this sample
                img_positions = (input_ids[batch_idx] == img_token_id).nonzero(
                    as_tuple=True
                )[0]

                # Replace image tokens with projected embeddings
                if len(img_positions) != len(projected):
                    raise ValueError(
                        f"Mismatch: {len(img_positions)} image tokens but {len(projected)} embeddings"
                    )
                inputs_embeds[batch_idx, img_positions] = projected

            return inputs_embeds

        # Handle single tensor (original behavior)
        projected_embeds = self.mm_feature_proj(image_embeds.to(self.dtype))

        if projected_embeds.dim() == 3:
            projected_embeds = projected_embeds.flatten(0, 1)

        img_token_id = self.config.img_token_id
        if img_token_id is None:
            raise ValueError("img_token_id not set in config")

        inputs_embeds[input_ids == img_token_id] = projected_embeds
        return inputs_embeds

    @torch.no_grad()
    def fast_greedy_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[KV_CACHE] = None,
        max_length: int = 100,
        drop_generated_kv_cache: bool = False,
        output_hidden_states: bool = False,
        use_dst_head: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, KV_CACHE, Optional[CausalLMOutputWithPast]]:
        """Fast greedy generation with KV cache (ProAssist pattern).

        Args:
            use_dst_head: If True, use dst_generation_head; otherwise use speaking_generation_head

        Returns:
            - generated token ids
            - updated KV cache
            - first step outputs (contains binary head logits if output_hidden_states=True)
        """
        if (
            not hasattr(self, "inplace_output_ids")
            or self.inplace_output_ids.shape[1] < max_length
        ):
            self.inplace_output_ids = torch.full(
                (1, max_length), -100, dtype=torch.long, device=self.device
            )

        past_key_values_to_return = past_key_values
        token_idx = 0
        first_outputs = None

        # Select which generation head to use
        if use_dst_head:
            if self.dst_generation_head is not None:
                generation_head = self.dst_generation_head
            else:
                generation_head = self.lm_head
        else:
            if self.speaking_generation_head is not None:
                generation_head = self.speaking_generation_head
            else:
                generation_head = self.lm_head

        for i in range(max_length):
            # Get hidden states from base model
            base_outputs = self.model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
                return_dict=True,
            )

            hidden_states = (
                base_outputs.hidden_states[-1]
                if base_outputs.hidden_states
                else base_outputs[0]
            )
            past_key_values = base_outputs.past_key_values

            # Apply selected generation head
            logits = generation_head(hidden_states)

            # Binary heads (only on first step if requested)
            if i == 0 and output_hidden_states:
                first_outputs = {
                    "logits": logits,
                    "past_key_values": past_key_values,
                    "hidden_states": base_outputs.hidden_states,
                }
                if self.speaking_decision_head:
                    first_outputs["speaking_logits"] = self.speaking_decision_head(
                        hidden_states
                    ).squeeze(-1)
                if self.dst_update_head:
                    first_outputs["dst_update_logits"] = self.dst_update_head(
                        hidden_states
                    ).squeeze(-1)
                past_key_values_to_return = past_key_values

            new_token_id = logits[:, -1:].argmax(dim=-1)
            self.inplace_output_ids[:, i] = new_token_id
            token_idx = i

            if new_token_id.item() == self.config.eos_token_id:
                break

            inputs_embeds = self.get_input_embeddings()(new_token_id)

        if not drop_generated_kv_cache:
            past_key_values_to_return = past_key_values

        return (
            self.inplace_output_ids[:, : token_idx + 1],
            past_key_values_to_return,
            first_outputs,
        )


class DSTProActLlamaForCausalLM(LlamaForCausalLM, DSTProActModelMixin):
    """DST-extended Llama model with multimodal support."""

    config_class = DSTProActLlamaConfig
    _keys_to_ignore_on_load_missing = [
        "mm_projector",
        "speaking_decision_head",
        "dst_update_head",
        "speaking_generation_head",
        "dst_generation_head",
    ]

    def __init__(self, config: DSTProActLlamaConfig) -> None:
        super().__init__(config)
        self.mm_projector = None
        self.speaking_decision_head = None
        self.dst_update_head = None
        self.speaking_generation_head = None
        self.dst_generation_head = None
        logger.info("✓ DSTProActLlamaForCausalLM initialized")

    def init_multimodal_modules(self) -> None:
        """Initialize multimodal projection and decision heads."""
        self._init_multimodal_modules(
            self.config.vision_hidden_size, self.config.hidden_size
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[KV_CACHE] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        speaking_gen_labels: Optional[torch.LongTensor] = None,
        dst_gen_labels: Optional[torch.LongTensor] = None,
        speaking_labels: Optional[torch.BoolTensor] = None,
        dst_update_labels: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        """Forward pass with multimodal fusion and binary heads."""
<<<<<<< HEAD
        output_hidden_states = (
            self.config.use_speaking_decision_head
            or self.config.use_dst_update_head
            or output_hidden_states
        )

=======
        # Always need hidden states for binary heads
        output_hidden_states = True
        
>>>>>>> 1b884d44130d507cba92db0474451da5ab992235
        if inputs_embeds is None:
            inputs_embeds = self.joint_embed(input_ids, image_embeds)

        # Truncate during training
        if (
            self.training
            and self.config.max_seq_len > 0
            and inputs_embeds.shape[1] > self.config.max_seq_len
        ):
            max_len = self.config.max_seq_len
            inputs_embeds = inputs_embeds[:, :max_len]
            if labels is not None:
                labels = labels[:, :max_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_len]
            if speaking_labels is not None:
                speaking_labels = speaking_labels[:, :max_len]
            if dst_update_labels is not None:
                dst_update_labels = dst_update_labels[:, :max_len]

        # Call base model without lm_head (get hidden states only)
        base_outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache if not self.training else False,
            output_attentions=output_attentions,
            output_hidden_states=True,  # Always need hidden states for separate heads
            return_dict=True,
        )

        hidden_states = (
            base_outputs.hidden_states[-1]
            if base_outputs.hidden_states
            else base_outputs[0]
        )

        # Create outputs object to populate
        outputs = type("ModelOutput", (), {})()
        outputs.past_key_values = base_outputs.past_key_values
        outputs.hidden_states = (
            base_outputs.hidden_states if output_hidden_states else None
        )
        outputs.attentions = base_outputs.attentions if output_attentions else None

        # Binary head outputs
        outputs.speaking_logits = None
        outputs.dst_update_logits = None
        if self.speaking_decision_head:
            outputs.speaking_logits = self.speaking_decision_head(
                hidden_states
            ).squeeze(-1)
        if self.dst_update_head:
            outputs.dst_update_logits = self.dst_update_head(hidden_states).squeeze(-1)

        # Generation head routing: use separate heads for speaking vs DST
        # During training, we know which tokens belong to which task from labels
        # During inference, we'll select based on binary head predictions

        if self.training and (
            speaking_gen_labels is not None or dst_gen_labels is not None
        ):
            # Training: compute logits from both heads (if separate heads enabled) or single head
            if self.config.use_separate_generation_heads:
                speaking_logits = self.speaking_generation_head(hidden_states)
                dst_logits = self.dst_generation_head(hidden_states)
                # Store both for loss computation
                outputs.speaking_generation_logits = speaking_logits
                outputs.dst_generation_logits = dst_logits
                # Keep unified logits for backward compatibility (use speaking by default)
                outputs.logits = speaking_logits
            else:
                # Single head mode: use lm_head for both tasks
                unified_logits = self.lm_head(hidden_states)
                outputs.speaking_generation_logits = unified_logits
                outputs.dst_generation_logits = unified_logits
                outputs.logits = unified_logits
        else:
            # Inference: need to determine which head to use
            # If we have binary predictions, use them; otherwise default to speaking
            # For now, use speaking head by default (will be refined in fast_greedy_generate)
            if self.speaking_generation_head:
                outputs.logits = self.speaking_generation_head(hidden_states)
            else:
                # Fallback to original lm_head if separate heads not initialized
                outputs.logits = self.lm_head(hidden_states)

        # Compute losses
        loss = None
        log_dict = {}

        # Handle separate generation labels for DST and speaking
        # Use separate generation heads for each task
        # In causal LM, logits at position i predict token at position i+1
        # So we need to shift: use logits[:-1] to predict labels[1:]
        if (
            speaking_gen_labels is not None
            and hasattr(outputs, "speaking_generation_logits")
            and outputs.speaking_generation_logits is not None
        ):
            # Shift logits and labels for causal LM
            shift_logits = outputs.speaking_generation_logits[..., :-1, :].contiguous()
            shift_labels = speaking_gen_labels[..., 1:].contiguous()

            # Compute LM loss for assistant responses using speaking generation head
            mask = shift_labels != -100
            if mask.any():
                speaking_gen_loss = ce_loss(
                    outputs.logits[..., :-1, :][mask],
                    speaking_gen_labels[..., 1:][mask],
                )
                loss = speaking_gen_loss
                log_dict["speaking_gen_loss"] = speaking_gen_loss.item()

        if (
            dst_gen_labels is not None
            and hasattr(outputs, "dst_generation_logits")
            and outputs.dst_generation_logits is not None
        ):
            # Shift logits and labels for causal LM
            shift_logits = outputs.dst_generation_logits[..., :-1, :].contiguous()
            shift_labels = dst_gen_labels[..., 1:].contiguous()

            # Compute LM loss for DST updates using DST generation head
            mask = shift_labels != -100
            if mask.any():
                dst_gen_loss = ce_loss(
                    outputs.logits[..., :-1, :][mask], dst_gen_labels[..., 1:][mask]
                )
                loss = (loss + dst_gen_loss) if loss is not None else dst_gen_loss
                log_dict["dst_gen_loss"] = dst_gen_loss.item()

        # Fallback to standard labels if separate labels not provided
        if loss is None and labels is not None:
            # Shift for causal LM
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            lm_loss = ce_loss(shift_logits.flatten(0, 1), shift_labels.flatten())
            loss = lm_loss
            log_dict["lm_loss"] = lm_loss.item()

        if speaking_labels is not None and self.speaking_decision_head:
            # Follow ProAssist's approach: separate loss for positive and negative frames
            pos_mask = speaking_labels == 1
            neg_mask = speaking_labels == 0

            speaking_loss = 0.0

            # Loss for positive frames (should speak)
            if pos_mask.any():
                speaking_loss_pos = bce_loss(
                    outputs.speaking_logits[pos_mask], speaking_labels[pos_mask].float()
                )
                speaking_loss += speaking_loss_pos

            # Loss for negative frames (should not speak)
            if neg_mask.any():
                speaking_loss_neg = bce_loss(
                    outputs.speaking_logits[neg_mask], speaking_labels[neg_mask].float()
                )
                speaking_loss += speaking_loss_neg

            # Average to balance gradient contributions
            if pos_mask.any() and neg_mask.any():
                speaking_loss = speaking_loss / 2.0

            log_dict["speaking_binary_loss"] = (
                speaking_loss.item()
                if isinstance(speaking_loss, torch.Tensor)
                else speaking_loss
            )
            loss = (
                (loss + speaking_loss * self.config.binary_loss_weight)
                if loss is not None
                else (speaking_loss * self.config.binary_loss_weight)
            )

            # Compute metrics for speaking decision using sklearn
            mask = speaking_labels != self.config.ignore_id
            if mask.any():
                preds = (
                    (torch.sigmoid(outputs.speaking_logits[mask]) > 0.5)
                    .long()
                    .cpu()
                    .numpy()
                )
                targets = speaking_labels[mask].long().cpu().numpy()

                # Balanced accuracy
                log_dict["speaking_balanced_accuracy"] = float(
                    balanced_accuracy_score(targets, preds)
                )

                # Precision, Recall, F1 (zero_division=0 handles edge cases)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, preds, average="binary", zero_division=0
                )
                log_dict["speaking_precision"] = float(precision)
                log_dict["speaking_recall"] = float(recall)
                log_dict["speaking_f1"] = float(f1)

        if dst_update_labels is not None and self.dst_update_head:
            # Follow ProAssist's approach: separate loss for positive and negative frames
            pos_mask = dst_update_labels == 1
            neg_mask = dst_update_labels == 0

            dst_loss = 0.0

            # Loss for positive frames (should update DST)
            if pos_mask.any():
                dst_loss_pos = bce_loss(
                    outputs.dst_update_logits[pos_mask],
                    dst_update_labels[pos_mask].float(),
                )
                dst_loss += dst_loss_pos

            # Loss for negative frames (should not update DST)
            if neg_mask.any():
                dst_loss_neg = bce_loss(
                    outputs.dst_update_logits[neg_mask],
                    dst_update_labels[neg_mask].float(),
                )
                dst_loss += dst_loss_neg

            # Average to balance gradient contributions
            if pos_mask.any() and neg_mask.any():
                dst_loss = dst_loss / 2.0

            log_dict["dst_binary_loss"] = (
                dst_loss.item() if isinstance(dst_loss, torch.Tensor) else dst_loss
            )
            loss = (
                (loss + dst_loss * self.config.binary_loss_weight)
                if loss is not None
                else (dst_loss * self.config.binary_loss_weight)
            )

            # Compute metrics for DST update decision using sklearn
            mask = dst_update_labels != self.config.ignore_id
            if mask.any():
                preds = (
                    (torch.sigmoid(outputs.dst_update_logits[mask]) > 0.5)
                    .long()
                    .cpu()
                    .numpy()
                )
                targets = dst_update_labels[mask].long().cpu().numpy()

                # Balanced accuracy
                log_dict["dst_balanced_accuracy"] = float(
                    balanced_accuracy_score(targets, preds)
                )

                # Precision, Recall, F1 (zero_division=0 handles edge cases)
                precision, recall, f1, _ = precision_recall_fscore_support(
                    targets, preds, average="binary", zero_division=0
                )
                log_dict["dst_precision"] = float(precision)
                log_dict["dst_recall"] = float(recall)
                log_dict["dst_f1"] = float(f1)

        # Convert output to dict and add metrics directly (following DST SmolVLM pattern)
        output_dict = {
            "loss": loss,
            "logits": outputs.logits,
            "past_key_values": outputs.past_key_values,
            "hidden_states": outputs.hidden_states,
            "attentions": outputs.attentions,
            "speaking_logits": (
                outputs.speaking_logits if hasattr(outputs, "speaking_logits") else None
            ),
            "dst_update_logits": (
                outputs.dst_update_logits
                if hasattr(outputs, "dst_update_logits")
                else None
            ),
        }

        # Add all metrics directly to the output dict
        output_dict.update(log_dict)

        return output_dict


def trim_past_key_values(
    past_key_values: KV_CACHE, start: int, stop: int, batch_idx: int = -1
) -> KV_CACHE:
    """Select a slice of past key values."""
    if batch_idx == -1:
        return tuple(
            [(k[:, :, start:stop], v[:, :, start:stop]) for k, v in past_key_values]
        )
    return tuple(
        [
            (
                k[batch_idx : batch_idx + 1, :, start:stop],
                v[batch_idx : batch_idx + 1, :, start:stop],
            )
            for k, v in past_key_values
        ]
    )
