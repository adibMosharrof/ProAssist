"""
DST ProAct Modeling

Model classes for ProAssist-style DST architecture.
"""

import logging
from typing import Optional, Tuple
import random
import torch
import torch.nn as nn
from transformers import LlamaForCausalLM, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

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

        # Binary decision heads (only created if use_binary_decision_heads=True)
        # If False, we'll extract probabilities from LM head output instead
        if self.config.use_binary_decision_heads:
            # Speaking decision head
            if "linear" in self.config.binary_decision_head_type:
                self.speaking_decision_head = nn.Linear(lm_input_size, 1)
            else:
                self.speaking_decision_head = nn.Sequential(
                    nn.Linear(lm_input_size, lm_input_size // 2),
                    nn.GELU(),
                    nn.Linear(lm_input_size // 2, 1),
                )
            self.speaking_decision_head.to(self.device, self.dtype)

            # DST update decision head
            if "linear" in self.config.binary_decision_head_type:
                self.dst_update_head = nn.Linear(lm_input_size, 1)
            else:
                self.dst_update_head = nn.Sequential(
                    nn.Linear(lm_input_size, lm_input_size // 2),
                    nn.GELU(),
                    nn.Linear(lm_input_size // 2, 1),
                )
            self.dst_update_head.to(self.device, self.dtype)
            
            logger.info(f"✓ Initialized binary decision heads (type: {self.config.binary_decision_head_type})")
        else:
            # Token probability mode - no binary heads needed
            self.speaking_decision_head = None
            self.dst_update_head = None
            logger.info(f"✓ Using token probability approach (asst_token_id: {self.config.asst_gen_token_id}, dst_token_id: {self.config.dst_gen_token_id})")

        # Separate DST generation head (optional - controlled by config)
        # Speaking always uses lm_head
        if self.config.use_separate_generation_heads:
            vocab_size = self.config.vocab_size
            self.dst_generation_head = nn.Linear(lm_input_size, vocab_size, bias=False)

            # Initialize DST head from lm_head if it exists
            if hasattr(self, "lm_head") and self.lm_head is not None:
                logger.info("Initializing dst_generation_head from lm_head")
                self.dst_generation_head.weight.data = self.lm_head.weight.data.clone()
            else:
                logger.info("Initializing dst_generation_head with random weights")

            self.dst_generation_head.to(self.device, self.dtype)
            logger.info(f"✓ Initialized dst_generation_head (speaking uses lm_head)")
        else:
            # Single head mode - use lm_head for both tasks
            self.dst_generation_head = None
            logger.info(f"✓ Using single lm_head for both speaking and DST")

        logger.info(
            f"✓ Initialized multimodal modules ({mm_feature_size} -> {lm_input_size})"
        )

    def mm_feature_proj(self, features: torch.Tensor) -> torch.Tensor:
        """Project vision features to LLM embedding space."""
        return self.mm_projector(features)

    def _extract_token_probability(
        self, logits: torch.Tensor, token_id: Optional[int]
    ) -> Optional[torch.Tensor]:
        """Extract probability of a specific token from logits.
        
        Args:
            logits: Tensor of shape [batch, seq_len, vocab_size]
            token_id: Token ID to extract probability for
            
        Returns:
            Tensor of shape [batch, seq_len] with probabilities, or None if token_id is None
        """
        if token_id is None:
            return None
        
        # Get softmax probabilities: [batch, seq_len, vocab_size]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Extract probability of specified token: [batch, seq_len]
        return probs[..., token_id]

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
        # Do not clamp input_ids to config.vocab_size. 
        # When we add special tokens ([DST], [ASST]), their IDs > config.vocab_size.
        # The embedding layer is already resized to handle them.
        inputs_embeds = self.get_input_embeddings()(input_ids)

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
            use_dst_head: If True, use dst_generation_head; otherwise use lm_head

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
        # Speaking uses lm_head (natural language), DST uses dst_generation_head if available
        if use_dst_head:
            if self.dst_generation_head is not None:
                generation_head = self.dst_generation_head
            else:
                generation_head = self.lm_head
        else:
            # Always use lm_head for speaking
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

            # Check for EOS token using config.eos_token_id (set from tokenizer during loading)
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
        "dst_generation_head",
    ]

    def __init__(self, config: DSTProActLlamaConfig) -> None:
        super().__init__(config)
        self.mm_projector = None
        self.speaking_decision_head = None
        self.dst_update_head = None
        self.dst_generation_head = None
        logger.info("✓ DSTProActLlamaForCausalLM initialized")

    def init_multimodal_modules(self) -> None:
        """Initialize multimodal projection and decision heads."""
        self._init_multimodal_modules(
            self.config.vision_hidden_size, self.config.hidden_size
        )

    def _log_binary_decision(self, logits: torch.Tensor, decision_name: str) -> None:
        """Log binary decision metrics for positive frames.
        
        Args:
            logits: Logits from binary head for positive frames
            decision_name: Name of the decision (e.g., "SPEAKING", "DST")
        """
        if len(logits) == 0:
            return
            
        probs = torch.sigmoid(logits)
        
        # Randomly sample 1-3 frames to log
        num_to_log = min(random.randint(1, 3), len(logits))
        log_indices = random.sample(range(len(logits)), num_to_log)
        
        for idx in log_indices:
            logit_val = logits[idx].item()
            prob_val = probs[idx].item()
            logger.info(f"[TRAIN] Frame Binary Decision [{decision_name}] (positive):")
            logger.info(f"  Logit: {logit_val:.4f}")
            logger.info(f"  Sigmoid prob: {prob_val:.4f}")
            logger.info(f"  Label: 1 (should trigger {decision_name.lower()})")
            logger.info(f"  Correct: {prob_val > 0.5}")

    def _log_token_probability(self, probs: torch.Tensor, decision_name: str, token_name: str) -> None:
        """Log token probability metrics for positive frames.
        
        Args:
            probs: Probabilities for positive frames
            decision_name: Name of the decision (e.g., "SPEAKING", "DST")
            token_name: Token name (e.g., "[ASST]", "[DST]")
        """
        if len(probs) == 0:
            return
            
        num_to_log = min(3, len(probs))
        log_indices = random.sample(range(len(probs)), num_to_log)
        
        logger.info(f"[TRAIN] Frame Token Probability [{decision_name}] (positive):")
        for idx in log_indices:
            prob_val = probs[idx].item()
            logger.info(f"  {token_name} token prob: {prob_val:.4f}")
            logger.info(f"  Should {decision_name.lower()}: {prob_val > 0.5}")

    def _compute_binary_decision_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        decision_name: str,
        token_name: str,
    ) -> torch.Tensor:
        """Compute binary decision loss with optional logging.
        
        Args:
            logits: Logits or probabilities from decision module
            labels: Binary labels (0 or 1)
            decision_name: Name of decision (e.g., "SPEAKING", "DST")
            token_name: Token name for logging (e.g., "[ASST]", "[DST]")
            
        Returns:
            Loss value (tensor or float)
        """
        pos_mask = labels == 1
        neg_mask = labels == 0
        loss = 0.0
        
        # Convert token probabilities to logit space if needed
        if not self.config.use_binary_decision_heads:
            eps = 1e-7
            # Save clamped probs for logging before converting to logit space
            clamped_probs = torch.clamp(logits, eps, 1 - eps)
            logits = torch.logit(clamped_probs)  # Convert probabilities to logit space
        else:
            clamped_probs = None

        # Loss for positive frames
        if pos_mask.any():
            loss_pos = bce_loss(logits[pos_mask], labels[pos_mask].float())
            loss += loss_pos
            
            # Logging
            monitor_freq = getattr(self.config, 'monitor_log_freq', 0.001)
            if self.training and random.random() < monitor_freq:
                with torch.no_grad():
                    if self.config.use_binary_decision_heads:
                        self._log_binary_decision(logits[pos_mask], decision_name)
                    else:
                        # Log the actual clamped probabilities
                        self._log_token_probability(clamped_probs[pos_mask], decision_name, token_name)

        # Loss for negative frames
        if neg_mask.any():
            loss_neg = bce_loss(logits[neg_mask], labels[neg_mask].float())
            loss += loss_neg

        # Average to balance gradient contributions
        if pos_mask.any() and neg_mask.any():
            loss = loss / 2.0

        return loss

    def _compute_predictions_from_logits(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute predictions from logits, handling both binary heads and token probabilities.
        
        Args:
            logits: Logits or probabilities
            labels: Labels to mask
            
        Returns:
            Tuple of (predictions, targets)
        """
        mask = labels != self.config.ignore_id
        if not mask.any():
            return None, None

        if self.config.use_binary_decision_heads:
            # Binary heads: apply sigmoid
            preds = (
                torch.sigmoid(logits[mask]) > self.config.binary_threshold
            ).long()
        else:
            # Token probability: probabilities are already in [0, 1]
            preds = (logits[mask] > self.config.binary_threshold).long()

        targets = labels[mask].long()
        return preds, targets

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
        # Always need hidden states for binary heads
        output_hidden_states = True

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
            if speaking_gen_labels is not None:
                speaking_gen_labels = speaking_gen_labels[:, :max_len]
            if dst_gen_labels is not None:
                dst_gen_labels = dst_gen_labels[:, :max_len]

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

        # Binary head outputs or token probability outputs
        outputs.speaking_logits = None
        outputs.dst_update_logits = None
        
        if self.config.use_binary_decision_heads:
            # Binary head approach: separate heads for decisions
            if self.speaking_decision_head:
                outputs.speaking_logits = self.speaking_decision_head(
                    hidden_states
                ).squeeze(-1)
            if self.dst_update_head:
                outputs.dst_update_logits = self.dst_update_head(hidden_states).squeeze(-1)
        else:
            # Token probability approach: extract probabilities from LM head
            # Get logits from lm_head (we'll compute them below, so store for later)
            outputs._use_token_probability = True

        # Generation head routing: use separate heads for speaking vs DST
        # During training/eval with labels, we know which tokens belong to which task
        # During pure inference, we'll select based on binary head predictions

        if speaking_gen_labels is not None or dst_gen_labels is not None:
            # Training/Eval with labels: compute logits from both heads for loss computation
            if self.config.use_separate_generation_heads:
                # Use lm_head for speaking, dst_generation_head for DST
                speaking_logits = self.lm_head(hidden_states)
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
            
            # Extract token probabilities if using token probability approach
            if not self.config.use_binary_decision_heads:
                # Extract probability of special tokens from logits
                # speaking: probability of [ASST] token
                # dst: probability of [DST] token
                outputs.speaking_logits = self._extract_token_probability(
                    outputs.speaking_generation_logits, self.config.asst_gen_token_id
                )
                outputs.dst_update_logits = self._extract_token_probability(
                    outputs.dst_generation_logits, self.config.dst_gen_token_id
                )
        else:
            # Inference: need to determine which head to use
            # If we have binary predictions, use them; otherwise default to speaking
            if self.config.use_separate_generation_heads:
                # Use lm_head for speaking
                outputs.logits = self.lm_head(hidden_states)
            else:
                # Single head mode: use lm_head
                outputs.logits = self.lm_head(hidden_states)
            
            # Extract token probabilities if using token probability approach
            if not self.config.use_binary_decision_heads:
                outputs.speaking_logits = self._extract_token_probability(
                    outputs.logits, self.config.asst_gen_token_id
                )
                outputs.dst_update_logits = self._extract_token_probability(
                    outputs.logits, self.config.dst_gen_token_id
                )

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
            # Use views (no copy) - contiguous only when needed by loss function
            shift_labels = speaking_gen_labels[..., 1:]
            mask = shift_labels != -100
            
            # Combine with attention mask to exclude padding tokens
            if attention_mask is not None:
                # attention_mask is [batch, seq_len], shift it to match shifted labels
                shift_attention_mask = attention_mask[:, 1:].bool()
                mask = mask & shift_attention_mask

            if mask.any():
                # Only flatten/reshape what's needed for loss computation
                shift_logits = outputs.speaking_generation_logits[..., :-1, :]
                speaking_gen_loss = ce_loss(
                    shift_logits.reshape(-1, shift_logits.size(-1))[mask.reshape(-1)],
                    shift_labels.reshape(-1)[mask.reshape(-1)],
                )
                loss = speaking_gen_loss
                log_dict["speaking_gen_loss"] = speaking_gen_loss.item()
            else:
                # Even with empty mask, record a zero loss for logging
                log_dict["speaking_gen_loss"] = 0.0

        if (
            dst_gen_labels is not None
            and hasattr(outputs, "dst_generation_logits")
            and outputs.dst_generation_logits is not None
        ):
            # Shift logits and labels for causal LM
            # Use views (no copy) - contiguous only when needed by loss function
            shift_labels = dst_gen_labels[..., 1:]
            mask = shift_labels != -100
            
            # Combine with attention mask to exclude padding tokens
            if attention_mask is not None:
                # attention_mask is [batch, seq_len], shift it to match shifted labels
                shift_attention_mask = attention_mask[:, 1:].bool()
                mask = mask & shift_attention_mask

            if mask.any():
                # Only flatten/reshape what's needed for loss computation
                shift_logits = outputs.dst_generation_logits[..., :-1, :]
                
                # Compute loss (standard, memory efficient for backward)
                dst_gen_loss = ce_loss(
                    shift_logits.reshape(-1, shift_logits.size(-1))[mask.reshape(-1)],
                    shift_labels.reshape(-1)[mask.reshape(-1)],
                )
                loss = (loss + dst_gen_loss) if loss is not None else dst_gen_loss
                log_dict["dst_gen_loss"] = dst_gen_loss.item()
            else:
                # Even with empty mask, record a zero loss for logging
                log_dict["dst_gen_loss"] = 0.0
                
                # --- DEBUG LOGGING (Optimized for VRAM) ---
                # Log very rarely (controlled by config) to check generation quality
                monitor_freq = getattr(self.config, 'monitor_log_freq', 0.001)  # Default to 0.001 if not set
                if self.training and random.random() < monitor_freq:
                    with torch.no_grad():
                        # Optimize: argmax FIRST (reduce dim), then mask. 
                        # shift_logits: [B, T, V] -> [B, T]
                        all_preds = shift_logits.argmax(dim=-1)
                        
                        # Apply mask to the small tensor (for accuracy calc)
                        flat_mask = mask.view(-1)
                        flat_preds = all_preds.view(-1)[flat_mask]
                        flat_labels = shift_labels.reshape(-1)[flat_mask]
                        
                        acc = (flat_preds == flat_labels).float().mean()
                        
                        # Log basic stats
                        logger.info(f"\n[DST MONITOR] Loss: {dst_gen_loss.item():.4f} | Acc: {acc:.2%}")

                        # Check tokens if tokenizer is available
                        if hasattr(self, "debug_tokenizer") and self.debug_tokenizer:
                            try:
                                dst_id = self.debug_tokenizer.convert_tokens_to_ids("[DST]")
                                eos_id = self.debug_tokenizer.eos_token_id
                                
                                # Find a batch item that has [DST]
                                # input_ids is [B, T]
                                has_dst = (input_ids == dst_id)
                                rows, cols = torch.where(has_dst)
                                
                                if len(rows) > 0:
                                    # Pick the first occurrence
                                    b_idx = rows[0]
                                    start_idx = cols[0] # Position of [DST]
                                    
                                    # Look for EOS after [DST] in the LABELS
                                    # dst_gen_labels at [DST] pos is -100. Text starts at start_idx + 1.
                                    # Slice from start_idx to end
                                    sample_labels = dst_gen_labels[b_idx, start_idx:]
                                    
                                    # Find first EOS
                                    eos_matches = (sample_labels == eos_id).nonzero()
                                    
                                    if len(eos_matches) > 0:
                                        rel_eos_idx = eos_matches[0].item()
                                        
                                        # Extract TARGET (Labels)
                                        # from 1 to rel_eos_idx+1 (inclusive of EOS)
                                        # index 0 is -100 (DST token)
                                        target_ids = sample_labels[1 : rel_eos_idx + 1]
                                        
                                        # Extract PREDICTION (Logits)
                                        # all_preds is [B, T-1]. Match alignment.
                                        # input [DST] at 'start_idx' predicts 'start_idx' in 'shift_logits' (which maps to labels 'start_idx+1')
                                        # So preds for the sequence start at 'start_idx'.
                                        # Length is same as target_ids.
                                        pred_end_idx = start_idx + len(target_ids)
                                        if pred_end_idx <= all_preds.size(1):
                                            sample_preds = all_preds[b_idx, start_idx : pred_end_idx]
                                            
                                            # Debug: show raw tokens before decoding
                                            logger.info(f"  Target token IDs: {target_ids.tolist()}")
                                            logger.info(f"  Pred token IDs:   {sample_preds.tolist()}")
                                            
                                            # Decode with explicit handling of special tokens
                                            decoded_target = self.debug_tokenizer.decode(target_ids, skip_special_tokens=False)
                                            decoded_pred = self.debug_tokenizer.decode(sample_preds, skip_special_tokens=False)
                                            
                                            logger.info(f"  Target: {decoded_target!r}")
                                            logger.info(f"  Model : {decoded_pred!r}")
                                            
                                            if decoded_target == decoded_pred:
                                                logger.info("  ✓ Perfect Match")
                                            else:
                                                logger.info("  ✗ Mismatch")
                                        else:
                                            logger.warning("  Prediction index out of bounds (truncated sample?)")
                                    else:
                                        logger.info("  (Monitor) Found [DST] but no EOS in labels - split sequence?")
                                else:
                                    logger.info("  (Monitor) No [DST] in this batch.")

                            except Exception as e:
                                logger.warning(f"Monitor decoding failed: {e}")
                # ---------------------

        # Speaking text generation monitoring
        if (speaking_gen_labels is not None and 
            hasattr(outputs, "speaking_generation_logits") and 
            outputs.speaking_generation_logits is not None):
            monitor_freq = getattr(self.config, 'monitor_log_freq', 0.001)
            if self.training and random.random() < monitor_freq:
                with torch.no_grad():
                    # Get speaking generation loss from log_dict
                    speaking_gen_loss = log_dict.get("speaking_gen_loss", 0.0)
                    
                    # Compute accuracy for generation task
                    shift_labels = speaking_gen_labels[..., 1:]
                    mask = shift_labels != -100
                    
                    if attention_mask is not None:
                        shift_attention_mask = attention_mask[:, 1:].bool()
                        mask = mask & shift_attention_mask
                    
                    if mask.any():
                        shift_logits = outputs.speaking_generation_logits[..., :-1, :]
                        all_preds = shift_logits.argmax(dim=-1)
                        
                        flat_mask = mask.view(-1)
                        flat_preds = all_preds.view(-1)[flat_mask]
                        flat_labels = shift_labels.reshape(-1)[flat_mask]
                        
                        acc = (flat_preds == flat_labels).float().mean()
                        logger.info(f"\n[SPEAKING MONITOR] Loss: {speaking_gen_loss:.4f} | Acc: {acc:.2%}")
                        
                        # Decode and show text generation
                        if hasattr(self, "debug_tokenizer") and self.debug_tokenizer:
                            try:
                                eos_id = self.debug_tokenizer.eos_token_id
                                
                                # Extract first sample's speaking text
                                sample_labels = speaking_gen_labels[0]
                                eos_matches = (sample_labels == eos_id).nonzero()
                                
                                if len(eos_matches) > 0:
                                    end_idx = eos_matches[0].item()
                                    target_ids = sample_labels[:end_idx + 1]
                                    sample_preds = all_preds[0, :len(target_ids)]
                                    
                                    decoded_target = self.debug_tokenizer.decode(target_ids, skip_special_tokens=False)
                                    decoded_pred = self.debug_tokenizer.decode(sample_preds, skip_special_tokens=False)
                                    
                                    logger.info(f"  Target: {decoded_target!r}")
                                    logger.info(f"  Model : {decoded_pred!r}")
                                    
                                    if decoded_target == decoded_pred:
                                        logger.info("  ✓ Perfect Match")
                                    else:
                                        logger.info("  ✗ Mismatch")
                            except Exception as e:
                                logger.warning(f"Speaking monitor decoding failed: {e}")

        # Fallback to standard labels if separate labels not provided
        if loss is None and labels is not None:
            # Shift for causal LM - use views, reshape instead of flatten
            shift_logits = outputs.logits[..., :-1, :]
            shift_labels = labels[..., 1:]
            lm_loss = ce_loss(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss = lm_loss
            log_dict["lm_loss"] = lm_loss.item()

        if speaking_labels is not None and outputs.speaking_logits is not None:
            # Compute binary decision loss
            speaking_loss = self._compute_binary_decision_loss(
                outputs.speaking_logits,
                speaking_labels,
                "SPEAKING",
                "[ASST]",
            )

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

            # Store predictions and targets for metric computation in trainer
            preds, targets = self._compute_predictions_from_logits(
                outputs.speaking_logits, speaking_labels
            )
            if preds is not None:
                log_dict["speaking_preds"] = preds
                log_dict["speaking_targets"] = targets

        if dst_update_labels is not None and outputs.dst_update_logits is not None:
            # Compute binary decision loss
            dst_loss = self._compute_binary_decision_loss(
                outputs.dst_update_logits,
                dst_update_labels,
                "DST",
                "[DST]",
            )

            log_dict["dst_binary_loss"] = (
                dst_loss.item() if isinstance(dst_loss, torch.Tensor) else dst_loss
            )
            loss = (
                (loss + dst_loss * self.config.binary_loss_weight)
                if loss is not None
                else (dst_loss * self.config.binary_loss_weight)
            )

            # Store predictions and targets for metric computation in trainer
            preds, targets = self._compute_predictions_from_logits(
                outputs.dst_update_logits, dst_update_labels
            )
            if preds is not None:
                log_dict["dst_preds"] = preds
                log_dict["dst_targets"] = targets

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
