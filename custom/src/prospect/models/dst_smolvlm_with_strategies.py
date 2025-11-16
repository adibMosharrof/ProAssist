"""
DST SmolVLM with Multi-Task Learning

Extends SmolVLMWithStrategies (not SmolVLMForConditionalGeneration directly)
to include Dialog State Tracking (DST) prediction heads for multi-task learning
with speaking decisions, DST update decisions, and DST state updates.

This inheritance approach avoids code duplication and leverages the proven
functionality from SmolVLMWithStrategies (tested with VLM stream runner).
"""

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    to include 4 prediction heads:
    1. Speaking Decision (Binary: should I speak?)
    2. DST Update Decision (Binary: should I update DST?)
    3. Response Generation (Text output - inherited from SmolVLMWithStrategies)
    4. DST State Update (State classification)

    **Architecture:**
    ```
    Video Frames + Dialog History + Current DST → SmolVLM2 (VLM-based)
                                                  ↓
                                            [4 Training Heads]
                                                  ↓
        ┌─────────────┬─────────────┬─────────────┬─────────────┐
        │             │             │             │             │
        │  Speaking   │    DST      │  Response   │    DST      │
        │  Decision   │  Update     │ Generation  │    State    │
        │  (Binary)   │ Decision    │   (Text)    │  Update     │
        │             │  (Binary)   │             │ (States)    │
        └─────────────┴─────────────┴─────────────┴─────────────┘
    ```

    **Training**: All 4 heads are trained simultaneously with different loss functions.
    **Inference**: Binary decisions determine which action outputs to use.

    **Inheritance Benefits**:
    - Inherits proven joint_embed(), fast_greedy_generate() from SmolVLMWithStrategies
    - Avoids code duplication and leverages tested functionality
    - Only adds DST-specific multi-task learning heads
    """

    def __init__(self, config):
        """Initialize DST SmolVLM with multi-task heads"""
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

        # Create DST prediction heads in __init__ (proper PyTorch design)
        # These are now registered parameters that will be saved/loaded and optimized
        self.speaking_decision_head = nn.Linear(hidden_size, 2)
        self.dst_update_head = nn.Linear(hidden_size, 2)
        self.dst_state_head = nn.Linear(hidden_size, self.config.num_dst_states)

        # Initialize weights properly
        nn.init.xavier_uniform_(self.speaking_decision_head.weight)
        nn.init.xavier_uniform_(self.dst_update_head.weight)
        nn.init.xavier_uniform_(self.dst_state_head.weight)

        # Initialize biases
        nn.init.zeros_(self.speaking_decision_head.bias)
        nn.init.zeros_(self.dst_update_head.bias)
        nn.init.zeros_(self.dst_state_head.bias)

        logger.info(
            "DST SmolVLMWithStrategies: Extended SmolVLMWithStrategies with DST heads"
        )
        logger.info(f"DST heads initialized with hidden_size: {hidden_size}")
        logger.info(f"Focal loss criterion: alpha=0.25, gamma=2")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        past_key_values: Optional[KV_CACHE] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass with DST predictions

        Returns:
            Dict containing:
            - last_hidden_state: Base model hidden states (inherited)
            - logits: Base model logits (for text generation - inherited)
            - speaking_logits: Speaking decision logits (DST-specific)
            - dst_update_logits: DST update decision logits (DST-specific)
            - dst_state_logits: DST state prediction logits (DST-specific)
        """
        # Get base model outputs (including text generation logits)
        # This calls SmolVLMWithStrategies.forward() which calls SmolVLMForConditionalGeneration.forward()
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,  # Force hidden states output for DST heads
            **kwargs,
        )

        # Get last hidden state for DST heads
        # Use the correct attribute name for SmolVLMCausalLMOutputWithPast
        last_hidden_state = getattr(outputs, "last_hidden_state", None)

        # If last_hidden_state is not available, try other possible attributes
        if last_hidden_state is None:
            last_hidden_state = getattr(outputs, "hidden_states", None)
            if last_hidden_state and isinstance(last_hidden_state, tuple):
                last_hidden_state = last_hidden_state[
                    -1
                ]  # Get last layer's hidden states

        # If still not found, fall back to using hidden states directly
        if last_hidden_state is None:
            raise AttributeError(
                f"Cannot find hidden states in model output. Available attributes: {list(outputs.keys()) if hasattr(outputs, 'keys') else dir(outputs)}"
            )

        # Get DST predictions using pre-created heads (no dynamic creation!)
        speaking_logits = self.speaking_decision_head(last_hidden_state)
        dst_update_logits = self.dst_update_head(last_hidden_state)
        dst_state_logits = self.dst_state_head(last_hidden_state)

        return {
            **outputs,  # Include all inherited outputs (logits, last_hidden_state, etc.)
            "speaking_logits": speaking_logits,  # [batch, seq_len, 2]
            "dst_update_logits": dst_update_logits,  # [batch, seq_len, 2]
            "dst_state_logits": dst_state_logits,  # [batch, seq_len, num_dst_states]
        }

    def fast_greedy_generate_with_dst(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[KV_CACHE] = None,
        max_length: int = 100,
        include_dst: bool = True,
        **kwargs,
    ) -> Tuple[torch.Tensor, KV_CACHE, Dict[str, Any]]:
        """
        Enhanced fast_greedy_generate that includes DST predictions

        Inherits fast_greedy_generate() from SmolVLMWithStrategies, then adds DST predictions.

        Args:
            inputs_embeds: Can be embeddings OR tuple (input_ids, pixel_values, kwargs)
            past_key_values: KV cache from previous generation
            max_length: Maximum tokens to generate
            include_dst: Whether to include DST predictions

        Returns:
            (output_ids, past_key_values, dst_predictions) tuple
        """
        # Use inherited fast_greedy_generate from SmolVLMWithStrategies
        output_ids, new_cache = self.fast_greedy_generate(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            max_length=max_length,
            **kwargs,
        )

        # Get DST predictions for this generation step
        dst_predictions = {}
        if include_dst:
            with torch.no_grad():
                # We need to do a forward pass to get DST predictions
                if isinstance(inputs_embeds, tuple):
                    input_ids, pixel_values, extra_kwargs = inputs_embeds
                    # Use the generated output as input for DST prediction
                    dst_outputs = self.forward(
                        input_ids=output_ids,
                        pixel_values=pixel_values,
                        past_key_values=new_cache,
                        **extra_kwargs,
                    )
                else:
                    dst_outputs = self.forward(
                        inputs_embeds=inputs_embeds, past_key_values=new_cache
                    )

                # Get DST predictions
                speaking_probs = F.softmax(
                    dst_outputs.speaking_logits[:, -1, :], dim=-1
                )
                dst_update_probs = F.softmax(
                    dst_outputs.dst_update_logits[:, -1, :], dim=-1
                )
                dst_state_probs = F.softmax(
                    dst_outputs.dst_state_logits[:, -1, :], dim=-1
                )

                dst_predictions = {
                    "speaking_decision": speaking_probs,  # [2] probabilities
                    "dst_update": dst_update_probs,  # [2] probabilities
                    "dst_state": dst_state_probs,  # [num_dst_states] probabilities
                }

        return output_ids, new_cache, dst_predictions

    def get_dst_predictions(
        self,
        outputs: Dict[str, Any],
    ) -> Dict[str, torch.Tensor]:
        """
        Get DST predictions from model outputs

        Args:
            outputs: Model outputs from forward()

        Returns:
            Dict of DST predictions with probabilities and predictions
        """
        # Use the last sequence position for predictions
        last_idx = -1

        predictions = {}

        if "speaking_logits" in outputs:
            speaking_probs = F.softmax(outputs.speaking_logits[:, last_idx, :], dim=-1)
            predictions["speaking_decision"] = speaking_probs
            predictions["speaking_pred"] = speaking_probs.argmax(dim=-1)

        if "dst_update_logits" in outputs:
            dst_update_probs = F.softmax(
                outputs.dst_update_logits[:, last_idx, :], dim=-1
            )
            predictions["dst_update"] = dst_update_probs
            predictions["dst_update_pred"] = dst_update_probs.argmax(dim=-1)

        if "dst_state_logits" in outputs:
            dst_state_probs = F.softmax(
                outputs.dst_state_logits[:, last_idx, :], dim=-1
            )
            predictions["dst_state"] = dst_state_probs
            predictions["dst_state_pred"] = dst_state_probs.argmax(dim=-1)

        return predictions
