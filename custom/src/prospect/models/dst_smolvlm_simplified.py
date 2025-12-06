"""
DST SmolVLM with Minimal Complexity

Follows ProAssist pattern: Model = Architecture only
- Inherits generation from parent
- Runner handles inference logic
- Trainer handles loss/metrics

This class ONLY adds 2 binary decision heads.
Everything else is inherited.
"""

import logging
from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn

from prospect.models.smolvlm_with_strategies import SmolVLMWithStrategies

logger = logging.getLogger(__name__)

# Type alias for KV cache (legacy tuple format)
KV_CACHE = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]


class DSTSmolVLMWithStrategies(SmolVLMWithStrategies):
    """
    DST SmolVLM - Minimal Extension of SmolVLMWithStrategies
    
    Adds only 2 binary decision heads:
    - speaking_decision_head: Should the assistant speak?
    - dst_update_head: Should we update DST?
    
    Everything else (generation, caching, inference logic) is in the runner.
    Training losses and metrics are computed in the trainer.
    
    **Why minimal?**
    - Model = Architecture definition only
    - Inference = DSTStreamRunner (frame-by-frame generation)
    - Training = Trainer class (loss computation + metrics)
    
    Following ProAssist's separation of concerns.
    """

    def __init__(self, config):
        """Initialize with binary decision heads only"""
        super().__init__(config)
        
        # Get hidden size from config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = 1280  # SmolVLM2 default
            logger.warning(f"Could not determine hidden_size, using fallback: {hidden_size}")
        
        # Binary decision heads (that's it!)
        self.speaking_decision_head = nn.Linear(hidden_size, 1)
        self.dst_update_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.speaking_decision_head.weight)
        nn.init.xavier_uniform_(self.dst_update_head.weight)
        nn.init.zeros_(self.speaking_decision_head.bias)
        nn.init.zeros_(self.dst_update_head.bias)
        
        logger.info(f"✅ DSTSmolVLM initialized with binary decision heads")

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        past_key_values: Optional[KV_CACHE] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Forward pass: Get base LM outputs, add binary head predictions.
        
        This ONLY computes architecture outputs, not training losses.
        Training losses are computed in the trainer.
        """
        # Call parent forward to get LM outputs
        # Parent handles vision embedding projection automatically
        outputs = super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_embeds=image_embeds,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        
        # Get last hidden state for binary head predictions
        # From parent's forward: either from hidden_states[-1] or last_hidden_state
        if "hidden_states" in outputs and outputs["hidden_states"]:
            last_hidden_state = outputs["hidden_states"][-1]
        else:
            # Fallback: recompute if not available
            # This shouldn't happen if parent returns hidden_states
            last_hidden_state = outputs.get("last_hidden_state")
        
        # Add binary head outputs
        speaking_logits = self.speaking_decision_head(last_hidden_state)  # [batch, seq_len, 1]
        dst_update_logits = self.dst_update_head(last_hidden_state)       # [batch, seq_len, 1]
        
        outputs["speaking_logits"] = speaking_logits
        outputs["dst_update_logits"] = dst_update_logits
        
        return outputs
