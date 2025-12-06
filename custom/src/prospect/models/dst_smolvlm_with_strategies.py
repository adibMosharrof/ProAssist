"""
DST SmolVLM - Minimal Extension of SmolVLMWithStrategies

Adds only 2 binary decision heads to the base model:
- speaking_decision_head: Should the assistant speak?
- dst_update_head: Should we update DST?

Following ProAssist's philosophy:
- Model = Architecture definition only
- Generation = Inherited from parent (fast_greedy_generate)
- Inference = DSTStreamRunner handles frame-by-frame logic
- Training = Trainer class handles loss computation and metrics

Minimal complexity: This class only extends with binary heads.
Everything else follows ProAssist's proven patterns.
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
import torch
import torch.nn as nn
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from prospect.models.smolvlm_with_strategies import SmolVLMWithStrategies

logger = logging.getLogger(__name__)

# Type alias for KV cache (legacy tuple format)
KV_CACHE = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]


class DSTSmolVLMWithStrategies(SmolVLMWithStrategies):
    """
    DST SmolVLM - Minimal extension with binary decision heads.
    
    Inherits from SmolVLMWithStrategies:
    - vision_projector: Projects image embeddings to LLM space
    - joint_embed(): Combines text + image inputs
    - fast_greedy_generate(): Token-by-token generation with KV cache
    
    Adds:
    - speaking_decision_head: Sigmoid binary classifier
    - dst_update_head: Sigmoid binary classifier
    
    Usage in inference:
    1. Runner calls joint_embed(input_ids, image_embeds)
    2. Runner calls fast_greedy_generate(embeddings, kv_cache) [inherited]
    3. Runner checks binary head logits from outputs
    4. Runner decides whether to generate based on thresholds
    """

    def __init__(self, config):
        """Initialize with only binary decision heads"""
        super().__init__(config)
        
        # Get hidden size from config
        if hasattr(config, "text_config") and hasattr(config.text_config, "hidden_size"):
            hidden_size = config.text_config.hidden_size
        elif hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        else:
            hidden_size = 1280  # SmolVLM2 default
            logger.warning(f"Could not determine hidden_size, using fallback: {hidden_size}")
        
        # Binary decision heads (ONLY addition to parent class)
        self.speaking_decision_head = nn.Linear(hidden_size, 1)
        self.dst_update_head = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        nn.init.xavier_uniform_(self.speaking_decision_head.weight)
        nn.init.xavier_uniform_(self.dst_update_head.weight)
        nn.init.zeros_(self.speaking_decision_head.bias)
        nn.init.zeros_(self.dst_update_head.bias)
        
        logger.info(f"✅ DSTSmolVLM initialized (extends SmolVLMWithStrategies + binary heads)")

    def _fuse_input_embeds(
        self,
        input_ids: torch.LongTensor,
        image_embeds: torch.Tensor,
    ) -> torch.Tensor:
        """
        Manually fuse text embeddings and pre-computed image embeddings.
        This bypasses SmolVLMForConditionalGeneration's requirement for pixel_values.
        
        NOTE: User represents each image with exactly 1 embedding (CLS token), 
        not the standard 17+ patches. We map 1 image embedding to 1 <image> token.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            image_embeds: Pre-computed visual features.
                         Expected shapes:
                         - [batch_size * num_images, hidden_size]
                         - [batch_size, num_images, hidden_size]
                         - [batch_size, num_images, 1, hidden_size] (common from loose unsqueezing)
                         
        Returns:
            inputs_embeds: Combined embeddings [batch_size, seq_len, hidden_size]
        """
        # Get text embeddings from the base Llama model
        inputs_embeds = self.model.embed_tokens(input_ids)
        
        # Find where image tokens are
        # SmolVLM uses <image> token ID from config
        image_token_id = self.config.image_token_id
        
        # Create mask for image tokens
        image_mask = (input_ids == image_token_id)
        
        # Verify we have enough image embeddings
        num_images_in_text = image_mask.sum()
        
        if num_images_in_text == 0:
            return inputs_embeds
            
        # Robustly flatten image_embeds to [total_images, hidden_size]
        if image_embeds.dim() == 4: # [B, N, 1, H]
            image_embeds = image_embeds.squeeze(2) # -> [B, N, H]
            
        if image_embeds.dim() == 3: # [B, N, H]
            image_embeds = image_embeds.flatten(0, 1) # -> [B*N, H]
            
        # Check if we have matching counts
        if image_embeds.shape[0] != num_images_in_text:
            # If we have a mismatch, it's likely a configuration error 
            # (e.g. tokenizer adding multiple tokens per image vs 1 embedding)
            logger.warning(f"Mismatch in image tokens ({num_images_in_text}) vs embeddings ({image_embeds.shape[0]}). "
                           f"Assuming 1 embedding per image token and truncating/padding if necessary.")
        
        # Convert to same dtype
        image_embeds = image_embeds.to(dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        
        # Fill embeddings
        # We can use masked scatter or index assignment
        # Flatten inputs_embeds to [batch*seq_len, hidden] for easier assignment
        batch_size, seq_len, hidden = inputs_embeds.shape
        flat_embeds = inputs_embeds.view(-1, hidden)
        flat_mask = image_mask.view(-1)
        
        # Indices where mask is True
        indices = torch.nonzero(flat_mask).squeeze()
        
        # Assign
        # If lengths match exactly (ideal case)
        if image_embeds.shape[0] == indices.shape[0]:
            flat_embeds[indices] = image_embeds
        else:
            # Safe assignment for mismatched lengths
            count = min(image_embeds.shape[0], indices.shape[0])
            if count > 0:
                flat_embeds[indices[:count]] = image_embeds[:count]
            
        return flat_embeds.view(batch_size, seq_len, hidden)

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
        
        This ONLY computes model outputs. Training losses are computed in the trainer.
        """
        # Call parent forward - handles vision projection and LM output automatically
        # Force output_hidden_states to ensure we get hidden states for binary heads
        if "output_hidden_states" in kwargs:
            kwargs.pop("output_hidden_states")
            
        # Ensure past_key_values is a DynamicCache if it's a tuple (legacy)
        # This is needed because LlamaModel (in newer transformers) expects an object with get_seq_length()
        if past_key_values is not None and isinstance(past_key_values, tuple):
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)

        # MANUAL EMBEDDING FUSION (Fix for "pixel_values required" error)
        # If we have image_embeds but NO pixel_values, the base SmolVLMForConditionalGeneration
        # will error out because it expects pixel_values when image tokens are present.
        # We bypass this by manually fusing embeddings and calling the internal self.model directly.
        if image_embeds is not None and pixel_values is None and input_ids is not None:
            # 1. Fuse embeddings
            inputs_embeds = self._fuse_input_embeds(input_ids, image_embeds)
            
            # 2. Call internal LlamaModel (self.model) directly
            # This bypasses SmolVLM's vision processing logic entirely
            outputs = self.model(
                input_ids=None, # We provide inputs_embeds instead
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True, # We need this for binary heads
                return_dict=True,
                **kwargs
            )
            
            # 3. Project logits using the LM head (standard CausalLM step)
            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            
            # 4. Wrap in standard output class
            # Note: outputs is BaseModelOutputWithPast, we need CausalLMOutputWithPast
            base_outputs = outputs
            outputs = CausalLMOutputWithPast(
                loss=None,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            
        else:
            # Standard path: let base class handle it (requires pixel_values if images present)
            outputs = super().forward(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_embeds=image_embeds,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                output_hidden_states=True,
                **kwargs,
            )
        
        # Get last hidden state for binary head predictions
        # Prefer explicit last_hidden_state, fallback to last item in hidden_states
        if hasattr(outputs, 'last_hidden_state') and outputs.last_hidden_state is not None:
            last_hidden_state = outputs.last_hidden_state
        elif "hidden_states" in outputs and outputs["hidden_states"]:
            last_hidden_state = outputs["hidden_states"][-1]
        else:
            # If no hidden states available, try to get them from the output object
            last_hidden_state = getattr(outputs, 'hidden_states', [None])[-1] if hasattr(outputs, 'hidden_states') else None
        
        # Add binary head outputs (only addition to parent output)
        if last_hidden_state is not None:
            speaking_logits = self.speaking_decision_head(last_hidden_state)  # [batch, seq_len, 1]
            dst_update_logits = self.dst_update_head(last_hidden_state)       # [batch, seq_len, 1]
            
            outputs["speaking_logits"] = speaking_logits
            outputs["dst_update_logits"] = dst_update_logits
        
        return outputs
