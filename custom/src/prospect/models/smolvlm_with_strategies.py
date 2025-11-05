"""
SmolVLM with Approach 1: Runner-based compression (ProAssist pattern)

Simple wrapper around SmolVLMForConditionalGeneration that maintains
compatibility with ProAssist's design. Compression happens in the runner
before calling generate(), not inside the model.

Implements fast_greedy_generate() to bypass DynamicCache issues.
"""

import logging
from typing import Optional, Tuple
import torch
from transformers import SmolVLMForConditionalGeneration


logger = logging.getLogger(__name__)

# Type alias for KV cache (legacy tuple format)
KV_CACHE = Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...]


class SmolVLMWithStrategies(SmolVLMForConditionalGeneration):
    """
    SmolVLM wrapper for ProAssist-style KV cache management.
    
    **Approach 1 Implementation (ProAssist Pattern)**
    
    This class follows ProAssist's design:
    1. Compression happens in the runner BEFORE calling generate()
    2. Uses fast_greedy_generate() to bypass DynamicCache/cache_position issues
    3. Strategies are decoupled and managed by the runner
    
    Usage:
        ```python
        # Load model
        model = SmolVLMWithStrategies.from_pretrained("HuggingFaceTB/SmolVLM2-Instruct")
        
        # Get embeddings
        inputs_embeds = model.joint_embed(input_ids=input_ids, pixel_values=pixel_values)
        
        # Generate with cache (ProAssist pattern)
        output_ids, new_cache = model.fast_greedy_generate(
            inputs_embeds, 
            past_key_values=cache,
            max_length=100
        )
        ```
    """
    
    def __init__(self, config):
        """Initialize model"""
        super().__init__(config)
        
        # Config fields for compatibility
        if not hasattr(config, 'max_seq_len'):
            config.max_seq_len = 4096
        
        # Cache for inplace output generation
        self.inplace_output_ids = None
        
        logger.info(f"Initialized SmolVLMWithStrategies (Approach 1 - ProAssist pattern)")
        logger.info(f"Max sequence length: {config.max_seq_len}")
    
    def joint_embed(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Create joint embeddings from text tokens and images (ProAssist pattern).
        
        For SmolVLM, we simply return the text+image inputs as is, and let
        forward() handle the vision-language fusion internally.
        
        Args:
            input_ids: Text token IDs [batch, seq_len]
            pixel_values: Image pixels [batch, channels, height, width]
            **kwargs: Additional arguments (e.g., image_sizes)
            
        Returns:
            Inputs for forward() - we pass through input_ids and pixel_values
            separately, not as embeddings, because SmolVLM needs both.
        """
        # SmolVLM's forward() expects input_ids + pixel_values,
        # not pre-computed embeddings. So we just pass them through.
        # The forward() call will handle the vision-language fusion.
        
        # Return a special marker to indicate we need to use input_ids mode
        # This is handled in fast_greedy_generate
        return (input_ids, pixel_values, kwargs)
    
    @torch.no_grad()
    def fast_greedy_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[KV_CACHE] = None,
        max_length: int = 100,
        verbose: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, KV_CACHE]:
        """
        Fast greedy generation using forward() calls (ProAssist pattern).
        
        This bypasses the Transformers generate() API and DynamicCache,
        avoiding cache_position issues when using compressed caches.
        
        Args:
            inputs_embeds: Can be embeddings OR tuple (input_ids, pixel_values, kwargs)
            past_key_values: KV cache from previous generation (tuple format)
            max_length: Maximum tokens to generate
            verbose: Print debug info
            
        Returns:
            (output_ids, past_key_values) tuple in legacy format
        """
        from transformers.cache_utils import DynamicCache
        
        # Allocate output buffer (reuse if possible)
        if (
            self.inplace_output_ids is None
            or self.inplace_output_ids.shape[1] < max_length
        ):
            self.inplace_output_ids = torch.full(
                (1, max_length), -100, dtype=torch.long, device=self.device
            )
        
        # Check if inputs_embeds is actually a tuple from joint_embed
        use_ids_mode = isinstance(inputs_embeds, tuple)
        if use_ids_mode:
            input_ids, pixel_values, extra_kwargs = inputs_embeds
            first_forward_kwargs = {'input_ids': input_ids, 'pixel_values': pixel_values, **extra_kwargs}
        else:
            first_forward_kwargs = {'inputs_embeds': inputs_embeds}
        
        # Convert tuple cache to DynamicCache for forward() if needed
        if past_key_values is not None and isinstance(past_key_values, tuple):
            cache_for_forward = DynamicCache.from_legacy_cache(past_key_values)
        else:
            cache_for_forward = past_key_values
        
        past_key_values_to_return = None
        
        for i in range(max_length):
            # Forward pass with cache
            if i == 0:
                # First iteration: use input_ids or inputs_embeds
                outputs = self.forward(
                    **first_forward_kwargs,
                    past_key_values=cache_for_forward,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                # Subsequent iterations: use token embeddings
                outputs = self.forward(
                    inputs_embeds=inputs_embeds,
                    past_key_values=cache_for_forward,
                    use_cache=True,
                    return_dict=True,
                )
            
            cache_for_forward = outputs.past_key_values
            
            if i == 0:
                # Store initial cache (convert back to tuple format)
                if hasattr(cache_for_forward, 'to_legacy_cache'):
                    past_key_values_to_return = cache_for_forward.to_legacy_cache()
                else:
                    past_key_values_to_return = cache_for_forward
            
            # Greedy sampling: take argmax
            new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
            self.inplace_output_ids[:, i] = new_token_id
            
            if verbose:
                decoded = self.config.tokenizer.decode(new_token_id[0]) if hasattr(self.config, 'tokenizer') else f"token_{new_token_id.item()}"
                logger.debug(f"Step {i}: {decoded}")
            
            # Check for EOS
            if new_token_id == self.config.eos_token_id:
                break
            
            # Prepare next input (embed the new token)
            inputs_embeds = self.get_input_embeddings()(new_token_id)
        
        # Return final cache in tuple format
        if hasattr(cache_for_forward, 'to_legacy_cache'):
            past_key_values_to_return = cache_for_forward.to_legacy_cache()
        else:
            past_key_values_to_return = cache_for_forward
        
        return self.inplace_output_ids[:, : i + 1], past_key_values_to_return
