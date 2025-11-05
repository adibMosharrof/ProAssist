# KV Cache Manipulation for SmolVLM Dialog Generation

## Overview

This document provides strategies for manipulating past key values in transformer models without fully reimplementing the `generate()` method. The goal is to handle token limit constraints in long dialog generation with video frames by implementing drop-all/drop-middle strategies with summarization.

## Background

When generating long dialogs with vision-language models like SmolVLM:
- You pass in: generated dialog + prompt + video frames
- You hit token limits and need to compress the KV cache
- ProAssist manually implemented `generate()` to manipulate past key values
- You want a cleaner approach without full reimplementation

## Approach 1: Transformers Cache Classes ⭐ (Recommended for Simple Cases)

### Description
Use the built-in `DynamicCache` class from transformers to manipulate the KV cache directly.

### Implementation

```python
from transformers import DynamicCache
import torch

def drop_middle_from_cache(past_key_values, n_keep_start, n_keep_end):
    """
    Drop middle tokens from KV cache while preserving start and end.
    
    Args:
        past_key_values: Tuple of layer caches
        n_keep_start: Number of tokens to keep from start
        n_keep_end: Number of tokens to keep from end
    
    Returns:
        Modified cache with middle tokens dropped
    """
    if isinstance(past_key_values, DynamicCache):
        cache = past_key_values
    else:
        # Convert tuple to DynamicCache
        cache = DynamicCache()
        for layer_idx, (key, value) in enumerate(past_key_values):
            cache.update(key, value, layer_idx)
    
    # Modify each layer's cache
    for layer_idx in range(len(cache)):
        key = cache.key_cache[layer_idx]
        value = cache.value_cache[layer_idx]
        
        seq_len = key.shape[2]  # [batch, heads, seq_len, head_dim]
        
        if seq_len > n_keep_start + n_keep_end:
            # Drop middle
            new_key = torch.cat([
                key[:, :, :n_keep_start, :],
                key[:, :, -n_keep_end:, :]
            ], dim=2)
            
            new_value = torch.cat([
                value[:, :, :n_keep_start, :],
                value[:, :, -n_keep_end:, :]
            ], dim=2)
            
            cache.key_cache[layer_idx] = new_key
            cache.value_cache[layer_idx] = new_value
    
    return cache

# Usage during generation
past_key_values = None

for step in range(max_generation_steps):
    outputs = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        use_cache=True,
        return_dict=True,
    )
    
    past_key_values = outputs.past_key_values
    
    # Check cache size and compress if needed
    if past_key_values:
        cache_length = past_key_values[0][0].shape[2]
        if cache_length > max_cache_length:
            past_key_values = drop_middle_from_cache(
                past_key_values, 
                n_keep_start=1024,
                n_keep_end=512
            )
    
    # Get next token
    next_token = outputs.logits[:, -1:, :].argmax(dim=-1)
    
    if next_token.item() == tokenizer.eos_token_id:
        break
    
    input_ids = next_token
```

### Pros
- Uses official transformers utilities
- Clean and maintainable
- Compatible with future library updates

### Cons
- Less control over generation process
- Manual loop required instead of using `model.generate()`

---

## Approach 2: Custom Model with Overridden Methods ⭐⭐ (Recommended for Complex Cases)

### Description
Create a custom model class that overrides `prepare_inputs_for_generation()` to automatically compress the cache.

### Implementation

```python
from transformers import AutoModelForVision2Seq
from transformers import DynamicCache
import torch

class SmolVLMWithCacheCompression(AutoModelForVision2Seq):
    def __init__(self, config, max_cache_length=2048, compression_strategy='drop_middle'):
        super().__init__(config)
        self.max_cache_length = max_cache_length
        self.compression_strategy = compression_strategy
        self.n_visual_tokens = None  # Set this after processing images
        self.n_system_tokens = None  # Set this based on your prompt
    
    def set_special_token_counts(self, n_visual_tokens, n_system_tokens):
        """Set the number of visual and system tokens to preserve"""
        self.n_visual_tokens = n_visual_tokens
        self.n_system_tokens = n_system_tokens
    
    def prepare_inputs_for_generation(
        self, 
        input_ids, 
        past_key_values=None, 
        attention_mask=None,
        **kwargs
    ):
        # Compress cache if needed
        if past_key_values and self._should_compress(past_key_values):
            past_key_values = self._compress_cache(past_key_values)
            
            # Update attention mask accordingly
            if attention_mask is not None:
                attention_mask = self._update_attention_mask(
                    attention_mask, 
                    past_key_values
                )
        
        # Call parent's prepare method
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            **kwargs
        )
        
        return model_inputs
    
    def _should_compress(self, past_key_values):
        """Check if cache compression is needed"""
        if not past_key_values:
            return False
        
        # Get cache length from first layer
        cache_length = past_key_values[0][0].shape[2]
        return cache_length > self.max_cache_length
    
    def _compress_cache(self, past_key_values):
        """Compress the KV cache using the specified strategy"""
        if self.compression_strategy == 'drop_middle':
            return self._drop_middle(past_key_values)
        elif self.compression_strategy == 'drop_all_except_special':
            return self._drop_all_except_special(past_key_values)
        else:
            raise ValueError(f"Unknown compression strategy: {self.compression_strategy}")
    
    def _drop_middle(self, past_key_values):
        """
        Drop middle tokens while preserving:
        - Visual tokens (start)
        - System prompt (after visual)
        - Recent conversation (end)
        """
        n_keep_start = (self.n_visual_tokens or 0) + (self.n_system_tokens or 256)
        n_keep_end = 512  # Keep recent context
        
        compressed_cache = []
        
        for layer_cache in past_key_values:
            key, value = layer_cache
            seq_len = key.shape[2]
            
            if seq_len > n_keep_start + n_keep_end:
                new_key = torch.cat([
                    key[:, :, :n_keep_start, :],
                    key[:, :, -n_keep_end:, :]
                ], dim=2)
                
                new_value = torch.cat([
                    value[:, :, :n_keep_start, :],
                    value[:, :, -n_keep_end:, :]
                ], dim=2)
            else:
                new_key, new_value = key, value
            
            compressed_cache.append((new_key, new_value))
        
        return tuple(compressed_cache)
    
    def _drop_all_except_special(self, past_key_values):
        """
        Drop all tokens except visual and system tokens.
        Useful before summarization.
        """
        n_keep = (self.n_visual_tokens or 0) + (self.n_system_tokens or 256)
        
        compressed_cache = []
        
        for layer_cache in past_key_values:
            key, value = layer_cache
            
            new_key = key[:, :, :n_keep, :]
            new_value = value[:, :, :n_keep, :]
            
            compressed_cache.append((new_key, new_value))
        
        return tuple(compressed_cache)
    
    def _update_attention_mask(self, attention_mask, past_key_values):
        """Update attention mask to match compressed cache size"""
        new_cache_length = past_key_values[0][0].shape[2]
        
        # Truncate or adjust attention mask
        if attention_mask.shape[1] > new_cache_length:
            # Use same drop strategy
            n_keep_start = (self.n_visual_tokens or 0) + (self.n_system_tokens or 256)
            n_keep_end = 512
            
            new_mask = torch.cat([
                attention_mask[:, :n_keep_start],
                attention_mask[:, -n_keep_end:]
            ], dim=1)
            
            return new_mask
        
        return attention_mask

# Usage
model = SmolVLMWithCacheCompression.from_pretrained(
    "HuggingFaceTB/SmolVLM-Instruct",
    max_cache_length=2048,
    compression_strategy='drop_middle'
)

# Set special token counts after encoding
n_visual_tokens = 729  # Example: 27x27 image patches
n_system_tokens = len(tokenizer.encode(system_prompt))
model.set_special_token_counts(n_visual_tokens, n_system_tokens)

# Now use model.generate() normally - compression happens automatically!
outputs = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    max_new_tokens=1024,
    do_sample=True,
    temperature=0.7,
)
```

### Pros
- Seamless integration with `model.generate()`
- Automatic compression during generation
- Clean API - no manual loops needed
- Preserves all generate() features (beam search, sampling, etc.)

### Cons
- Requires model subclassing
- More complex initial setup

---

## Approach 3: Manual Chunking with Summarization ⭐⭐⭐ (Best for Dialog with Summarization)

### Description
When you need to do summarization at compression points, use explicit chunking with cache resets.

### Implementation

```python
class DialogGeneratorWithSummarization:
    def __init__(self, model, tokenizer, max_cache_length=2048):
        self.model = model
        self.tokenizer = tokenizer
        self.max_cache_length = max_cache_length
        self.n_visual_tokens = None
        self.n_system_tokens = None
    
    def set_special_tokens(self, n_visual_tokens, n_system_tokens):
        self.n_visual_tokens = n_visual_tokens
        self.n_system_tokens = n_system_tokens
    
    def generate_with_summarization(
        self,
        initial_prompt,
        pixel_values,
        max_total_tokens=4096,
        chunk_size=1024,
        summarization_fn=None,
    ):
        """
        Generate long dialog with periodic summarization.
        
        Args:
            initial_prompt: Initial system prompt + dialog history
            pixel_values: Video frames
            max_total_tokens: Total tokens to generate
            chunk_size: Generate this many tokens before considering summarization
            summarization_fn: Function to summarize dialog history
        
        Returns:
            Full generated text
        """
        # Encode initial inputs
        inputs = self.tokenizer(
            initial_prompt,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.model.device)
        
        input_ids = inputs['input_ids']
        attention_mask = inputs.get('attention_mask')
        
        all_generated_ids = []
        past_key_values = None
        total_generated = 0
        
        # Store visual embeddings separately if needed
        visual_cache = None
        
        while total_generated < max_total_tokens:
            # Generate a chunk
            outputs = self.model.generate(
                input_ids=input_ids,
                pixel_values=pixel_values if visual_cache is None else None,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                max_new_tokens=min(chunk_size, max_total_tokens - total_generated),
                return_dict_in_generate=True,
                output_scores=True,
                use_cache=True,
            )
            
            generated_ids = outputs.sequences[:, input_ids.shape[1]:]
            all_generated_ids.append(generated_ids)
            total_generated += generated_ids.shape[1]
            
            # Check if we need to compress/summarize
            if outputs.past_key_values:
                cache_length = outputs.past_key_values[0][0].shape[2]
                
                if cache_length > self.max_cache_length:
                    print(f"Cache length {cache_length} exceeds limit. Summarizing...")
                    
                    # Get all text generated so far
                    all_text = self.tokenizer.decode(
                        torch.cat([input_ids] + all_generated_ids, dim=1)[0],
                        skip_special_tokens=False
                    )
                    
                    # Summarize if function provided
                    if summarization_fn:
                        summary = summarization_fn(all_text)
                        
                        # Re-encode with summary
                        new_inputs = self.tokenizer(
                            summary,
                            return_tensors="pt",
                            add_special_tokens=True
                        ).to(self.model.device)
                        
                        input_ids = new_inputs['input_ids']
                        attention_mask = new_inputs.get('attention_mask')
                        past_key_values = None  # Reset cache
                        
                        # Clear generated history since we summarized
                        all_generated_ids = []
                    else:
                        # Just drop middle without summarization
                        past_key_values = self._drop_middle_cache(
                            outputs.past_key_values
                        )
                        
                        # Update input_ids to continue from last token
                        input_ids = outputs.sequences[:, -1:]
                else:
                    # Continue with existing cache
                    past_key_values = outputs.past_key_values
                    input_ids = outputs.sequences[:, -1:]
            else:
                break  # No more generation needed
            
            # Check for EOS
            if generated_ids[0, -1].item() == self.tokenizer.eos_token_id:
                break
        
        # Decode final output
        final_text = self.tokenizer.decode(
            torch.cat([input_ids] + all_generated_ids, dim=1)[0],
            skip_special_tokens=True
        )
        
        return final_text
    
    def _drop_middle_cache(self, past_key_values):
        """Drop middle tokens from cache"""
        n_keep_start = (self.n_visual_tokens or 0) + (self.n_system_tokens or 256)
        n_keep_end = 512
        
        compressed_cache = []
        
        for layer_cache in past_key_values:
            key, value = layer_cache
            seq_len = key.shape[2]
            
            if seq_len > n_keep_start + n_keep_end:
                new_key = torch.cat([
                    key[:, :, :n_keep_start, :],
                    key[:, :, -n_keep_end:, :]
                ], dim=2)
                
                new_value = torch.cat([
                    value[:, :, :n_keep_start, :],
                    value[:, :, -n_keep_end:, :]
                ], dim=2)
            else:
                new_key, new_value = key, value
            
            compressed_cache.append((new_key, new_value))
        
        return tuple(compressed_cache)

# Example summarization function
def summarize_dialog(text):
    """Summarize dialog history using a separate model or LLM"""
    # Option 1: Use a summarization model
    # summary_model = pipeline("summarization")
    # summary = summary_model(text, max_length=512)[0]['summary_text']
    
    # Option 2: Use LLM for structured summarization
    summary_prompt = f"""Summarize the following dialog, preserving key information:

{text}

Summary:"""
    
    # Call your summarization method
    summary = call_summarization_api(summary_prompt)
    
    return summary

# Usage
generator = DialogGeneratorWithSummarization(
    model=model,
    tokenizer=tokenizer,
    max_cache_length=2048
)

# Set based on your video encoding
generator.set_special_tokens(
    n_visual_tokens=729,  # e.g., 27x27 patches per frame
    n_system_tokens=len(tokenizer.encode(system_prompt))
)

output = generator.generate_with_summarization(
    initial_prompt=prompt,
    pixel_values=video_frames,
    max_total_tokens=4096,
    chunk_size=1024,
    summarization_fn=summarize_dialog
)
```

### Pros
- Full control over when/how to summarize
- Can track dialog turns explicitly
- Handles very long conversations
- Preserves semantic coherence through summarization

### Cons
- Most complex implementation
- Requires separate summarization logic
- More computational overhead

---

## Approach 4: Logits Processor Hook

### Description
Use the `LogitsProcessor` callback mechanism to intercept and modify cache during generation.

### Implementation

```python
from transformers import LogitsProcessor, LogitsProcessorList

class KVCacheCompressor(LogitsProcessor):
    def __init__(
        self, 
        max_cache_length=2048,
        n_visual_tokens=0,
        n_system_tokens=256,
        compression_strategy='drop_middle'
    ):
        self.max_cache_length = max_cache_length
        self.n_visual_tokens = n_visual_tokens
        self.n_system_tokens = n_system_tokens
        self.compression_strategy = compression_strategy
        self.compression_count = 0
    
    def __call__(self, input_ids, scores, **kwargs):
        """
        Called during generation to process logits.
        We use this hook to compress the cache.
        """
        # Note: We can't directly access past_key_values here in standard setup
        # This is a limitation of the LogitsProcessor approach
        
        # This would need custom model integration to work properly
        return scores

# This approach has limitations and is not recommended
# Kept here for completeness
```

### Pros
- Integrates with generation callbacks

### Cons
- Limited access to cache in standard implementation
- Requires significant workarounds
- **Not recommended** - use Approach 2 or 3 instead

---

## Comparison Table

| Approach | Complexity | Control | Works with generate() | Best For |
|----------|-----------|---------|----------------------|----------|
| 1. Cache Classes | Low | Medium | No (manual loop) | Simple compression without summarization |
| 2. Custom Model | Medium | High | Yes | Automatic compression during generation |
| 3. Manual Chunking | High | Highest | Partially | Dialog with summarization, tracking turns |
| 4. Logits Processor | Medium | Low | Yes | Not recommended |

---

## Recommendations for Your ProAssist Paper

Based on your requirements (SmolVLM + dialog + video frames + summarization):

### **Primary Recommendation: Approach 3 (Manual Chunking with Summarization)**

This is best because:
1. You need summarization capability (not just dropping tokens)
2. You're working with dialogs where turn tracking is important
3. You need to preserve video frame tokens consistently
4. Most flexibility for experimentation

### **Alternative: Approach 2 (Custom Model) + Approach 3 hybrid**

For cleaner code:
1. Use Approach 2 for automatic drop-middle compression
2. Add explicit summarization checkpoints at dialog turn boundaries
3. Reset cache after summarization

```python
# Hybrid approach
model = SmolVLMWithCacheCompression.from_pretrained(...)
generator = DialogGeneratorWithSummarization(model, tokenizer)

# Automatic compression happens during generation
# Explicit summarization at turn boundaries
output = generator.generate_with_summarization(
    initial_prompt=prompt,
    pixel_values=frames,
    summarization_fn=your_summarization_fn
)
```

---

## Implementation Checklist

- [ ] Determine visual token count from SmolVLM's vision encoder
- [ ] Measure system prompt token count
- [ ] Decide on compression strategy: drop-middle vs drop-all-then-summarize
- [ ] Implement or choose summarization method
- [ ] Test cache compression doesn't break attention masks
- [ ] Verify visual tokens are preserved across compressions
- [ ] Profile memory usage before/after compression
- [ ] Test on multi-turn dialogs with video frames
- [ ] Compare output quality with/without compression

---

## Additional Tips for SmolVLM

### Visual Token Preservation

```python
def get_visual_token_count(model, pixel_values):
    """Get the number of tokens used for visual encoding"""
    with torch.no_grad():
        # Process images through vision encoder
        vision_outputs = model.vision_model(pixel_values)
        
        # Count tokens (varies by model architecture)
        # For ViT-based: typically (img_size / patch_size)^2 per image
        n_tokens = vision_outputs.shape[1]
    
    return n_tokens
```

### Attention Mask Handling

Always update attention masks when modifying cache:

```python
def update_attention_mask_for_compression(attention_mask, keep_start, keep_end):
    """Update attention mask after dropping middle tokens"""
    if attention_mask is None:
        return None
    
    new_mask = torch.cat([
        attention_mask[:, :keep_start],
        attention_mask[:, -keep_end:]
    ], dim=1)
    
    return new_mask
```

### Memory Monitoring

```python
def get_cache_memory_usage(past_key_values):
    """Calculate memory usage of KV cache"""
    if not past_key_values:
        return 0
    
    total_bytes = 0
    for layer_cache in past_key_values:
        key, value = layer_cache
        total_bytes += key.element_size() * key.nelement()
        total_bytes += value.element_size() * value.nelement()
    
    return total_bytes / (1024 ** 2)  # Convert to MB

# Usage
print(f"Cache size: {get_cache_memory_usage(past_key_values):.2f} MB")
```

---

## Conclusion

For your ProAssist paper with SmolVLM, I recommend:

1. **Start with Approach 3** (Manual Chunking with Summarization) for maximum control
2. Consider implementing **Approach 2** (Custom Model) as a cleaner alternative once you validate the approach
3. Always preserve visual tokens at the start of the sequence
4. Implement summarization at natural dialog turn boundaries
5. Monitor memory usage and generation quality throughout

This approach will give you the flexibility needed for research while maintaining clean, reproducible code.