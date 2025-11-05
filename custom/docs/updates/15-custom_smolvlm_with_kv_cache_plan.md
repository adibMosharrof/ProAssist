# Custom SmolVLM2 Model with KV Cache Accumulation - Implementation Plan

**Date**: November 3, 2025  
**Status**: ✅ IMPLEMENTATION COMPLETE (See: `17-implementation_status.md` for current status)  
**Goal**: Extend SmolVLM2 with ProAssist-style KV cache management for streaming inference

---

## 📋 IMPLEMENTATION COMPLETE

This plan has been **fully implemented**. For current status, issues, and next steps, see:
- **Status Document**: `custom/docs/updates/17-implementation_status.md`
- **Error Analysis**: `custom/docs/updates/16-e2e_test_errors_and_fixes.md`
- **Test Suite**: `custom/src/prospect/tests/README.md`

**Quick Summary**:
- ✅ Custom model with joint_embed() and fast_greedy_generate()
- ✅ Custom processor with streaming support
- ✅ VLM runner integration with KV cache accumulation
- ✅ All context strategies implemented (drop_all, drop_middle, summarize_and_drop)
- ✅ Unit tests passing (8/8)
- ✅ Test organization restructured following dst_data_builder pattern
- ⚠️ E2E tests blocked by disk space issue

---

# Original Implementation Plan (For Reference)

## Overview

Create a custom SmolVLM2 model that extends the base HuggingFace model with:
1. `joint_embed()` - Combine text + image embeddings
2. `fast_greedy_generate()` - Custom generation loop with KV cache control
3. Context overflow strategies (drop_all, drop_middle, summarize_and_drop)

This will enable proper context accumulation and strategy activation for PROSPECT.

## ProAssist Architecture Analysis

### ProAssist's Custom Model Structure

```
ProActLlamaForCausalLM
├─ Inherits: LlamaForCausalLM (HF base)
├─ Mixin: ProActModelMixin
│   ├─ joint_embed() - Combines text + vision embeddings
│   ├─ fast_greedy_generate() - Custom generation with KV cache
│   ├─ mm_feature_proj() - Vision feature projection
│   └─ visual_embed() - Vision encoder
└─ Config: ProActLlamaConfig
    ├─ exceed_context_handling: Strategy type
    ├─ max_seq_len: Context window size
    └─ Special tokens: img_token, img_sep_token, etc.
```

### Key Methods

**1. joint_embed()** (modeling_proact.py:70-88)
```python
def joint_embed(self, input_ids, images=None, image_embeds=None):
    # Get text embeddings
    inputs_embeds = self.get_input_embeddings()(input_ids)
    
    # If images provided, encode and project them
    if images is not None:
        image_embeds = self.visual_embed(images)
        image_embeds = self.mm_feature_proj(image_embeds)
    
    # Replace image token positions with image embeddings
    inputs_embeds[input_ids == self.config.img_token_id] = image_embeds
    
    return inputs_embeds
```

**2. fast_greedy_generate()** (modeling_proact.py:91-180)
```python
def fast_greedy_generate(self, inputs_embeds, past_key_values, max_length=100, ...):
    past_key_values_to_return = past_key_values
    
    for i in range(max_length):
        # Forward pass with KV cache
        outputs = self.forward(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        
        # After first token, save KV cache
        if i == 0:
            past_key_values_to_return = past_key_values
        
        # Get next token
        new_token_id = outputs.logits[:, -1:].argmax(dim=-1)
        
        # Check for EOS
        if new_token_id == self.config.eos_token_id:
            break
        
        # Embed next token for next iteration
        inputs_embeds = self.get_input_embeddings()(new_token_id)
    
    return output_ids, past_key_values_to_return
```

## SmolVLM2 Architecture

### Current SmolVLM2 Structure

```
SmolVLMForConditionalGeneration
├─ Inherits: Idefics2ForConditionalGeneration
├─ Components:
│   ├─ vision_model: SiglipVisionModel
│   ├─ connector: MLP projection
│   ├─ text_model: Llama-based language model
│   └─ image_token_id: Special token for images
└─ Methods:
    ├─ forward() - Standard forward pass
    └─ generate() - Standard HF generation
```

### SmolVLM2 Image Processing

SmolVLM2 uses a different approach than ProAssist:
- Vision encoder: SiglipVisionModel (not CLIP)
- Connector: Simple MLP (not custom projector)
- Image tokens: Embedded directly in text sequence

## Implementation Plan

### Phase 1: Create Custom SmolVLM Model (3-4 hours)

#### File: `custom/src/prospect/models/custom_smolvlm.py`

```python
"""Custom SmolVLM2 with ProAssist-style KV cache management"""

import torch
import torch.nn as nn
from transformers import SmolVLMForConditionalGeneration
from typing import Optional, Tuple

from prospect.models.configuration_custom_smolvlm import CustomSmolVLMConfig


class CustomSmolVLMMixin:
    """Mixin for SmolVLM2 with custom generation and KV cache management"""
    
    def joint_embed(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Combine text and image embeddings.
        
        Similar to ProAssist's joint_embed but adapted for SmolVLM2's architecture.
        
        Args:
            input_ids: Text token IDs
            pixel_values: Raw image pixels (if not pre-encoded)
            image_embeds: Pre-encoded image embeddings
            
        Returns:
            Combined embeddings ready for forward pass
        """
        # Get text embeddings
        text_embeds = self.get_input_embeddings()(input_ids)
        
        # Process images if provided
        if pixel_values is not None:
            # Encode images through vision model
            vision_outputs = self.vision_model(pixel_values)
            image_embeds = vision_outputs.last_hidden_state
            
            # Project to text embedding space
            image_embeds = self.connector(image_embeds)
        
        if image_embeds is not None:
            # Replace image token positions with image embeddings
            # SmolVLM2 uses image_token_id to mark where images go
            image_token_mask = input_ids == self.config.image_token_id
            text_embeds[image_token_mask] = image_embeds.flatten(0, 1)
        
        return text_embeds
    
    @torch.no_grad()
    def fast_greedy_generate(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        max_length: int = 100,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, Tuple]:
        """
        Custom greedy generation with explicit KV cache management.
        
        Similar to ProAssist's fast_greedy_generate but for SmolVLM2.
        
        Args:
            inputs_embeds: Input embeddings from joint_embed()
            past_key_values: Accumulated KV cache from previous frames
            max_length: Maximum tokens to generate
            eos_token_id: End of sequence token
            pad_token_id: Padding token
            
        Returns:
            (output_ids, past_key_values_to_return)
        """
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id or eos_token_id
        
        # Initialize output buffer
        batch_size = inputs_embeds.shape[0]
        output_ids = torch.full(
            (batch_size, max_length),
            pad_token_id,
            dtype=torch.long,
            device=inputs_embeds.device
        )
        
        # Save KV cache after processing input
        past_key_values_to_return = past_key_values
        
        for i in range(max_length):
            # Forward pass
            outputs = self.language_model(
                inputs_embeds=inputs_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Update KV cache
            past_key_values = outputs.past_key_values
            
            # After first token, save the KV cache (includes input context)
            if i == 0:
                past_key_values_to_return = past_key_values
            
            # Get next token (greedy)
            next_token_logits = outputs.logits[:, -1, :]
            next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Store generated token
            output_ids[:, i] = next_token_id.squeeze(-1)
            
            # Check for EOS
            if next_token_id.item() == eos_token_id:
                # Trim output to actual length
                output_ids = output_ids[:, :i+1]
                break
            
            # Prepare next iteration: embed the generated token
            inputs_embeds = self.get_input_embeddings()(next_token_id)
        
        return output_ids, past_key_values_to_return


class CustomSmolVLMForConditionalGeneration(
    SmolVLMForConditionalGeneration,
    CustomSmolVLMMixin
):
    """
    Custom SmolVLM2 model with ProAssist-style KV cache management.
    
    Extends SmolVLMForConditionalGeneration with:
    - joint_embed(): Combine text + image embeddings
    - fast_greedy_generate(): Custom generation with KV cache control
    """
    
    config_class = CustomSmolVLMConfig
    
    def __init__(self, config: CustomSmolVLMConfig):
        super().__init__(config)
        # Additional initialization if needed
```

#### File: `custom/src/prospect/models/configuration_custom_smolvlm.py`

```python
"""Configuration for custom SmolVLM2 model"""

from transformers import Idefics2Config
from enum import Enum


class ExceedContextHandling(Enum):
    """Context overflow handling strategies"""
    DROP_ALL = "drop_all"
    DROP_MIDDLE = "drop_middle"
    SUMMARIZE_AND_DROP = "summarize_and_drop"


class CustomSmolVLMConfig(Idefics2Config):
    """
    Configuration for CustomSmolVLM with context management.
    
    Extends Idefics2Config (SmolVLM2's base) with ProAssist-style settings.
    """
    
    def __init__(
        self,
        *,
        max_seq_len: int = 4096,
        exceed_context_handling: str = "drop_all",
        reserved_seq_len: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.max_seq_len = max_seq_len
        self.exceed_context_handling = exceed_context_handling
        self.reserved_seq_len = reserved_seq_len
        
        # Validate strategy
        if exceed_context_handling not in ExceedContextHandling._value2member_map_:
            raise ValueError(
                f"Unsupported exceed_context_handling: {exceed_context_handling}"
            )
    
    @property
    def exceed_context_handling_strategy(self) -> ExceedContextHandling:
        return ExceedContextHandling(self.exceed_context_handling)
```

### Phase 2: Create Custom Processor (1-2 hours)

#### File: `custom/src/prospect/models/processing_custom_smolvlm.py`

```python
"""Custom processor for SmolVLM2 with streaming support"""

import torch
from transformers import AutoProcessor
from typing import List, Dict, Tuple


class CustomSmolVLMProcessor:
    """
    Processor for CustomSmolVLM with streaming support.
    
    Handles:
    - Frame-by-frame input preparation
    - Chat template formatting
    - Token sequence management
    """
    
    def __init__(self, base_processor: AutoProcessor):
        self.base_processor = base_processor
        self.tokenizer = base_processor.tokenizer
        self.image_processor = base_processor.image_processor
    
    def get_input_sequence(
        self,
        num_images: int,
        messages: List[Dict[str, str]],
        first_turn: bool = False
    ) -> Tuple[torch.LongTensor, str]:
        """
        Prepare input sequence for streaming inference.
        
        Args:
            num_images: Number of images in this turn
            messages: List of message dicts (role, content)
            first_turn: Whether this is the first turn
            
        Returns:
            (input_ids, input_str)
        """
        # Format text messages
        if messages:
            if first_turn:
                # Apply full chat template for first turn
                input_str_txt = self._apply_chat_template(messages)
            else:
                # Append messages for subsequent turns
                input_str_txt = ""
                for msg in messages:
                    input_str_txt += self._add_message(msg)
        else:
            input_str_txt = ""
        
        # Add image tokens
        input_str_img = self._add_img_tokens(num_images)
        input_str = input_str_txt + input_str_img
        
        # Tokenize
        input_ids = self.tokenizer(
            input_str,
            return_tensors="pt",
            add_special_tokens=False
        )["input_ids"]
        
        return input_ids, input_str
    
    def _apply_chat_template(self, messages: List[Dict]) -> str:
        """Apply chat template (SmolVLM2 format)"""
        # SmolVLM2 uses standard chat template
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        return formatted
    
    def _add_message(self, message: Dict) -> str:
        """Add a single message"""
        role = message["role"]
        content = message["content"]
        # Simple format: "Role: content\n"
        return f"{role.capitalize()}: {content}\n"
    
    def _add_img_tokens(self, num_images: int) -> str:
        """Add image placeholder tokens"""
        # SmolVLM2 uses <image> token
        return "<image>" * num_images
    
    def add_last_assistant_message(
        self,
        model_inputs: Dict[str, torch.Tensor],
        last_msg: Optional[torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Prepend last assistant message to current input.
        
        This is used to maintain dialogue context in KV cache.
        """
        if last_msg is None:
            return model_inputs
        
        input_ids = model_inputs["input_ids"]
        
        # Concatenate: [last_msg] + [current_input]
        input_ids = torch.cat([last_msg, input_ids], dim=-1)
        model_inputs["input_ids"] = input_ids
        
        return model_inputs
    
    def cleanup_text(self, text: str) -> Tuple[str, Optional[str]]:
        """Clean up generated text"""
        # Remove special tokens
        text = text.strip()
        
        # Remove image tokens
        text = text.replace("<image>", "")
        
        # Extract role if present
        if ":" in text:
            parts = text.split(":", 1)
            if parts[0].lower() in ["assistant", "user", "system"]:
                return parts[1].strip(), parts[0].lower()
        
        return text, None
```

### Phase 3: Update VLM Runner (2 hours)

#### File: `custom/src/prospect/runners/vlm_stream_runner.py`

**Changes needed**:

1. **Import custom model**:
```python
from prospect.models.custom_smolvlm import CustomSmolVLMForConditionalGeneration
from prospect.models.processing_custom_smolvlm import CustomSmolVLMProcessor
from prospect.models.configuration_custom_smolvlm import CustomSmolVLMConfig
```

2. **Load custom model instead of standard**:
```python
def __init__(self, ...):
    # Create custom config
    config = CustomSmolVLMConfig.from_pretrained(
        model_name,
        max_seq_len=kwargs.get('max_seq_len', 4096),
        exceed_context_handling=context_strategy_type,
        reserved_seq_len=kwargs.get('reserved_seq_len', 128),
    )
    
    # Load custom model
    self.model = CustomSmolVLMForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        cache_dir=cache_dir,
    )
    
    # Create custom processor
    base_processor = AutoProcessor.from_pretrained(model_name, ...)
    self.processor = CustomSmolVLMProcessor(base_processor)
```

3. **Use joint_embed + fast_greedy_generate**:
```python
def _generate_dialogue_with_cache(self, frame, prev_substep, curr_substep):
    prompt = self.dialogue_generation_prompt.format(...)
    
    # Prepare inputs
    inputs = self.processor(images=frame, text=prompt, return_tensors="pt")
    inputs = {k: v.to(self.device) for k, v in inputs.items()}
    
    # Add last message if exists
    if self.last_msg_tokens is not None:
        inputs = self.processor.add_last_assistant_message(
            inputs, self.last_msg_tokens
        )
    
    # Get embeddings
    input_embeds = self.model.joint_embed(**inputs)
    
    # Generate with KV cache
    output_ids, self.past_key_values = self.model.fast_greedy_generate(
        input_embeds,
        self.past_key_values,
        max_length=self.max_new_tokens,
    )
    
    # Store last token for next iteration
    self.last_msg_tokens = output_ids[:, -1:]
    
    # Check overflow and apply strategy
    if self.context_strategy:
        cache_len = self._get_cache_length()
        if self.context_strategy.should_reduce_cache(cache_len):
            self._apply_context_strategy(inputs)
    
    # Decode
    dialogue = self.processor.tokenizer.decode(output_ids[0], ...)
    return dialogue
```

### Phase 4: Update Context Strategies (1 hour)

#### Update `summarize_and_drop.py`

```python
def _generate_summary(self, model, processor, current_frame, ...):
    """Generate summary using custom model"""
    
    # Prepare summary prompt
    summary_inputs = dict(current_frame)
    summary_text = f"<image>{self.summary_prompt}"
    summary_ids = processor.tokenizer(summary_text, ...)["input_ids"]
    summary_inputs["input_ids"] = summary_ids
    
    # Get embeddings
    input_embeds = model.joint_embed(**summary_inputs)
    
    # Generate summary with accumulated KV cache
    output_ids, _ = model.fast_greedy_generate(
        input_embeds,
        past_key_values,  # Contains all history!
        max_length=self.summary_max_length,
    )
    
    # Decode and clean
    summary = processor.tokenizer.decode(output_ids[0], ...)
    return summary
```

## Implementation Steps

### Step 1: Create Model Files (Day 1 - Morning)

1. ✅ Create `custom/src/prospect/models/` directory
2. ✅ Implement `configuration_custom_smolvlm.py`
3. ✅ Implement `custom_smolvlm.py` with mixin
4. ✅ Implement `processing_custom_smolvlm.py`
5. ✅ Add `__init__.py` for module

### Step 2: Test Custom Model (Day 1 - Afternoon)

1. ✅ Create test script to load custom model
2. ✅ Verify `joint_embed()` works
3. ✅ Verify `fast_greedy_generate()` works
4. ✅ Compare output with standard SmolVLM2

### Step 3: Integrate with Runner (Day 1 - Evening)

1. ✅ Update `vlm_stream_runner.py` to use custom model
2. ✅ Update `_generate_dialogue_with_cache()` to use new methods
3. ✅ Test with single video

### Step 4: Test Strategies (Day 2 - Morning)

1. ✅ Test drop_all with KV cache
2. ✅ Test drop_middle with KV cache
3. ✅ Test summarize_and_drop with KV cache
4. ✅ Verify overflow handling is triggered
5. ✅ Compare metrics across strategies

### Step 5: Documentation (Day 2 - Afternoon)

1. ✅ Document custom model architecture
2. ✅ Add usage examples
3. ✅ Update README
4. ✅ Create migration guide

## Key Differences: ProAssist vs. CustomSmolVLM

| Aspect | ProAssist | CustomSmolVLM |
|--------|-----------|---------------|
| **Base Model** | LlamaForCausalLM | SmolVLMForConditionalGeneration |
| **Vision Encoder** | CLIP (separate) | SiglipVisionModel (built-in) |
| **Projector** | Custom 2-layer MLP | Built-in connector |
| **Image Tokens** | Custom `<image>` + sep | Standard `<image>` |
| **Chat Format** | LLaMA3 format | SmolVLM2 format |
| **Special Logic** | "Not talk" threshold | Standard generation |

## Testing Plan

### Test 1: Verify joint_embed()

```python
# Test script
from prospect.models.custom_smolvlm import CustomSmolVLMForConditionalGeneration
import torch
from PIL import Image

model = CustomSmolVLMForConditionalGeneration.from_pretrained(...)
processor = ...

# Prepare inputs
image = Image.open("test.jpg")
text = "<image>What do you see?"
inputs = processor(images=image, text=text, return_tensors="pt")

# Test joint_embed
embeds = model.joint_embed(**inputs)
print(f"Embeddings shape: {embeds.shape}")  # Should be [1, seq_len, hidden_dim]
```

### Test 2: Verify fast_greedy_generate()

```python
# Test generation
embeds = model.joint_embed(**inputs)
output_ids, kv_cache = model.fast_greedy_generate(
    embeds,
    past_key_values=None,
    max_length=50
)

print(f"Generated: {processor.tokenizer.decode(output_ids[0])}")
print(f"KV cache size: {kv_cache[0][0].shape[2]} tokens")
```

### Test 3: Verify KV Cache Accumulation

```python
# Test accumulation across multiple frames
past_kv = None

for frame in frames[:10]:
    inputs = processor(images=frame, text="<image>Describe", ...)
    embeds = model.joint_embed(**inputs)
    output_ids, past_kv = model.fast_greedy_generate(embeds, past_kv, ...)
    
    print(f"Frame {i}: KV cache = {past_kv[0][0].shape[2]} tokens")

# Expected output:
# Frame 0: KV cache = 1000 tokens
# Frame 1: KV cache = 2000 tokens
# Frame 2: KV cache = 3000 tokens
# ...
```

### Test 4: Verify Strategy Activation

```python
# Test with drop_all strategy
runner = VLMStreamRunner(
    model_name="HuggingFaceTB/SmolVLM2-Instruct",
    context_strategy_type="drop_all",
    use_kv_cache=True,
    max_seq_len=4096,
)

# Run on long video
results = runner.run_inference_on_video(video)

# Check logs for:
# - "KV cache overflow: XXXX tokens, applying drop_all strategy"
# - "After drop_all: KV cache = 0 tokens"
```

## Expected Behavior After Implementation

### Without Overflow (Short Video)

```
Frame 1: Generate dialogue
  - KV cache: 0 → 1200 tokens
  - Strategy: Not triggered

Frame 2: Generate dialogue
  - KV cache: 1200 → 2400 tokens
  - Strategy: Not triggered

Frame 3: Generate dialogue
  - KV cache: 2400 → 3600 tokens
  - Strategy: Not triggered
```

### With Overflow (Long Video)

```
Frame 1-3: (as above)
  - KV cache grows: 0 → 1200 → 2400 → 3600

Frame 4: Generate dialogue
  - KV cache: 3600 → 4800 tokens
  - ⚠️ OVERFLOW! (exceeds 4096 - 128 = 3968)
  - Strategy: drop_all triggered
  - KV cache: 4800 → 0 tokens
  - Last msg: Preserved

Frame 5: Generate dialogue
  - KV cache: 0 → 1200 tokens
  - Strategy: Not triggered
  - (Cycle repeats)
```

## Benefits of Custom Model

1. ✅ **Full KV cache control** - Explicit management like ProAssist
2. ✅ **Strategy activation** - Overflow handling actually works
3. ✅ **Context accumulation** - Model sees history
4. ✅ **Extensible** - Easy to add custom logic (e.g., "not talk" threshold)
5. ✅ **Compatible** - Works with existing strategy system

## Risks and Mitigations

### Risk 1: SmolVLM2 Architecture Complexity
- **Risk**: SmolVLM2 has complex vision-language integration
- **Mitigation**: Study Idefics2 architecture carefully, test incrementally

### Risk 2: Breaking Standard Functionality
- **Risk**: Custom model might break standard HF features
- **Mitigation**: Inherit properly, override minimally, test thoroughly

### Risk 3: Performance Degradation
- **Risk**: Custom generation loop might be slower
- **Mitigation**: Use torch.no_grad(), optimize critical paths

## Alternative: Simpler Approach

If custom model is too complex, we could:

### Option A: Use ProAssist's Trained Model
- Load ProAssist's model weights
- Use their runner directly
- Strategies work out of the box
- **Downside**: Not using SmolVLM2

### Option B: Simplified KV Cache (No Custom Model)
- Use standard HF generate() with `past_key_values`
- Accept limitations (not perfect accumulation)
- Implement basic overflow detection
- **Downside**: Not as clean as ProAssist

## Recommendation

**Implement the custom model** following ProAssist's pattern:
1. It's the cleanest solution
2. Gives us full control
3. Enables all strategies properly
4. Sets foundation for future enhancements (DST, etc.)
5. Estimated time: 1-2 days

## Files to Create

1. `custom/src/prospect/models/__init__.py`
2. `custom/src/prospect/models/configuration_custom_smolvlm.py`
3. `custom/src/prospect/models/custom_smolvlm.py`
4. `custom/src/prospect/models/processing_custom_smolvlm.py`
5. `custom/src/prospect/models/test_custom_model.py` (test script)

## Files to Modify

1. `custom/src/prospect/runners/vlm_stream_runner.py` - Use custom model
2. `custom/src/prospect/context_strategies/summarize_and_drop.py` - Use joint_embed
3. `custom/config/prospect/model/smolvlm2.yaml` - Add custom model flag

## Success Criteria

✅ Custom model loads successfully  
✅ `joint_embed()` produces correct embeddings  
✅ `fast_greedy_generate()` generates text  
✅ KV cache accumulates across frames  
✅ Overflow triggers strategy  
✅ All three strategies work correctly  
✅ Metrics comparable to stateless baseline  

## Next Steps

1. Review this plan
2. Approve approach
3. Begin implementation with Phase 1
4. Test incrementally
5. Document findings

Ready to proceed?
