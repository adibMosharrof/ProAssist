# DST Training Batch Creation Plan - Complete Architecture

## Overview

This document describes the complete DST (Dialog State Tracking) training architecture, including batch creation strategy, multi-task learning approach, and how training relates to inference. The architecture follows ProAssist's proven patterns while extending them for DST-aware multi-task learning.

## 🎯 Training Architecture Summary

### **Training vs Inference Pattern**

| Aspect | Training | Inference |
|--------|----------|-----------|
| **Memory** | Stateless (no KV cache) | Stateful (with KV cache) |
| **Context** | Frame + conversation snapshot | Accumulated conversation + context |
| **Compression** | None | DST-aware summarization |
| **Data Flow** | Independent batches | Sequential frame processing |
| **Memory Usage** | Predictable, bounded | Grows over time, needs compression |

### **Key Insight**: 
**Training** teaches the model *what to learn*, **Inference** teaches it *how to apply* that knowledge with DST-aware context management.

## 🧠 DST Multi-Task Learning Architecture

### **Model Architecture**
```
Input: Video Frame + Conversation Context + DST State
    ↓
SmolVLM2 (Base VLM)
    ↓
[4 Training Heads] ← Multi-task learning
    ├─ Speaking Decision Head (Binary: should I speak?)
    ├─ DST Update Decision Head (Binary: should I update DST?)
    ├─ Response Generation Head (Text output - standard LM)
    └─ DST State Update Head (3-state classification)
    ↓
Output: Multi-task predictions + combined loss
```

### **Training Loss Computation**
```python
total_loss = (
    language_loss * 1.0 +                    # Standard language modeling
    speaking_loss * weight +                 # Focal loss (class imbalance)
    dst_update_loss * weight +               # Focal loss (class imbalance)
    dst_state_loss * weight                  # Cross entropy
)
```

## 📊 Batch Creation Strategy

### **Frame-Level Sampling Approach**

Each training sample represents one frame at a specific timestamp:

```python
training_sample = {
    # Video context (frame at timestamp t)
    "video_frame": tensor,                    # Shape: (3, 224, 224)
    "frame_timestamp": float,                 # e.g., 45.5 seconds
    "fps": int,                              # e.g., 2 fps
    
    # Conversation context (up to timestamp t)
    "conversation": List[Dict],              # dialogue history so far
    "input_ids": tensor,                     # tokenized conversation (max 4096 tokens)
    "attention_mask": tensor,                # attention mask for tokens
    
    # DST context at timestamp t
    "dst_state": Dict,                       # current DST state
    "dst_annotations": List,                 # DST data active at timestamp
    
    # Multi-task labels for this timestamp
    "speaking_labels": int,                  # Should speak? (0/1)
    "dst_update_labels": int,               # Should update DST? (0/1)
    "dst_state_labels": int,                # Current state (0/1/2)
    "response_labels": str,                 # Ground truth response (if speaking)
    
    # Metadata
    "video_id": str,                        # Video identifier
    "frame_idx": int,                       # Frame index in video
}
```

### **Video → Training Samples Conversion**

```
📹 Video with conversation + DST:
Timeline: [-----|-----|-----|-----|-----|-----]
Frames:   [F1  ][F2  ][F3  ][F4  ][F5  ][F6  ]
DST:      [Step1→Step2→Step3→Step4→Step5→Step6]
Speaking: [--------1---------][---1---][--1---]

🎓 Training samples generated:
Sample 1: F1 + Conversation_1 + DST_State_1 + Labels_1
Sample 2: F2 + Conversation_2 + DST_State_2 + Labels_2  
Sample 3: F3 + Conversation_3 + DST_State_3 + Labels_3
...
Sample 6: F6 + Conversation_6 + DST_State_6 + Labels_6
```

## 🔍 Frame Sampling Strategy

### **Sampling Points**

**1. Key Transition Frames:**
- DST state changes (step → step)
- Task phase transitions (prep → assembly → cleanup)
- Critical action boundaries

**2. Speaking Event Frames:**
- Frames where assistant speaks (±0.5s window)
- Decision-making moments (when to intervene)
- Teaching opportunities

**3. Regular Sampling:**
- Every N frames for continuity
- Uniform coverage of video timeline
- Balanced distribution across video

**4. Random Sampling:**
- Mix of above strategies
- Augment training data diversity
- Prevent overfitting to specific patterns

### **Sampling Configuration**
```yaml
sampling:
  strategy: "hybrid"          # key_frames + regular + speaking_events
  key_frame_interval: 25      # Sample every 25th frame (major events)
  speaking_window: 1.0        # ±1.0s around speaking events
  regular_interval: 50        # Sample every 50th frame (maintenance)
  random_sampling_ratio: 0.2  # 20% random samples
  max_frames_per_video: 200   # Cap samples per video
```

## 📝 Conversation Context Processing

### **Context Truncation Strategy**

**Maximum Length**: 4096 tokens (following ProAssist pattern)

**Truncation Priority** (from most to least important):
1. **Recent dialogue** (last 200 turns) - highest priority
2. **DST-related turns** (steps, decisions, completions)
3. **Speaker instructions** (critical guidance)
4. **Historical context** (older relevant information)

**Truncation Algorithm**:
```python
def truncate_conversation(conversation, max_tokens=4096):
    # 1. Keep recent dialogue (up to 200 turns)
    recent = conversation[-200:]
    
    # 2. Extract turns with tokenization
    turns_with_tokens = []
    current_tokens = 0
    
    for turn in reversed(recent):  # Start from most recent
        turn_tokens = len(tokenizer.encode(turn["content"]))
        if current_tokens + turn_tokens <= max_tokens:
            turns_with_tokens.append(turn)
            current_tokens += turn_tokens
        else:
            break
    
    # 3. Return reversed to maintain chronological order
    return list(reversed(turns_with_tokens))
```

## 🏷️ Multi-Task Label Generation

### **1. Speaking Decision Labels**

**Label**: `1` if assistant speaks within ±0.5s of frame timestamp, else `0`

```python
def generate_speaking_labels(conversation, timestamp):
    """Check if assistant speaks around this timestamp"""
    for turn in conversation:
        if turn["from"] == "assistant":
            turn_time = turn.get("timestamp", 0)
            if abs(turn_time - timestamp) <= 0.5:
                return 1
    return 0
```

### **2. DST Update Decision Labels**

**Label**: `1` if DST state changes within ±1.0s of frame timestamp, else `0`

```python
def generate_dst_update_labels(dst_annotations, timestamp):
    """Check if DST state changes around this timestamp"""
    before_state = get_dst_state_at_time(dst_annotations, timestamp - 1.0)
    after_state = get_dst_state_at_time(dst_annotations, timestamp + 1.0)
    return 1 if before_state != after_state else 0
```

### **3. DST State Classification Labels**

**Label**: Current DST state at timestamp
- `0`: Not Started
- `1`: In Progress  
- `2`: Completed

```python
def get_dst_state_label(dst_annotations, timestamp):
    """Get current DST state at timestamp"""
    active = get_active_dst_annotations(dst_annotations, timestamp)
    if not active:
        return 0  # Not Started
    
    # Check if any task is completed
    for annotation in active:
        if annotation["end_ts"] < timestamp:
            return 2  # Completed
    
    return 1  # In Progress
```

### **4. Response Generation Labels**

**Label**: Ground truth assistant response (if speaking), else empty string

```python
def generate_response_labels(conversation, timestamp):
    """Get ground truth response if speaking at timestamp"""
    for turn in conversation:
        if turn["from"] == "assistant":
            turn_time = turn.get("timestamp", 0)
            if abs(turn_time - timestamp) <= 0.5:
                return turn["value"]
    return ""  # No speaking event
```

## 🎯 Batch Formation

### **Training Batch Structure**

```python
batch = {
    # Vision inputs
    "pixel_values": torch.tensor(batch_size × 3 × 224 × 224),
    "attention_mask": torch.tensor(batch_size × max_seq_len),
    
    # Text inputs
    "input_ids": torch.tensor(batch_size × max_seq_len),
    "labels": torch.tensor(batch_size × max_seq_len),  # For language modeling
    
    # DST task labels
    "speaking_labels": torch.tensor(batch_size),        # Binary (0/1)
    "dst_update_labels": torch.tensor(batch_size),     # Binary (0/1)
    "dst_state_labels": torch.tensor(batch_size),      # Multi-class (0/1/2)
    
    # Metadata
    "timestamps": torch.tensor(batch_size),           # Frame timestamps
    "video_ids": List[str],                           # Video identifiers
    "frame_indices": torch.tensor(batch_size),        # Frame indices
}
```

### **Training Loop**

```python
for batch in dataloader:
    # Forward pass through DST model
    outputs = model(
        pixel_values=batch["pixel_values"],
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
    )
    
    # Compute multi-task loss
    loss = trainer.compute_loss(model, batch, outputs)
    
    # Backpropagation
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## 🏗️ Implementation Components

### **Data Pipeline Architecture**

```
Raw Video + Conversation + DST → DSTTrainingDataset
                                        ↓
Frame Sampling + Context Processing → Training Samples
                                        ↓
DSTDataCollator → Batched Tensors
                                        ↓
DSTCustomTrainer → Multi-task Loss → Model Update
```

### **Key Components**

**1. DSTTrainingDataset** (`custom/src/prospect/data_sources/dst_training_dataset.py`)
- Loads video frames, conversations, and DST data
- Performs frame-level sampling
- Generates conversation context up to timestamp

**2. DSTDataCollator** (`custom/src/prospect/data_sources/dst_data_collator.py`)
- Collates individual samples into batches
- Handles tokenization and padding
- Manages variable-length sequences

**3. DSTCustomTrainer** (`custom/src/prospect/train/dst_custom_trainer.py`)
- Computes multi-task loss with focal loss
- Manages training loop and optimization
- Handles checkpointing and logging

**4. DSTSmolVLMWithStrategies** (`custom/src/prospect/models/dst_smolvlm_with_strategies.py`)
- Extends SmolVLMWithStrategies with 4 DST heads
- Inherits proven VLM functionality
- Supports context compression for inference

## 📈 Training Benefits

### **Advantages of Frame-Level Sampling**

✅ **Matches Inference Pattern**: Training sees same structure as VLM stream runner  
✅ **Stateless Training**: Each sample independent, predictable memory usage  
✅ **Multi-Task Learning**: All 4 heads trained simultaneously  
✅ **DST-Aware**: Model learns when to update state based on video progress  
✅ **Scalable**: Multiple samples per video → rich training data  
✅ **Balanced Sampling**: Key frames + regular frames + speaking events  

### **Memory Efficiency**

- **Training**: Predictable memory (batches independent)
- **Inference**: Adaptive memory (context compression when needed)
- **No State Pollution**: Training doesn't interfere with inference state management

## 🎯 Next Steps

### **Implementation Priorities**

1. **DSTTrainingDataset**: Implement frame sampling and context processing
2. **DSTDataCollator**: Add proper tokenization and batching
3. **DSTCustomTrainer**: Verify multi-task loss computation
4. **Testing**: Create comprehensive test suite
5. **Training Scripts**: Update configuration and runner scripts

### **Configuration Updates**

- Update `custom/config/prospect/data_source/dst_training.yaml`
- Modify `custom/runner/run_dst_training.sh` for new training mode
- Add sampling parameters to configuration

### **Validation Strategy**

1. **Unit Tests**: Test each component individually
2. **Integration Tests**: Test complete training pipeline
3. **Inference Tests**: Verify trained model works with VLM stream runner
4. **Performance Tests**: Measure training speed and memory usage

This architecture provides a solid foundation for DST-aware multi-task learning while maintaining the proven ProAssist training patterns for reliability and performance.