# Simplified DST Training - ProAssist Pattern

## Understanding ProAssist's Approach

After examining `mmassist/train/` files, here's how ProAssist actually trains:

### 1. **Frame-Based Training**
Instead of temporal sequences, ProAssist uses **frame indices**. We adapt this by embedding frame information directly in conversation turns:
```python
conversation = [
    {"role": "assistant", "content": "Let's start...", "start_frame": 0, "end_frame": 5},
    {"role": "assistant", "content": "Now attach...", "start_frame": 5, "end_frame": 10},
    {"role": "DST_UPDATE", "content": [...], "start_frame": 10, "end_frame": 11}
]
```

### 2. **Negative Frame Sampling**
```python
# Control how many "silent" frames to learn from
neg_frame_sampling_rate = 0.5  # Learn from 50% of frames when user is silent
```

### 3. **Learnable Token Ranges**
- **Assistant turns**: Always compute language modeling loss
- **Frame tokens**: Only compute loss for **sampled frames** (based on `neg_frame_sampling_rate`)
- **No temporal BCE loss** - just language modeling for assistant turns

### 4. **Training Loss Structure**
```python
loss = language_modeling_loss(
    assistant_content_tokens,     # Always compute loss
    frame_tokens[sampled_indices] # Only for sampled frames
)
```

## ProAssist Progress Summaries vs DST State Context

### **ProAssist's Approach: Progress Summaries**
ProAssist provided context through **progress summaries** in each conversation turn:
```json
{
  "conversation": [
    {
      "role": "assistant",
      "content": "Let's start by attaching the chassis",
      "progress": "Task has not started yet. No steps completed."
    },
    {
      "role": "assistant",
      "content": "Great! Now attach the wheel to the chassis",
      "progress": "Step 1 (Attach chassis) is in progress. Step 2 (Attach wheel) not started."
    }
  ]
}
```

### **DST Approach: Embedded State Context**
We replace progress summaries with **DST state information** embedded directly in conversation turns:
```json
{
  "conversation": [
    {
      "role": "assistant",
      "content": "Let's start by attaching the chassis",
      "dst_context": {"S1": "not_started", "S2": "not_started"}
    },
    {
      "role": "assistant",
      "content": "Great! Now attach the wheel to the chassis",
      "dst_context": {"S1": "in_progress", "S2": "not_started"}
    }
  ]
}
```

### **Key Differences**
- **ProAssist**: Natural language progress descriptions
- **DST**: Structured state mappings (step_id → state)
- **Advantage**: DST provides machine-readable state that can be directly used for training loss computation

## Why This Works Better

### ✅ **Handles Imbalance Naturally**
- Most frames are "silent" (no talking)
- `neg_frame_sampling_rate` controls learning from silent frames
- No need for complex temporal BCE loss

### ✅ **Follows Proven Pattern**
- Uses ProAssist's validated approach
- Frame-based processing is what the models expect
- Simpler architecture with less overhead

### ✅ **Efficient Training**
- Only processes selected frames for loss computation
- Leverages existing ProAssist infrastructure
- Direct token-level supervision

### ✅ **Structured State Context**
- DST state provides clear, structured context instead of natural language summaries
- Enables direct loss computation on state predictions
- Maintains temporal continuity across conversation splits

### ✅ **Split-Safe Context Preservation**
- When conversations are split due to token limits, DST state ensures continuity
- Each training sample receives correct initial DST state in metadata
- DST_CONTEXT turns maintain state information across segment boundaries

## Simplified DST Training Implementation

### **Enhanced Conversation Format**

custom/docs/training/6.1-prospect_training_data.json

the above json file contains a sample data point.

**Note**: Frame information is embedded directly in each conversation turn. DST context is included in both assistant turns (for state awareness) and DST_UPDATE events (showing state at transition time). Roles use DSTRole enum values. For token efficiency, string values get converted to integers during model training (assistant=1, DST_UPDATE=3, etc.).

### **Training Losses**
1. **Language Modeling Loss**: Assistant content (same as ProAssist)
2. **DST State Classification**: Predict current DST state from assistant turns with dst_context
3. **DST Transition Prediction**: Predict state transitions from DST_UPDATE events

**DST State Context Usage**: Unlike ProAssist's progress summaries that were just context, DST state information serves dual purposes:
- **Context for the model**: Provides structured state information in assistant turns
- **Training supervision**: Enables direct loss computation on state predictions and transitions

### **Simplified Architecture**
```
Input: Conversation turns with embedded frames + DST context
↓
Language Model (processes text + frame tokens)
Language Modeling Loss ← Assistant content + frame tokens
↓
DST Heads (attached to model)
DST State Loss ← dst_context in assistant turns
DST Transition Loss ← DST_UPDATE event content
```

## Key Benefits

### ✅ **Much Simpler**
- No temporal sequence management
- No BCE loss for timing decisions  
- Follows proven ProAssist pattern

### ✅ **More Efficient**
- Uses existing ProAssist infrastructure
- Frame-based processing
- Direct token-level supervision

### ✅ **Better Foundation**
- Leverages what works in ProAssist
- Easier to integrate with existing models
- Less complex architecture

## Implementation Steps

### 1. **Dataset Conversion**
- Convert enhanced format to ProAssist-style conversation
- Add `dst_state` and `dst_transition` to frame turns
- Include DST context in assistant turns

### 2. **Model Architecture** 
- Add DST heads to ProAssist model
- Frame tokens → DST state classification
- Frame tokens → DST transition prediction

### 3. **Training**
- Same training loop as ProAssist
- Add DST loss components
- Use `neg_frame_sampling_rate` for frame selection

This approach is much simpler, follows proven patterns, and avoids the complexity of temporal sequences while still enabling DST training.