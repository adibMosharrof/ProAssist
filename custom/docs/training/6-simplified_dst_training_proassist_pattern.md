# Simplified DST Training - ProAssist Pattern

## Understanding ProAssist's Approach

After examining `mmassist/train/` files, here's how ProAssist actually trains:

### 1. **Frame-Based Training**
Instead of temporal sequences, ProAssist uses **frame indices**:
```python
conversation = [
    {"role": "frames", "start": 0, "end": 5},      # Frame 0-4
    {"role": "assistant", "content": "Let's start..."},
    {"role": "frames", "start": 5, "end": 10},     # Frame 5-9  
    {"role": "assistant", "content": "Now attach..."},
    {"role": "frames", "start": 10, "end": 15}     # Frame 10-14
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

## Simplified DST Training Implementation

### **Enhanced Conversation Format**
```json
{
  "conversation": [
    {
      "role": "frames", 
      "start": 0, 
      "end": 5,
      "dst_state": {"S1": "not_started", "S2": "not_started"}  # NEW: Add DST context
    },
    {
      "role": "assistant", 
      "content": "Let's start by attaching the chassis",
      "dst_context": {"S1": "not_started", "S2": "not_started"}
    },
    {
      "role": "frames",
      "start": 97, 
      "end": 118,
      "dst_transition": [{"id": "S1", "transition": "start"}]  # NEW: State transition
    },
    {
      "role": "assistant",
      "content": "Great! Now attach the wheel to the chassis",
      "dst_context": {"S1": "in_progress", "S2": "not_started"}
    }
  ],
  "dst": [
    {"id": "S1", "name": "Attach chassis", "start_ts": 97.2, "end_ts": 118.7},
    {"id": "S2", "name": "Attach wheel", "start_ts": 130.7, "end_ts": 152.1}
  ]
}
```

### **Training Losses**
1. **Language Modeling Loss**: Assistant content (same as ProAssist)
2. **DST State Classification**: Frame tokens with DST context
3. **DST Transition Prediction**: Frame tokens with transition info

### **Simplified Architecture**
```
Input: Frames + Conversation text
↓
Language Model (for assistant content)
Language Modeling Loss ← Assistant turns
↓
DST Heads (attached to model)
DST State Loss ← Frames with DST context
DST Transition Loss ← Frames with transitions
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