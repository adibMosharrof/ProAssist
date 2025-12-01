# Token-Level Training Plan for Prospect DST (Minimal Path)

## Goal
Get training running **this week** with token-level binary decisions to benchmark if DST helps with model performance.

---

## Why Token-Level (Instead of Windowed)

Your existing `test_training.json` already has:
- ✅ Full conversation with all turns and timestamps
- ✅ Frame ranges (`start_frame`, `end_frame`) per turn
- ✅ DST state annotations (`dst_state`)
- ✅ Speaking/DST events clearly marked (`role: "assistant"`, `role: "DST_UPDATE"`)

**Prospect's advantage over ProAssist**:
- ProAssist uses a single "silence" head with **inverted semantics** (predicts "should be silent")
- Prospect has **two separate direct heads** (speaking decision + DST update decision)
- Your labels align directly with what you want to predict → **no semantic inversion needed**
- Simpler loss computation: straightforward BCE without frame mask weighting

---

## Minimal Implementation Plan (1-2 days)

Your dataset already has everything needed! The conversation contains:
- `start_frame`, `end_frame` for each turn
- `role` field identifying whether turn is `assistant`, `DST_UPDATE`, `user`, or `system`
- `neg_frame_sampling_rate` for controlling negative frame sampling (already in your data structure!)

We just need to leverage this to generate per-token binary labels with **direct semantics** and **negative sampling**.

### Handling Negative Frames with Sampling

Most frames in a clip are "negative" (no speaking, no DST update). Without handling this class imbalance, the model trains on mostly silent frames and could be biased.

**Solution: Negative Frame Sampling**
- Always include positive frames (label=1)
- Randomly sample negative frames (label=0) at rate `neg_frame_sampling_rate`
- This balances the dataset without discarding data

**Example**:
```
Total frames: 1000
Speaking frames: 100 (frames where assistant speaks)
Silent frames: 900 (remaining frames)

With neg_frame_sampling_rate=0.5:
- Keep all 100 speaking frames
- Keep only ~450 of the 900 silent frames
- Total supervised: 550 frames instead of 1000
- Positive ratio: 100/550 = 18% (much better than 100/1000 = 10%)
```

**Implementation**: Mark frames with label=-100 (ignore index) if they don't pass the sampling threshold. PyTorch's BCE loss automatically ignores these frames.

### Phase 1: Modify DSTTrainingDataset (2-3 hours)

**File**: `custom/src/prospect/data_sources/dst_training_dataset.py` (modify existing)

**Task**: In `__getitem__()`, add token-level label generation after frames are loaded

**Algorithm** (uses existing frame ranges + negative sampling):
```
Initialize labels to -100 (ignore_index) for all frames

1. Sample negative frames:
   For each frame index 0 to total_frames:
     - If random() < neg_frame_sampling_rate:
       - Mark as label=0 (negative, no speaking/DST update)
     - Else:
       - Keep as -100 (ignore in loss)

2. Mark positive frames (override sampled negatives):
   For each turn in conversation:
     - If role == "assistant":
       - Set speaking_labels[start_frame:end_frame] = 1
     - If role == "DST_UPDATE":
       - Set dst_update_labels[start_frame:end_frame] = 1
     - Both labels might be 1 for overlapping frames

3. Add to return dict:
   {
       ...existing fields...
       "speaking_labels": [num_frames],      # 1=assistant, 0=sampled negative, -100=ignored
       "dst_update_labels": [num_frames],    # 1=DST_UPDATE, 0=sampled negative, -100=ignored
   }
```

**Key Methods** (add to `DSTTrainingDataset` class):

```python
def _generate_token_level_labels(
    self, turns: List[Dict], total_frames: int, neg_sampling_rate: float = 0.5
) -> tuple:
    """
    Generate per-token binary labels with negative frame sampling.
    
    Direct semantics:
    - speaking_labels[i] = 1 if frame i is covered by assistant turn
    - speaking_labels[i] = 0 if frame i is sampled as negative
    - speaking_labels[i] = -100 if frame i is not sampled (ignored in loss)
    
    Same for dst_update_labels.
    
    Args:
        turns: List of conversation turns (each has start_frame, end_frame, role)
        total_frames: Total number of frames loaded for this clip
        neg_sampling_rate: Probability to include a negative frame (0.5 = 50%)
    
    Returns:
        (speaking_labels, dst_update_labels)
        Each tensor shape: [total_frames], dtype=long
    """
    import random
    
    # Initialize with -100 (ignore index for loss functions)
    speaking_labels = torch.full(total_frames, -100, dtype=torch.long)
    dst_update_labels = torch.full(total_frames, -100, dtype=torch.long)
    
    # Sample negative frames
    # For each frame, decide whether to supervise it (if it's negative)
    for frame_idx in range(total_frames):
        if random.random() < neg_sampling_rate:
            # Include this frame as a negative example
            speaking_labels[frame_idx] = 0
            dst_update_labels[frame_idx] = 0
    
    # Mark positive frames (assistant and DST_UPDATE)
    # These are always supervised, regardless of sampling
    for turn in turns:
        if "start_frame" not in turn or "end_frame" not in turn:
            continue
        
        start_frame = turn["start_frame"]
        end_frame = turn["end_frame"]
        
        # Clamp to valid frame range
        start_frame = max(0, min(start_frame, total_frames - 1))
        end_frame = max(0, min(end_frame, total_frames))
        
        if start_frame < end_frame:
            if turn["role"] == "assistant":
                # Mark frames as positive for speaking
                speaking_labels[start_frame:end_frame] = 1
                # Ensure dst_update_labels are set to at least 0 if not already labeled
                mask = dst_update_labels[start_frame:end_frame] == -100
                dst_update_labels[start_frame:end_frame] = torch.where(
                    mask,
                    torch.tensor(0),
                    dst_update_labels[start_frame:end_frame]
                )
            elif turn["role"] == "DST_UPDATE":
                # Mark frames as positive for DST update
                dst_update_labels[start_frame:end_frame] = 1
                # Ensure speaking_labels are set to at least 0 if not already labeled
                mask = speaking_labels[start_frame:end_frame] == -100
                speaking_labels[start_frame:end_frame] = torch.where(
                    mask,
                    torch.tensor(0),
                    speaking_labels[start_frame:end_frame]
                )
    
    return speaking_labels, dst_update_labels
```

**Modify `__getitem__()` return dict**:
```python
# In __getitem__, after creating the sample dict (around line 265):

# Generate token-level labels from frame ranges with negative sampling
speaking_labels, dst_update_labels = \
    self._generate_token_level_labels(
        valid_turns, 
        frame_count,
        neg_sampling_rate=self.neg_frame_sampling_rate  # Use the rate from config
    )

sample = {
    "video_uid": clip["video_uid"],
    ...existing fields...
    "frames": frames_tensor,
    # ADD THESE 2 FIELDS:
    "speaking_labels": speaking_labels,
    "dst_update_labels": dst_update_labels,
}

return sample
```

---

### Phase 2: Modify DSTDataCollator (1-2 hours)

**File**: `custom/src/prospect/data_sources/dst_data_collator.py` (modify existing)

**Task**: Extract binary labels from samples, batch them with padding. Respect -100 (ignore index).

**Add to `_process_multimodal_batch` method** (after extracting frame_tensors_list):

```python
# Extract token-level binary labels from samples
# Labels are -100 (ignore), 0 (negative), or 1 (positive)
speaking_labels_list = [s.get("speaking_labels", torch.full((0,), -100, dtype=torch.long)) for s in samples]
dst_update_labels_list = [s.get("dst_update_labels", torch.full((0,), -100, dtype=torch.long)) for s in samples]

# Pad to same length, using -100 as padding value (ignore index)
speaking_labels = self._pad_label_sequences(speaking_labels_list, pad_value=-100)
dst_update_labels = self._pad_label_sequences(dst_update_labels_list, pad_value=-100)
```

**Add helper method to `DSTDataCollator`**:

```python
def _pad_label_sequences(
    self, sequences: List[torch.Tensor], pad_value=-100
) -> torch.Tensor:
    """
    Pad label sequences to same length.
    
    Args:
        sequences: List of label tensors (shape [seq_len])
        pad_value: Value to use for padding (default -100 for ignore index)
    
    Returns:
        Padded tensor of shape [batch_size, max_len]
    """
    if not sequences or all(s.numel() == 0 for s in sequences):
        return torch.full((len(sequences), 0), pad_value, dtype=torch.long)
    
    max_len = max(s.shape[0] if s.dim() > 0 else 0 for s in sequences)
    
    padded = torch.full((len(sequences), max_len), pad_value, dtype=torch.long)
    
    for i, seq in enumerate(sequences):
        if seq.numel() > 0:
            seq_len = min(seq.shape[0], max_len)
            padded[i, :seq_len] = seq[:seq_len]
    
    return padded
```

**Add to batch dict return** (around line 110):

```python
batch = {
    # ... existing fields ...
    "speaking_labels": speaking_labels,        # [batch, seq_len] with values -100, 0, 1
    "dst_update_labels": dst_update_labels,    # [batch, seq_len] with values -100, 0, 1
}
```

```python
# Extract token-level binary labels from samples
speaking_labels_list = [s.get("speaking_labels", torch.zeros(0)) for s in samples]
dst_update_labels_list = [s.get("dst_update_labels", torch.zeros(0)) for s in samples]

# Pad to same length
speaking_labels = self._pad_label_sequences(speaking_labels_list)
dst_update_labels = self._pad_label_sequences(dst_update_labels_list)
```

**Add helper method to `DSTDataCollator`**:

```python
def _pad_label_sequences(
    self, sequences: List[torch.Tensor], pad_value=0
) -> torch.Tensor:
    """Pad label sequences to same length."""
    if not sequences or all(s.numel() == 0 for s in sequences):
        return torch.zeros((len(sequences), 0), dtype=torch.long)
    
    max_len = max(s.shape[0] if s.dim() > 0 else 0 for s in sequences)
    dtype = sequences[0].dtype if sequences[0].numel() > 0 else torch.long
    
    padded = torch.full((len(sequences), max_len), pad_value, dtype=dtype)
    
    for i, seq in enumerate(sequences):
        if seq.numel() > 0:
            seq_len = min(seq.shape[0], max_len)
            padded[i, :seq_len] = seq[:seq_len]
    
    return padded
```

**Add to batch dict return** (around line 110):

```python
batch = {
    # ... existing fields ...
    "speaking_labels": speaking_labels,
    "dst_update_labels": dst_update_labels,
}
```

---

### Phase 3: Modify DSTSmolVLMWithStrategies (1-2 hours)

**File**: `custom/src/prospect/models/dst_smolvlm_with_strategies.py` (modify existing)

**Task**: Update forward pass with direct BCE loss. Handle -100 (ignore index) properly.

**Add parameters to forward signature**:

```python
def forward(
    self,
    input_ids=None,
    pixel_values=None,
    labels=None,
    speaking_labels=None,        # NEW: [batch, seq_len] with -100/0/1
    dst_update_labels=None,       # NEW: [batch, seq_len] with -100/0/1
    **kwargs
):
```

**Replace binary loss computation** (find existing speaking/dst loss code, replace with direct BCE):

```python
if speaking_labels is not None and dst_update_labels is not None:
    # Use reduction='none' initially to handle ignore index manually
    bce_loss_fn = nn.BCEWithLogitsLoss(reduction='none')
    
    # ===== SPEAKING LOSS =====
    speaking_logits_flat = speaking_logits.squeeze(-1)  # [batch, seq_len]
    speaking_labels_flat = speaking_labels.float()      # [batch, seq_len]
    
    speak_loss_all = bce_loss_fn(speaking_logits_flat, speaking_labels_flat)
    
    # Ignore -100 labels (unsampled negative frames)
    speak_mask = speaking_labels_flat != -100
    if speak_mask.any():
        speak_loss = speak_loss_all[speak_mask].mean()
    else:
        speak_loss = 0.0
    
    # ===== DST UPDATE LOSS =====
    dst_logits_flat = dst_update_logits.squeeze(-1)     # [batch, seq_len]
    dst_labels_flat = dst_update_labels.float()         # [batch, seq_len]
    
    dst_loss_all = bce_loss_fn(dst_logits_flat, dst_labels_flat)
    
    # Ignore -100 labels (unsampled negative frames)
    dst_mask = dst_labels_flat != -100
    if dst_mask.any():
        dst_loss = dst_loss_all[dst_mask].mean()
    else:
        dst_loss = 0.0
    
    # Combine with LM loss
    lm_loss = outputs.loss if hasattr(outputs, 'loss') else 0.0
    loss = speak_loss + dst_loss + lm_loss
```

**Key points**:
- Use `reduction='none'` to get per-element losses
- Mask out -100 values (unsampled frames) before averaging
- Average only over sampled frames (labels != -100)

---

---

## Summary: 3 Changes with Negative Sampling

| Change | File | Effort | What It Does |
|--------|------|--------|------------|
| Label generation + sampling | `dst_training_dataset.py` | ~60 lines | Generate labels with -100 for unsampled negatives, 0 for sampled negatives, 1 for positives |
| Label batching | `dst_data_collator.py` | ~30 lines | Pad labels to batch max length, use -100 as padding value |
| Loss computation with ignore index | `dst_smolvlm_with_strategies.py` | ~35 lines | Direct BCE loss, mask out -100 values before averaging |

**Total effort**: ~4-6 hours of development

---

## Label Value Semantics

**After Phase 1 (Dataset)**:
- `label = -100`: Frame was not sampled as negative (ignored in loss)
- `label = 0`: Frame was sampled as negative (or overlaps with no positive event)
- `label = 1`: Frame overlaps with assistant or DST_UPDATE event

**After Phase 2 (Collator)**:
- Padding also uses -100 (consistent ignore index)
- Batch shape: `[batch_size, max_seq_len]`

**Phase 3 (Model)**:
- Only compute loss over frames where `label != -100`
- Average loss over sampled frames only
- -100 values never contribute to gradients

---

## Why This Works

1. **Handles class imbalance**: By sampling negatives, you balance positive:negative ratio
2. **Efficient**: Only compute loss for sampled frames, not all frames
3. **Standard PyTorch**: -100 is the default ignore_index in many loss functions
4. **Flexible**: Adjust `neg_frame_sampling_rate` per dataset (val=1.0, train=0.5)
5. **ProAssist-compatible**: Similar to their approach but simpler (no label inversion)
