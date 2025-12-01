# Negative Frame Subsampling Strategy

## Overview

The assistant model must learn to guide the user through a task. However, in a typical task video, the assistant only speaks occasionally - most frames are "silent" moments where no action is needed. This creates a severe class imbalance problem:

- **Positive frames**: Moments when the assistant speaks (assistant utterance) or updates task state (DST_UPDATE)
  - These are rare (~5-20% of total frames)
  - Model MUST learn from these

- **Negative frames**: Silent moments when the assistant should NOT speak or update DST
  - These are abundant (~80-95% of total frames)
  - Including all of them during training wastes computation and skews loss

## The Problem: Class Imbalance

Without negative frame subsampling:
1. Model sees mostly silent frames
2. Loss dominated by negative class (silence)
3. Binary classifiers (speaking_binary, dst_binary) biased toward predicting "silent"
4. Model learns to rarely speak (safer than speaking incorrectly)
5. Training is inefficient and unreliable

## The Solution: Negative Frame Subsampling

We strategically sample a subset of negative frames during training while keeping all positive frames:

### Positive Frames (Always Learn)
- **Assistant turns**: Frames during assistant utterances
  - Learn to generate: correct assistant response text
  - Learn to predict: speaking_binary = 1
  
- **DST_UPDATE turns**: Frames when task state updates
  - Learn to generate: correct DST JSON update
  - Learn to predict: dst_binary = 1

### Negative Frames (Sample Strategically)
- **All frames not in assistant or DST_UPDATE turns**: 
  - This includes user/system turns, silent moments, gaps between turns, scene transitions
  - Sample: neg_frame_sampling_rate × total_negative_frames
  - Learn to predict: speaking_binary = 0, dst_binary = 0
  - Learn to recognize: when NOT to speak or update

### Sampling Rate

The `neg_frame_sampling_rate` controls the proportion of negative frames included:

- **Training**: `neg_frame_sampling_rate = 0.5`
  - Include ~50% of negative frames
  - Balances class distribution without full computation
  - Approximate ratio: for every 10 positive frames, sample ~10 negative frames
  
- **Validation**: `neg_frame_sampling_rate = 1.0`
  - Include ALL frames (positive and negative)
  - Gives unbiased evaluation on real data distribution
  - Estimates actual performance on deployment
  
- **Test**: `neg_frame_sampling_rate = 1.0`
  - Full realistic distribution

## Data Structure in Our Implementation

Our conversation data structure embeds frame information in each turn:

```json
{
  "conversation": [
    {
      "role": "system",
      "content": "...",
      "start_frame": 0,
      "end_frame": 10
    },
    {
      "role": "user",
      "content": "...",
      "start_frame": 10,
      "end_frame": 25
    },
    {
      "role": "assistant",
      "content": "...",
      "start_frame": 25,
      "end_frame": 35
    },
    {
      "role": "DST_UPDATE",
      "content": "{...}",
      "start_frame": 35,
      "end_frame": 40
    }
  ]
}
```

**Frame Classification:**

We classify frames by examining the **entire video timeline** and marking what's happening:

1. **Positive frames**: Any frames where model takes action
   - Frames in `assistant` turns → model generates text response
   - Frames in `DST_UPDATE` turns → model updates task state
   - Always include ALL positive frames in training (sampling_rate = 1.0)

2. **Negative frames**: All other frames in the video
   - Frames in `user` or `system` turns → someone else is speaking, no model action needed
   - **Frames with no turn at all** → silent/idle moments, scene transitions, waiting periods
   - These are the frames where model should learn to NOT speak and NOT update DST
   - Sample subset based on `neg_frame_sampling_rate`

The key insight: Negative frames aren't limited to user/system turns. The gaps between turns (uncovered frames) are also negative frames where the assistant must recognize "no action needed."

## Implementation Strategy

### Per-Turn Sampling

For each turn in the conversation:

1. **Assistant & DST_UPDATE turns**:
   - Always include all frames
   - Generate full learning labels for both model text generation and binary classification

2. **All other frames** (negative frames - user/system/silent/gaps):
   - Identify all frames NOT covered by assistant or DST_UPDATE turns
   - Calculate: `num_learn_frames = int(neg_frame_sampling_rate × num_negative_frames)`
   - Random sampling: Select `num_learn_frames` frames uniformly from all negative frames
   - Include selected frames in learning
   - Set binary labels: `speaking_binary = 0`, `dst_binary = 0` (no action needed)

### Learning Objectives During Negative Frames

When we sample and include negative frames in training:
- **speaking_gen**: No token targets (language generation not applicable)
- **speaking_binary**: Target = 0 (don't speak)
- **dst_gen**: No token targets (no DST update needed)
- **dst_binary**: Target = 0 (don't update state)

The binary classifiers learn to recognize these frames as "no action needed" moments.

## Benefits

1. **Faster Training**: Reduced compute per epoch (especially with sampling_rate < 1.0)
2. **Better Generalization**: Balanced class distribution prevents bias toward always-silent predictions
3. **Realistic Performance**: Validation with full distribution shows real-world performance
4. **Flexibility**: Sampling rate can be tuned:
   - Higher rate (e.g., 0.8): More conservative, better binary classification
   - Lower rate (e.g., 0.3): Faster training, focused on generation tasks

## Practical Example

**Video with 1000 frames:**
- Assistant speaks in frames 100-110 (10 frames) → **Positive**
- DST updates in frames 250-260 (10 frames) → **Positive**
- Everything else → **Negative** (980 frames)

**Training with neg_frame_sampling_rate=0.5:**
- Learn from: 10 + 10 + (980 × 0.5) = 510 frames total
- Class ratio: ~20 positive : 490 negative (still imbalanced, but much better than 20:980)

**Validation with neg_frame_sampling_rate=1.0:**
- Learn from: 10 + 10 + 980 = 1000 frames
- Class ratio: exact real distribution (20:980)
- True test of model's ability to recognize silence

## Related Components

- `DSTTrainingDataset.get_learn_ranges_separated()`: Handles frame sampling logic
- `DSTDataCollator._get_learn_ranges_for_dst()`: Converts frame ranges to token positions
- `dst_smolvlm_with_strategies.py`: Computes losses only for sampled frames (via label tensor masking)
