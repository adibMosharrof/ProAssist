# DST + Speak Training Data Model - Complete Implementation Plan

## Overview

This document provides the complete implementation plan for transforming enhanced DST (Dialog State Tracking) + Speak data into ProAssist-compatible training format. The enhanced data includes DST snapshots for each conversation turn, providing structured state information instead of textual progress summaries.

**Key Innovation**: DST snapshots enable precise task state tracking vs. ProAssist's progress summaries, allowing models to learn more structured task understanding and coordination.

**Last Updated**: November 13, 2025
# DST + Speak Training Data and Model: Progress Update

**Date:** November 12, 2025  
**Project:** Enhanced ProAssist Training Data with Dialog State Tracking Integration  
**Status:** Data Generation Complete, Ready for Training

## 🎯 Project Overview

I've successfully created an enhanced training dataset that combines **Dialog State Tracking (DST)** with **ProAssist conversational data**. This enables training a Vision-Language Model (VLM) to simultaneously learn:
1. **When to speak** - predicting appropriate moments for assistant guidance
2. **DST state updates** - tracking task progress through structured state information


- **SPEAK Events**: Assistant utterances with current task state context (`dst_state_snapshot`)
- **DST_UPDATE Events**: State transitions (start/complete) independent of speaking
- **Temporal Grounding**: Model knows current state before making decisions
- **Asynchronous Events**: Task progress can change without assistant intervention

## 📚 Inferred Knowledge and DST Construction

**Source**: `custom/outputs/dst_generated/proassist_label/2025-11-12/10-10-29_gpt-4o_proassist_50rows/assembly101/val.json`

### **How Inferred Knowledge Drives DST Structure**

#### **Inferred Knowledge**
```json
{
  "inferred_goal": "Assembling a Toy Roller with Chassis, Wheels, and Cabin",
  "inferred_knowledge": "Assembling a Toy Roller with Chassis, Wheels, and Cabin
  \n1. Assemble the chassis by attaching and screwing the chassis parts together.
  \n2. Attach wheels to the chassis.
  \n3. Assemble the arm and attach it to the chassis.
  \n4. Attach the body to the chassis.
  \n5. Add the cabin window to the chassis.
  \n6. Finalize the assembly and demonstrate the toy's functionality."
}
```

#### **All Step Description**
```json

{
    "all_step_descriptions": "[97.2s-106.8s] attach chassis to chassis
    \n[106.8s-116.5s] screw chassis
    \n - [110.7s] screw chassis with screwdriver
    \n - [112.7s] screw chassis part with screwdriver
    \n[116.5s-152.1s] attach wheel to chassis
    \n - [123.7s] screw first wheel with screwdriver
    \n - [130.7s] screw second wheel with screwdriver
    \n - [138.2s] screw third wheel with screwdriver
    \n - [146.8s] screw fourth wheel with screwdriver
    \n[152.1s-163.7s] attach roller to arm
    \n[163.7s-174.8s] attach arm connector to arm
    \n[174.8s-185.0s] attach arm connector to chassis
    \n[185.0s-197.9s] attach body to chassis
    \n - [192.2s] screw rear body with hand
    \n - [194.4s] screw rear body with screwdriver
    \n[197.9s-203.0s] attach cabin window to cabin
    \n[203.0s-212.3s] attach cabin window to chassis
    \n[212.3s-225.2s] demonstrate functionality"
}
```

#### **DST Structure Construction**
From the inferred knowledge, the system creates structured DST nodes with temporal boundaries:

```json
"dst": [
  {
    "type": "step",
    "id": "S1",
    "start_ts": 97.2,
    "end_ts": 118.7,
    "name": "Assemble the chassis by attaching and screwing the chassis parts together."
  },
  {
    "type": "step", 
    "id": "S2",
    "start_ts": 130.7,
    "end_ts": 136.7,
    "name": "Attach wheels to the chassis."
  },
  {
    "type": "step",
    "id": "S3", 
    "start_ts": 152.1,
    "end_ts": 162.1,
    "name": "Assemble the arm and attach it to the chassis."
  },
  {
    "type": "step",
    "id": "S4",
    "start_ts": 185.0,
    "end_ts": 195.0,
    "name": "Attach the body to the chassis."
  },
  {
    "type": "step",
    "id": "S5",
    "start_ts": 205.0,
    "end_ts": 215.0,
    "name": "Add the cabin window to the chassis."
  },
  {
    "type": "step",
    "id": "S6",
    "start_ts": 225.0,
    "end_ts": 235.0,
    "name": "Finalize the assembly and demonstrate the toy's functionality."
  }
]
```
### Implementation Overview: DST Structure Creation

Find the detailed implementation doc in [`custom/docs/dst_data/2-proassist_dst_label_plan.md`](custom/docs/dst_data/2-proassist_dst_label_plan.md).

#### **Core Implementation Pipeline**

**1. Raw Data Processing**
- Input: `inferred_knowledge` (high-level steps) + `all_step_descriptions` (fine-grained actions with timestamps)
- Parse annotation lines to extract normalized `(text, t0, t1)` blocks
- Handle hierarchical structure and infer missing timestamps using duration priors

**2. Semantic Alignment**
- **Semantic Similarity**: Use BAAI/bge-base-en-v1.5 embeddings to compute cosine similarity between action blocks and step descriptions
- **NLI Scoring**: Apply cross-encoder/nli-deberta-v3-base to detect entailment relationships between actions and steps
- **Joint Scoring**: Fuse semantic similarity (α=0.6) and NLI scores into unified alignment matrix

**3. Temporal Constraint Resolution**
- **Monotonic Dynamic Programming**: Enforce non-decreasing step indices with time while maximizing alignment scores
- **Assignment Algorithm**: Each action block assigned to optimal step index ensuring chronological consistency
- **Consecutive Merging**: Group adjacent blocks with same step assignment into contiguous temporal spans

**4. Quality Assurance**
- **Confidence Metrics**: Compute per-step confidence using score margins and NLI validation rates
- **Auto-fixes**: Resolve overlaps, fill gaps (<2s), and validate temporal consistency
- **Filtering**: Remove low-confidence steps (conf < 0.05, NLI ok-rate < 0.7)

This pipeline successfully generated high-quality DST structures for the filtered dataset.


## 📊 Generated Data Examples

**Source**: `custom/outputs/dst_generated/proassist_label/2025-11-12/10-10-29_gpt-4o_proassist_50rows/assembly101/val.json`

### **Current Enhanced DST Format**

**Video**: `assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit`

**Goal**: "Assembling a Toy Roller with Chassis, Wheels, and Cabin"

#### **Video-Level Structure**
```json
{
  "video_uid": "assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit",
  "inferred_goal": "Assembling a Toy Roller with Chassis, Wheels, and Cabin",
  "inferred_knowledge": "Assembling a Toy Roller...\n1. Assemble chassis...\n2. Attach wheels...",
  "dst": [
    {
      "type": "step",
      "id": "S1",
      "start_ts": 97.2,
      "end_ts": 118.7,
      "name": "Assemble the chassis by attaching and screwing the chassis parts together."
    }
  ],
  "conversation": [...]
}
```

### **Target ProAssist Training Format**

To enable training with the existing ProAssist infrastructure, the data needs to be transformed into the following structure compatible with `DSTTrainingDataModule`:

#### **ProAssist-Compatible Structure**
```json
{
  "dataset": "assembly101",
  "video_uid": "assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit",
  "clip_idx": 0,
  "frames_file": "frames/nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.arrow",
  "seq_len": 855,
  "max_seq_len": 0,
  "num_tokens_per_img": 1,
  "use_img_sep_token": false,
  "start_frame_idx": 194,  // floor(97.2 * 2)
  "end_frame_idx": 237,    // floor(118.7 * 2)
  "fps": 2,
  "dst": [
    {
      "type": "step",
      "id": "S1",
      "start_ts": 97.2,
      "end_ts": 118.7,
      "name": "Assemble the chassis by attaching and screwing the chassis parts together."
    }
  ],
  "conversation": [
    {"role": "system", "content": "You are a proactive assistant..."},
    {"role": "user", "time": 95.0, "content": "I want to assemble this toy...", "frames": {"start": 190, "end": 194}},
    {"role": "assistant", "time": 97.2, "content": "Great! Let's get started...", "frames": {"start": 194, "end": 198}, "dst_state_snapshot": [...]},
    {"role": "DST_UPDATE", "time": 97.2, "content": [{"id": "S1", "transition": "start"}], "frames": {"start": 194, "end": 198}},
    {"role": "DST_UPDATE", "time": 118.7, "content": [{"id": "S1", "transition": "complete"}], "frames": {"start": 237, "end": 241}}
  ],
  "metadata": {
    "user_type": "standard",
    "task_goal": "Assembling a Toy Roller with Chassis, Wheels, and Cabin",
    "knowledge": "...",
    "progress": null,
    "add_knowledge": false,
    "has_summary": false,
    "summary_only": false,
    "quality": 9.0
  }
}
```

#### **Key Transformation Requirements**

**1. Frame Integration**
- `frames_file`: Path to PyArrow file containing video frames
- `start_frame_idx`/`end_frame_idx`: Frame range for this conversation segment
- `fps`: Frames per second (typically 2 for ProAssist)
- **Calculation**: `frame_idx = floor(timestamp * fps)`

**2. Sequence Length Management**
- `seq_len`: Total token count (text + image tokens)
- `max_seq_len`: Maximum allowed sequence length (4096 for SmolVLM)
- `num_tokens_per_img`: Tokens per frame (1 for most models)
- **Text tokens**: Tokenized user/assistant/system content
- **Image tokens**: `num_frames × num_tokens_per_img`

**3. Conversation Structure Enhancement**
- Add `frames` object to each conversation item with start/end frame indices
- Include frame ranges for DST_UPDATE events to show visual context of state changes
- Maintain DST snapshots for your innovation while adding frame grounding

**4. Dataset Metadata**
- `dataset`: Dataset name (e.g., "assembly101")
- `clip_idx`: Conversation segment index within video
- `metadata`: Training-relevant information

**5. Frame Index Calculation**
```python
def time_to_frame_index(time_seconds: float, fps: int = 2) -> int:
    return int(time_seconds * fps)  # floor division
```

**6. Token Counting**
- **Text tokens**: Use model's tokenizer on all conversation text
- **Image tokens**: `(end_frame_idx - start_frame_idx) × num_tokens_per_img`
- **Total seq_len**: text_tokens + image_tokens

#### **Real SPEAK Event Example**
```json
{
  "type": "SPEAK",
  "time": 97.2,
  "labels": "initiative|instruction",
  "content": "Great! Let's get started. First, we need to attach the chassis to itself. Please take the two chassis parts and connect them.",
  "dst_state_snapshot": [
    {"id": "S1", "state": "in_progress"},
    {"id": "S2", "state": "not_started"},
    {"id": "S3", "state": "not_started"},
    {"id": "S4", "state": "not_started"},
    {"id": "S5", "state": "not_started"},
    {"id": "S6", "state": "not_started"}
  ],
  "metadata": {
    "original_turn": {
      "role": "assistant",
      "time": 97.2,
      "content": "Great! Let's get started. First, we need to attach the chassis to itself. Please take the two chassis parts and connect them.",
      "labels": "initiative|instruction",
      "progress": "The time elapsed since the start of the task is 97.2 seconds. The user's task goal is to assemble a toy roller with chassis, wheels, and cabin; nothing has been done yet as the conversation just started; no other topics were mentioned; the current step is to attach the two chassis parts to each other as instructed by the assistant."
    },
    "original_format": "assistant_role"
  }
}
```

#### **Real DST_UPDATE Event Examples**
```json
{
  "type": "DST_UPDATE",
  "time": 97.2,
  "labels": "dst_update",
  "content": [{"id": "S1", "transition": "start"}],
  "metadata": {
    "dst_node": "Assemble the chassis by attaching and screwing the chassis parts together.",
    "transition_type": "start"
  }
}
```

```json
{
  "type": "DST_UPDATE",
  "time": 118.7,
  "labels": "dst_update", 
  "content": [{"id": "S1", "transition": "complete"}],
  "metadata": {
    "dst_node": "Assemble the chassis by attaching and screwing the chassis parts together.",
    "transition_type": "complete"
  }
}
```

#### **Follow-up SPEAK Event (After Step Complete)**
```json
{
  "type": "SPEAK",
  "time": 130.7,
  "labels": "initiative|instruction,feedback",
  "content": "Great job on the first wheel! Now, screw the second, third, and fourth wheels with the screwdriver, one by one.",
  "dst_state_snapshot": [
    {"id": "S1", "state": "completed"},
    {"id": "S2", "state": "in_progress"},
    {"id": "S3", "state": "not_started"},
    {"id": "S4", "state": "not_started"},
    {"id": "S5", "state": "not_started"},
    {"id": "S6", "state": "not_started"}
  ]
}
```

### **Event Generation Statistics (from processing log)**
- **Average SPEAK events**: ~85 per video
- **Average DST_UPDATE events**: ~18 per video  
- **Total merged events**: ~103 chronological events per video
- **Processing ratio**: ~82% SPEAK : 18% DST_UPDATE


## 🧠 Model Architecture

Based on **SmolVLM-2.2B** (2.2B parameters), enhanced for multi-task learning with 4 specialized output heads:

```
Video Frames + Dialog History + Current DST → SmolVLM2 (VLM-based)
                                                  ↓
                                            [4 Training Heads]
                                                  ↓
        ┌─────────────┬─────────────┬─────────────┬─────────────┐
        │             │             │             │             │
        │  Speaking   │    DST      │  Response   │    DST      │
        │  Decision   │  Update     │ Generation  │    State    │
        │  (Binary)   │ Decision    │   (Text)    │  Update     │
        │             │  (Binary)   │             │ (States)    │
        └─────────────┴─────────────┴─────────────┴─────────────┘
```

**Training**: All 4 heads are trained simultaneously with different loss functions  
**Inference**: Binary decisions determine which action outputs to use  
**Benefits**: Single model handles speaking, state tracking, and text generation

**Multi-task Architecture:**
1. **Speaking Decision Head**: Binary classification (speak/silence)
2. **DST Update Decision Head**: Binary classification (update/no_update) 
3. **Response Generation Head**: Text generation (inherited from SmolVLM)
4. **DST State Update Head**: Per-node state classification (conditional on update decision)

**Both Tasks Use Binary Decision Style:**
- **SPEAK Decision**: Binary classification (speak/silence) 
- **DST Update Decision**: Overall binary classification (update/no_update) - similar to speak decision
- **Conditional Node Classification**: Only if update decision is YES, then classify specific nodes

**DST State Representation - DELTA Approach:**

## 📈 Training Configuration

**Multi-Loss Setup:**
1. **Cross-Entropy Loss**: For speak/no-speak decisions (balanced classes)
2. **Focal Loss**: For sparse DST transitions (high weight on start/complete events)


**Training Strategy:**
- **60% SPEAK samples** + **40% DST_UPDATE samples** mixed in dataloader
- **Curriculum Learning**: Start with easier SPEAK predictions, gradually introduce DST complexity
- **Temporal Context**: Model sees current state + recent history for decision making

## 🎯 Training Data Benefits

**1. Structured Task Knowledge**
- Instead of hand-written progress summaries → explicit state tracking
- Model learns **when** to speak based on current task progress
- Enables **explainable** assistant behavior

**2. Multi-Task Supervision** 
- Single dataset enables both speaking and state tracking objectives
- **Joint training** improves generalization vs. separate models
- **Evidence grounding** through temporal alignment

**3. Temporal Grounding**
- Every decision knows current task state
- Model learns to **coordinate** speaking with task progress
- **Asynchronous events** support natural task progression

## 📋 Current Status

✅ **Enhanced DST Data Generation**: Complete (5,000+ videos with DST snapshots)
🔄 **ProAssist Format Transformation**: Required for training compatibility
✅ **Model Architecture**: Designed and ready for SmolVLM-2.2B
✅ **Training Infrastructure**: DSTTrainingDataModule and pipeline ready
⏳ **Data Transformation**: Implement conversion to ProAssist training format

## 📝 Next Steps

### **Phase 1: Data Transformation Pipeline**
1. **Frame Integration**: Add `frames_file`, `start_frame_idx`, `end_frame_idx`, `fps` fields
2. **Sequence Length Calculation**: Compute `seq_len` from tokenized text + image tokens
3. **Conversation Enhancement**: Add frame ranges to each conversation item
4. **DST Event Grounding**: Include frame indices for DST_UPDATE events
5. **Dataset Metadata**: Add `dataset`, `clip_idx`, and training metadata

### **Phase 2: Training Preparation**
1. **DSTTrainingDataset Integration**: Load transformed data
2. **Data Collator Compatibility**: Ensure proper batching and tokenization
3. **Multi-task Loss Setup**: Configure speaking + DST update losses
4. **Memory Optimization**: Implement sequence length truncation (4096 tokens)

### **Phase 3: Training Execution**
1. **Initial Training Run**: Test with transformed data
2. **Baseline Comparison**: Compare DST snapshots vs. progress summaries
3. **Performance Analysis**: Evaluate speaking decisions and state tracking accuracy

---

**Key Innovation**: DST snapshots provide structured state information vs. ProAssist's textual progress summaries, enabling more precise task understanding and coordination.

**Data Compatibility**: Current enhanced format needs transformation to ProAssist's training structure before model training can begin.