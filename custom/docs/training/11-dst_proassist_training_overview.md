# DST-Enhanced ProAssist Training: System Overview

## Executive Summary

I'm building **DST ProAssist**, an extension of the ProAssist video LLM that integrates **Dialogue State Tracking (DST)** into the model's decision-making pipeline. This work addresses a key limitation in existing multimodal assistants: they lack structured understanding of task progression and state management during long, procedural interactions.

---

## Problem Statement

### Current Limitation: Stateless Multimodal LLMs
ProAssist and similar video LLMs operate **frame-by-frame** without explicit state tracking. They make independent decisions at each frame:
- **When should the assistant speak?** (Speaking decision)
- **What should the assistant say?** (Response generation)

But they never explicitly ask: **"What is the current state of the task?"** or **"Has progress been made?"**

### The Gap
In **procedural videos** (assembly, cooking, repairs), understanding task progression is critical:
- Step 1 → In Progress → Complete → Move to Step 2
- Current state determines what advice to give next

### Why This Matters
- **Without DST**: Model might repeat instructions or give contextually inappropriate guidance
- **With DST**: Model understands where we are in the workflow and provides progressive, step-aware assistance

---

## My Approach: DST ProAssist

### Architecture Overview

```
Video Frames (Assembly/Cooking)
         ↓
    SigLIP Vision Encoder
    (Pre-computed Embeddings: 1152-dim)
         ↓
    Multimodal Projector
    (Frame Embedding → LLM Space)
         ↓
    Llama-3.2-3B Language Model
    (Base LLM backbone)
         ↓
    ┌─────────────────────────────────────────┐
    │   Multi-Task Prediction Heads           │
    ├─────────────────────────────────────────┤
    │ 1. Language Model Head                  │
    │    → Generate next token (DST or text)  │
    │                                         │
    │ 2. Speaking Decision Head (Binary)      │
    │    → Should assistant speak now?        │
    │                                         │
    │ 3. DST Update Decision Head (Binary)    │
    │    → Should task state be updated?      │
    └─────────────────────────────────────────┘
         ↓
    Multi-Task Loss Computation
    (Balanced negative sampling)
```

### Key Components

#### 1. **Multimodal Input Processing**
- **Vision**: Pre-computed SigLIP embeddings (1152-dim) from video frames
- **Text**: Task description (DST schema) + dialogue history + special tokens
- **Fusion**: Frame embeddings projected into LLM embedding space, replacing `<image>` tokens

#### 2. **Role-Based Token Structure**
Sequences are built with semantic role tokens:

```
[System Instruction: DST Overview + Current State]
<image> S1→start [DST] completed <eos> [ASST] "Now attach the... " <eos>
<image> [ASST] "Use your wrench..." <eos>
<image> S2→start [DST] in_progress <eos> [ASST] "Next step..." <eos>
```

- `<image>`: Frame boundary (where binary decisions are made)
- `[DST]`: DST update prefix (model learns to generate state transitions)
- `[ASST]`: Assistant response prefix (model learns to generate helpful text)
- Special tokens added to tokenizer vocabulary

#### 3. **Four-Task Training Objective**

| Task | Output | Label | Purpose |
|------|--------|-------|---------|
| **Speaking Gen Loss** | Next token logits | Text after `[ASST]` | Learn response generation |
| **DST Gen Loss** | Next token logits | Text after `[DST]` | Learn state transition language |
| **Speaking Binary Loss** | Binary logit (0/1) | Label at `<image>` | Learn when to speak |
| **DST Update Binary Loss** | Binary logit (0/1) | Label at `<image>` | Learn when to update state |

#### 4. **Balanced Loss Computation (Following ProAssist)**
```python
# Separate loss for positive and negative examples
speaking_loss_pos = BCE(logits[speaking==1], 1)
speaking_loss_neg = BCE(logits[speaking==0], 0)

# Average to prevent class imbalance bias
speaking_loss = (loss_pos + loss_neg) / 2
```

This ensures balanced gradient updates despite imbalanced data (many frames without speaking).

---

## Novelty vs. ProAssist

### ProAssist (Baseline)
- ✅ Multi-task LLM + vision for procedural video
- ✅ Speaking decision head (when to speak)
- ✅ Response generation
- ❌ No explicit state tracking
- ❌ No DST component

### DST ProAssist (This Work)
- ✅ All ProAssist capabilities
- ✅ **Two decision heads** (speaking + DST update)
- ✅ **Separate generation streams** (responses + DST updates)
- ✅ **DST context in system prompt** (model aware of task schema and current state)
- ✅ **Balanced loss computation** (refined from ProAssist's NFS strategy)

### Key Innovations
1. **Dual Decision Architecture**: 
   - ProAssist: 1 binary head (speaking)
   - DST ProAssist: 2 binary heads (speaking + DST update)

2. **State-Aware Context**:
   - System instruction includes full DST schema (all steps)
   - System instruction includes current progress (e.g., "S1: in_progress, S2: not_started")
   - Updates after clip_idx=0 reflect actual state progression

3. **Separate Generation Targets**:
   - [DST] prefix supervises state transition learning (e.g., "S1→complete")
   - [ASST] prefix supervises dialogue response learning
   - Independent loss streams enable focused optimization

4. **Refined Negative Frame Sub-sampling**:
   - Separate loss computation for positive vs. negative frames
   - Averaging prevents gradient dominance by majority class
   - Mirrors ProAssist's NFS but applied symmetrically to both decision types

---

## Training Pipeline

### Data Pipeline
1. **Input**: Assembly101 dataset (procedural videos with annotations)
2. **Sparse Format**: JSON with only event frames (DST updates, speaking events)
3. **Collator Processing**:
   - Load pre-computed SigLIP embeddings (frame → 1152-dim vector)
   - Parse conversation (system → DST → user → assistant turns)
   - Build continuous sequences with role tokens
   - Apply negative frame sub-sampling (e.g., 10% of silent frames)
   - Create 4 label tensors (speaking_gen, dst_gen, speaking_binary, dst_binary)

### Model Loading & Optimization
- **Base LLM**: Llama-3.2-3B-Instruct
- **Quantization**: Optional 4-bit NF4 (QLoRA, 75% memory reduction)
- **LoRA**: Low-rank adaptation (rank=128, α=256) on 7 attention/MLP modules
- **Frozen**: Base model parameters stay fixed
- **Trainable**: Only multimodal projector + 2 decision heads + LoRA adapters

### Training Configuration
- **Batch Size**: 4 samples/device
- **Gradient Accumulation**: 2 steps (effective batch = 8)
- **Learning Rate**: 2e-5 (bfloat16 mixed precision)
- **Warmup**: 100 steps
- **Epochs**: 2 (with early stopping)
- **Logging**: W&B integration for loss tracking across 4 tasks

---

## Multimodal Architecture in Action

### How Vision and Language Integrate

```python
# 1. Vision Processing (Offline, Pre-computed)
video_frames → SigLIP Encoder → embeddings [batch, num_frames, 1152]
# Stored in .arrow files for training efficiency

# 2. Text Processing (Online, During Training)
"S1 → Step 2 → S3" (task schema) → Tokenizer → [101, 245, 89, ...]
"The current state is..." (dialogue state) → Tokenizer → [512, 413, ...]

# 3. Multimodal Fusion (During Forward Pass)
input_ids = [vocab_tokens, <image>, vocab_tokens, <image>, ...]
                            ↓
embeddings = Embedding_Layer(input_ids)
            where <image> tokens get replaced by SigLIP embeddings

# 4. Transformer Processing
embeddings → Llama-3.2 Transformer → hidden_states [-1] (last layer)
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
            LM_Head (vocab)    Speaking_Head    DST_Update_Head
           (logits for text)   (binary: 0/1)    (binary: 0/1)

# 5. Multi-Task Loss
final_loss = speaking_gen_loss + dst_gen_loss + speaking_binary_loss + dst_binary_loss
```

### Why This Design?
- **Pre-computed embeddings**: Training only requires LLM forward passes (faster)
- **On-demand projection**: Images only projected when needed (memory efficient)
- **Shared backbone**: Single LLM learns from all 4 tasks simultaneously
- **Specialized heads**: Each decision type (speaking vs. DST) gets dedicated output layer

---

## Results & Insights

### Current Status
- ✅ Model loads and trains end-to-end
- ✅ All 4 losses compute correctly
- ✅ Binary loss averaging prevents gradient bias
- 🔄 Convergence analysis in progress (monitoring recall improvement)

### Key Learnings
1. **Class Imbalance Matters**: Without balanced loss computation (ProAssist's NFS), model biases toward predicting "0"
2. **Negative Sampling**: 10% of negative frames is crucial (not all frames get gradients)
3. **System Instructions**: DST context in initial prompt helps model understand task scope
4. **Role Tokens**: Separating [DST] and [ASST] content enables independent optimization

---

## Use Cases

### Where DST ProAssist Excels
1. **Procedural Videos**: Assembly, cooking, repairs, tutorials
2. **Multi-Step Tasks**: Anything with clear progression (step 1 → 2 → 3)
3. **State-Dependent Advice**: "Next, attach the..." (only valid if previous step complete)
4. **Context-Aware Dialogue**: Model knows what to repeat vs. what to advance

### Example: Assembly Tutorial
```
Frame 1: "Assemble the chassis..."
    DST Decision: S1→start (mark as in progress)
    Speaking Decision: Yes ("Let's attach the step...")

Frame 50: Chassis is assembled
    DST Decision: S1→complete, S2→start (move to next step)
    Speaking Decision: Yes ("Great! Now attach the body...")

Frame 100: (Silent assembly, no events)
    DST Decision: No update
    Speaking Decision: No (let user work)
```

---

## Why This Matters (Interview Ready)

### The Problem I'm Solving
Most video LLMs treat each frame independently. **DST ProAssist** adds structured state tracking, enabling the model to understand task progression and provide step-aware, contextually appropriate guidance.

### My Contribution
- Extended ProAssist with **dual decision heads** (speaking + DST update)
- Implemented **multi-stream generation** (responses + state transitions)
- Engineered **balanced loss computation** to handle class imbalance
- Created **modular pipeline** (collator → model → trainer) for reproducibility

### What Makes It Novel
Unlike general-purpose video LLMs, DST ProAssist explicitly models **task state**, making it ideal for procedural understanding where context and progression matter.

### Technical Rigor
- Follows ProAssist's proven techniques (negative frame sub-sampling, special tokens)
- Extends methodically (2 heads instead of 1, separate loss streams)
- Validates each component (loss computation tested against ProAssist's patterns)

---

## Files Structure

```
custom/src/prospect/
├── train/
│   └── train_dst_proassist.py          # Main training pipeline
├── data_sources/
│   ├── dst_proassist_collator.py       # Data collation (sequence building)
│   └── dst_proassist_dataset.py        # Dataset loading
├── models/
│   └── dst_proact/
│       ├── modeling.py                 # Model forward pass + loss computation
│       ├── configuration.py            # Config management
│       └── ...
└── train/
    └── dst_proassist_trainer.py        # HuggingFace Trainer customization
```

---

## Next Steps

1. **Convergence Analysis**: Monitor speaking/DST recall over training epochs
2. **Inference Pipeline**: Implement frame-by-frame generation with state tracking
3. **Evaluation**: Benchmark on unseen videos (accuracy of state predictions + response quality)
4. **Extensions**: Curriculum learning, contrastive losses, multi-dataset training
