# 🧠 Pretraining Plan: Video–Text Alignment for DST-Grounded VLM

## 1. Overview

**Goal**

Pretrain a Vision-Language Model (VLM) to learn strong alignment between **video frames** and **action text** derived from annotated DST data.  
This stage builds domain-specific multimodal representations that later enhance:

- Dialog State Tracking (DST) prediction  
- Evidence Grounding (frame retrieval)  
- Progress Summarization (structured reasoning)

**High-Level Idea**

Each DST node describes what the user is doing (*“Attach wheel to chassis”*) and when it occurs.  
By aligning short video clips around those timestamps with their corresponding text, the VLM learns to connect *visual motion patterns* with *procedural language* before any downstream fine-tuning.

---

## 2. Motivation

Generic VLMs (e.g., SmolVLM-2.2B) are trained on broad web data where image–text pairs cover diverse but shallow concepts.  
They lack understanding of **task-specific, step-wise actions**.

### Problems in the base model
- Visual embeddings are not tuned to small, repetitive task actions.  
- Text embeddings are generic and unaware of task hierarchies.  
- Downstream modules must learn alignment from scratch, requiring more data and compute.

### Benefits of this pretraining
- Builds **action-aware embeddings** that capture fine-grained temporal cues.  
- Provides a **shared space** where visual and textual representations of the same step are close.  
- Improves downstream **DST accuracy**, **grounding precision**, and **summary faithfulness**.

---

## 3. Conceptual Design

### Core Objective
> Learn embeddings such that *video clips* and *texts* describing the same action are close, while unrelated pairs are distant.

This is achieved through **contrastive learning** with careful handling of positives and negatives.

---

## 4. Data Preparation

Each training sample comes from the annotated DST dataset.

| Component | Source | Example |
|------------|---------|----------|
| **Video clip** | Frames near action timestamp (`start_ts`, `end_ts`) | 4–8 frames centered ±2 s |
| **Action text** | DST node string | “Attach wheel to chassis” |
| **Hierarchy text (optional)** | Parent or substep description | “Assemble wheel module” |

**Pair construction**

1. For each action node:  
   Extract a short clip + text description.  
2. Treat this as a *positive* pair (✓).  
3. Use clips from **other videos** as negatives (✗).  
4. Treat nearby steps in the *same video* as *soft negatives* or ignore them.

---

## 5. Model Architecture

### 5.1 Base Model

The model is a pretrained **SmolVLM-2.2B-Instruct**, which already integrates:

- A **vision encoder** for frames (ViT-based visual tower)  
- A **language backbone** for text (Transformer decoder)  
- A **cross-modal fusion module** that connects both modalities via attention  

We do **not** build new encoders.  
Instead, we reuse this full multimodal architecture and fine-tune it slightly on our ProAssist video–text pairs to specialize it for procedural action understanding.

---

### 5.2 What Is Updated

Since SmolVLM is already well-aligned on general vision–language data, we only **lightly adapt** it using LoRA or selective fine-tuning:

| Component | Action | Purpose |
|------------|---------|----------|
| **Vision tower** | Frozen | Generic low-level perception is already strong |
| **Cross-modal fusion layers** | Fine-tuned or LoRA-adapted | Teach how ProAssist actions visually appear |
| **Top language blocks** | Optionally LoRA-adapted | Capture phrasing and terminology of action texts |
| **Projection layer (added head)** | Trainable | Produces a single pooled embedding per clip/text for contrastive loss |

This design preserves the pretrained knowledge while encouraging better **alignment between task-specific frames and action descriptions**.

---

### 5.3 How It Processes a Sample

1. **Video Input:**  
   - Sample 4–8 frames around each action timestamp (e.g., ±2 s window).  
   - Feed the frames through the vision path of SmolVLM.  
   - Obtain fused multimodal tokens.

2. **Text Input:**  
   - Provide the DST action text (e.g., “Attach wheel to chassis”).  
   - Tokenize and embed through the language backbone.

3. **Fusion & Pooling:**  
   - SmolVLM’s internal attention layers fuse vision and text.  
   - Extract the `[CLS]` or pooled multimodal embedding.

4. **Projection Head:**  
   - Pass pooled embeddings through a small MLP head to map them to a normalized latent space (e.g., 1024-D).  
   - These projected embeddings are used in the **contrastive loss** to align matching video–text pairs.


---

## 6. Learning Objective

### 6.1 Contrastive Alignment (InfoNCE / CLIP-style)

For a batch B of pairs \((v_i, x_i)\):

\[
\mathcal{L}_{\text{ITC}} =
-\frac{1}{B} \sum_i
\left[
\log \frac{\exp(\mathbf{v}_i \cdot \mathbf{x}_i / \tau)}
{\sum_j w_{ij}\exp(\mathbf{v}_i \cdot \mathbf{x}_j / \tau)}
+
\log \frac{\exp(\mathbf{x}_i \cdot \mathbf{v}_i / \tau)}
{\sum_j w_{ij}\exp(\mathbf{x}_i \cdot \mathbf{v}_j / \tau)}
\right]
\]

where \(w_{ij}\) weights negatives:
- 1 for cross-video pairs,  
- λ (0.3–0.5) for same-video different actions,  
- 0 for neighboring or semantically similar steps.

### 6.2 Optional Auxiliary Heads
| Head | Input | Output | Purpose |
|------|--------|---------|----------|
| **ITM (Video-Text Matching)** | `[v_i ‖ x_i]` | Binary | Stabilizes alignment |
| **Captioning (optional)** | video → text | Reconstructs action phrase | Regularization |

Total loss:  
\(\mathcal{L} = \mathcal{L}_{\text{ITC}} + \beta \mathcal{L}_{\text{ITM}} + \gamma \mathcal{L}_{\text{caption}}\)  
( β = 0.5, γ = 0.2 )

---

## 7. Training Setup

| Setting | Value |
|----------|--------|
| Model | SmolVLM-2.2B-Instruct |
| Precision | bfloat16 |
| Batch size | 32 pairs (contrastive) |
| Frames per clip | 4 – 8 |
| Learning rate | 1e-4 (cosine decay) |
| Optimizer | AdamW (weight decay 1e-4) |
| Epochs | 5 – 15 (early stop on R@5) |
| Temperature τ | learnable, init 0.07 |
| Gradient clip | 1.0 |
| Mixed precision | Yes |
| Negatives | cross-video and masked same-video |
| Checkpoint metric | Video ↔ Text Recall@5 |

---

## 8. Forward Pass Example

**Input:**  
Video segment (122 s – 128 s) showing “Attach wheel to chassis” action  
**Steps:**

1. Encode frames → pooled visual embedding `v_i`  
2. Encode text → embedding `x_i`  
3. Compute similarity matrix with batch texts and videos  
4. Apply contrastive loss to maximize `sim(v_i, x_i)` and minimize others  

**Output:**  
Aligned embeddings where:  
- `sim(v_i, x_i)` ≫ `sim(v_i, x_j ≠ i)`  
- Action-specific clusters form in embedding space.  

---

## 9. Expected Outcome

After this pretraining stage, the VLM will:

- Encode procedural frames and textual actions into a **shared, action-aware space**.  
- Serve as a **strong initialization** for subsequent DST prediction, evidence grounding, and progress summarization.  
- Require less fine-tuning data and achieve **better alignment and interpretability** in the ProAssist pipeline.

---
