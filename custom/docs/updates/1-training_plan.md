# 🧠 Training Plan: Vision-Language Progress Summarization with DST Grounding

---

## 1. Overview

### 🎯 Goal  
Develop a **Vision-Language Model (VLM)** that can:

1. Observe **video frames** and **dialogue context** in an egocentric instructional task.  
2. Interpret **task structure** via a predefined **Dialog State Tree (DST)**.  
3. Predict which steps/substeps/actions are **Completed (C)**, **In Progress (IP)**, or **Not Started (NS)**.  
4. Generate a **structured progress summary** describing what has been accomplished, what is ongoing, and what remains.  
5. Optionally, **point to visual evidence** (frames) that justify its predictions.

---

## 2. Motivation

Traditional **ProAssist** used:
- A **large LLaMA-based LLM** for language reasoning.  
- A separate **image encoder** for visual perception.  
- A **4K-token summarization window** for context.  

Our approach:
- Replace with a **single lightweight VLM** (e.g., `SmolVLM2-2.2B-Instruct`).  
- Use **joint vision-language embeddings** for alignment.  
- Achieve comparable or better results with **1–2K tokens** by incorporating **DST structure** and **temporal grounding**.

---

## 3. Core Learning Objectives

The model jointly learns **three complementary tasks**:

| Objective | Description | Output |
|------------|--------------|---------|
| **A. DST State Prediction** | Predict C/IP/NS for each step, substep, or action node. | Per-node state labels |
| **B. Evidence Grounding (Pointer Head)** | Identify which frames support each active node. | Frame indices or timestamps |
| **C. Progress Summarization** | Generate a structured JSON summary + concise progress note. | JSON text |

These objectives make the model both **accurate** and **interpretable**.

---

## 4. Data Representation

### 4.1 Inputs

Each training sample (a video window) contains:

| Component | Example | Tokens/Frames |
|------------|----------|---------------|
| **Video frames** | 16–24 frames @ 1 fps (e.g., 97s–112s) | Vision input |
| **DST schema** | List of steps/substeps/actions with text descriptions | 300–500 tokens |
| **Dialogue context** | Last 10–15 turns (summarized) | ≤500 tokens |
| **Memory JSON** | Previous summary of the task | ≤200 tokens |
| **Captions / Hints** | Optional object-verb captions | ≤100 tokens |

---

### 4.2 Labels

Derived automatically from your annotated DST JSON:

- **Node states**: `y_state[i] ∈ {C, IP, NS}` at time τ  
  (using the rule: `C if end_ts ≤ τ; IP if start_ts ≤ τ < end_ts; else NS`)
- **Evidence frames**: indices `y_frame[i]` = frames within `[start_ts, end_ts]`
- **Summary JSON**: ground-truth structured summary for the window

---

## 5. Model Architecture

A **multi-head VLM** built on a pretrained base like `SmolVLM2-2.2B-Instruct`.

### 5.1 Base VLM
- Vision encoder (ViT-G/14) → patch embeddings.  
- Text decoder with cross-modal attention.  
- Already trained for image/video–text alignment.

### 5.2 Added Components

#### 1️⃣ DST Node Encoder
Encodes each DST node:
```
"S2.1: Attach wheel to chassis"
```
→ hidden vector `v_i`.

Used as conditioning tokens in both classification and generation.

#### 2️⃣ Graph State Head
- Input: pooled visual + node embeddings  
- Output: per-node 3-way logits (C/IP/NS)  
- Loss: Cross-Entropy per node

#### 3️⃣ Evidence Pointer Head
- Input: node embeddings `v_i` and frame embeddings `f_j`  
- Output: attention distribution `p_{i,j} = softmax_j(sim(v_i, f_j))`  
- Loss: CE alignment with ground-truth frames

#### 4️⃣ Progress Generator
Language decoder that generates:
- Structured JSON (steps completed, in-progress, blocked)
- Natural-language progress note (≤120 tokens)

---

## 6. Training Objectives

Total loss:
```
L = λ1 * L_state + λ2 * L_evidence + λ3 * L_summary
```

### A. State Loss
```
L_state = Σ_i CE(y_state[i], ŷ_state[i])
```

### B. Evidence Loss
```
L_evidence = -Σ_i Σ_j y_frame[i,j] * log(p[i,j])
```

### C. Summary Loss
```
L_summary = CE(y_json_tokens, ŷ_json_tokens)
```

**Recommended weights:**  
`λ1 = 1.0`, `λ2 = 0.5`, `λ3 = 1.0`

---

## 7. Training Setup

| Setting | Value |
|----------|--------|
| **Model** | SmolVLM2-2.2B-Instruct |
| **Precision** | bfloat16 |
| **Batch size** | 2–4 windows per GPU |
| **LoRA ranks** | 16–32 |
| **Trainable layers** | Top 8 language blocks + cross-modal layers |
| **Learning rate** | 1e-4 with cosine decay |
| **Optimizer** | AdamW |
| **Epochs** | 1–2 (early stop on val F1 of current step) |

---

## 8. Forward Pass Example

**Input (time window 116.5s–152.1s):**
- 20 frames (wheel assembly phase)
- DST: S1–S6 text descriptions
- Previous memory JSON
- Optional captions: “hand holding wheel”, “screwdriver tightening wheel”

**Model sees:**
```
[Frames]
[DST Nodes: S1...S6]
[Dialogue Context]
[Memory JSON]
→ Encoded via SmolVLM2 backbone
```

**Outputs:**
```text
S1: Completed
S2: In Progress
S3–S6: Not Started
```

**Evidence pointer:**
- For node S2.1 (“Attach wheel”), points to frames near 123.7s–146.8s.

**Progress summary:**
```json
{
  "completed_steps": ["S1_Assemble chassis"],
  "current_step": {
    "step_id": "S2_Attach wheels",
    "evidence": [{"frame": 5, "t": "124s"}, {"frame": 10, "t": "139s"}]
  },
  "next_actions": ["Attach arm connector to arm"],
  "progress_note": "Chassis assembly completed. Now attaching wheels to the chassis using a screwdriver."
}
```

---

## 9. Evaluation Metrics

| Category | Metric | Description |
|-----------|---------|-------------|
| **DST State Prediction** | Accuracy, F1 | Node-level C/IP/NS classification |
| **Current Step Accuracy** | Accuracy | Most active step at τ |
| **Evidence Grounding** | Frame F1 (±2s) | Overlap between predicted & gold frames |
| **Progress Summarization** | BLEU / ROUGE / Human Utility | Faithfulness, usefulness |
| **Efficiency** | Tokens vs. performance | Compare 1K / 2K / 4K context |
| **Interpretability** | Qualitative | Node→Frame heatmaps |

---

## 10. Evaluation Example

**Gold:**
- Active: S2.1 (Attach wheel)  
- Evidence: frames 123s–147s  
- Summary: “Attaching wheels to chassis with screwdriver.”

**Model Output:**
- Active: S2.1  
- Evidence: frames 124s, 139s  
- Summary: “Wheels are being attached to the chassis using a screwdriver.”

✅ Step correctness: **True**  
✅ Frame overlap: **2/2 correct (F1 = 1.0)**  
✅ Summary: **Semantically faithful**

---

## 11. Why This Works

- ✅ **DST as structure** → model updates states, not rebuilds trees.  
- ✅ **Evidence pointer** → explicit visual grounding.  
- ✅ **JSON outputs** → structured, factual summaries under tight context.  
- ✅ **LoRA fine-tuning** → efficient on 2×V100 GPUs.  
- ✅ **Evaluation-ready** → measurable interpretability.

---

## 12. Optional Extensions

| Idea | Description |
|------|--------------|
| **Curriculum training** | Train state → add summarization later. |
| **Contrastive alignment** | Add InfoNCE between node/frame embeddings. |
| **Pseudo-labeling** | Use larger VLM (Qwen2.5-VL-7B) as teacher. |
| **Memory updates** | Feed previous JSON summaries as state. |
| **Heatmap visualizer** | Display node-frame attention for interpretability. |

---

## 13. Training Flow Summary

**Step 1 — Data Preparation:**  
- Slide windows across video (16–24 frames).  
- Derive DST node labels and frame evidence.  
- Generate progress summaries.

**Step 2 — Model Input:**  
- Encode video, DST schema, dialogue, and memory.

**Step 3 — Multitask Learning:**  
- Predict node states.  
- Identify evidence frames.  
- Generate structured summary.

**Step 4 — Joint Optimization:**  
- Combine `L_state + L_evidence + L_summary` with tuned λ’s.

**Step 5 — Evaluation:**  
- Quantitative: F1, frame grounding, summarization metrics.  
- Qualitative: visual grounding, examples, ablations.

---

## 14. Expected Outcomes

A **lightweight, explainable** progress summarizer that:

- Operates within **1.5–2K tokens**.  
- Accurately tracks **DST-based task progression**.  
- Produces **concise structured summaries**.  
- Grounds claims in **visual evidence**.  
- Scales efficiently on **SmolVLM2-2.2B** or similar models.

---

**End of Document**
