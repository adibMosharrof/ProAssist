# 📘 ProAssist Dataset Label Generation and Alignment Plan

## 1. Goal
Create **clean, high-level step (DST) timestamps** from noisy, fine-grained procedural annotations.  
We only use:

- **`inferred_knowledge`** → defines high-level steps (`S1...SK`)
- **`all_step_descriptions`** → provides noisy, low-level actions with timestamps

### Objective
Generate a temporally ordered, high-confidence mapping between fine-grained actions and high-level DST steps **without any manual lexicon** or LLM hallucinations.

---

## 2. Overall Pipeline
1. **Parse raw annotation lines** to obtain normalized `(text, t0, t1)` blocks.  
2. **Compute semantic similarity** between each block text and step text.  
3. **Compute NLI entailment scores** between each (block, step) pair.  
4. **Fuse semantic + NLI scores** into a joint matrix.  
5. **Run monotonic dynamic programming (DP)** to enforce temporal order and assign blocks → steps.  
6. **Merge assigned blocks** per step into contiguous time spans.  
7. **Compute confidence metrics** and filter noisy data.  
8. **Export per-frame or per-segment labels** for training and evaluation.

---

## 3. Parsing and Timestamp Inference

### 3.1 Input
Typical raw snippet:
```
[97.2s-106.8s] attach chassis to chassis
[106.8s-116.5s] screw chassis
[116.5s-152.1s] attach wheel to chassis
 - [123.7s] screw first wheel with screwdriver
 - [130.7s] screw second wheel with screwdriver
[152.1s] attach roller to arm
[163.7s-174.8s] attach arm connector to arm
[174.8s-185.0s] attach arm connector to chassis
...
```

### 3.2 Parsing Rules
1. **Detect timestamps**  
   - `[a–b]` → interval block  
   - `[a]` → point block (missing end time)
2. **Track hierarchy**  
   - Lines beginning with `"-"` → child of previous top-level block.
3. **Infer parent intervals**  
   - Parent `[t0,t1]` = span covering its children ± 1 s padding.
4. **Infer end time for point blocks**  
   - `t1 = min(next_top_level_start – 0.2, t0 + prior_duration(text))`
   - `prior_duration` defaults:  
     | Verb group | Duration (s) |
     |-------------|-------------|
     | screw/tighten | 6 |
     | attach/place/insert/connect | 10 |
     | demonstrate/show/roll | 12 |
5. **Fix overlaps** → cut midpoint between consecutive intervals.  
6. **Merge micro-gaps (< 2 s)** when texts are semantically similar.  
7. **Clamp and quantize** to video duration and frame grid.

### 3.3 Output
List of cleaned blocks:
```python
[
  {"text": "attach chassis to chassis", "t0":97.2, "t1":106.8},
  {"text": "screw chassis", "t0":106.8, "t1":116.5},
  {"text": "attach wheel to chassis", "t0":116.5, "t1":152.1},
  ...
]
```

---

## 4. Semantic Similarity

### 4.1 Embeddings
Use any sentence-level bi-encoder, e.g.:
```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("BAAI/bge-base-en-v1.5")
```

### 4.2 Similarity
Compute cosine similarity between block texts and high-level step titles:

\[
S_{ik} = \cos(E_b(i), E_s(k))
\]

Optional positional prior (helps preserve global order):

\[
S'_{ik} = S_{ik} + \lambda \cdot \mathcal N\!\left(\frac{i}{I};\frac{k}{K},\sigma\right)
\]
Typical: λ = 0.1, σ = 0.25

---

## 5. NLI Scoring

### 5.1 Concept
Use an off-the-shelf Natural-Language-Inference model to check if a block *entails* belonging to a step.

- **Premise:** block/action text  
- **Hypothesis:** `"This action is part of step: '<step text>'."`

### 5.2 Example Code
```python
from sentence_transformers import CrossEncoder
import numpy as np

nli = CrossEncoder("cross-encoder/nli-deberta-v3-base")  # [contradiction, neutral, entailment]

blocks = ["attach wheel to chassis", "attach body to chassis"]
steps  = ["Attach wheels to the chassis.", "Attach the body to the chassis."]

pairs, index_map = [], []
for i,b in enumerate(blocks):
    for k,s in enumerate(steps):
        pairs.append((b, f"This action is part of step: '{s}'"))
        index_map.append((i,k))

probs = nli.predict(pairs, convert_to_numpy=True)
P_c, P_n, P_e = probs[:,0], probs[:,1], probs[:,2]
N = P_e - P_c                     # scalar NLI score
```

- `p_entail` high → strong match  
- `p_contra` high → likely different step  
- Combine as: `N[i,k] = p_entail - p_contra` (range ≈ –1 … +1)

---

## 6. Joint Scoring

Normalize row-wise (z-score) and combine:

\[
J_{ik} = \alpha \cdot \widehat{S'}_{ik} + (1-\alpha)\cdot \widehat{N}_{ik}
\]

Typical α = 0.6.

---

## 7. Monotonic Dynamic Programming

### 7.1 Goal
Assign each block to a step index `k_i ∈ {1…K}` such that:
- step indices are **non-decreasing** with time
- total joint score is maximized

### 7.2 Core Algorithm (stay or move)
```python
def monotonic_dp(J):
    I,K = J.shape
    NEG = -1e9
    dp  = np.full((I,K), NEG)
    ptr = np.full((I,K), -1, dtype=int)
    dp[0,0] = J[0,0]
    for i in range(1,I):
        for k in range(K):
            stay = dp[i-1,k]
            move = dp[i-1,k-1] if k>0 else NEG
            if move > stay:
                dp[i,k] = move + J[i,k]; ptr[i,k] = k-1
            else:
                dp[i,k] = stay + J[i,k]; ptr[i,k] = k
    # backtrack
    k_seq = [np.argmax(dp[-1])]
    for i in range(I-1,0,-1):
        k_seq.insert(0, ptr[i, k_seq[0]])
    return np.array(k_seq)
```

### 7.3 Merge Consecutive Assignments
Group consecutive blocks with the same step index → final step spans:

```
S1: [97.2, 116.5]
S2: [116.5, 152.1]
S3: [152.1, 185.0]
S4: [185.0, 197.9]
S5: [197.9, 212.3]
S6: [212.3, 225.2]
```

---

## 8. Confidence & Filtering

### 8.1 Per-Block Margins
\[
\Delta^J_i = J_{i,a_i} - \max_{j\neq a_i} J_{i,j}
\]

### 8.2 Step-Level Confidence
\[
\text{conf}_k = \text{mean}_{i:a_i=k}(\Delta^J_i)
\]
\[
\text{nli\_ok}_k = \frac{\#\{i:a_i=k,\;p_e≥0.6,\;p_c≤0.2\}}{\#\{i:a_i=k\}}
\]

### 8.3 Stability (optional)
Re-run with a different encoder → compute IoU overlap of step spans.  
Keep steps where:

| Metric | Threshold |
|---------|------------|
| conf_k | ≥ 0.05 |
| nli_ok_k | ≥ 0.7 |
| stability_k | ≥ 0.6 |

Low-confidence steps can be dropped or down-weighted during training.

---

## 9. Audits and Auto-Fixes
1. **Order:** guaranteed by DP  
2. **Overlap:** cut midpoint between consecutive spans  
3. **Gaps (< 2 s):** absorb into neighbors  
4. **Duration priors:** learn mean±2σ per step index; flag outliers  
5. **Coverage:** warn if uncovered video portion > 10 %

---

## 10. Outputs
- **`step_spans.json`**
  ```json
  {
    "video_uid": "assembly_9011",
    "steps": [
      {"id":1, "name":"Assemble chassis", "t0":97.2, "t1":116.5, "conf":0.92},
      {"id":2, "name":"Attach wheels", "t0":116.5, "t1":152.1, "conf":0.88},
      ...
    ]
  }
  ```
- **Optional per-frame CSV** for segmentation models (sampled at 3 fps)
- **Audit log** with confidence, NLI ok-rates, and flags

---

## 11. Benefits
✅ **No hand-crafted lexicons**  
✅ **No LLM timestamp hallucination**  
✅ **Domain-agnostic**—works across multiple assembly tasks  
✅ **Confidence-aware labels** for robust training  
✅ **Automatic consistency checks** for order, overlap, and coverage  

---

## 12. Future Extensions
- Add **cross-encoder re-ranking** trained on high-confidence pairs.  
- Use **NLI contradictions** to detect mislabeled or out-of-domain blocks.  
- Integrate **evidence pointers** (most salient frames) later for explainability.

---

## 13. Implementation Plan

### Overview
A modular, reproducible generator will be added to the ProAssist DST pipeline, following the above plan. This generator will:
- Take `inferred_knowledge` and `all_step_descriptions` as input.
- Parse, align, and merge blocks using semantic similarity, NLI, and monotonic DP.
- Output YAML files with step spans and a detailed audit log (including all intermediate matrices).
- Optionally export per-frame CSVs for segmentation models.

### Steps
1. **New Generator Class**
   - Implement as `ProAssistDSTLabelGenerator` in the codebase.
   - Register in the generator factory/config for easy selection.

2. **Dependencies**
   - Add `sentence-transformers`, `cross-encoder`, `pyyaml`, `numpy`, and `pandas` to requirements.

3. **Parsing**
   - Parse annotation lines into normalized blocks with timestamps, handling hierarchy and duration inference as described.

4. **Embedding & Similarity**
   - Use `BAAI/bge-base-en-v1.5` for block/step embeddings and cosine similarity.
   - Optionally add positional prior.

5. **NLI Scoring**
   - Use `cross-encoder/nli-deberta-v3-base` for entailment scoring between blocks and steps.

6. **Joint Scoring & Assignment**
   - Normalize and combine scores, then run monotonic DP for block-to-step assignment.

7. **Merging & Output**
   - Merge assignments into step spans, compute confidence metrics, and auto-fix overlaps/gaps.
   - Output YAML with step spans and audit log (matrices, flags, confidence, coverage, etc.).
   - Optionally output per-frame CSV.

8. **Integration & Testing**
   - Integrate with Hydra config and CLI runner.
   - Test on provided data and validate outputs.

### Notes
- No hand-crafted lexicons or LLM timestamp hallucination.
- All thresholds and model choices are configurable for future tuning.
- Audit log ensures transparency and reproducibility.

---

**Author:** Adib Mosharrof  
**Project:** ProAssist DST Label Generation  
**Version:** v1.0  (Updated Nov 2025)
