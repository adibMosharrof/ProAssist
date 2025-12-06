# Revised DST Inference Plan: Frame-Based Streaming with State Injection

## 1. Philosophy: True Streaming Simulation

The existing ProAssist `VLMStreamRunner` iterates frame-by-frame, detecting transitions and managing KV cache explicitly. This simulates a real-time agent.

**Correct Approach:** We must implement a **Frame-Based Streaming Runner** that:
1. Iterates through the video embeddings frame-by-frame.
2. Maintains a continuous **KV Cache** (Visual + Text history).
3. At each step, checks **Binary Heads** (`speaking_decision`, `dst_update`).
4. **Context Overflow Strategy:** When the KV cache fills up, we **Drop** the tensor cache and **Replace** the context with the **Full DST Schema (Updated)**.

---

## 2. Data Format & Model Architecture

### 2.1 Training Data Format

The training data contains conversations with interleaved assistant and DST_UPDATE turns.

**System Prompt Example (with DST state):**
```json
{
  "role": "system",
  "content": "You are a helpful assistant.\n\nDialogue Context:\nCurrent step states - Step S1: in_progress, Step S6: not_started, Step S2: not_started, Step S5: not_started, Step S9: not_started, Step S4: not_started, Step S8: not_started, Step S7: not_started, Step S3: not_started",
  "start_frame": 0,
  "end_frame": 1,
  "labels": "system|generic"
}
```

**Key Insight:** The DST state is included in the system prompt at the start of each clip. This provides the model with the current task progress context.

**DST_UPDATE Turn Example:**
```json
{
  "role": "DST_UPDATE",
  "content": [
    {
      "id": "S7",
      "transition": "complete"
    }
  ],
  "time": 902.3,
  "start_frame": 1798,
  "end_frame": 1804,
  "dst_state": {
    "S7": "not_started"
  },
  "labels": "initiative|dst_update,dst_complete,steps_1"
}
```

**Assistant Turn Example (following DST_UPDATE):**
```json
{
  "role": "assistant",
  "content": "Now, reattach the rear bumper to the chassis. Make sure it's securely attached.",
  "time": 902.3,
  "labels": "assistant|generic",
  "start_frame": 1804,
  "end_frame": 1804,
  "dst_state": {
    "S7": "completed",
    "S8": "in_progress"
  }
}
```

**Turn Ordering:** At the same timestamp, DST_UPDATE turns always come BEFORE assistant turns. This is enforced by `proassist_style.py`.

### 2.2 DST Update Format

The model generates DST updates as **text**, not JSON:
- **Format:** `"S1->start"` or `"S2->complete"`
- **Single update per turn:** Each DST_UPDATE turn contains exactly one update
- **Transitions:** `start`, `complete` (mapped to `in_progress`, `completed` states)

**Implementation:** See `DSTTrainingDataset.get_dst_update_str()`:
```python
def get_dst_update_str(self, dst_update_content: List[Dict]) -> str:
    """Convert DST update to text: "S1->start" or "S2->complete" """
    update = dst_update_content[0]  # Take first update
    return f"{update['id']}->{update['transition']}"
```

### 2.3 Model Architecture: `DSTSmolVLMWithStrategies`

The model extends `SmolVLMWithStrategies` with multi-task heads:

```
Video Frames + Dialog History + Current DST → SmolVLM2 (VLM-based)
                                                  ↓
                                          [Multi-Task Heads]
                                                  ↓
        ┌─────────────────┬─────────────────┬─────────────────────┐
        │                 │                 │                     │
        │  Speaking       │    DST Update   │  Text Generation    │
        │  Decision       │    Decision     │  (Response OR       │
        │  (Binary Head)  │  (Binary Head)  │   DST Update Text)  │
        │  [Linear(h,1)]  │  [Linear(h,1)]  │  (LM Head)          │
        └─────────────────┴─────────────────┴─────────────────────┘
```

**Binary Heads:**
- `speaking_decision_head`: Linear(hidden_size, 1) → Binary decision (sigmoid > 0.5)
- `dst_update_head`: Linear(hidden_size, 1) → Binary decision (sigmoid > 0.5)

**Important:** Both binary heads operate on `last_hidden_state` which is the **full sequence** of hidden states `[batch, seq_len, hidden_size]`. Due to causal attention in the transformer, the hidden state at each position contains information from ALL previous tokens (both visual and text). This is the same approach as ProAssist's W2T head.

During inference, we check the decision at the **last position** (`[:, -1, :]`) which has attended to all prior context.

### 2.4 Vision Feature Handling

**ProAssist Pattern (CLS Token Strategy):**
1. Pre-extract [CLS] tokens from SmolVLM2 vision encoder (2048-dim)
2. Save to disk as float16 embeddings
3. During training/inference, load embeddings and project to LLM space via trainable MLP

**Forward Pass Flow:**
```python
# 1. Load pre-computed embeddings
image_embeds = sample["embeddings"]  # [num_frames, 2048]

# 2. Project to LLM space via trainable MLP
vision_embeds = self.vision_projector(image_embeds)  # [num_frames, hidden_size]

# 3. Replace <image> tokens with projected embeddings
input_embeds = text_model.embed_tokens(input_ids)
for pos in image_token_positions:
    input_embeds[pos] = vision_embeds[frame_idx]

# 4. Forward through LLM (returns KV cache)
outputs = text_model(inputs_embeds=input_embeds, past_key_values=kv_cache)
```

---

## 3. Streaming Inference Architecture

### 3.1 Data Loading

- **Source:** `DSTTrainingDataset` (provides precomputed embeddings `[T, 2048]`)
- **Stream Simulation:** Iterate through the `embeddings` tensor frame-by-frame
- **DST Schema:** Available in `sample["dst"]` as list of step definitions

### 3.2 The `DSTStreamRunner`

#### Core Loop
```python
def run_inference_on_video(self, sample):
    embeddings = sample["embeddings"]  # [T, 2048]
    kv_cache = None
    dst_state = {}  # Running state: {"S1": "completed", "S2": "in_progress"}
    dst_schema = sample["dst"]  # List of step definitions
    last_msg = None
    
    outputs = []
    
    for frame_idx in range(len(embeddings)):
        # 1. Prepare Input (frame embedding + optional last message)
        model_inputs = self.prepare_inputs(embeddings[frame_idx], last_msg)
        
        # 2. Forward Pass
        model_outputs = self.model.forward_inference(
            image_embeds=model_inputs["image_embeds"],
            input_ids=model_inputs["input_ids"],
            past_key_values=kv_cache
        )
        
        # 3. Update KV Cache
        kv_cache = model_outputs.past_key_values
        last_msg = None
        
        # 4. Check DST Update Decision
        dst_update_triggered = False
        if torch.sigmoid(model_outputs.dst_update_logits[:, -1]) > self.dst_threshold:
            # Generate DST update text: "S1->complete"
            dst_text = self.generate_text(model_outputs, mode="dst")
            dst_state = self.update_state(dst_state, dst_text)
            dst_update_triggered = True
        
        # 5. Check Speaking Decision
        gen_text = ""
        if torch.sigmoid(model_outputs.speaking_logits[:, -1]) > self.speaking_threshold:
            # Generate assistant response
            gen_text = self.generate_text(model_outputs, mode="response")
        
        # 6. Store output for evaluation
        outputs.append(FrameOutput(
            gen=gen_text,
            ref=self._get_ref_at_frame(sample, frame_idx),
            frame_idx_in_stream=frame_idx,
            timestamp_in_stream=frame_idx / self.fps,
            dst_update=dst_text if dst_update_triggered else None,
            dst_state=dict(dst_state),
        ))
        
        # 7. Context Management (KV Cache Overflow)
        if self._should_refresh_context(kv_cache):
            kv_cache = None
            # Rebuild system prompt with updated DST state
            last_msg = self._build_updated_schema_prompt(dst_schema, dst_state)
    
    return outputs
```

### 3.3 Generation Modes

The model uses the **same LM head** for both DST updates and assistant responses. The "mode" is determined by which binary head triggers:

| Condition | Action | Generated Content |
|-----------|--------|-------------------|
| `dst_update_logits > threshold` | Generate DST update | `"S3->complete"` |
| `speaking_logits > threshold` | Generate response | `"Great job! Now attach..."` |
| Both triggered | Generate both sequentially | DST first, then response |
| Neither triggered | Skip generation | Empty string |

**Important:** DST_UPDATE and assistant turns can occur at the same timestamp. In training data, DST_UPDATE always precedes assistant at the same time. During inference, we respect this order.

### 3.4 DST State Parsing

**Parsing DST Update Text:**
```python
def parse_dst_update(self, text: str) -> tuple[str, str] | None:
    """Parse "S1->complete" into (step_id, transition)."""
    text = text.strip()
    if "->" not in text:
        return None
    parts = text.split("->", 1)
    if len(parts) != 2:
        return None
    step_id = parts[0].strip()
    transition = parts[1].strip()
    step_id = parts[0].strip()
    transition = parts[1].strip()
    return (step_id, transition)
```

**Flexible Parsing:**
We allow flexible spacing around the arrow (e.g., `S1 -> start` or `S1->start`) to be robust to minor generation variations.

```python
def update_state(self, current_state: dict, dst_text: str) -> dict:
    """Update DST state based on generated text."""
    parsed = self.parse_dst_update(dst_text)
    if parsed is None:
        logger.warning(f"Invalid DST update format: {dst_text}")
        return current_state
    
    step_id, transition = parsed
    # Map transition to state
    if transition == "start":
        current_state[step_id] = "in_progress"
    elif transition == "complete":
        current_state[step_id] = "completed"
    return current_state
```

**Malformed Output Handling:**
- If the model generates invalid DST text (no "->", unknown format), we log a warning and skip the update.
- This maintains system robustness while preserving metrics accuracy.

### 3.5 Context Overflow Strategy: Full DST Schema Refresh

When KV cache exceeds `max_seq_len - reserved_seq_len`:

1. **Drop:** Clear the entire KV cache
2. **Re-inject:** Build a new system prompt with the full DST schema + current state
3. **Continue:** Next iteration starts with `[Updated Schema] + [Next Frame]`

**Building Updated Schema Prompt:**
```python
def _build_updated_schema_prompt(self, dst_schema: list, current_state: dict) -> str:
    """
    Build system prompt with full task knowledge and current progress.
    
    Example output:
    "You are a helpful assistant.
    
    Task: Assembling a Toy Excavator Model
    Steps:
    - S1: Assemble the base structure (COMPLETED)
    - S2: Assemble the turntable (COMPLETED)
    - S3: Attach the cabin (IN_PROGRESS)
    - S4: Assemble the boom (NOT_STARTED)
    ...
    
    Current step states - Step S1: completed, Step S2: completed, Step S3: in_progress, ..."
    """
    lines = ["You are a helpful assistant.\n"]
    lines.append(f"Task: {self.inferred_goal}\n")
    lines.append("Steps:")
    
    for step in dst_schema:
        step_id = step["id"]
        step_name = step["name"]
        state = current_state.get(step_id, "not_started").upper()
        lines.append(f"- {step_id}: {step_name} ({state})")
    
    # Add current state in original format for consistency
    state_str = ", ".join([f"Step {k}: {v}" for k, v in current_state.items()])
    lines.append(f"\nDialogue Context:\nCurrent step states - {state_str}")
    
    return "\n".join(lines)
```

**Key Insight:** Unlike ProAssist's "Summary + Task Knowledge" approach, we use "Full DST Schema + Current State". The DST state itself serves as the memory of what happened, eliminating the need for a generated summary.

---

## 4. KV Cache Management

### 4.1 Multimodal KV Cache

**Confirmed:** The model's `forward()` method combines text and vision embeddings into a single `inputs_embeds` tensor before passing to the LLM. Therefore:

- The LLM's `past_key_values` contains attention keys/values for **both** modalities
- By passing `past_key_values` back at each step, we preserve full multimodal history
- Vision embeddings are inserted at `<image>` token positions

**Code Reference (from `DSTSmolVLMWithStrategies.forward()`):**
```python
# Get text embeddings
input_embeds = text_model.embed_tokens(input_ids)

# Replace image tokens with vision features
if vision_embeds is not None:
    for batch_idx in range(input_ids.shape[0]):
        image_positions = torch.where(image_mask[batch_idx])[0]
        for pos in image_positions:
            input_embeds[batch_idx, pos] = vision_embeds[frame_idx]

# Forward through text model (returns KV cache)
text_outputs = text_model(
    inputs_embeds=input_embeds,
    past_key_values=past_key_values,
    use_cache=True,
)
```

### 4.2 Streaming with KV Cache

During inference, we incrementally build context:

```
Frame 0: [System Prompt] + [<image>] → KV cache stores [sys + img0]
Frame 1: [<image>] → KV cache stores [sys + img0 + img1]
Frame 2: [<image>] + [Generated Response] → KV cache stores [sys + img0 + img1 + img2 + gen]
...
```

When cache overflows, we reset and inject the updated schema.

---

## 5. Evaluation Metrics

### 5.1 ProAssist Metrics (Reused)

We reuse ProAssist's evaluation infrastructure for turn-taking and NLG metrics:

| Metric | Description | Source |
|--------|-------------|--------|
| `jaccard_index` | Matched / (Matched + Missed + Redundant) | `find_match()` |
| `precision` | Matched / (Matched + Redundant) | `find_match()` |
| `recall` | Matched / (Matched + Missed) | `find_match()` |
| `F1` | 2 * P * R / (P + R) | Derived |
| `missing_rate` | Missed / (Matched + Missed) | `find_match()` |
| `redundant_rate` | Redundant / (Matched + Redundant) | `find_match()` |
| `BLEU-1/2/3/4` | N-gram overlap | `NLGEval` |
| `METEOR` | Alignment-based metric | `NLGEval` |
| `CIDEr` | Consensus-based metric | `NLGEval` |

### 5.2 DST-Specific Metrics (New)

#### Binary Decision Metrics

Similar to ProAssist's "when to speak" evaluation, we evaluate:

1. **Speaking Decision:** Did the model correctly decide WHEN to speak?
2. **DST Update Decision:** Did the model correctly decide WHEN to update DST?

| Metric | Description |
|--------|-------------|
| `speaking_balanced_accuracy` | Balanced Accuracy of speaking binary decision |
| `speaking_precision` | Precision of speaking predictions |
| `speaking_recall` | Recall of speaking predictions |
| `speaking_f1` | F1 score for speaking decision |
| `dst_update_accuracy` | Accuracy of DST update binary decision |
| `dst_update_precision` | Precision of DST update predictions |
| `dst_update_recall` | Recall of DST update predictions |
| `dst_update_recall` | Recall of DST update predictions |
| `dst_update_f1` | F1 score for DST update decision |
| `dst_update_balanced_acc` | **Balanced Accuracy** (handles class imbalance) |

**Note:** We use `balanced_accuracy_score` for both speaking and DST update decisions to account for the high imbalance (most frames are silent/no-update).

#### DST Content Metrics

For DST updates, we use **exact matching** instead of semantic similarity (unlike assistant responses):

**Matching Criteria:**
1. **Step ID Match:** `S1` == `S1` (exact string match)
2. **Transition Match:** `complete` == `complete` (exact keyword match)

**Implementation:**
```python
def evaluate_dst_update(pred_text: str, ref_text: str) -> dict:
    """
    Evaluate DST update prediction vs reference.
    
    pred_text: "S3->complete"
    ref_text: "S3->complete"
    """
    pred_parsed = parse_dst_update(pred_text)  # ("S3", "complete")
    ref_parsed = parse_dst_update(ref_text)    # ("S3", "complete")
    
    if pred_parsed is None or ref_parsed is None:
        return {"step_match": False, "transition_match": False, "exact_match": False}
    
    step_match = pred_parsed[0] == ref_parsed[0]
    transition_match = pred_parsed[1] == ref_parsed[1]
    exact_match = step_match and transition_match
    
    return {
        "step_match": step_match,
        "transition_match": transition_match,
        "exact_match": exact_match,
    }
```

| Metric | Description |
|--------|-------------|
| `dst_step_accuracy` | % of DST updates with correct step ID |
| `dst_transition_accuracy` | % of DST updates with correct transition |
| `dst_exact_match` | % of DST updates with both correct |
| `dst_joint_goal_accuracy` | % of clips where ALL DST updates are correct |

### 5.3 Evaluation Flow

```python
def evaluate_sample(predictions: List[FrameOutput], sample: dict) -> dict:
    """
    Evaluate a single sample's predictions.
    
    1. Turn-taking evaluation (ProAssist pattern):
       - Use find_match() to bipartite match predictions to references
       - Compute Jaccard, Precision, Recall, F1
       
    2. NLG evaluation (for matched pairs):
       - Use semantic similarity threshold (0.5) to filter matches
       - Compute BLEU, METEOR, CIDEr on filtered pairs
       
    3. DST evaluation:
       - Binary decision metrics (speaking, dst_update)
       - DST content metrics (step/transition exact match)
    """
```

---

## 6. File Structure

```
custom/
├── config/
│   ├── inference/                    # NEW: Separate inference config folder
│   │   └── dst_inference.yaml        # Main Hydra config for inference
│   └── prospect/
│       ├── model/
│       │   └── dst_smolvlm2.yaml     # Model config (shared)
│       └── data_source/
│           └── dst_eval.yaml         # Evaluation data source
│
└── src/prospect/
    ├── models/
    │   └── dst_smolvlm_with_strategies.py
    └── inference/                    # NEW: Dedicated inference package
        ├── __init__.py
        ├── run_inference.py          # Main entry point (Hydra app)
        ├── runners/
        │   ├── __init__.py
        │   └── dst_stream_runner.py  # Frame-by-frame streaming runner
        ├── evaluators/
        │   ├── __init__.py
        │   └── dst_evaluator.py      # Parallel evaluator wrapper
        └── metrics/
            ├── __init__.py
            └── dst_metrics.py        # DST-specific metrics
```

---

## 7. Efficiency Considerations

### 7.1 Why Sequential Frame Processing is Unavoidable

**Important:** Both ProAssist and our DST inference process frames **sequentially within each video**. This is not a limitation of our implementation but a fundamental requirement of streaming inference:

1. **KV Cache Dependency:** Each frame's hidden states depend on ALL previous frames via the KV cache. Frame N cannot be processed until frames 0 to N-1 are in the cache.

2. **Causal Decision Making:** The binary heads (speaking, DST update) need to see the full context up to the current frame. Processing frames in parallel would break causality.

3. **Autoregressive Generation:** When the model decides to speak, it generates tokens one-by-one, each depending on previous tokens.

### 7.2 Parallelism Strategies

#### Strategy 1: Multi-Video Parallelism (Required)
We will process **multiple videos concurrently**, with each video assigned to a different GPU. This ensures maximum throughput while maintaining the sequential nature of frame processing within each video.

```python
class ParallelDSTEvaluator:
    def evaluate_all(self, dataset):
        # Use Hydra's launcher or multiprocessing to distribute videos
        # Each worker process handles a subset of videos on a specific GPU
        ...
```

**Speedup:** Near-linear with number of GPUs (e.g., 4 GPUs → ~4x faster)

#### Strategy 2: Sparse Generation
The binary heads act as a **filter** - generation only happens when triggered:

```python
for frame_idx in range(len(embeddings)):
    # FAST: Forward pass for binary decision (~10ms)
    outputs = self.model.forward(image_embeds=frame_embed, past_key_values=kv_cache)
    kv_cache = outputs.past_key_values
    
    # Check binary heads
    speaking = torch.sigmoid(outputs.speaking_logits[:, -1]) > threshold
    dst_update = torch.sigmoid(outputs.dst_update_logits[:, -1]) > threshold
    
    # SLOW: Generation only when triggered (~100-500ms, but rare)
    if speaking:
        response = self.generate(...)
    if dst_update:
        dst_text = self.generate(...)
```

**Insight:** If speaking/DST triggers are sparse (e.g., 5% of frames), most frames are just fast forward passes.

### 7.3 Timing Estimates

Based on typical SmolVLM2 performance on A100 GPU:

| Operation | Time per Frame | Notes |
|-----------|---------------|-------|
| Forward pass (with KV cache) | ~5-15ms | Incremental, only new tokens |
| Binary head decision | ~0.1ms | Two linear layers |
| Text generation (50 tokens) | ~100-200ms | Autoregressive |
| KV cache management | ~1ms | Concatenation/trimming |

**Example Video (1000 frames, 5% speaking rate):**
- Forward passes: 1000 × 10ms = 10s
- Binary decisions: 1000 × 0.1ms = 0.1s
- Generations: 50 × 150ms = 7.5s
- **Total: ~18s per video**

**With 4 GPUs processing videos in parallel:**
- 100 videos / 4 GPUs × 18s = ~7.5 minutes total

### 7.4 Recommended Configuration

```yaml
# custom/config/inference/dst_inference.yaml

inference:
  # Multi-video parallelism
  num_gpus: 4
  videos_per_gpu: 1  # Process one video at a time per GPU
  
  # Prefill optimization
  prefill_frames: 32  # Process first N frames together
  
  # Binary decision thresholds
  speaking_threshold: 0.5
  dst_update_threshold: 0.5
  
  # Generation settings
  max_new_tokens: 128
  do_sample: false  # Greedy for speed
  
  # KV cache management
  max_seq_len: 4096
  reserved_seq_len: 512
```

---

## 8. Summary

### 8.1 Key Differences from ProAssist

| Aspect | ProAssist | DST Inference |
|--------|-----------|---------------|
| Speaking Decision | W2T probability head | Binary classification head (sigmoid) |
| Task Progress | Generated summary | DST state dictionary |
| Context Refresh | Summary + Task Knowledge | Full DST Schema + Current State |
| Output Types | Response only | Response AND DST update |
| Content Evaluation | Semantic similarity | Exact match (for DST) |

### 8.2 Components to Implement

| Component | Source | Notes |
|-----------|--------|-------|
| Dataset | `DSTTrainingDataset` | Same class for train/eval |
| `FrameOutput` | `mmassist.eval.runners.stream_inference` | Reuse directly |
| `find_match()` | `mmassist.eval.evaluators.pred_match` | Reuse directly |
| `NLGEval` | `mmassist.eval.metrics.nlg_scorer` | Reuse directly |
| `DSTStreamRunner` | **NEW** | Frame-by-frame streaming with KV cache |
| `DSTEvaluator` | **NEW** | ProAssist metrics + DST metrics |
| `DSTMetricsCalculator` | **NEW** | Binary decision + exact match metrics |

### 8.3 Metrics Summary

1. **Turn-Taking Metrics** (from ProAssist):
   - Jaccard Index, Precision, Recall, F1, Missing Rate, Redundant Rate

2. **NLG Metrics** (from ProAssist):
   - BLEU-1/2/3/4, METEOR, CIDEr

3. **DST Binary Decision Metrics** (NEW):
   - Speaking: Accuracy, Precision, Recall, F1
   - DST Update: Accuracy, Precision, Recall, F1

4. **DST Content Metrics** (NEW):
   - Step Accuracy, Transition Accuracy, Exact Match, Joint Goal Accuracy
