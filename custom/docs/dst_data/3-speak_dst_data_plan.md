# DST Data Creation Plan for ProAssist Extension

## 1. Objective

The goal is to build a unified dataset that integrates **Dialog State Tracking (DST)** annotations with the original **ProAssist** dialogue and video data. Instead of using hand-written progress summaries (as done in ProAssist), the model will rely on **DST-based structured task knowledge** and **per-turn DST state updates**.

This dataset will be used for pretraining and fine-tuning a **Vision-Language Model (VLM)** (e.g., SmolVLM-2.2B) for downstream tasks such as:

* When-to-speak prediction
* DST state update prediction
* Evidence grounding (optional, derived dynamically)
* Progress summarization (implicit through DST tracking)

---

## 2. Key Idea

Each conversation will contain **two types of entries**:

1. **SPEAK events** — when the assistant should generate a spoken instruction or response.
2. **DST_UPDATE events** — when the task state changes (e.g., a step starts or completes), regardless of whether the assistant speaks.

Both event types are time-aligned with the video and share the same DST knowledge context.

---

## 3. Data Sources

* **Base data:** ProAssist dataset (video frames + dialogue + timestamps).
* **DST annotations:** Generated automatically from procedural knowledge (step descriptions + inferred timestamps).
* **Additional info:** Frame timestamps, node hierarchy, and object/action metadata.

---

## 4. Data Structure

Each conversation will be an array of chronological events. Every entry represents either a **SPEAK** or **DST_UPDATE** action.

### 4.1. SPEAK Event

Represents a moment when the assistant speaks to guide the user.

```json
{
  "type": "SPEAK",
  "time": 112.7,
  "labels": "initiative|instruction",
  "content": "Next, attach the wheels to the chassis. Take the first wheel and attach it to the chassis. Then, screw it with the screwdriver.",
  "dst_state_snapshot": [
    {"id": "S1", "state": "done"},
    {"id": "S2", "state": "ongoing"},
    {"id": "S3", "state": "not_started"}
  ]
}
```

**Fields:**

* `type`: Always `SPEAK` for assistant turns.
* `time`: Timestamp (seconds from start).
* `labels`: Dialogue act label(s) (`initiative|instruction`, etc.).
* `content`: Text of the assistant’s utterance.
* `dst_state_snapshot`: Current state of all DST nodes.

### 4.2. DST_UPDATE Event

Represents a change in the task state, whether or not the assistant speaks.

```json
{
  "type": "DST_UPDATE",
  "time": 130.7,
  "labels": "dst_update",
  "content": [
    {"id": "S2", "transition": "start"},
    {"id": "S1", "transition": "complete"}
  ]
}
```

**Fields:**

* `type`: Always `DST_UPDATE`.
* `time`: Timestamp when the change occurs.
* `labels`: Constant value `dst_update`.
* `content`: List of DST node transitions.

**Note:** No explicit evidence span is stored. The temporal grounding or evidence window can be computed dynamically using the timestamp and a configurable ±Δ parameter during training or evaluation.

---

## 5. DST Schema

Each video is associated with a DST plan describing the procedural structure of the task.

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
  }
]
```

This structure defines the ground-truth temporal boundaries for state transitions.

---

## 6. Target Labels

Each time chunk or event will have multilabel DST targets (Option B approach):

For each node `i`, predict one of:

* `no_change`
* `start`
* `complete`
* `error`

This allows independent, per-node supervision and avoids forcing overlap with speaking events.

---

## 7. Label Construction

From the DST timestamps:

1. When the current window crosses a node’s `start_ts`, assign `start`.
2. When it crosses `end_ts`, assign `complete`.
3. If no boundaries are crossed, assign `no_change`.

Transitions are sparse → use **focal loss** with higher weight on `start` and `complete`.

Example per-node target vector for one frame:

```json
{
  "S1": "complete",
  "S2": "start",
  "S3": "no_change"
}
```

---

## 8. Mixing Event Types

During training:

* Mix `SPEAK` and `DST_UPDATE` samples in the same dataloader (e.g., 60%/40%).
* Each item contains:

  * Sampled frames
  * Static DST knowledge (list of step names)
  * Current DST state
  * Event type token (`<SPEAK>` or `<DST_UPDATE>`)
  * Corresponding target (either text or transitions)

---

## 9. Inference Flow

At inference time, for each time window:

1. Predict whether to **speak** (`p_speak`).
2. Predict **DST transitions** per node.
3. Update internal DST state based on predicted transitions.
4. Generate assistant text only when `p_speak` exceeds a threshold.
5. If grounding is needed, dynamically compute a ±Δ temporal window around the event timestamp to extract supporting frames.

---

## 10. Benefits

* Maintains a single unified conversation timeline.
* Handles asynchronous events (state can change without speech, or vice versa).
* Replaces heuristic progress summaries with **structured, explainable DST states**.
* Enables multitask supervision (speak, update, ground, summarize).
* Grounding evidence spans are computed dynamically → flexible and cleaner dataset.
* Compatible with ProAssist’s Negative Frame Subsampling and chunking scheme.

---

## 11. Next Steps

1. Implement data writer to generate the `conversations` array.
2. Validate timestamps alignment with video frames.
3. Add schema checks for node consistency.
4. Visualize event alignment (timeline view).
5. Integrate dataset into VLM pretraining pipeline.
