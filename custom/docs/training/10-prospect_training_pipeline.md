# PROSPECT DST Training Pipeline: From Raw Data to Model Output

## Overview

This document traces the complete flow of data through the PROSPECT DST (Dialogue State Tracking) training pipeline, from raw JSON training data to model predictions and loss computation. Each stage includes visual diagrams to help you understand how a conversation is processed end-to-end.

---

## 1. Raw Training Data Format

### Input: conversation.json

The training data is organized as clips, where each clip represents one training sample with a complete conversation and associated DST state. Here's an example from the actual training data:

```json
{
  "video_uid": "assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit",
  "inferred_goal": "Assembling a Toy Roller with Chassis, Wheels, and Cabin",
  "conversation": [
    {
      "role": "assistant",
      "content": "Now, let's attach the cabin window to the cabin and then to the chassis. Take the cabin window and secure it to both parts using your screwdriver.",
      "time": 197.9,
      "labels": "assistant|generic",
      "start_frame": 394,
      "end_frame": 396,
      "dst_state": {
        "S1": "completed",
        "S2": "completed",
        "S3": "completed",
        "S4": "completed",
        "S5": "in_progress"
      }
    },
    {
      "role": "DST_UPDATE",
      "content": [
        {
          "id": "S5",
          "transition": "complete"
        }
      ],
      "time": 212.3,
      "start_frame": 423,
      "end_frame": 425,
      "dst_state": {
        "S1": "completed",
        "S2": "completed",
        "S3": "completed",
        "S4": "completed",
        "S5": "completed",
        "S6": "not_started"
      }
    }
  ],
  "dst": [
    {
      "type": "step",
      "id": "S5",
      "start_ts": 197.9,
      "end_ts": 212.3,
      "name": "Add the cabin window to the chassis."
    }
  ],
  "dataset": "assembly101",
  "start_frame_idx": 193,
  "end_frame_idx": 451
}
```

### Key Components:

1. **conversation**: List of turns, each with:
   - `role`: "assistant", "user", "DST_UPDATE", or "system"
   - `content`: Text of the turn (for assistant/user) or structured DST updates (for DST_UPDATE)
   - `start_frame`, `end_frame`: Frame indices corresponding to this turn
   - `dst_state`: Current DST state at this turn

2. **dst**: Global list of DST steps being tracked in this conversation

3. **Frame references**: `start_frame_idx` and `end_frame_idx` point to specific frames in the video

---

## 2. Dataset Loading & Preprocessing

### Stage 1: Loading a Clip from JSON

When the dataloader requests a sample, the dataset performs three key steps:

1. **Load JSON and retrieve one clip** - The training data is stored in JSON files. Each clip in the JSON represents one complete conversation with associated DST state and frame metadata.

2. **Get unique clip ID** - Each clip is assigned a unique identifier that's used to find the corresponding precomputed vision embeddings.

3. **Load precomputed vision embeddings** - The embeddings were pre-extracted from video frames and stored in pickle files. Since `use_precomputed_embeddings=True` is the default, the dataset loads all frames for the clip as a numpy array of shape `[num_frames, 2048]`, where 2048 is the dimension of the [CLS] token from SmolVLM2.

#### Pipeline Diagram: Dataset Loading Stage

```
┌──────────────────────────────────────┐
│  Training JSON File                  │
│  ├─ Clip 0                          │
│  │  ├─ start_frame_idx: 193         │
│  │  ├─ end_frame_idx: 451           │
│  │  ├─ conversation: [...]          │
│  │  └─ dst: [...]                   │
│  ├─ Clip 1                          │
│  └─ ...                             │
└──────────────────────────────────────┘
           ↓ (Load Clip #0)
┌──────────────────────────────────────┐
│  Clip Data                           │
│  ├─ video_uid: "assembly_..."       │
│  ├─ clip_id: unique identifier      │
│  ├─ conversation: [                 │
│  │   {role: "assistant", ...},      │
│  │   {role: "DST_UPDATE", ...}      │
│  │ ]                                │
│  └─ dst: [step definitions]         │
└──────────────────────────────────────┘
           ↓ (Load embeddings by clip_id)
┌──────────────────────────────────────┐
│  Precomputed Embeddings Pickle File  │
│  {clip_id}_embeddings.pkl            │
│  = numpy array [259, 2048]           │
│                                      │
│  [Frame 0 embedding (2048-dim)]      │
│  [Frame 1 embedding (2048-dim)]      │
│  ...                                 │
│  [Frame 258 embedding (2048-dim)]    │
└──────────────────────────────────────┘
           ↓ (Return sample)
┌──────────────────────────────────────┐
│  Sample Dictionary                   │
│  {                                   │
│    "conversation": [...],            │
│    "embeddings": [259, 2048] tensor, │
│    "dst": [...],                     │
│    "clip_id": "...",                 │
│    "sample_idx": 0                   │
│  }                                   │
└──────────────────────────────────────┘
```

**Key Details at this Stage:**
- Each sample includes the full conversation history (all turns)
- Embeddings cover the entire clip (259 frames in this example)
- The sample is a Python dictionary with all information needed for batch collation
- Embeddings are already computed (no vision encoder needed during training)

---

## 3. Batch Collation & Label Generation

### Stage 2: Preparing Samples for the Model

The collator takes multiple samples from the dataset and combines them into a single batch ready for the model. This involves five main steps:

**Step 1: Extract embeddings and count frames**
For each sample in the batch, extract the embeddings tensor and note how many frames it contains.

**Step 2: Format conversations with temporal frame interleaving**
Each conversation is formatted using `format_conversation_with_ranges()`. This method:
- Formats text with temporal interleaving: [turn_1_frames][turn_1_text] [turn_2_frames][turn_2_text]...
- Returns **four separate range lists** computed in a single pass:
  1. `speaking_ranges`: Character ranges for assistant turn text
  2. `dst_ranges`: Character ranges for DST_UPDATE turn text  
  3. `negative_ranges`: Character ranges for sampled non-assistant frames
  4. Implicitly: Image token ranges for all turns

This ensures perfect alignment between formatted text and ranges, since both are computed together. Frame ranges are **INCLUSIVE**: `[start_frame, end_frame]` means frames start through end (inclusive on both ends), so `num_frames = end_frame - start_frame + 1`.

**Step 3: Tokenize the formatted text**
The formatted text is tokenized using the LLM's tokenizer. The tokenizer returns both token IDs and offset mappings that track which characters in the original text correspond to which tokens. This mapping is critical for creating labels.

**Step 4: Pad embeddings and flatten**
Since different turns have different numbers of frames, all embeddings are collected and flattened in the order they appear in the conversation text. This ensures frame-to-token alignment.

**Step 5: Create learning labels using character ranges**
The collator converts the four character ranges (speaking, dst, negative) into token-level label tensors using the offset mappings from tokenization:

- **speaking_ranges → speaking_gen_labels + speaking_labels**
  - Identify token positions corresponding to speaking_ranges
  - Create `speaking_gen_labels`: token IDs for LM loss (next-token prediction)
  - Create `speaking_labels`: value of 1 for binary "should speak?" classification

- **dst_ranges → dst_gen_labels + dst_update_labels**
  - Identify token positions corresponding to dst_ranges
  - Create `dst_gen_labels`: token IDs for LM loss (next-token prediction)
  - Create `dst_update_labels`: value of 1 for binary "should update DST?" classification

- **negative_ranges → speaking_gen_labels**
  - If negative frame sampling is enabled, identify negative frame tokens
  - Include these in `speaking_gen_labels` for training (learn when NOT to speak)
  - All other positions default to -100 (ignored by loss functions)

#### Pipeline Diagram: Batch Collation Stage

```
┌────────────────────────────────────────────────────────────┐
│  Multiple Samples from DataLoader                          │
│  Each contains: conversation, embeddings [N, 2048], etc.   │
└────────────────────────────────────────────────────────────┘
         ↓ (Extract embeddings)
┌────────────────────────────────────────────────────────────┐
│  Step 1: Conversation Data                                 │
│  {                                                         │
│    "conversation": [                                       │
│      {role: "system", content: "You are helpful..."},      │
│      {role: "assistant", content: "Let's start",           │
│       start_frame: 5, end_frame: 7},                       │
│      {role: "DST_UPDATE", content: [{id: "S1", ...}],      │
│       start_frame: 10, end_frame: 11}                      │
│    ],                                                      │
│    "embeddings": [29, 2048]                               │
│  }                                                         │
└────────────────────────────────────────────────────────────┘
         ↓ (Format with single source of truth)
┌────────────────────────────────────────────────────────────┐
│  Step 2: format_conversation_with_ranges() Output          │
│                                                            │
│  Returns: (formatted_text, speaking_ranges, dst_ranges,   │
│            negative_ranges)                               │
│                                                            │
│  formatted_text:                                           │
│  <image><image><image>Assistant: Let's start...            │
│  <image><image>[DST_UPDATE] S1->completed...               │
│                                                            │
│  speaking_ranges: [range(start_1, end_1)]                 │
│    (char positions for "Assistant: Let's start...")        │
│                                                            │
│  dst_ranges: [range(start_2, end_2)]                       │
│    (char positions for "[DST_UPDATE] S1->completed...")    │
│                                                            │
│  negative_ranges: [...]                                    │
│    (char positions for sampled non-assistant frames)       │
│                                                            │
│  Key: Single method ensures text and ranges are aligned    │
│       Frame range is INCLUSIVE: num_frames = end - start+1 │
│       All 4 ranges computed in one pass, zero mismatch risk │
└────────────────────────────────────────────────────────────┘
         ↓ (Tokenize)
┌────────────────────────────────────────────────────────────┐
│  Step 3: Tokenized with Offset Mapping                     │
│                                                            │
│  input_ids:     [tok1, tok2, tok3, ..., tok_N]            │
│  offset_mapping: [(char_start, char_end), ...]            │
│                                                            │
│  Tokens include both image tokens and text tokens         │
│  in their natural temporal order                          │
└────────────────────────────────────────────────────────────┘
         ↓ (Extract embeddings by turn)
┌────────────────────────────────────────────────────────────┐
│  Step 4: Flattened Embeddings by Turn Order                │
│                                                            │
│  Turn 1 embeddings: [3, 2048]  (frames 5-7)              │
│  Turn 2 embeddings: [2, 2048]  (frames 10-11)            │
│                                                            │
│  Flattened: [5, 2048]  (turn 1 then turn 2 frames)       │
│  Ordering: Exactly matches order of <image> tokens       │
│           in tokenized sequence                           │
└────────────────────────────────────────────────────────────┘
         ↓ (Create labels from offset mappings)
┌────────────────────────────────────────────────────────────┐
│  Step 5: Label Tensors from Range Mapping                  │
│                                                            │
│  speaking_gen_labels (from speaking_ranges):              │
│    [-100, -100, ..., tok_k, tok_k+1, ..., -100, ...]     │
│    (only assistant turn tokens are NOT -100)             │
│    (next-token prediction LM loss)                        │
│                                                            │
│  speaking_labels (from speaking_ranges):                  │
│    [-100, -100, ..., 1, 1, ..., 1, -100, ...]           │
│    (1 for assistant turn tokens, binary classification)   │
│                                                            │
│  dst_gen_labels (from dst_ranges):                         │
│    [-100, ..., tok_m, tok_m+1, ..., -100, ...]           │
│    (only DST_UPDATE tokens are NOT -100)                 │
│    (next-token prediction LM loss)                        │
│                                                            │
│  dst_update_labels (from dst_ranges):                      │
│    [-100, ..., 1, 1, ..., 1, -100, ...]                  │
│    (1 for DST_UPDATE tokens, binary classification)       │
│                                                            │
│  Note: negative_ranges are optionally included in          │
│        speaking_gen_labels for training on negative frames │
└────────────────────────────────────────────────────────────┘
         ↓ (Stack and return batch)
┌────────────────────────────────────────────────────────────┐
│  Final Batch Dictionary:                                   │
│  {                                                         │
│    "input_ids": [batch_size, seq_len],                    │
│    "attention_mask": [batch_size, seq_len],              │
│    "image_embeds": [total_frames, 2048],                  │
│      (flattened: all turn embeddings concatenated)        │
│    "speaking_gen_labels": [batch_size, seq_len],          │
│      (LM loss for assistant turns only)                   │
│    "speaking_labels": [batch_size, seq_len],              │
│      (binary: 1 for assistant, -100 elsewhere)            │
│    "dst_gen_labels": [batch_size, seq_len],               │
│      (LM loss for DST_UPDATE turns only)                  │
│    "dst_update_labels": [batch_size, seq_len],            │
│      (binary: 1 for DST_UPDATE, -100 elsewhere)           │
│  }                                                         │
└────────────────────────────────────────────────────────────┘
```

**Key Details at this Stage:**
- **Format and ranges together**: `format_conversation_with_ranges()` computes both text and ranges in a single pass
- **Four separate ranges**: speaking_ranges, dst_ranges, negative_ranges, and implicit image token ranges
- **Temporal interleaving**: Image tokens precede the conversation text they correspond to (one image per frame)
- **Frame range convention**: INCLUSIVE on both ends: `[start_frame, end_frame]` means frames start through end inclusive
- **Frame order preservation**: Embeddings are extracted and flattened in the exact order frames appear in the conversation
- **Negative frame sampling**: Non-assistant frames are randomly sampled during training (controlled by `neg_frame_sampling_rate`) to handle imbalanced data
- **Multi-task labels**: Four label tensors guide four different learning objectives (2 LM losses + 2 binary classifiers)
- **Selective masking**: All non-learnable regions (image tokens, user text, system text) are masked with -100

#### Label Tensor Breakdown (4-Loss Setup)

All 4 label tensors are created by the collator and used by the model:

| Label Tensor | Source Range | Purpose | Value at Learnable Positions | Value at Other Positions |
|---|---|---|---|---|
| `speaking_gen_labels` | speaking_ranges + (negative_ranges optional) | Next-token prediction LM loss for assistant turns | Token ID (next token) | -100 (ignored) |
| `speaking_labels` | speaking_ranges | Binary: should assistant speak? | 1 (yes, assistant speaking) | -100 (ignored) |
| `dst_gen_labels` | dst_ranges | Next-token prediction LM loss for DST updates | Token ID (next token) | -100 (ignored) |
| `dst_update_labels` | dst_ranges | Binary: should DST be updated? | 1 (yes, DST updating) | -100 (ignored) |

**Training Flow:**
- **Assistant turns**: Use both `speaking_gen_labels` (LM) and `speaking_labels` (binary decision)
- **DST_UPDATE turns**: Use both `dst_gen_labels` (LM) and `dst_update_labels` (binary decision)
- **Other content**: All labels are -100 (ignored by loss functions)

**Example Label Assignment for a Sequence (4-Objective Setup):**
```
Token sequence:  [system] [image][image] [asst_tok1][asst_tok2] [image][image] [dst_tok1][dst_tok2] [user_tok1]
                    ↓         ↓       ↓         ↓         ↓        ↓      ↓       ↓        ↓        ↓

speaking_gen_labels: [-100] [-100][-100] [asst_tok2][asst_tok3] [-100][-100] [-100]   [-100]   [-100]
                       (shifted by 1)          ↑ used for speaking_gen_loss

speaking_labels: [-100]   [-100][-100]   [1]      [1]      [-100][-100] [-100]   [-100]   [-100]
                                          ↑ used for speaking_binary_loss

dst_gen_labels:  [-100]   [-100][-100]  [-100]   [-100]    [-100][-100] [dst_tok2][dst_tok3] [-100]
                   (shifted by 1)                                           ↑ used for dst_gen_loss

dst_update_labels: [-100]   [-100][-100]  [-100]   [-100]    [-100][-100]   [1]      [1]    [-100]
                                                                              ↑ used for dst_binary_loss
```

**What Happens During Training:**

- **Speaking Generation Loss** (from `speaking_gen_labels`):
  - LM head converts hidden states to vocab logits
  - Compute cross-entropy loss at assistant positions
  - Model learns: "After these tokens, predict the next assistant token"

- **Speaking Binary Loss** (from `speaking_labels`):
  - speaking_decision_head converts hidden states to binary logits
  - Compute BCE loss at assistant positions
  - Model learns: "At these positions, should assistant speak?"

- **DST Generation Loss** (from `dst_gen_labels`):
  - LM head converts hidden states to vocab logits
  - Compute cross-entropy loss at DST_UPDATE positions
  - Model learns: "After these tokens, predict the next DST update token"

- **DST Binary Loss** (from `dst_update_labels`):
  - dst_update_head converts hidden states to binary logits
  - Compute BCE loss at DST_UPDATE positions
  - Model learns: "At these positions, should DST be updated?"

All 4 losses are weighted and summed for backpropagation.

---

## 4. Model Processing & Forward Pass

### Stage 3: Processing the Batch Through the Model

The model receives the batch and processes it through three parallel pathways that ultimately fuse together:

**Vision Pathway:**
- Receives precomputed embeddings of shape `[batch, max_frames, 2048]`
- These are already [CLS] token representations extracted from video frames
- Applies a trainable vision projector to align the vision embedding space with the text embedding space

**Text Pathway:**
- Receives tokenized text as input IDs
- Converts tokens to their embedding vectors using the LLM's token embedding layer

**Fusion:**
- The `<image>` tokens in the text are identified
- They are replaced with the corresponding vision embeddings from the vision pathway
- This creates a mixed representation of text and vision

**LLM Processing:**
- The fused embeddings are processed through the language model
- The model attends to both text and vision information
- Output is a sequence of hidden representations

**Prediction & Loss:**
- Token logits are computed from the hidden representations
- Language modeling loss is calculated by comparing predicted logits with label targets
- Loss is computed only on tokens marked for learning (non-(-100) labels)

#### Pipeline Diagram: Model Forward Pass Stage

```
┌─────────────────────────────────────────────────────────────┐
│  Batch from Collator                                        │
│  {                                                          │
│    "input_ids": [batch, seq_len],                          │
│    "image_embeds": [batch, max_frames, 2048],              │
│    "attention_mask": [batch, seq_len],                     │
│    "labels": [batch, seq_len]                              │
│  }                                                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
          ┌────────────┴────────────┐
          ↓                         ↓
   VISION PATHWAY            TEXT PATHWAY
          │                         │
   ┌──────────────┐          ┌──────────────┐
   │ Precomputed  │          │  Token IDs   │
   │ Embeddings   │          │              │
   │ [B,F,2048]   │          │ [B, seq_len] │
   └──────┬───────┘          └──────┬───────┘
          │                         │
          ↓                         ↓
   ┌──────────────┐          ┌──────────────┐
   │Vision        │          │Token         │
   │Projector     │          │Embeddings    │
   │              │          │              │
   │Linear(2048)  │          │embed_tokens  │
   │ ↓ GELU       │          │              │
   │Linear(2048)  │          │              │
   └──────┬───────┘          └──────┬───────┘
          │                         │
          ↓                         ↓
   ┌──────────────┐          ┌──────────────┐
   │ Projected    │          │ Text         │
   │ Vision       │          │ Embeddings   │
   │ [B,2048]     │          │ [B,seq,2048] │
   └──────┬───────┘          └──────┬───────┘
          │                         │
          └────────────┬────────────┘
                       ↓
              ┌─────────────────┐
              │  FUSION LAYER   │
              │                 │
              │ For each <image>│
              │ token position: │
              │ Replace with    │
              │ vision embedding│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   Fused Input   │
              │ [B,seq_len,2048]│
              │                 │
              │ Text tokens +   │
              │ Vision at <img> │
              │ positions       │
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   LLM Forward   │
              │   (Multi-layer) │
              │                 │
              │ Attention over  │
              │ fused embeddings│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │ Hidden States   │
              │ [B,seq_len,2048]│
              └────────┬────────┘
                       ↓
              ┌─────────────────┐
              │   LM Head       │
              │                 │
              │ Linear layer to │
              │ vocab size      │
              └────────┬────────┘
                       ↓
              ┌──────────────────┐
              │    Logits        │
              │[B,seq_len,V_size]│
              └────────┬─────────┘
                       ↓
         ┌─────────────────────────┐
         │   Loss Computation      │
         │                         │
         │ Compare logits[:-1]     │
         │ with labels[1:]         │
         │ (shift for next-token   │
         │  prediction)            │
         │                         │
         │ Ignore -100 labels      │
         └────────┬────────────────┘
                  ↓
              ┌────────┐
              │ Loss   │
              │(scalar)│
              └────────┘
```

**Key Details at this Stage:**
- **No vision encoder**: The vision model is frozen on the CPU; only precomputed embeddings are used
- **Projector alignment**: Transforms 2048-dim embeddings to match LLM hidden size
- **Seamless fusion**: Vision and text information are combined at the token position level
- **Efficient attention**: The LLM can attend to both text and vision information naturally
- **Selective loss**: Only tokens marked for learning contribute to the loss

---

## 5. Complete Data Flow: End-to-End Example

Let's trace one complete example conversation through the entire pipeline:

**Raw Conversation (from JSON):**
- System instruction: "You are a helpful assistant..."
- Assistant turn: "Now, let's attach the cabin window..." (start_frame=394, end_frame=396, inclusive → 3 frames)
- DST_UPDATE turn: Updates step S5 to "completed" (start_frame=423, end_frame=425, inclusive → 3 frames)

**At Dataset Loading:**
- Clip is loaded from JSON with all 29 frames of embeddings [29, 2048]
- Sample dictionary contains the full conversation and all embeddings

**At Batch Collation - format_conversation_with_ranges():**
- Conversation is formatted with temporal interleaving AND ranges computed:
  ```
  <image><image><image>Assistant: Now, let's attach the cabin window...
  <image><image><image>[DST_UPDATE] Step S5 completed
  ```
- Returns:
  - `formatted_text`: The above string
  - `speaking_ranges`: [range(0, 105)] (char positions of assistant text)
  - `dst_ranges`: [range(105, 150)] (char positions of DST_UPDATE text)
  - `negative_ranges`: [] (no non-assistant frames to sample)

**After Tokenization:**
```
input_ids:       [tok_img1, tok_img2, tok_img3, tok_asst, ..., tok_img4, tok_img5, tok_img6, tok_dst, ...]
offset_mapping:  [(char_start, char_end), ...]  (maps each token to char positions in formatted_text)
```

**Embedding Extraction and Flattening:**
- Turn 1 frames [394, 396] (inclusive) → 3 frames: [3, 2048]
- Turn 2 frames [423, 425] (inclusive) → 3 frames: [3, 2048]
- Flattened: concatenate → [6, 2048]
- Order: Turn 1 frames (3) followed by Turn 2 frames (3), exactly matching image token order

**Label Creation (using character ranges → token ranges via offset_mapping):**
```
speaking_gen_labels: [-100, -100, tok_asst, ..., tok_asst, -100, -100, -100, ...]
                      (LM loss: predict next token after assistant text)

speaking_labels:     [-100, -100, 1, ..., 1, -100, -100, -100, ...]
                      (binary: 1 for assistant tokens)

dst_gen_labels:      [-100, -100, -100, ..., tok_dst, ..., tok_dst, -100, ...]
                      (LM loss: predict next token after DST text)

dst_update_labels:   [-100, -100, -100, ..., 1, ..., 1, -100, ...]
                      (binary: 1 for DST_UPDATE tokens)
```

**At Model Forward Pass (4-Objective Setup):**
- Image embeddings [6, 2048] are projected through the vision projector → [6, 2048]
- Text tokens are converted to embeddings (full sequence)
- The 6 image token positions are replaced with the projected vision embeddings in order:
  - Positions of tok_img1, tok_img2, tok_img3 get embeddings from frames 394, 395, 396
  - Positions of tok_img4, tok_img5, tok_img6 get embeddings from frames 423, 424, 425
- The fused representation is processed through the LLM
- Loss is computed using four objectives:
  - **Speaking Generation Loss** (from `speaking_gen_labels`): Model learns to predict next token of assistant response (LM loss)
  - **Speaking Binary Loss** (from `speaking_labels`): Model learns to predict whether assistant should speak (binary classification)
  - **DST Generation Loss** (from `dst_gen_labels`): Model learns to predict next token of DST update (LM loss)
  - **DST Binary Loss** (from `dst_update_labels`): Model learns to predict whether DST should be updated (binary classification)
  - **Final Loss**: Weighted sum of all 4 losses

---

## 6. Multi-Task Learning Objectives

The model is trained with **FOUR** primary learning objectives happening simultaneously, derived from the four label tensors created by the collator:

**Objective 1: Speaking Generation Loss (LM Loss)**
- Model learns to predict the next token during **assistant turns only**
- Label tensor: `speaking_gen_labels` (from `speaking_ranges`)
- Helps the model generate appropriate assistant responses
- Loss function: Cross-entropy (next-token prediction)
- Applied at: All token positions where `speaking_gen_labels != -100`

**Objective 2: Speaking Decision Loss (Binary Classification)**
- Model learns to predict whether the **assistant should speak** at each token position
- Label tensor: `speaking_labels` (from `speaking_ranges`)
- Binary decision: speak (1) or don't speak (0)
- Loss function: Binary cross-entropy with logits
- Applied at: All token positions where `speaking_labels != -100`

**Objective 3: DST Generation Loss (LM Loss)**
- Model learns to predict the next token during **DST_UPDATE turns only**
- Label tensor: `dst_gen_labels` (from `dst_ranges`)
- Helps the model generate appropriate DST state updates as JSON
- Loss function: Cross-entropy (next-token prediction)
- Applied at: All token positions where `dst_gen_labels != -100`

**Objective 4: DST Binary Loss (Binary Classification)**
- Model learns to predict whether **DST should be updated** at each token position
- Label tensor: `dst_update_labels` (from `dst_ranges`)
- Binary decision: update (1) or don't update (0)
- Loss function: Binary cross-entropy with logits
- Applied at: All token positions where `dst_update_labels != -100`

**How They Work Together:**
- The collator computes 4 label tensors in a single pass from character ranges:
  - `speaking_ranges` → `speaking_gen_labels` + `speaking_labels`
  - `dst_ranges` → `dst_gen_labels` + `dst_update_labels`
- During forward pass, the model processes each token and generates hidden states
- From the same hidden states, three separate heads make predictions:
  - **LM head**: Predicts logits for all vocabulary (used for both `speaking_gen_labels` and `dst_gen_labels`)
  - **speaking_decision_head**: Predicts binary logits for "should assistant speak?"
  - **dst_update_head**: Predicts binary logits for "should DST update?"
- Only tokens marked with non-(-100) labels in the respective label tensor contribute to their loss
- All 4 losses are computed and weighted, then summed for a single backpropagation step

**Loss Combination Strategy:**
```
Total Loss = 
    speaking_gen_weight × CE(speaking_gen_logits, speaking_gen_labels) +
    speaking_binary_weight × BCE(speaking_decision_logits, speaking_labels) +
    dst_gen_weight × CE(dst_gen_logits, dst_gen_labels) +
    dst_binary_weight × BCE(dst_update_logits, dst_update_labels)
```

Where CE = Cross-Entropy (LM loss, ignores -100), BCE = Binary Cross-Entropy (classification, ignores -100)

All weights are configurable in the training config (default: 1.0 for all).

---

## 7. Training Validation Checklist

✅ **Data Quality**
- Each sample is independent (no data leakage between clips)
- Frame indices are correctly extracted for each turn
- DST state snapshots are preserved throughout
- Precomputed embeddings are validated: shape [num_frames, 2048], normalized values

✅ **Model Architecture**
- Vision encoder is frozen (no gradients computed)
- Vision model is on CPU (memory efficient)
- Vision projector properly aligns embeddings
- Text and vision pathways fuse seamlessly
- Loss computation uses proper next-token shifting

✅ **Data Pipeline**
- Variable-length embeddings are padded correctly
- Tokenization respects sequence length limits
- Offset mapping ensures accurate label placement
- All batch tensors have correct shapes and dtypes

✅ **Memory Efficiency**
- Lazy loading: clips loaded on-demand
- Precomputed embeddings: no vision encoding during training
- Padding: embeddings padded within batch, not globally

---

## 8. Summary: Pipeline Architecture

The PROSPECT training pipeline follows a **clean separation of concerns**:

1. **Offline Phase** (before training): Vision embeddings are extracted from video frames and saved to pickle files
2. **Dataset Phase**: JSON clips are loaded and paired with their precomputed embeddings
3. **Batch Phase**: Clips are formatted, tokenized, and organized into batches
4. **Model Phase**: Batches are processed through the vision-text fused model
5. **Loss Phase**: Language modeling and binary decision losses are computed

This architecture ensures:
- **Efficiency**: Vision features are pre-extracted; no encoder needed during training
- **Modularity**: Each stage can be understood and debugged independently
- **Clarity**: Data transformations are explicit and traceable
- **Scalability**: New data or modifications only affect relevant stages

