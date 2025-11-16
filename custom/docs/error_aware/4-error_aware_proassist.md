## Motivation

Human procedural tasks (e.g., assembly, repair, cooking) are non-linear: users make mistakes, skip steps, or redo parts.
ProAssist models what to say and when to speak, assuming correct sequential progress — it can’t detect when the user deviates from the intended procedure.

New insight:
If we introduce a Dialog State Tracking (DST) representation of task progress — where each state encodes what substep(s) should be complete, in progress, or pending — then the assistant can explicitly compare the current observed state vs. the expected one and detect deviation (error).

That DST structure becomes a consistency checker rather than just a structured summary.

## Data

Existing datasets in Proassist: Wtag, Assembly101 already contain information about errors (2-3%) error turns.

We can initially start with these to build the POC.

custom/outputs/dst_generated/proassist_label/2025-11-06/15-52-50_gpt-4o_proassist_50rows/assembly101/val.json

Above is an example json data file with the DST annotations added.

We will use this data to train and evaluate the DST-augmented ProAssist model.

Details on training the DST plan is given in the documents below

- custom/docs/training/2-e2e_training_plan.md
- custom/docs/training/3-dst_training_implementation_plan.md

### Synthetic Error Data Generation

To build error states, we can modify the steps in the inferred_knowledge present in the json data.
The inferred knowledge contains the expected sequence of steps, which we can alter to simulate common user errors such as skipping a step, repeating a step, or performing steps out of order.
We will have to modify the conversational dialogs as well to handle the changes.
The video will be untouched to maintain the original context and focus on the error simulation in the task steps.

#### Using DST to detect errors

Initially the model will be provided a DST which is based on the inferred knowledge. 
All the nodes will have not started state.

✅ Step 1: Generate corresponding DST states

The model will predict the observed DST (DST_observed) based on the previous DST, conversation history and video frames from the current KV cache.

From the ground truth DST, we can derive the expected DST (DST_gt_expected).

Next we can label frames with is_error = 1 if DST_observed is not equal to DST_gt_expected.


✅ Step 2: Generate recovery targets

When is_error = 1, label the corrective next step (from the original DST sequence).

This becomes the target output for your “what to say” or “how to recover” module.

So you now have paired data:

(video frames, audio, DST_expected, DST_observed) → (error_flag, correction_instruction)

### Training Tasks

We can cast this as a two-stage multitask setup:

Error Detection (binary)

Input: multimodal encoding + current DST state.

Output: probability of deviation from expected state.

Error Correction (sequence generation)

Input: same + predicted error type.

Output: natural language suggestion or next correct DST substep.

This will be another decision head in the architecture, where we predict the error type and generate the correction.

Optionally, we can add:

Auxiliary loss: state reconstruction (predict expected DST given current observations).

Contrastive loss: between correct vs. corrupted sequences (align correct temporal orderings).


### Binary Decision Making

We have 3 binary decisions:
    - should speak
    - should update DST
    - is error

For each binary decision, we will do something similar to the retrieve and re-rank approach.

Given the current frame window, we will predict whether each decision should be true or false.

Next if the decision is true, we will give look into the current frames to determine which frame constitutes the positive decision.

This will allow the model to first identify at a high level whether something major has changed, then look closely at the frames to find what exactly changed.

### Metrics

Error Detection Accuracy / F1

Correction Accuracy / BLEU / Semantic Match

Temporal Alignment Score — overlap of predicted vs. ground-truth step boundaries.

User Simulation Metrics — in future work, simulated “user correction success rate.”


### Baselines

| Model                                   | Description                                        | Purpose                                                 |
| --------------------------------------- | -------------------------------------------------- | ------------------------------------------------------- |
| **ProAssist-Base (Summarization)**      | Original model trained to summarize user progress. | Tests whether summarization implicitly captures errors. |
| **ProAssist-DST (Ours)**                | Predicts explicit DST and detects deviations.      | Core novelty.                                           |
| **Temporal Contrastive Model**          | Learns sequence order via contrastive learning.    | Tests if order modeling alone suffices.                 |
| **Vision-Only / Speech-Only Ablations** | Remove modalities.                                 | Shows contribution of multimodal fusion.                |
