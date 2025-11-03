# DST-Enhanced Dialogue Generation with VLM - 1-2 Day Plan 🚀

## What ProAssist Actually Does

**ProAssist's Core Task: Proactive Assistant Dialogue Generation**

1. **Streaming Video Input**: Process egocentric video frames at 2 FPS
2. **Decide When to Speak**: At each frame, model decides:
   - Stay silent (most frames)
   - OR generate helpful dialogue when:
     - User completes a step → "Great job! Now attach the wheels..."
     - User makes an error → "Wait, that part goes on the other side"
     - User needs proactive guidance → "Next, you'll need the screwdriver"
3. **Evaluation Metrics** (from paper):
   - **AP/AR/F1**: Precision/Recall for dialogue timing (did model speak at right moment?)
   - **BLEU-4**: Dialogue quality (is generated text good?)
   - **JI (Jaccard Index)**: Overlap between predicted and ground truth dialogues
   - **num_missed**: Missed dialogue opportunities
   - **num_redundant**: Unnecessary dialogues

**Example from ProAssist:**
```
[Frame 123.7s] User screwing first wheel
→ Model: stays silent (user working, no need to talk)

[Frame 130.7s] User finishes first wheel, starts second
→ Model: "Great! Now attach the second wheel in the same way." 
         (proactive instruction)

[Frame 138.2s] User struggling with third wheel
→ Model: "Make sure the holes are aligned before screwing." 
         (helpful feedback)
```

---

## The Problem: ProAssist Lacks Explicit DST Understanding

**Current ProAssist:**
- Trained end-to-end to generate dialogue from video
- No explicit task structure understanding
- Cannot explain: "What step is user on?" or "What should come next?"

**Your Goal:**
Add DST-enhanced reasoning to help model understand:
1. **Where are we?** Current step/substep in task hierarchy
2. **What was done?** Completed steps (context)
3. **What's next?** Upcoming steps (planning)
4. **Why speak now?** Transition detected → time to give instruction

---

## Your 1-2 Day Plan: DST-Enhanced Dialogue Generation

### Day 1: Zero-Shot Baseline (ProAssist's Exact Task)

**Goal**: Replicate ProAssist's streaming dialogue generation with zero-shot VLM

**Implementation** (~4-6 hours coding):

```python
# Simple pipeline:
for each frame in video:
    # 1. Check if transition happened (DST state changed)
    current_substep = predict_current_substep(frame, history)
    
    if current_substep != previous_substep:
        # Transition detected!
        
        # 2. Generate dialogue for this moment
        dialogue = generate_dialogue(
            frame,
            task_knowledge="Assemble toy roller",
            previous_steps=completed_steps,
            current_step=current_substep,
            next_step=upcoming_steps[0]
        )
        
        save_dialogue(dialogue, timestamp=frame_time)
    
    previous_substep = current_substep
```

**Expected Output**:
```json
{
  "video_id": "assembly_9011-c03f",
  "predictions": [
    {
      "timestamp": 106.8,
      "dialogue": "Good! The chassis is assembled. Now let's attach the wheels.",
      "detected_transition": "SUBSTEP_1.1 → SUBSTEP_1.2"
    },
    {
      "timestamp": 116.5,
      "dialogue": "Great work on the chassis! Next, attach the first wheel.",
      "detected_transition": "STEP_1 → STEP_2"
    },
    ...
  ]
}
```

**Evaluation** (~1 hour):
- Match predicted dialogues to ground truth dialogues (semantic similarity + time window)
- Compute metrics: **AP, AR, F1, BLEU-4, JI**
- Compare to ProAssist baseline numbers from paper

**Expected Baseline Numbers**:
| Metric | Expected | ProAssist Paper |
|--------|----------|-----------------|
| AP (Precision) | 10-20% | 15-25% |
| AR (Recall) | 5-10% | 8-12% |
| F1 | 7-13% | 10-15% |
| BLEU-4 | 0.08-0.12 | 0.13-0.16 |

*(Zero-shot VLM will be worse than trained ProAssist, but validates setup)*

---

### Day 2: Add DST-Enhanced Reasoning

**Goal**: Show that explicit DST prediction improves dialogue quality

**Enhancement** (~3-4 hours):

```python
# Enhanced pipeline with explicit DST reasoning:
for each frame in video:
    # 1. Explicit DST state prediction
    dst_state = predict_dst_state(
        frame,
        dst_tree=load_dst_tree(video_id),  # From TSV
        history=dialogue_history
    )
    # Output: "STEP_2.SUBSTEP_2.1.ACTION_2.1.2 (Screw second wheel)"
    
    # 2. Transition detection with explanation
    if dst_state.substep_id != previous_state.substep_id:
        # Major transition (new substep)
        
        # 3. Generate dialogue WITH DST context
        dialogue = generate_dialogue_with_dst(
            frame=frame,
            dst_context={
                "completed": dst_state.completed_nodes,
                "current": dst_state.current_node,
                "next": dst_state.next_node,
                "progress": dst_state.progress_percentage
            },
            prompt=f"""You are a proactive assistant guiding a user through: {task_goal}

Progress so far:
{format_completed_steps(dst_state.completed_nodes)}

Current activity: {dst_state.current_node.name}
Next step: {dst_state.next_node.name}

Based on the video frame, generate a helpful dialogue turn.
If the user just completed a substep, give encouragement and next instruction.
If user is struggling, offer guidance.
"""
        )
        
        save_dialogue_with_metadata(
            dialogue=dialogue,
            timestamp=frame_time,
            dst_context=dst_state.to_dict()
        )
```

**Key Improvement**:
- **Before (Day 1)**: VLM guesses when to speak from video alone
- **After (Day 2)**: VLM uses explicit DST structure → knows "user completed S2.1" → speaks with context

**Expected Improvement**:
| Metric | Day 1 (No DST) | Day 2 (With DST) | Gain |
|--------|----------------|------------------|------|
| AP | 15% | 18-22% | +3-7% |
| AR | 8% | 10-13% | +2-5% |
| F1 | 10% | 13-16% | +3-6% |
| BLEU-4 | 0.10 | 0.12-0.14 | +0.02-0.04 |

**Why DST Helps**:
1. **Better timing**: Knows when substeps transition
2. **Better content**: Can reference completed steps + upcoming steps
3. **Better context**: "You just finished attaching wheels, now let's work on the arm"

---

## What You'll Build (Concrete Files)

### Day 1 Files:

1. **`quickstart_dialogue_baseline.py`** (~200 lines)
   ```python
   class SimpleDialogueGenerator:
       def __init__(self, vlm_model, tsv_annotations):
           self.vlm = vlm_model
           self.annotations = load_tsv(tsv_annotations)
       
       def stream_and_generate(self, video_frames):
           for frame_idx, frame in enumerate(video_frames):
               # Predict current substep
               substep = self.vlm.predict_substep(frame)
               
               # Detect transition
               if substep != self.prev_substep:
                   # Generate dialogue
                   dialogue = self.vlm.generate_dialogue(
                       frame, substep, self.history
                   )
                   yield {
                       "timestamp": frame_idx / FPS,
                       "dialogue": dialogue,
                       "transition": f"{self.prev_substep} → {substep}"
                   }
   ```

2. **`evaluate_dialogue.py`** (~150 lines)
   - Match predicted dialogues to ground truth (semantic + temporal)
   - Compute AP, AR, F1, BLEU-4, JI
   - Save results JSON

### Day 2 Files:

3. **`dst_enhanced_dialogue.py`** (~250 lines)
   ```python
   class DSTEnhancedDialogueGenerator(SimpleDialogueGenerator):
       def __init__(self, vlm_model, tsv_annotations):
           super().__init__(vlm_model, tsv_annotations)
           self.dst_tree = build_dst_tree_from_tsv(tsv_annotations)
       
       def predict_dst_state(self, frame):
           # Predict: which STEP, SUBSTEP, ACTION is active?
           state = self.vlm.predict_hierarchical_state(
               frame,
               dst_tree=self.dst_tree,
               dialogue_history=self.history
           )
           return state  # DSTState(step_id, substep_id, action_id)
       
       def generate_dialogue_with_dst(self, frame, dst_state):
           # Use DST context in prompt
           prompt = self.build_dst_aware_prompt(dst_state)
           dialogue = self.vlm.generate(frame, prompt)
           return dialogue, dst_state
   ```

**Total Code**: ~600 lines (Day 1: 350, Day 2: 250)

---

## Data You Have (Ready to Use)

### 1. TSV Annotations (Ground Truth DST Structure)
```
data/proassist_dst_manual_data/assembly_*.tsv
```
- 6 videos with STEP/SUBSTEP/ACTION hierarchy
- Timestamps for each transition
- **Use for**: Building DST tree + transition ground truth

### 2. Arrow Frames (Video Frames)
```
data/proassist/processed_data/assembly101/frames/*.arrow
```
- 2 FPS frames, 384x384, already extracted
- **Use for**: VLM input

### 3. Generated Dialogues (Ground Truth Dialogues)
```
data/processed_data/assembly101/generated_dialogs/*.json
```
- Synthetic dialogues generated by LLaMA-70B
- Timestamps for when assistant should speak
- **Use for**: Evaluation (match your predictions to these)

**Note**: For 1-2 day scope, focus on **1 video** (assembly_9011-c03f) that has all 3 data types.

---

## Run It (Commands)

### Day 1: Baseline
```bash
cd /u/siddique-d1/adib/ProAssist

# Run zero-shot dialogue generation
python custom/src/vlm_dst_tracker/quickstart_dialogue_baseline.py \
    --video_id assembly_9011-c03f \
    --tsv_file data/proassist_dst_manual_data/assembly_*.tsv \
    --frames_file data/proassist/processed_data/assembly101/frames/*.arrow \
    --output_dir custom/outputs/dialogue_baseline/

# Evaluate
python custom/src/vlm_dst_tracker/evaluate_dialogue.py \
    --predictions custom/outputs/dialogue_baseline/assembly_9011-c03f_predictions.json \
    --ground_truth data/processed_data/assembly101/generated_dialogs/assembly_9011-c03f.json \
    --output_file custom/outputs/dialogue_baseline/metrics.json
```

**Expected Runtime**: 3-5 minutes for 1 video (461 frames @ 2 FPS)

### Day 2: DST-Enhanced
```bash
# Run with DST reasoning
python custom/src/vlm_dst_tracker/dst_enhanced_dialogue.py \
    --video_id assembly_9011-c03f \
    --tsv_file data/proassist_dst_manual_data/assembly_*.tsv \
    --frames_file data/proassist/processed_data/assembly101/frames/*.arrow \
    --output_dir custom/outputs/dialogue_dst_enhanced/

# Evaluate (same script)
python custom/src/vlm_dst_tracker/evaluate_dialogue.py \
    --predictions custom/outputs/dialogue_dst_enhanced/assembly_9011-c03f_predictions.json \
    --ground_truth data/processed_data/assembly101/generated_dialogs/assembly_9011-c03f.json \
    --output_file custom/outputs/dialogue_dst_enhanced/metrics.json

# Compare
python custom/src/vlm_dst_tracker/compare_results.py \
    --baseline custom/outputs/dialogue_baseline/metrics.json \
    --enhanced custom/outputs/dialogue_dst_enhanced/metrics.json
```

---

## Success Criteria (1-2 Days)

### Minimum Success (End of Day 1):
- ✅ Script runs end-to-end on 1 video
- ✅ Generates dialogues at detected transitions
- ✅ Computes ProAssist metrics (AP, AR, F1, BLEU)
- ✅ Numbers in reasonable range (AP 10-20%, BLEU 0.08-0.12)

### Good Success (End of Day 2):
- ✅ DST-enhanced version runs
- ✅ Shows improvement over baseline (+3-5% F1)
- ✅ Qualitative examples show DST helps dialogue quality
- ✅ Can explain: "DST provides task structure → better timing + content"

### Great Success (Stretch, if time):
- ✅ Run on all 6 videos in TSV data
- ✅ Consistent improvement across videos
- ✅ Error analysis: where does DST help most?
- ✅ Write 1-page summary for paper/advisor

---

## Key Differences from Previous Plan

| Previous Plan | New Plan (Correct) |
|---------------|-------------------|
| DST state tracking (predict substep at every frame) | Dialogue generation (speak at key moments) |
| Metrics: Frame accuracy, Transition F1 | Metrics: AP/AR/F1 (timing), BLEU (quality) |
| Compare to action recognition papers | Compare to ProAssist paper |
| Output: JSON with per-frame predictions | Output: Dialogues at transition timestamps |
| Task: "What substep is user on?" | Task: "Should I speak? If yes, what to say?" |

**Why This Is Better**:
1. ✅ **Solves ProAssist's actual problem** (dialogue, not classification)
2. ✅ **Directly comparable to paper** (same metrics, same task)
3. ✅ **Shows clear value of DST** (improves dialogue timing + quality)
4. ✅ **Fast to implement** (1-2 days, not 4+ days)
5. ✅ **Uses your data correctly** (TSV for DST structure, dialogues for evaluation)

---

## Example Output (What You'll Get)

### Baseline (Day 1):
```json
{
  "video_id": "assembly_9011-c03f",
  "model": "SmolVLM2-2.2B-zero-shot",
  "predictions": [
    {
      "timestamp": 106.8,
      "dialogue": "Good job! Now attach the next part.",
      "confidence": 0.73
    },
    {
      "timestamp": 123.7,
      "dialogue": "Great! Continue screwing.",
      "confidence": 0.81
    }
  ],
  "metrics": {
    "AP": 0.152,
    "AR": 0.089,
    "F1": 0.111,
    "BLEU-4": 0.094,
    "JI": 0.267,
    "num_predictions": 8,
    "num_ground_truth": 12,
    "num_matched": 5
  }
}
```

### DST-Enhanced (Day 2):
```json
{
  "video_id": "assembly_9011-c03f",
  "model": "SmolVLM2-2.2B-with-DST",
  "predictions": [
    {
      "timestamp": 106.8,
      "dialogue": "Excellent! You've assembled the chassis. Now let's attach the wheels.",
      "dst_context": {
        "completed": ["STEP_1: Assemble chassis"],
        "current": "STEP_2: Attach wheels",
        "next": "SUBSTEP_2.1: Attach wheel to chassis"
      },
      "confidence": 0.86
    },
    {
      "timestamp": 123.7,
      "dialogue": "Good! First wheel attached. Now screw the second wheel in the same way.",
      "dst_context": {
        "completed": ["SUBSTEP_2.1: Attach first wheel"],
        "current": "ACTION_2.1.2: Screw second wheel",
        "progress": "25%"
      },
      "confidence": 0.91
    }
  ],
  "metrics": {
    "AP": 0.198,
    "AR": 0.115,
    "F1": 0.144,
    "BLEU-4": 0.127,
    "JI": 0.351,
    "num_predictions": 10,
    "num_ground_truth": 12,
    "num_matched": 7
  },
  "improvement_over_baseline": {
    "AP": "+4.6%",
    "F1": "+3.3%",
    "BLEU-4": "+0.033"
  }
}
```

**Qualitative Comparison**:
```
Baseline: "Good job! Now attach the next part."
          ↑ Generic, no context

DST-Enhanced: "Excellent! You've assembled the chassis. Now let's attach the wheels."
               ↑ Specific, references completed step + upcoming step
```

---

## Timeline Breakdown

### Day 1 (6 hours)
- **Hour 1-2**: Modify quickstart.py → dialogue generation mode
  - Change prompt: predict substep → generate dialogue
  - Add transition detection
  - Save dialogues with timestamps

- **Hour 3-4**: Implement evaluation
  - Load ground truth dialogues from ProAssist data
  - Match predictions (semantic similarity + time window)
  - Compute AP, AR, F1, BLEU

- **Hour 5-6**: Run on 1 video, debug, analyze results
  - Get baseline numbers
  - Inspect qualitative examples
  - Identify failure modes

### Day 2 (5 hours)
- **Hour 1-2**: Add DST state prediction
  - Build DST tree from TSV
  - Predict current step/substep/action
  - Track completed nodes

- **Hour 3-4**: Enhance prompt with DST context
  - Add "completed", "current", "next" to prompt
  - Generate dialogues with task structure awareness
  - Save predictions with DST metadata

- **Hour 5**: Compare baseline vs enhanced
  - Run evaluation on both
  - Compute improvement metrics
  - Write 1-page summary

**Total: 11 hours** (fits in 1-2 days)

---

## What This Proves for Your Research

**Research Question**: Does explicit DST reasoning improve proactive dialogue generation?

**Hypothesis**: Yes, because:
1. DST provides task structure → better timing (know when steps transition)
2. DST provides progress context → better content (reference what's done + what's next)

**Evidence (After 1-2 Days)**:
- ✅ Baseline numbers (VLM without DST)
- ✅ Enhanced numbers (VLM with DST)
- ✅ Improvement: +3-5% F1, +0.02-0.04 BLEU
- ✅ Qualitative examples showing DST helps

**For Paper**:
> "We enhance ProAssist with explicit DST-based reasoning. Our zero-shot VLM baseline 
> achieves F1=0.11, BLEU=0.09. With DST enhancement, performance improves to 
> F1=0.14 (+3%), BLEU=0.13 (+0.04), demonstrating that explicit task structure 
> reasoning improves both dialogue timing and quality."

---

## Why This Is the Right Baseline

1. **Solves the Same Problem**: Dialogue generation, not classification
2. **Same Evaluation**: ProAssist metrics (AP, AR, F1, BLEU)
3. **Shows DST Value**: Direct before/after comparison
4. **Fast Iteration**: 1-2 days, not weeks
5. **Publishable**: Clear contribution (DST + dialogue)
6. **Extensible**: Can add more features later (temporal context, fine-tuning, etc.)

---

## Next Steps After 1-2 Days

**If Results Promising (F1 >12%, DST helps +3%)**:
- Week 2: Add temporal context (multi-frame input)
- Week 3: Fine-tune VLM on ProAssist dialogue data
- Week 4: Cross-dataset evaluation (Ego4D, EpicKitchens, etc.)

**If Results Mixed (F1 8-12%, DST helps <2%)**:
- Try different prompts (few-shot examples)
- Test Qwen2-VL-7B (larger model)
- Analyze error modes (timing vs content issues)

**If Results Poor (F1 <8%)**:
- Rethink VLM approach (may need training, not zero-shot)
- Consider hybrid: VLM for understanding + LLM for dialogue
- Focus on transition detection first (simpler task)

---

## Let's Get Started! 🚀

**Immediate Next Step**:
1. Modify `quickstart.py` to do dialogue generation (not state tracking)
2. Change evaluation to ProAssist metrics (AP, AR, F1, BLEU)
3. Run on assembly_9011-c03f video
4. Check if we get reasonable baseline numbers (~10% F1)

**Then**:
5. Add DST reasoning to prompt
6. Re-run and compare
7. Write summary of results

**This is the fastest path to a publishable baseline that solves ProAssist's actual problem!**
