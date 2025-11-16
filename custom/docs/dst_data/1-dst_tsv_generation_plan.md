# DST Generation Plan: Deterministic TSV + LLM Correction Pipeline

## Overview
This plan describes a **hybrid deterministic + LLM-assisted approach** for generating clean DST TSV files from raw step descriptions (`all_step_descriptions`) and task summaries (`inferred_knowledge`).

Instead of letting the LLM generate the full structure (which leads to repetition and inconsistencies), we:
1. **Deterministically parse** timestamps and actions from text.
2. **Construct a hierarchical TSV scaffold** using rule-based grouping.
3. **Use an LLM only to clean and normalize names** while keeping IDs and timestamps fixed.

This ensures stable, interpretable, and reproducible DST structures that can still benefit from LLM language cleanup.

---

## Pipeline Overview

| Stage | Description | Method |
|--------|--------------|--------|
| 1 | Parse timestamps and actions from text | Regex-based extraction |
| 2 | Group into hierarchy (steps, substeps, actions) | Temporal containment + heuristics |
| 3 | Deduplicate and normalize | Deterministic rules |
| 4 | Generate TSV scaffold | Deterministic writer |
| 5 | LLM correction | Controlled prompt for name cleanup |
| 6 | Post-validation | Re-run consistency checks |

---

## Stage 1: Deterministic Scaffold Generation

### Parsing
Use regex patterns to extract timestamp ranges and instant events.

```
\[([0-9.]+)s-([0-9.]+)s\]\s*(.+)   # range (step/substep)
-\s*\[([0-9.]+)s\]\s*(.+)          # instant action
```

### Create flat records
Each parsed line becomes a record:
- **type**: inferred as `substep` (for ranges) or `action` (for instants)
- **start_ts**, **end_ts**
- **name**: raw description

Example (raw extraction):

| type | start_ts | end_ts | name |
|------|-----------|--------|------|
| substep | 97.2 | 106.8 | attach chassis to chassis |
| action | 110.7 | 110.7 | screw chassis with screwdriver |

---

## Stage 2: Hierarchical Grouping Heuristics

### Rules
1. **Containment:**  
   - Action belongs to the tightest enclosing range.
   - Substeps belong to the nearest enclosing longer range.

2. **Promotion:**  
   Promote a `substep` to `step` if:
   - Duration ≥ 30–45 seconds  
   - Name contains keywords (e.g., *chassis, wheels, arm, body, cabin, demo*)  
   - It has ≥2 substeps/actions.

3. **ID assignment:**  
   - Steps: `S1`, `S2`, … (sorted by start time)  
   - Substeps: `S1.1`, `S1.2`, …  
   - Actions: `S1.1.a`, `S1.1.b`, …  

4. **Action attachment:**  
   Each action must be fully contained in a parent substep or step.

---

## Stage 3: Deduplication & Normalization Rules

1. **Merge near-duplicates:**  
   - If two actions have the same normalized verb/object and occur within `0.6s`, merge them into one with `(x2)` notation.

2. **Verb canonicalization:**  
   | Original | Normalized |
   |-----------|------------|
   | attach, mount | attach |
   | screw, tighten | screw |
   | hand screw | screw (hand) |

3. **Snap jitter:**  
   If an action time is within `0.3s` of its parent’s boundary, snap to the boundary.

4. **Deduplicate identical timestamps:**  
   Keep one row, append `(xN)` to the name if needed.

---

## Stage 4: Generate the TSV Scaffold

Write the flat hierarchical structure using Python’s `csv` module.

**Columns:**  
`type`, `id`, `start_ts`, `end_ts`, `name`

Example output:

```tsv
type	id	start_ts	end_ts	name
step	S1	94.4	153.6	Attach interior and wheels
substep	S1.1	94.4	105.2	Attaching interior
action	S1.1.a	94.4	105.2	attach interior to chassis
substep	S1.2	105.2	153.6	Attaching wheels
action	S1.2.a	105.2	153.6	attach wheel to chassis
action	S1.2.b	111.0	111.0	screw first wheel with hand
```

---

## Stage 5: LLM Cleanup (Controlled Correction)

Use the LLM for **language refinement only**.  
Do **not** allow it to change timestamps, IDs, or hierarchy.

### Example prompt

```
You are given a TSV representing a 3-level procedural task tree.
Rules:
- Do NOT modify timestamps or IDs.
- Keep the same columns: type, id, start_ts, end_ts, name.
- Only improve naming: unify style, merge duplicates, and ensure parent names summarize their children.
- For identical actions at same time, merge into one and mark with (xN).
- Never invent or remove rows.
Return a TSV with the same format.
```

---

## Stage 6: Post-LLM Validation

After LLM correction:
- Re-validate containment and ordering.
- Ensure IDs and timestamps are unchanged.
- Ensure hierarchy depth (STEP > SUBSTEP > ACTION) is consistent.
- Skip LLM-corrected TSV if validation fails; fallback to deterministic version.

---

## Benefits

✅ Deterministic parsing ensures reproducibility  
✅ LLM only polishes language, avoiding hallucinated timestamps  
✅ Hierarchical IDs and constraints prevent repetition  
✅ Easy to integrate with downstream DST validator or training data prep

---

## Summary Workflow

```
Raw step descriptions  →  Regex parse  →  Rule-based grouping
                                       ↓
                             Deterministic TSV scaffold
                                       ↓
                           LLM-controlled cleanup (names only)
                                       ↓
                        Validated final DST TSV (ready for model)
```

---
