"""
DST Generation Prompt Template
"""


DST_GENERATION_PROMPT_TEMPLATE = """You construct a 3-level Dialog State Tree (DST) for procedural tasks.

TASK
Produce a DST with exactly 3 levels: Step → Substep → Action (actions are the leaves).

IMPORTANT: Use a TIMESTAMP-FIRST approach:
1. First, analyze all timestamps in [all_step_descriptions] and sort activities chronologically
2. Identify temporal boundaries (time gaps) to determine where steps should start/end
3. Group temporally adjacent activities into steps, ensuring no overlap between steps
4. Then assign meaningful names to steps based on the activities they contain

CRITICAL CONSTRAINT (NO EXTERNAL KNOWLEDGE)
    Use ONLY the information present in the two inputs below:
    - Steps must come from the numbered items in [inferred_knowledge].
    - Substeps and Actions must come directly from [all_step_descriptions].
    Do NOT add actions, objects, tools, quantities, or timestamps that are not explicitly present in the inputs.

    OUTPUT FORMAT (JSON only; no prose)
    {{
    "steps": [
        {{
        "step_id": "S1",
        "name": "<High-level outcome from inferred_knowledge>",
        "timestamps": {{ "start_ts": <float>, "end_ts": <float> }},
        "substeps": [
            {{
            "sub_id": "S1.1",
            "name": "<Phase/segment derived from descriptions>",
            "timestamps": {{ "start_ts": <float>, "end_ts": <float> }},
            "actions": [
                {{
                "act_id": "S1.1.a",
                "name": "<Atomic action text copied/paraphrased from descriptions>",
                "args_schema": {{
                    "object": <str|null>,
                    "tool": <str|null>,
                    "qty": <str|number|null>
                }},
                "timestamps": {{ "start_ts": <float>, "end_ts": <float> }}
                }}
            ]
            }}
        ]
        }}
    ]
    }}

    TEMPORAL & STRUCTURAL RULES
    1) TIMESTAMPS ARE PRIMARY: Analyze timestamps FIRST to determine step boundaries before considering semantic content.
    2) Use the exact timestamps from [all_step_descriptions]; do not invent or modify them.
    3) Steps are sequential and non-overlapping across the whole video (no step can start before the previous step ends).
    4) Containment:
       - substep is fully within its parent step,
       - action is fully within its parent substep,
       - start_ts ≤ end_ts for every node.
    5) Parent timestamps are the envelope of their children:
       step.start = min(child.starts), step.end = max(child.ends); same for substeps.
    6) Order all children by start_ts.
    7) CRITICAL: Before creating steps, sort all activities by timestamp and group them into non-overlapping time windows.    SEMANTIC CLARITY (avoid repetition across levels)
    - Step name = outcome (noun phrase). e.g., "Prepare the rice".
    - Substep name = phase/tactic (gerund). e.g., "Measuring and washing".
    - Action name = atomic operation (imperative). e.g., "Measure rice with cup".
    - Avoid duplicate/near-duplicate names across adjacent levels. Broaden parent labels if needed.
    - Prefer ≥2 actions per substep when descriptions allow; otherwise keep 1 action but ensure the substep label is a broader phase.

    GROUPING GUIDANCE (TIMESTAMP-DRIVEN)
    - PRIMARY RULE: Group activities into steps based on TEMPORAL PROXIMITY, not semantic similarity.
    - Scan [all_step_descriptions] and identify natural temporal boundaries (gaps in time).
    - Each step should contain a consecutive sequence of activities in time.
    - If activities are temporally interleaved (e.g., activity A: 100-200, activity B: 150-250), they MUST be in the same step.
    - Step boundaries occur at temporal gaps, not at semantic transitions.
    - Within each step, group substeps by temporal adjacency and shared context.
    - NEVER create a step that would overlap in time with the previous or next step.

    ARGS SCHEMA
    - For each action, set args_schema fields to explicit values only if they appear in the descriptions; otherwise use null.

    VALIDATION REMINDERS
    - No external knowledge beyond the two inputs.
    - Sequential, non-overlapping steps.
    - Parent timestamps fully contain children.
    - JSON must be valid; no comments or extra text.
    {failure_section}

    Now use these inputs to construct the DST.

    [inferred_knowledge]
    {inferred_knowledge}

    [all_step_descriptions]
    {all_step_descriptions}
    """


def create_dst_prompt(
    inferred_knowledge: str,
    all_step_descriptions: str,
    previous_failure_reason: str = "",
) -> str:
    """
    Create DST generation prompt from template.
    
    Args:
        inferred_knowledge: High-level step descriptions
        all_step_descriptions: Detailed action descriptions with timestamps
        previous_failure_reason: Optional error message from previous attempt
        
    Returns:
        Formatted prompt string
    """
    failure_section = (
        f"\n\nPREVIOUS ATTEMPT FAILED. FIX THESE ISSUES:\n{previous_failure_reason}\n"
        if previous_failure_reason
        else ""
    )
    
    return DST_GENERATION_PROMPT_TEMPLATE.format(
        failure_section=failure_section,
        inferred_knowledge=inferred_knowledge,
        all_step_descriptions=all_step_descriptions,
    )
