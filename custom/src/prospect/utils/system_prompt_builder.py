"""Shared system prompt building logic for training and inference.

Ensures that training and inference use identical system prompt formatting.
"""

from typing import Dict, List


def build_system_prompt(dst_schema: List[Dict], current_state: Dict = None) -> str:
    """Build system prompt matching training collator format.
    
    Args:
        dst_schema: List of step dicts with 'id' and 'name' keys
        current_state: Dict mapping step_id to state (e.g., {'S1': 'completed'})
        
    Returns:
        Formatted system prompt string matching training format
    """
    if current_state is None:
        current_state = {}
    
    lines = []
    
    # 1. DST Task Overview (matches collator line 271-278)
    if dst_schema:
        lines.append("DST Task Overview:")
        for step in dst_schema:
            step_id = step.get("id", "")
            step_name = step.get("name", "")
            if step_id and step_name:
                lines.append(f"{step_id}: {step_name}")
        lines.append("")  # Blank line after overview
    
    # 2. Proactive assistant message (from training data)
    lines.append("You are a proactive assistant. Pay close attention to the user's actions and provide relevant information proactively.")
    lines.append("")  # Blank line
    
    # 3. Dialogue State with all steps
    lines.append("Dialogue State:")
    
    # Initialize all steps as "not_started" if not in current_state
    full_state = {}
    for step in dst_schema:
        step_id = step["id"]
        full_state[step_id] = current_state.get(step_id, "not_started")
    
    # Sort steps alphanumerically for consistent state string (matches collator logic)
    sorted_state = sorted(
        full_state.items(),
        key=lambda x: (int(x[0][1:]) if x[0][1:].isdigit() else x[0]),
    )
    state_parts = [f"Step {k}: {v}" for k, v in sorted_state]
    state_str = ", ".join(state_parts)
    
    lines.append(f"Current step states - {state_str}")
    
    return "\n".join(lines)
