"""
DST Label Mappings for Integer Encoding

This module provides mappings from text labels to integer IDs for DST training.
Converting text labels to integers makes model training more efficient.
"""

# DST State Mappings (from dst_state_snapshot)
DST_STATE_TO_ID = {
    "not_started": 0,
    "in_progress": 1, 
    "completed": 2,
}

ID_TO_DST_STATE = {
    0: "not_started",
    1: "in_progress", 
    2: "completed",
}

# DST Transition Mappings (from DST_UPDATE content)
DST_TRANSITION_TO_ID = {
    "no_change": 0,
    "start": 1,
    "complete": 2,
    "error": 3,
}

ID_TO_DST_TRANSITION = {
    0: "no_change",
    1: "start",
    2: "complete", 
    3: "error",
}

# Event Type Mappings
EVENT_TYPE_TO_ID = {
    "SPEAK": 0,
    "DST_UPDATE": 1,
}

ID_TO_EVENT_TYPE = {
    0: "SPEAK",
    1: "DST_UPDATE",
}


def encode_dst_state(state: str) -> int:
    """Convert DST state text to integer ID."""
    return DST_STATE_TO_ID.get(state, 0)  # Default to 0 (not_started) if unknown


def decode_dst_state(state_id: int) -> str:
    """Convert DST state integer ID to text."""
    return ID_TO_DST_STATE.get(state_id, "not_started")


def encode_dst_transition(transition: str) -> int:
    """Convert DST transition text to integer ID."""
    return DST_TRANSITION_TO_ID.get(transition, 0)  # Default to 0 (no_change) if unknown


def decode_dst_transition(transition_id: int) -> str:
    """Convert DST transition integer ID to text."""
    return ID_TO_DST_TRANSITION.get(transition_id, "no_change")


def encode_event_type(event_type: str) -> int:
    """Convert event type text to integer ID."""
    return EVENT_TYPE_TO_ID.get(event_type, 0)  # Default to 0 (SPEAK) if unknown


def decode_event_type(event_type_id: int) -> str:
    """Convert event type integer ID to text."""
    return ID_TO_EVENT_TYPE.get(event_type_id, "SPEAK")


def encode_dst_state_snapshot(snapshot: list) -> list:
    """Encode a DST state snapshot with integer IDs instead of text."""
    if not snapshot:
        return snapshot
    
    encoded_snapshot = []
    for state_item in snapshot:
        encoded_item = {
            "id": state_item.get("id", ""),
            "state": encode_dst_state(state_item.get("state", "not_started"))
        }
        encoded_snapshot.append(encoded_item)
    
    return encoded_snapshot


def encode_dst_update_content(content: list) -> list:
    """Encode DST update content with integer IDs instead of text."""
    if not content:
        return content
    
    encoded_content = []
    for transition_item in content:
        encoded_item = {
            "id": transition_item.get("id", ""),
            "transition": encode_dst_transition(transition_item.get("transition", "no_change"))
        }
        encoded_content.append(encoded_item)
    
    return encoded_content


def encode_enhanced_conversation(conversation: list) -> list:
    """Encode an entire enhanced conversation with integer mappings."""
    if not conversation:
        return conversation
    
    encoded_conversation = []
    for event in conversation:
        encoded_event = event.copy()
        
        # Encode event type
        encoded_event["type"] = encode_event_type(event.get("type", "SPEAK"))
        
        # Encode DST state snapshot for SPEAK events
        if event.get("type") == "SPEAK" and "dst_state_snapshot" in event:
            encoded_event["dst_state_snapshot"] = encode_dst_state_snapshot(
                event["dst_state_snapshot"]
            )
        
        # Encode DST update content for DST_UPDATE events
        elif event.get("type") == "DST_UPDATE" and "content" in event:
            encoded_event["content"] = encode_dst_update_content(event["content"])
        
        encoded_conversation.append(encoded_event)
    
    return encoded_conversation


def get_label_counts() -> dict:
    """Get the count of each label type for analysis."""
    return {
        "dst_states": len(DST_STATE_TO_ID),
        "dst_transitions": len(DST_TRANSITION_TO_ID), 
        "event_types": len(EVENT_TYPE_TO_ID),
    }