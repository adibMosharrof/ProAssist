"""
DST Enums Module

This module defines enums for efficient token representation of DST transitions and states.
Instead of using string representations, we use compact integer codes to save tokens.
"""

from enum import IntEnum


class DSTTransition(IntEnum):
    """Enum for DST transition types with compact integer codes"""
    START = 0      # Step starts (not_started -> in_progress)
    COMPLETE = 1   # Step completes (in_progress -> completed)
    PAUSE = 2      # Step paused (in_progress -> paused)
    RESUME = 3     # Step resumed (paused -> in_progress)
    CANCEL = 4     # Step cancelled (any state -> cancelled)
    RESET = 5      # Step reset (any state -> not_started)


class DSTState(IntEnum):
    """Enum for DST state types with compact integer codes"""
    NOT_STARTED = 0    # Step not yet started
    IN_PROGRESS = 1    # Step is currently being worked on
    COMPLETED = 2      # Step has been completed
    PAUSED = 3         # Step is temporarily paused
    CANCELLED = 4      # Step was cancelled
    FAILED = 5         # Step failed during execution


class DSTRole(IntEnum):
    """Enum for conversation roles with compact integer codes"""
    USER = 0           # User turn
    ASSISTANT = 1      # Assistant response (SPEAK)
    SYSTEM = 2         # System prompt
    DST_UPDATE = 3     # DST state update event


class DSTLabel(IntEnum):
    """Enum for DST event labels with compact integer codes"""
    # Base labels
    INITIATIVE = 0     # Proactive behavior
    REACTIVE = 1       # Reactive response
    
    # Intent labels  
    INSTRUCTION = 2    # Providing instructions
    FEEDBACK = 3       # Giving feedback
    INFO_SHARING = 4   # Sharing information
    CORRECTION = 5     # Making corrections
    
    # DST-specific labels
    DST_UPDATE = 6     # DST state update
    DST_START = 7      # Step start transition
    DST_COMPLETE = 8   # Step completion transition
    DST_MULTIPLE = 9   # Multiple transitions
    DST_STATE_CHANGE = 10  # General state change


# Mapping functions for string to enum conversion
def transition_from_string(transition_str: str) -> DSTTransition:
    """Convert transition string to DSTTransition enum"""
    mapping = {
        "start": DSTTransition.START,
        "complete": DSTTransition.COMPLETE,
        "pause": DSTTransition.PAUSE,
        "resume": DSTTransition.RESUME,
        "cancel": DSTTransition.CANCEL,
        "reset": DSTTransition.RESET,
    }
    return mapping.get(transition_str.lower(), DSTTransition.START)


def state_from_string(state_str: str) -> DSTState:
    """Convert state string to DSTState enum"""
    mapping = {
        "not_started": DSTState.NOT_STARTED,
        "in_progress": DSTState.IN_PROGRESS,
        "completed": DSTState.COMPLETED,
        "paused": DSTState.PAUSED,
        "cancelled": DSTState.CANCELLED,
        "failed": DSTState.FAILED,
    }
    return mapping.get(state_str.lower(), DSTState.NOT_STARTED)


def role_from_string(role_str: str) -> DSTRole:
    """Convert role string to DSTRole enum"""
    mapping = {
        "user": DSTRole.USER,
        "assistant": DSTRole.ASSISTANT,
        "system": DSTRole.SYSTEM,
        "speak": DSTRole.ASSISTANT,  # SPEAK maps to ASSISTANT
        "dst_update": DSTRole.DST_UPDATE,
    }
    return mapping.get(role_str.lower(), DSTRole.USER)


# Token counting helpers
def count_transition_tokens(transition: DSTTransition) -> int:
    """Return token count for a transition (single integer = 1 token)"""
    return 1


def count_state_tokens(state: DSTState) -> int:
    """Return token count for a state (single integer = 1 token)"""
    return 1


def count_role_tokens(role: DSTRole) -> int:
    """Return token count for a role (single integer = 1 token)"""
    return 1


def count_dst_content_tokens(content) -> int:
    """
    Count tokens for DST content using efficient integer encoding
    
    Args:
        content: DST content (list of dicts or single dict)
        
    Returns:
        Token count using compact integer encoding
    """
    if not content:
        return 0
    
    # Handle single transition
    if isinstance(content, dict):
        return count_single_transition_tokens(content)
    
    # Handle list of transitions
    if isinstance(content, list):
        total_tokens = 0
        for item in content:
            if isinstance(item, dict):
                total_tokens += count_single_transition_tokens(item)
        return total_tokens
    
    # Fallback for other types
    return 1


def count_single_transition_tokens(transition_dict: dict) -> int:
    """
    Count tokens for a single transition using compact encoding
    
    Args:
        transition_dict: Dict with 'id', 'transition' keys
        
    Returns:
        Token count (should be 2-3 tokens total: 1 for id, 1 for transition type)
    """
    if not isinstance(transition_dict, dict):
        return 1
    
    tokens = 0
    
    # Step ID (assume 1 token for integer ID)
    step_id = transition_dict.get("id", "")
    if step_id:
        tokens += 1  # Step ID as compact identifier
    
    # Transition type (1 token for enum)
    transition_str = transition_dict.get("transition", "start")
    transition_enum = transition_from_string(transition_str)
    tokens += count_transition_tokens(transition_enum)
    
    return tokens