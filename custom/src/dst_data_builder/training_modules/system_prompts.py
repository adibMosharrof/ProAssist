"""
System Prompt Variations for DST Training Data Generation

This module contains predefined system prompt variations used for diversity
in DST training data generation.
"""

SYSTEM_PROMPT_VARIATIONS = [
    "You are a helpful assistant.",
    "You are a proactive assistant. Pay close attention to the user's actions and provide relevant information proactively.",
    "You are a helpful and proactive assistant. Always be ready to assist and provide useful information ahead of time.",
    "You are an assistant that anticipates user needs. Provide assistance before being asked when appropriate.",
]


def get_system_prompt_variations():
    """
    Get the list of system prompt variations.

    Returns:
        List of system prompt strings
    """
    return SYSTEM_PROMPT_VARIATIONS.copy()


def get_random_system_prompt():
    """
    Get a random system prompt variation.

    Returns:
        Random system prompt string
    """
    import random
    return random.choice(SYSTEM_PROMPT_VARIATIONS)