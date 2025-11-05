"""Custom models for PROSPECT with KV cache management"""

from prospect.models.smolvlm_with_strategies import SmolVLMWithStrategies
from prospect.models.processing_custom_smolvlm import CustomSmolVLMProcessor


__all__ = [
    "SmolVLMWithStrategies",  # Approach 2: Strategy injection
    "CustomSmolVLMProcessor",
]
