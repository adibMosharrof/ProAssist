"""
LoRA utilities for DST ProAssist training.

Provides LoRA configuration and model wrapping following ProAssist's approach.
"""

import logging
from typing import Optional
from peft import LoraConfig, get_peft_model
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


def get_lora_config(
    lora_r: int = 128,
    lora_alpha: int = 256,
    lora_dropout: float = 0.05,
    target_modules: Optional[list] = None,
    modules_to_save: Optional[list] = None,
) -> LoraConfig:
    """Create LoRA configuration.
    
    Args:
        lora_r: LoRA rank (default: 128, ProAssist uses 128)
        lora_alpha: LoRA alpha (default: 256, ProAssist uses 256)
        lora_dropout: Dropout probability (default: 0.05)
        target_modules: Modules to apply LoRA to (default: all linear layers in LLM)
        modules_to_save: Additional modules to save (e.g., mm_projector, binary heads)
    
    Returns:
        LoraConfig object
    """
    if target_modules is None:
        # Default: target all linear layers in the LLM
        # This matches ProAssist's default behavior
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ]
    
    return LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        task_type="CAUSAL_LM",
        modules_to_save=modules_to_save,
    )


def apply_lora_to_model(
    model: PreTrainedModel,
    lora_config: LoraConfig,
) -> PreTrainedModel:
    """Apply LoRA to model and return PEFT model.
    
    Args:
        model: Base model to apply LoRA to
        lora_config: LoRA configuration
    
    Returns:
        PEFT model with LoRA adapters
    """
    logger.info(f"Applying LoRA with config: {lora_config}")
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()
    return peft_model


def print_trainable_parameters(model: PreTrainedModel) -> None:
    """Print the number of trainable parameters in the model.
    
    Args:
        model: Model to analyze
    """
    all_params = model.num_parameters()
    trainable_params = model.num_parameters(only_trainable=True)
    logger.info(
        f"trainable params: {trainable_params:,d} || "
        f"all params: {all_params:,d} || "
        f"trainable%: {100 * trainable_params / all_params:.4f}"
    )
