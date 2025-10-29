"""
GPT Generator Factory - Creates the appropriate GPT generator based on configuration
"""

import os
import logging
from typing import Dict, Any, Optional

from dst_data_builder.validators.structure_validator import StructureValidator
from dst_data_builder.validators.timestamps_validator import TimestampsValidator
from dst_data_builder.validators.id_validator import IdValidator
from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator
from dst_data_builder.gpt_generators.single_gpt_generator import SingleGPTGenerator
from dst_data_builder.gpt_generators.batch_gpt_generator import BatchGPTGenerator



class GPTGeneratorFactory:
    """Factory for creating GPT generators based on configuration"""

    @staticmethod
    def create_generator(
        generator_type: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: Optional[int] = None,
        generator_cfg: Optional[Dict[str, Any]] = None,
    ) -> BaseGPTGenerator:
        """
        Create a GPT generator based on the specified type

        Args:
            generator_type: Type of generator ("single" or "batch")
            api_key: OpenAI API key
            model_name: GPT model name
            temperature: Temperature for GPT responses
            max_tokens: Maximum tokens for GPT responses

        Returns:
            Appropriate GPT generator instance

        Raises:
            ValueError: If generator_type is not supported or configuration is invalid
        """

        # Validate generator type
        if generator_type.lower() not in ["single", "batch"]:
            raise ValueError(
                f"Unsupported generator type: {generator_type}. Supported types: 'single', 'batch'"
            )
        api_key_map = {
            "single": "OPENROUTER_API_KEY",
            "batch": "OPENAI_API_KEY",
        }
        api_key = os.getenv(api_key_map.get(generator_type.lower()), None)

        # Prepare validators if present inside generator_cfg
        validators = None
        if generator_cfg and "validators" in generator_cfg:
            validators = generator_cfg.get("validators")
        else:
            # Hard-code default validators for now
            validators = [
                StructureValidator(),
                TimestampsValidator(),
                IdValidator(),
            ]

        # Create the appropriate generator
        if generator_type.lower() == "single":
            return SingleGPTGenerator(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                validators=validators,
                max_retries=max_retries,
            )
        elif generator_type.lower() == "batch":
            # Extract optional batch-specific settings from generator_cfg
            batch_size = None
            check_interval = None
            save_intermediate = None
            if generator_cfg:
                batch_size = generator_cfg.get("batch_size", None)
                check_interval = generator_cfg.get("check_interval", None)
                save_intermediate = generator_cfg.get("save_intermediate", None)

            return BatchGPTGenerator(
                api_key=api_key,
                model_name=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                batch_size=batch_size,
                check_interval=check_interval,
                save_intermediate=save_intermediate,
                max_retries=max_retries,
                validators=validators,
            )
