"""
Base GPT Generator - Abstract base class for GPT-based DST generation
"""

import json
import openai
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from dst_data_builder.datatypes.dst_output import DSTOutput
import re
import time
import logging


class BaseGPTGenerator(ABC):
    """Abstract base class for GPT-based DST generation"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        validators: list = None,
    ):
        self.client = openai.OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Initialize validators: expect list of BaseValidator instances
        self.validators = validators or []

    def create_dst_prompt(
        self, inferred_knowledge: str, all_step_descriptions: str
    ) -> str:
        """Create a prompt for the LLM to generate DST structure"""
        return f"""
You are an expert at creating Dialog State Tracking (DST) structures for procedural tasks.

Given the following information, create a hierarchical DST structure:

INFERRED KNOWLEDGE (high-level steps):
{inferred_knowledge}

DETAILED STEP DESCRIPTIONS (with timestamps):
{all_step_descriptions}

Please create a DST structure in the following JSON format:

{{
    "steps": [
        {{
            "step_id": "S1",
            "name": "High-level step name from inferred knowledge",
            "substeps": [
                {{
                    "sub_id": "S1.1",
                    "name": "Substep name derived from detailed descriptions",
                    "timestamps": {{
                        "start_ts": 10.5,
                        "end_ts": 25.3
                    }},
                    "actions": [
                        {{
                            "act_id": "S1.1.a",
                            "name": "Specific action description",
                            "args_schema": {{
                                "object": "object_being_manipulated",
                                "tool": "tool_used_if_any",
                                "qty": quantity_or_null
                            }},
                            "timestamps": {{
                                "start_ts": 10.5,
                                "end_ts": 15.2
                            }}
                        }}
                    ]
                }}
            ]
        }}
    ]
}}

Rules:
1. Use the inferred_knowledge to determine the main steps (S1, S2, S3, etc.)
2. Use the detailed step descriptions to create substeps and actions with proper timestamps
3. Extract objects, tools, and quantities from the descriptions
4. Ensure timestamps are properly aligned between steps, substeps, and actions
5. Make action names descriptive and specific
6. Use null for args_schema fields when information is not available

Return only the JSON structure, no additional text.
"""

    def _clean_json_response(self, content: str) -> str:
        """Clean common JSON formatting issues in GPT responses"""
        # Remove markdown code blocks
        content = re.sub(r"```json\s*", "", content)
        content = re.sub(r"```\s*$", "", content)

        # Fix trailing commas before closing braces/brackets
        content = re.sub(r",(\s*[}\]])", r"\1", content)

        return content.strip()

    # `_validate_dst_structure` removed: validation is handled externally by the
    # project's validators (StructureValidator/TimestampsValidator). The
    # generator no longer performs structural validation here.

    def _attempt_dst_generation(
        self, inferred_knowledge: str, all_step_descriptions: str
    ) -> Dict[str, Any]:
        """
        Single attempt at DST generation (used by both single and batch generators)

        Args:
            inferred_knowledge: High-level step descriptions
            all_step_descriptions: Detailed timestamped descriptions

        Returns:
            DST structure dictionary or None if generation fails
        """
        prompt = self.create_dst_prompt(inferred_knowledge, all_step_descriptions)

        logger = logging.getLogger(__name__)

        try:
            logger.info(f"Making GPT API call with {len(prompt)} character prompt...")

            # Call OpenAI API
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at creating hierarchical task structures.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            # Parse response
            dst_content = response.choices[0].message.content.strip()
            logger.info(f"Received response length: {len(dst_content)} characters")

            # Try to extract JSON if there's extra text
            if not dst_content.startswith("{"):
                # Find JSON in the response
                json_match = re.search(r"\{.*\}", dst_content, re.DOTALL)
                if json_match:
                    dst_content = json_match.group()
                    logger.info(
                        f"Extracted JSON portion: {len(dst_content)} characters"
                    )

            # Clean up common JSON issues
            dst_content = self._clean_json_response(dst_content)

            # Try to parse JSON
            try:
                dst_structure = json.loads(dst_content)
            except json.JSONDecodeError as json_error:
                logger.exception(f"JSON parsing failed: {json_error}")
                return None

            # Structural validation is handled externally by the validators.

            logger.info("✅ Successfully generated DST structure")
            return dst_structure

        except Exception as e:
            logger.exception(f"Error in DST generation: {e}")
            return None

    def _run_validators(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Run configured validators against dst_structure.

        Returns (True, "") if all validators pass, otherwise (False, message).
        """
        for v in self.validators:
            try:
                ok, msg = v.validate(dst_structure)
            except Exception as e:
                return False, f"Validator raised exception: {e}"

            if not ok:
                return False, msg or "Validator failed"

        return True, ""

    @abstractmethod
    def generate_dst_outputs(
        self, file_paths: List[str]
    ) -> Dict[str, Optional[DSTOutput]]:
        """Generate DST outputs for a list of input file paths.

        Implementations should handle file reading, validation, batching, calling the
        underlying GPT generation methods, and constructing DSTOutput objects (or
        None for failures).
        """
        pass

    @abstractmethod
    def generate_multiple_dst_structures(
        self, items: List[Tuple[str, str, str]], max_retries: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate DST structures for multiple items

        Args:
            items: List of tuples (input_file, inferred_knowledge, all_step_descriptions)
            max_retries: Maximum number of retry attempts for single generation

        Returns:
            Dictionary mapping input_file -> DST structure or None
        """
        pass
