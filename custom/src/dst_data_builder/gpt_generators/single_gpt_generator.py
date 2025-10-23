"""
Single GPT Generator - Processes one file at a time with retry logic
"""

import json
from typing import Dict, Any, List, Tuple
import time
import logging
from .base_gpt_generator import BaseGPTGenerator
from dst_data_builder.datatypes.dst_output import DSTOutput


class SingleGPTGenerator(BaseGPTGenerator):
    """Single-file GPT generator with retry logic"""

    def generate_multiple_dst_structures(
        self, items: List[Tuple[str, str, str]], max_retries: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate DST structures for multiple items by processing them one by one

        Args:
            items: List of tuples (input_file, inferred_knowledge, all_step_descriptions)
            max_retries: Maximum number of retry attempts for single generation

        Returns:
            Dictionary mapping input_file -> DST structure or None
        """
        results = {}

        logger = logging.getLogger(__name__)

        for input_file, inferred_knowledge, all_step_descriptions in items:
            logger.info(f"--- Processing {input_file} ---")

            dst_structure = None

            # Try generation with retries
            for attempt in range(max_retries + 1):
                logger.info(
                    f"Attempt {attempt + 1}/{max_retries + 1} for {input_file}..."
                )

                dst_structure = self._attempt_dst_generation(
                    inferred_knowledge, all_step_descriptions
                )

                if dst_structure:
                    # Run validators if configured
                    valid, v_msg = self._run_validators(dst_structure)
                    if not valid:
                        logger.warning(
                            f"Validator rejected DST for {input_file}: {v_msg}"
                        )
                        dst_structure = None
                    else:
                        logger.info(
                            f"✅ Successfully generated DST for {input_file} on attempt {attempt + 1}"
                        )
                    break
                elif attempt < max_retries:
                    logger.info(f"Retrying {input_file}...")
                    time.sleep(1)  # Brief pause before retry
                else:
                    logger.error(
                        f"❌ Failed to generate DST for {input_file} after {max_retries + 1} attempts"
                    )

            results[input_file] = dst_structure

        return results

    def generate_dst_outputs(self, file_paths: List[str]) -> Dict[str, DSTOutput]:
        """Read files, generate DST structures one-by-one, and return DSTOutput instances.

        Returns a mapping input_file -> DSTOutput or None on failure.
        """
        items = []
        raw_map = {}

        for input_path in file_paths:
            try:
                with open(input_path, "r") as f:
                    try:
                        import json as _json
                    except Exception:
                        _json = __import__("json")

                    data = _json.load(f)
            except Exception as e:
                logging.getLogger(__name__).exception(
                    f"Failed to read {input_path}: {e}"
                )
                raw_map[input_path] = None
                continue

            inferred_knowledge = data.get("inferred_knowledge", "")
            parsed_anns = data.get("parsed_video_anns", {})
            all_step_descriptions = parsed_anns.get("all_step_descriptions", "")

            if not inferred_knowledge or not all_step_descriptions:
                logging.getLogger(__name__).warning(
                    f"Missing required fields in {input_path}"
                )
                raw_map[input_path] = None
                continue

            items.append((input_path, inferred_knowledge, all_step_descriptions))
            raw_map[input_path] = data

        results = self.generate_multiple_dst_structures(items)

        outputs = {}
        for input_file, dst_structure in results.items():
            raw = raw_map.get(input_file)
            if dst_structure and raw:
                outputs[input_file] = DSTOutput.from_data_and_dst(
                    raw, dst_structure, self.model_name
                )
            else:
                outputs[input_file] = None

        # Ensure every requested file_path is represented in outputs
        for input_path in file_paths:
            outputs.setdefault(input_path, None)

        return outputs
