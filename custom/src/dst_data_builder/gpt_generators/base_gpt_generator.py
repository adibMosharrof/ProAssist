"""
Base GPT Generator - Abstract base class for GPT-based DST generation
"""

import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

from dst_data_builder.datatypes.dst_output import DSTOutput
from dst_data_builder.utils.hydra_utils import get_hydra_output_dir
from dst_data_builder.validators.json_parsing_validator import JSONParsingValidator
from dst_data_builder.gpt_generators.dst_generation_prompt import create_dst_prompt
from dst_data_builder.gpt_generators.openai_api_client import OpenAIAPIClient


class BaseGPTGenerator(ABC):
    """Abstract base class for GPT-based DST generation"""

    def __init__(
        self,
        generator_type: str = "gpt",
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: int = 2,
        validators: list = None,
    ):
        # Instance-level logger for this generator and subclasses
        self.logger = logging.getLogger(__name__)

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        # Retries configuration
        self.max_retries = int(max_retries) if max_retries is not None else 2

        # Initialize JSON parsing validator (separate from structural validators)
        self.json_validator = JSONParsingValidator()

        # Initialize validators: expect list of BaseValidator instances
        # These validators run AFTER JSON parsing succeeds
        self.validators = validators or []

        # Create the OpenAI API client (handles API key from environment)
        self.api_client = OpenAIAPIClient(
            generator_type=generator_type,
            logger=self.logger
        )

    async def _attempt_dst_generation(
        self, inferred_knowledge: str, all_step_descriptions: str, previous_failure_reason: str = ""
    ) -> Tuple[bool, str]:
        """
        Single attempt at DST generation - API call only, returns raw response.

        Returns:
            Tuple of (success: bool, result: str)
            - On success: (True, raw_api_response_string)
            - On failure: (False, error_message)
        """
        if previous_failure_reason:
            self.logger.info("Including failure section in prompt with reason: %s", previous_failure_reason)

        prompt = create_dst_prompt(inferred_knowledge, all_step_descriptions, previous_failure_reason)

        # Use API client to make the call
        return await self.api_client.generate_completion(
            prompt=prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

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

    def _log_validator_statistics(self):
        """Log statistics from validators that support it."""
        if not self.validators:
            return
        
        from dst_data_builder.validators.timestamps_validator import TimestampsValidator
        
        for validator in self.validators:
            if isinstance(validator, TimestampsValidator):
                stats = validator.get_post_processing_stats()
                if stats.get("total_fixes", 0) > 0:
                    self.logger.info("="*60)
                    self.logger.info("📊 TimestampsValidator Post-Processing Statistics:")
                    self.logger.info(f"   Total timestamp violations fixed: {stats['total_fixes']}")
                    self.logger.info("="*60)
                break

    async def _try_generate_and_validate(self, inferred_knowledge: str, all_step_descriptions: str, previous_failure_reason: str = "") -> Tuple[bool, Any, str, Any]:
        """
        Generation + JSON parsing + validation with unified error handling.

        Returns (ok, dst_structure, reason_if_failed, generated_content)
        """
        # Step 1: API call (returns raw string)
        success, raw_response = await self._attempt_dst_generation(inferred_knowledge, all_step_descriptions, previous_failure_reason)
        if not success:
            # API error
            return False, None, raw_response, None

        # Step 2: Parse JSON with json_validator
        parse_ok, parse_error = self.json_validator.validate(raw_response)
        if not parse_ok:
            # JSON parsing failed - return detailed error for retry
            return False, None, parse_error, raw_response
        
        # Get parsed dict
        dst_structure = self.json_validator.parsed_result

        # Step 3: Run structural validators
        valid, v_msg = self._run_validators(dst_structure)
        if not valid:
            return False, dst_structure, f"validator_rejected: {v_msg}", dst_structure

        return True, dst_structure, "", None

    def _read_input_files(self, file_paths: List[str]) -> Tuple[List[Tuple[str, str, str]], Dict[str, Any]]:
        """Read and validate input files. Returns (items_list, raw_data_map)."""
        items: List[Tuple[str, str, str]] = []
        raw_map: Dict[str, Any] = {}

        for input_path in file_paths:
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                self.logger.exception(f"Failed to read {input_path}: {e}")
                raw_map[input_path] = None
                continue

            # Handle both single video objects and lists of videos
            video_objects = []
            if isinstance(data, list):
                # Filtered format: list of video objects
                video_objects = data
            elif isinstance(data, dict):
                # Original format: single video object
                video_objects = [data]
            else:
                self.logger.warning(f"Unexpected data format in {input_path}: {type(data)}")
                raw_map[input_path] = None
                continue

            # Process each video object
            for i, video_obj in enumerate(video_objects):
                # Handle different data structures
                inferred_knowledge = video_obj.get("inferred_knowledge", "")
                
                # Try different locations for all_step_descriptions
                all_step_descriptions = ""
                if "parsed_video_anns" in video_obj and isinstance(video_obj["parsed_video_anns"], dict):
                    all_step_descriptions = video_obj["parsed_video_anns"].get("all_step_descriptions", "")
                elif "conversations" in video_obj:
                    # Extract step descriptions from conversations
                    conversations = video_obj.get("conversations", [])
                    step_descriptions = []
                    for conv in conversations:
                        if isinstance(conv, dict) and "conversation" in conv:
                            for turn in conv["conversation"]:
                                if turn.get("role") == "user" and turn.get("content"):
                                    step_descriptions.append(f"[{turn.get('time', 0)}] {turn.get('content', '')}")
                    all_step_descriptions = "\n".join(step_descriptions)

                if not inferred_knowledge or not all_step_descriptions:
                    self.logger.warning(f"Missing required fields in {input_path} video {i}")
                    continue

                # Create unique identifier for this video
                video_id = f"{input_path}#video_{i}"
                items.append((video_id, inferred_knowledge, all_step_descriptions))
                raw_map[video_id] = video_obj

        return items, raw_map

    def _save_dst_output(self, result: Optional[DSTOutput], input_path: str, dst_output_dir: Path) -> bool:
        """Save a single DST output to file. Returns True if successful."""
        try:
            if "#video_" in input_path:
                # Handle individual videos from filtered files
                file_path, video_id = input_path.split("#video_")
                video_idx = int(video_id)
                out_name = f"dst_{Path(file_path).stem}_video_{video_idx}.json"
            else:
                # Handle original single video files
                out_name = f"dst_{Path(input_path).name}"
            
            out_file = dst_output_dir / out_name

            if result is None:
                self.logger.warning("Generation failed for %s", input_path)
                return False

            payload = result.to_dict() if hasattr(result, "to_dict") else result

            with open(out_file, "w") as f:
                json.dump(payload, f, indent=2)

            self.logger.info("✅ Saved: %s", out_file)
            return True

        except Exception as e:
            self.logger.exception("Failed to save output for %s: %s", input_path, e)
            return False

    def _save_failed_response(self, raw_response: str, input_path: str, error_reason: str, dst_output_dir: Path) -> None:
        """Save raw response for failed JSON parsing to enable later analysis."""
        try:
            # Create a subdirectory for failed responses
            failed_dir = dst_output_dir / "failed_responses"
            failed_dir.mkdir(exist_ok=True)
            
            # Save the raw response
            out_name = f"failed_{Path(input_path).stem}.txt"
            out_file = failed_dir / out_name
            
            with open(out_file, "w", encoding="utf-8") as f:
                f.write(f"=== FAILURE INFO ===\n")
                f.write(f"Input file: {input_path}\n")
                f.write(f"Error: {error_reason}\n")
                f.write(f"\n=== RAW RESPONSE ===\n")
                f.write(raw_response)
            
            self.logger.debug("💾 Saved failed response: %s", out_file)
            
        except Exception as e:
            self.logger.error("Failed to save failed response for %s: %s", input_path, e)

    async def generate_and_save_dst_outputs(
        self, 
        file_paths: List[str], 
        dst_output_dir: Path,
        batch_size: int = 5
    ) -> Tuple[int, int]:
        """
        Generate DST outputs and save them incrementally in batches with global retry logic.
        
        Strategy:
        1. Process all files in batches (attempt 1)
        2. Collect all failures across batches
        3. Retry all failures together in subsequent attempts
        
        Args:
            file_paths: List of input file paths to process
            dst_output_dir: Directory to save output files
            batch_size: Number of files to process in parallel
            
        Returns:
            Tuple of (processed_count, failed_count)
        """
        processed = 0
        failed = 0
        remaining_files = list(file_paths)
        
        # Track failure info for saving raw responses
        failure_info: Dict[str, Tuple[str, Any]] = {}  # {input_path: (error_reason, raw_content)}
        
        for attempt in range(self.max_retries + 1):
            if not remaining_files:
                break
                
            attempt_num = attempt + 1
            self.logger.info("=" * 60)
            self.logger.info("Starting attempt %d/%d: %d files to process", attempt_num, self.max_retries + 1, len(remaining_files))
            self.logger.info("=" * 60)
            
            attempt_failures = []
            
            # Process all remaining files in batches
            for i in range(0, len(remaining_files), batch_size):
                batch_files = remaining_files[i:i + batch_size]
                batch_num = i // batch_size + 1
                total_batches = (len(remaining_files) + batch_size - 1) // batch_size
                
                self.logger.info("🔄 Processing batch %d/%d: %d files", batch_num, total_batches, len(batch_files))
                
                try:
                    batch_outputs, batch_failure_info = await self.generate_dst_outputs(batch_files)
                    
                    # Update failure info with latest attempts
                    failure_info.update(batch_failure_info)
                    
                    for input_path in batch_files:
                        result = batch_outputs.get(input_path)
                        
                        if result and self._save_dst_output(result, input_path, dst_output_dir):
                            processed += 1
                        else:
                            failed += 1
                            attempt_failures.append(input_path)

                except Exception as e:
                    self.logger.exception("Failed to process batch %d: %s", batch_num, e)
                    # Mark all files in this batch as failed
                    attempt_failures.extend(batch_files)
                    failed += len(batch_files)
            
            # Update remaining files for next attempt
            remaining_files = attempt_failures
            
            if remaining_files:
                self.logger.warning("⚠️  Attempt %d: %d files failed, will retry", attempt_num, len(remaining_files))
            else:
                self.logger.info("✅ Attempt %d: All files processed successfully", attempt_num)

        # Save raw responses for all persistent failures
        json_parse_failures = 0
        for input_path, (error_reason, raw_content) in failure_info.items():
            if "JSON Parse Error" in error_reason and raw_content:
                try:
                    self._save_failed_response(raw_content, input_path, error_reason, dst_output_dir)
                    json_parse_failures += 1
                except Exception as e:
                    self.logger.error("Failed to save raw response for %s: %s", input_path, e)
        
        if json_parse_failures > 0:
            self.logger.info("💾 Saved %d failed responses to %s/failed_responses/", json_parse_failures, dst_output_dir)

        # Log post-processing statistics from validators (only once at the end)
        self._log_validator_statistics()

        return processed, failed

    async def generate_dst_outputs(self, file_paths: List[str]) -> Tuple[Dict[str, Optional[DSTOutput]], Dict[str, Tuple[str, Any]]]:
        """
        Read input files, generate DSTs for each, and return DSTOutput mapping.
        
        Returns:
            Tuple of (outputs, failure_info) where:
            - outputs: Dict mapping input_path to DSTOutput or None
            - failure_info: Dict mapping input_path to (error_reason, raw_content)
        """
        items, raw_map = self._read_input_files(file_paths)
        results, failure_info = await self.generate_multiple_dst_structures(items)

        outputs: Dict[str, Optional[DSTOutput]] = {}
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

        return outputs, failure_info

    async def generate_multiple_dst_structures(self, items: List[Tuple[str, str, str]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Tuple[str, Any]]]:
        """
        Generate DST structures for multiple items (single attempt, no retries).
        
        Retries are handled at a higher level by generate_and_save_dst_outputs().
        Subclasses should implement _execute_generation_round() to customize processing.

        Args:
            items: List of tuples (input_file, inferred_knowledge, all_step_descriptions)

        Returns:
            Tuple of (results, failure_info) where:
            - results: Dictionary mapping input_file -> DST structure or None
            - failure_info: Dictionary mapping input_file -> (error_reason, raw_content)
        """
        results: Dict[str, Any] = {inp[0]: None for inp in items}
        failure_info: Dict[str, Tuple[str, Any]] = {}
        failure_reasons: Dict[str, str] = {}
        
        try:
            successes, failures = await self._execute_generation_round(items, attempt_idx=0, failure_reasons=failure_reasons)
        except Exception as e:
            self.logger.exception("_execute_generation_round raised exception: %s", e)
            # On exception, all items fail
            return results, failure_info
        
        # Apply successes
        for inp, dst in successes.items():
            results[inp] = dst
            
        # Process failures and save their info
        if failures:
            for f in failures:
                input_file, _, _, reason, raw_content = f
                self.logger.warning("Failed: %s - %s", input_file, reason)
                failure_info[input_file] = (reason, raw_content)
        
        return results, failure_info

    @abstractmethod
    async def _execute_generation_round(
        self, 
        items: List[Tuple[str, str, str]], 
        attempt_idx: int, 
        failure_reasons: Dict[str, str]
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str, str, Any]]]:
        """
        Execute one round of generation for the given items.
        
        Must be implemented by subclasses to define how items are processed
        (e.g., sequentially, in parallel, in batches).
        
        Args:
            items: List of (input_file, inferred_knowledge, all_step_descriptions)
            attempt_idx: Current attempt index (0-based)
            failure_reasons: Dict to track failure reasons per input file
            
        Returns:
            Tuple of (successes_dict, failures_list)
            - successes_dict: {input_file: dst_structure}
            - failures_list: [(input_file, inferred_knowledge, all_step_descriptions, error_reason, generated_content), ...]
        """
        pass
