"""
Batch GPT Generator - Uses OpenAI's batch API for processing multiple files efficiently
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import time
import logging
from hydra.core.hydra_config import HydraConfig
from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator
from dst_data_builder.datatypes.dst_output import DSTOutput

# Module-level logger
logger = logging.getLogger(__name__)


class BatchGPTGenerator(BaseGPTGenerator):
    """Batch GPT generator using OpenAI's batch API"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        batch_size: int = None,
        check_interval: int = 60,
        save_intermediate: bool = False,
        max_retries: int = 2,
    ):
        super().__init__(api_key, model_name, temperature, max_tokens)
        # Configure batch options; prefer provided value, otherwise default
        self.batch_size = int(batch_size) if batch_size is not None else 100
        self.check_interval = int(check_interval) if check_interval is not None else 60
        self.save_intermediate = bool(save_intermediate)
        self.max_retries = int(max_retries) if max_retries is not None else 2

    def _create_batch_jsonl(
        self, items: List[Tuple[str, str, str]], batch_file: str
    ) -> None:
        """Create a JSONL file for batch processing"""
        logger.info(f"Creating batch file: {batch_file}")

        with open(batch_file, "w") as f:
            for input_file, inferred_knowledge, all_step_descriptions in items:
                request = self._create_batch_request(
                    input_file, inferred_knowledge, all_step_descriptions
                )
                f.write(json.dumps(request) + "\n")

        logger.info(f"✅ Created batch file with {len(items)} requests")

    def _create_batch_request(
        self, custom_id: str, inferred_knowledge: str, all_step_descriptions: str
    ) -> Dict[str, Any]:
        """Create a batch API request for a single DST generation"""
        prompt = self.create_dst_prompt(inferred_knowledge, all_step_descriptions)

        return {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are an expert at creating hierarchical task structures.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            },
        }

    def _upload_batch(self, batch_file: str) -> str:
        """Upload batch file to OpenAI and return batch ID"""
        logger.info(f"Uploading batch file: {batch_file}")

        with open(batch_file, "rb") as f:
            batch_input_file = self.client.files.create(file=f, purpose="batch")

        logger.info(f"✅ Uploaded batch file, got file ID: {batch_input_file.id}")

        # Create batch job
        batch_job = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )

        logger.info(f"✅ Created batch job: {batch_job.id}")
        return batch_job.id

    def _wait_for_batch_completion(
        self, batch_id: str, check_interval: int = 60
    ) -> Dict[str, Any]:
        """Wait for batch job to complete and return results"""
        logger.info(f"Waiting for batch {batch_id} to complete...")

        while True:
            batch_status = self.client.batches.retrieve(batch_id)
            status = batch_status.status

            logger.info(f"Batch status: {status}")

            if status == "completed":
                logger.info("✅ Batch completed!")
                return batch_status
            elif status == "failed":
                logger.error(f"❌ Batch failed: {batch_status}")
                return None
            elif status == "cancelled":
                logger.warning("❌ Batch was cancelled")
                return None
            else:
                logger.info(
                    f"Batch still processing... waiting {self.check_interval} seconds"
                )
                time.sleep(self.check_interval)

    def _download_batch_results(self, batch_id: str, output_file: str) -> None:
        """Download batch results to output file"""
        logger.info(f"Downloading batch results to: {output_file}")

        batch_status = self.client.batches.retrieve(batch_id)
        result_file_id = batch_status.output_file_id

        if not result_file_id:
            logger.error("❌ No output file ID found")
            return

        # Download the results
        result_content = self.client.files.content(result_file_id)
        result_content.write_to_file(output_file)

        logger.info(f"✅ Downloaded results to: {output_file}")

    def _parse_batch_results(self, results_file: str) -> Dict[str, Dict[str, Any]]:
        """Parse batch results file and extract DST structures"""
        logger.info(f"Parsing batch results from: {results_file}")

        results = {}

        with open(results_file, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)

                    custom_id = result.get("custom_id", "")
                    response = result.get("response", {})

                    # Determine whether this response contains a successful completion.
                    body = (
                        response.get("body", {}) if isinstance(response, dict) else {}
                    )

                    # success if explicit status == completed OR HTTP-like status_code 200/201
                    success = False
                    if response.get("status") == "completed":
                        success = True
                    elif isinstance(response.get("status_code"), int) and response.get(
                        "status_code"
                    ) in (200, 201):
                        success = True
                    elif isinstance(body, dict) and body.get("choices"):
                        # sometimes the batch result uses a direct response body with choices
                        success = True

                    if success:
                        # Extract the content from the response body (if present)
                        content = ""
                        try:
                            # body might be the raw completion object
                            content = (
                                body.get("choices", [{}])[0]
                                .get("message", {})
                                .get("content", "")
                            )
                        except Exception:
                            content = ""

                        # Parse the DST structure
                        if isinstance(content, str) and content.strip():
                            try:
                                # Clean and parse JSON
                                cleaned = self._clean_json_response(content)
                                dst_structure = json.loads(cleaned)

                                if self._validate_dst_structure(dst_structure):
                                        # Run additional validators if present
                                        try:
                                            valid, v_msg = self._run_validators(dst_structure)
                                        except Exception:
                                            valid, v_msg = True, ""

                                        if not valid:
                                            logger.error(
                                                f"❌ Validator rejected structure for {custom_id}: {v_msg}"
                                            )
                                            results[custom_id] = None
                                        else:
                                            results[custom_id] = dst_structure
                                            logger.info(f"✅ Parsed result for {custom_id}")
                                else:
                                    logger.error(
                                        f"❌ Invalid structure for {custom_id}"
                                    )
                                    # Save raw content for debugging
                                    try:
                                        batch_dir = Path(results_file).parent
                                        failed_path = (
                                            batch_dir
                                            / f"failed_raw_{Path(custom_id).name}.txt"
                                        )
                                        failed_path.write_text(content)
                                        logger.info(
                                            f"Saved raw invalid content to: {failed_path}"
                                        )
                                    except Exception:
                                        logger.exception(
                                            "Failed to write raw invalid content"
                                        )
                                    results[custom_id] = None
                            except json.JSONDecodeError:
                                logger.error(f"❌ Failed to parse JSON for {custom_id}")
                                # Save raw content for debugging
                                try:
                                    batch_dir = Path(results_file).parent
                                    failed_path = (
                                        batch_dir
                                        / f"failed_raw_{Path(custom_id).name}.txt"
                                    )
                                    failed_path.write_text(content)
                                    logger.info(
                                        f"Saved raw unparsable content to: {failed_path}"
                                    )
                                except Exception:
                                    logger.exception(
                                        "Failed to write raw unparsable content"
                                    )
                                results[custom_id] = None
                        else:
                            logger.warning(f"❌ Empty content for {custom_id}")
                            try:
                                batch_dir = Path(results_file).parent
                                failed_path = (
                                    batch_dir / f"failed_raw_{Path(custom_id).name}.txt"
                                )
                                failed_path.write_text("<EMPTY_CONTENT>")
                                logger.info(f"Saved empty marker to: {failed_path}")
                            except Exception:
                                logger.exception("Failed to write empty content marker")
                            results[custom_id] = None
                    else:
                        # Log and save detailed error info for easier debugging
                        # Prefer an explicit error.message if present, otherwise dump the body
                        error_obj = None
                        try:
                            error_obj = (
                                body.get("error") if isinstance(body, dict) else None
                            )
                        except Exception:
                            error_obj = None

                        if isinstance(error_obj, dict):
                            error_msg = error_obj.get("message")
                        else:
                            # Fallback to the body as a string
                            try:
                                error_msg = json.dumps(body)
                            except Exception:
                                error_msg = str(body)

                        logger.error(
                            f"❌ Batch request failed for {custom_id}: {error_msg}"
                        )
                        try:
                            batch_dir = Path(results_file).parent
                            failed_path = (
                                batch_dir / f"failed_error_{Path(custom_id).name}.json"
                            )
                            failed_path.write_text(
                                json.dumps(response.get("body", {}), indent=2)
                            )
                            logger.info(f"Saved detailed error body to: {failed_path}")
                        except Exception:
                            logger.exception("Failed to write detailed error body")
                        results[custom_id] = None

        return results

    def generate_multiple_dst_structures(
        self, items: List[Tuple[str, str, str]], max_retries: int = 2
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate DST structures for multiple items using OpenAI's batch API

        Args:
            items: List of tuples (input_file, inferred_knowledge, all_step_descriptions)
            max_retries: Maximum number of retry attempts (not used in batch mode)

        Returns:
            Dictionary mapping input_file -> DST structure or None
        """
        if not items:
            return {}

        logger.info(f"🚀 Starting batch processing of {len(items)} items...")

        # Resolve Hydra output directory (fallback to current working dir)
        try:
            hydra_cfg = HydraConfig.get()
            hydra_output_dir = getattr(hydra_cfg.runtime, "output_dir", None)
            base_dir = Path(hydra_output_dir) if hydra_output_dir else Path.cwd()
        except Exception:
            base_dir = Path.cwd()

        batch_dir = base_dir / "batch"
        batch_dir.mkdir(parents=True, exist_ok=True)

        # We'll attempt to process remaining_items up to max_retries times.
        remaining_items = list(items)
        final_results: Dict[str, Any] = {item[0]: None for item in items}

        for attempt in range(0, max_retries + 1):
            if not remaining_items:
                break

            attempt_ts = int(time.time())
            batch_file = str(
                batch_dir / f"batch_dst_requests_{attempt_ts}_attempt{attempt}.jsonl"
            )
            results_file = str(
                batch_dir / f"batch_dst_results_{attempt_ts}_attempt{attempt}.jsonl"
            )

            try:
                logger.info(
                    f"Attempt {attempt+1}/{max_retries+1}: processing {len(remaining_items)} items"
                )

                # Create batch JSONL for only remaining items
                self._create_batch_jsonl(remaining_items, batch_file)

                # Upload and create batch job
                batch_id = self._upload_batch(batch_file)

                # Wait for completion
                batch_status = self._wait_for_batch_completion(batch_id)

                if not batch_status:
                    logger.error("❌ Batch processing failed")
                    # Do not retry on complete batch failure; mark remaining as None and break
                    break

                # Download results
                self._download_batch_results(batch_id, results_file)

                # Parse results
                batch_results = self._parse_batch_results(results_file)

                # Update final_results and compute new remaining_items
                new_remaining: List[Tuple[str, str, str]] = []
                for (
                    input_file,
                    inferred_knowledge,
                    all_step_descriptions,
                ) in remaining_items:
                    parsed = batch_results.get(input_file)
                    if parsed:
                        final_results[input_file] = parsed
                    else:
                        # Keep this item for the next attempt
                        new_remaining.append(
                            (input_file, inferred_knowledge, all_step_descriptions)
                        )

                remaining_items = new_remaining

                logger.info(
                    f"Attempt {attempt+1} complete: {len(remaining_items)} items remaining"
                )

            except Exception as e:
                logger.error(f"❌ Error in batch processing attempt {attempt}: {e}")
                # On exception, keep remaining_items as-is and continue to next attempt

            finally:
                # Clean up temporary files for this attempt
                try:
                    if os.path.exists(batch_file):
                        os.remove(batch_file)
                        logger.info(f"🗑️ Cleaned up batch file: {batch_file}")

                    # Only remove results file if save_intermediate is False
                    if os.path.exists(results_file) and not getattr(
                        self, "save_intermediate", False
                    ):
                        os.remove(results_file)
                        logger.info(f"🗑️ Cleaned up results file: {results_file}")
                except Exception as e:
                    logger.warning(f"Warning: Could not clean up temporary files: {e}")

        return final_results

    def generate_dst_outputs(self, file_paths: List[str]) -> Dict[str, DSTOutput]:
        """Process the provided file paths in batches using the batch API.

        Returns a mapping input_file -> DSTOutput or None on failure.
        """
        # Read and prepare items
        items = []
        raw_map = {}

        for input_path in file_paths:
            try:
                with open(input_path, "r") as f:
                    data = json.load(f)
            except Exception as e:
                logger.exception(f"Failed to read {input_path}: {e}")
                raw_map[input_path] = None
                continue

            inferred_knowledge = data.get("inferred_knowledge", "")
            parsed_anns = data.get("parsed_video_anns", {})
            all_step_descriptions = parsed_anns.get("all_step_descriptions", "")

            if not inferred_knowledge or not all_step_descriptions:
                logger.warning(f"Missing required fields in {input_path}")
                raw_map[input_path] = None
                continue

            items.append((input_path, inferred_knowledge, all_step_descriptions))
            raw_map[input_path] = data

        outputs = {}

        # Chunk items according to self.batch_size
        for i in range(0, len(items), self.batch_size):
            chunk = items[i : i + self.batch_size]
            results = self.generate_multiple_dst_structures(
                chunk, max_retries=self.max_retries
            )

            for input_file, dst_structure in results.items():
                raw = raw_map.get(input_file)
                if dst_structure and raw:
                    outputs[input_file] = DSTOutput.from_data_and_dst(
                        raw, dst_structure, self.model_name
                    )
                else:
                    outputs[input_file] = None

        # Ensure every requested file_path is represented
        for input_path in file_paths:
            outputs.setdefault(input_path, None)

        return outputs
