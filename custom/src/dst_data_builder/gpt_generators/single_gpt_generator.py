"""
Single GPT Generator - Processes files in parallel with retry logic
"""

from typing import Dict, Any, List, Tuple
import asyncio
import logging

from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator


class SingleGPTGenerator(BaseGPTGenerator):
    """GPT generator that processes multiple files in parallel using asyncio"""

    def __init__(
        self,
        api_key: str,
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        validators: list = None,
        max_retries: int = 2,
    ):
        super().__init__(api_key, model_name, temperature, max_tokens, max_retries=max_retries, validators=validators)

    async def _execute_generation_round(self, remaining: List[Tuple[str, str, str]], attempt_idx: int, failure_reasons: Dict[str, str] = None):
        """Process all items in parallel using asyncio.gather."""
        successes: Dict[str, Any] = {}
        failures: List[Tuple[str, str, str, str, Any]] = []
        failure_reasons = failure_reasons or {}

        self.logger.info(f"Processing {len(remaining)} items in parallel")
        
        # Process all items concurrently
        tasks = []
        for input_file, inferred_knowledge, all_step_descriptions in remaining:
            previous_reason = failure_reasons.get(input_file, "")
            task = self._try_generate_and_validate(inferred_knowledge, all_step_descriptions, previous_reason)
            tasks.append((input_file, inferred_knowledge, all_step_descriptions, task))
        
        # Wait for all tasks to complete
        results = await asyncio.gather(*[t[3] for t in tasks], return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            input_file, inferred_knowledge, all_step_descriptions, _ = tasks[i]
            
            if isinstance(result, Exception):
                self.logger.error(f"Task failed for {input_file}: {result}")
                failures.append((input_file, inferred_knowledge, all_step_descriptions, f"task_exception: {result}", None))
                failure_reasons[input_file] = f"task_exception: {result}"
                continue
                
            ok, dst_structure, reason, generated_content = result
            if ok:
                successes[input_file] = dst_structure
                self.logger.info("✅ Successfully generated DST for %s on attempt %d", input_file, attempt_idx + 1)
            else:
                self.logger.warning("Failure for %s on attempt %d: %s", input_file, attempt_idx + 1, reason)
                failures.append((input_file, inferred_knowledge, all_step_descriptions, reason, generated_content))
                failure_reasons[input_file] = reason

        return successes, failures
