"""
LLM Ambiguous Handler Module

This module implements the LLM fallback phase for ambiguous cases in the hybrid DST algorithm.
When global similarity scoring identifies unclear boundaries, this module uses LLM reasoning
to resolve the ambiguity with comprehensive logging and cost tracking.
"""

import logging
import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

# Import GPT client for LLM calls
from dst_data_builder.gpt_generators.openai_api_client import OpenAIAPIClient


@dataclass
class AmbiguousBlock:
    """Represents an ambiguous block that needs LLM resolution"""

    block_id: int
    block_data: Dict[str, Any]
    similarity_scores: List[float]
    confidence: float
    top_alternatives: List[Tuple[int, float]]  # (step_index, score)


@dataclass
class LLMDecision:
    """Result of LLM decision for an ambiguous block"""

    block_id: int
    chosen_step_index: int
    confidence: float
    reasoning: str
    llm_used: bool = True


@dataclass
class BidirectionalConflict:
    """Represents a conflict between forward and backward passes"""

    block_id: int
    block_data: Dict[str, Any]
    forward_assignment: Dict[str, Any]  # step_index, confidence, source
    backward_assignment: Dict[str, Any]  # step_index, confidence, source
    conflict_reason: str  # "step_disagreement" or "low_confidence"


@dataclass
class LLMHandlingResult:
    """Result of LLM handling for ambiguous blocks"""

    decisions: List[LLMDecision]
    total_llm_calls: int
    success_count: int
    failure_count: int
    total_cost_estimate: float


class LLMAmbiguousHandler:
    """
    LLM Fallback for Ambiguous Cases: LLM fallback for ambiguous cases with comprehensive logging

    This class handles the LLM fallback phase for blocks that couldn't be clearly classified
    by the global similarity scoring. It uses batch querying for efficiency and provides
    detailed logging for cost tracking and debugging.
    """

    def __init__(self, config: DictConfig):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)

        # Configuration for LLM handling
        self.temperature = config.get("temperature", 0.1)
        self.max_tokens = config.get("max_tokens", 1000)
        self.batch_size = config.get("batch_size", 5)

        # Initialize LLM client
        self.llm_client = OpenAIAPIClient(
            generator_type="hybrid_dst_llm", logger=self.logger
        )

        # Cost tracking
        self.total_cost_estimate = 0.0

    def resolve_ambiguous_blocks(
        self, ambiguous_blocks: List[AmbiguousBlock], inferred_knowledge: List[str]
    ) -> LLMHandlingResult:
        """
        Resolve ambiguous blocks using LLM fallback

        Args:
            ambiguous_blocks: List of blocks classified as ambiguous
            inferred_knowledge: List of step descriptions for context

        Returns:
            LLMHandlingResult with decisions and statistics
        """
        if not ambiguous_blocks:
            self.logger.info("No ambiguous blocks to resolve")
            return LLMHandlingResult([], 0, 0, 0, 0.0)

        self.logger.info(
            "🤖 Starting LLM fallback for %d ambiguous blocks", len(ambiguous_blocks)
        )

        # Prepare contexts for all ambiguous blocks
        contexts = self._prepare_contexts(ambiguous_blocks, inferred_knowledge)

        # Log LLM call details
        self.logger.info(
            "📡 Preparing %d LLM calls for ambiguous case resolution", len(contexts)
        )
        for i, context in enumerate(contexts[:3]):  # Log first 3 contexts
            self.logger.debug("LLM Context %d: %s...", i + 1, context[:200] + "...")

        # Batch query LLM
        responses = self._batch_query_llm(contexts)

        self.logger.info("✅ Received %d LLM responses", len(responses))

        # Parse responses and extract decisions
        decisions = self._parse_decisions(
            responses, ambiguous_blocks, inferred_knowledge
        )

        success_count = len([d for d in decisions if d.chosen_step_index >= 0])
        failure_count = len(ambiguous_blocks) - success_count

        self.logger.info(
            "🎯 LLM resolved %d ambiguous blocks into %d successful decisions",
            len(ambiguous_blocks),
            success_count,
        )

        return LLMHandlingResult(
            decisions=decisions,
            total_llm_calls=len(contexts),
            success_count=success_count,
            failure_count=failure_count,
            total_cost_estimate=self.total_cost_estimate,
        )

    def resolve_bidirectional_conflicts(
        self,
        conflicts: List[BidirectionalConflict],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
        correctly_assigned_blocks: Dict[int, Dict[str, Any]],
    ) -> LLMHandlingResult:
        """
        Resolve conflicts between forward and backward passes using LLM

        Args:
            conflicts: List of bidirectional conflicts to resolve
            filtered_blocks: All filtered blocks for context
            inferred_knowledge: Step descriptions
            correctly_assigned_blocks: Block_idx -> assignment dict for non-conflicting blocks

        Returns:
            LLMHandlingResult with conflict resolution decisions
        """
        if not conflicts:
            self.logger.info("No bidirectional conflicts to resolve")
            return LLMHandlingResult([], 0, 0, 0, 0.0)

        self.logger.info(
            "🤖 Starting LLM resolution for %d bidirectional conflicts", len(conflicts)
        )

        # Prepare contexts for all conflicts
        contexts = self._prepare_bidirectional_contexts(
            conflicts, filtered_blocks, inferred_knowledge, correctly_assigned_blocks
        )

        # Log LLM call details
        self.logger.info(
            "📡 Preparing %d LLM calls for bidirectional conflict resolution",
            len(contexts),
        )
        for i, context in enumerate(contexts[:3]):  # Log first 3 contexts
            self.logger.debug(
                "Bidirectional Context %d: %s...", i + 1, context[:200] + "..."
            )

        # Batch query LLM
        responses = self._batch_query_llm(contexts)

        self.logger.info("✅ Received %d LLM responses for conflicts", len(responses))

        # Parse responses and extract decisions
        decisions = self._parse_bidirectional_decisions(
            responses, conflicts, inferred_knowledge
        )

        success_count = len([d for d in decisions if d.chosen_step_index >= 0])
        failure_count = len(conflicts) - success_count

        self.logger.info(
            "🎯 LLM resolved %d bidirectional conflicts into %d successful decisions",
            len(conflicts),
            success_count,
        )

        return LLMHandlingResult(
            decisions=decisions,
            total_llm_calls=len(contexts),
            success_count=success_count,
            failure_count=failure_count,
            total_cost_estimate=self.total_cost_estimate,
        )

    def _prepare_contexts(
        self, ambiguous_blocks: List[AmbiguousBlock], inferred_knowledge: List[str]
    ) -> List[str]:
        """
        Prepare LLM prompts for each ambiguous block

        Args:
            ambiguous_blocks: Ambiguous blocks to resolve
            inferred_knowledge: Step descriptions for context

        Returns:
            List of formatted prompts for LLM
        """
        contexts = []

        for block in ambiguous_blocks:
            context = self._format_ambiguous_block_context(block, inferred_knowledge)
            contexts.append(context)

        return contexts

    def _format_ambiguous_block_context(
        self, block: AmbiguousBlock, inferred_knowledge: List[str]
    ) -> str:
        """
        Format context for an ambiguous block

        Args:
            block: Ambiguous block to format
            inferred_knowledge: Step descriptions

        Returns:
            Formatted prompt for LLM
        """
        # Extract block content
        block_text = self._extract_block_text(block.block_data)

        # Get top alternative steps
        step_options = []
        for step_idx, score in block.top_alternatives[:3]:  # Top 3 options
            if step_idx < len(inferred_knowledge):
                step_text = inferred_knowledge[step_idx]
                step_options.append(
                    f"  {step_idx + 1}. {step_text} (similarity: {score:.3f})"
                )

        steps_text = "\n".join(step_options)

        # Format the prompt
        prompt = f"""You are resolving ambiguous dialogue segments for step detection in conversational AI.

BLOCK CONTENT: "{block_text}"

SIMILARITY SCORES: The block has these similarity scores with different steps:
{steps_text}

TASK: Choose the most appropriate step for this block. Consider:
1. Which step best matches the semantic meaning
2. The temporal context and flow
3. The overall conversation goal

RESPOND with JSON:
{{
    "chosen_step": <step_number>,
    "confidence": <0-1>,
    "reasoning": "<brief explanation>"
}}

Choose the step number (1, 2, 3, etc.) that best matches this block."""

        return prompt

    def _extract_block_text(self, block_data: Dict[str, Any]) -> str:
        """Extract text content from block data"""
        return str(block_data.get("text", block_data.get("content", "")))

    def _batch_query_llm(self, contexts: List[str]) -> List[str]:
        """
        Query LLM for all contexts in batches

        Args:
            contexts: List of formatted prompts

        Returns:
            List of LLM responses
        """
        all_responses = []

        # Process in batches to avoid overwhelming the API
        for i in range(0, len(contexts), self.batch_size):
            batch = contexts[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(contexts) + self.batch_size - 1) // self.batch_size

            self.logger.debug(
                "Processing LLM batch %d/%d (%d contexts)",
                batch_num,
                total_batches,
                len(batch),
            )

            # Process batch concurrently
            batch_responses = asyncio.run(self._query_batch_concurrently(batch))
            all_responses.extend(batch_responses)

        return all_responses

    async def _query_batch_concurrently(self, batch: List[str]) -> List[str]:
        """
        Query LLM for a batch of contexts concurrently

        Args:
            batch: List of contexts to query

        Returns:
            List of LLM responses
        """

        async def query_single_context(context: str) -> str:
            try:
                success, response = await self.llm_client.generate_completion(
                    prompt=context,
                    model="gpt-4o",  # Use same model as other generators
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

                if success:
                    return response
                else:
                    self.logger.warning("LLM query failed: %s", response)
                    return ""

            except Exception as e:
                self.logger.error("LLM query exception: %s", e)
                return ""

        # Execute all queries in the batch concurrently
        tasks = [query_single_context(context) for context in batch]
        responses = await asyncio.gather(*tasks, return_exceptions=True)

        # Convert exceptions to empty strings
        return [
            str(response) if not isinstance(response, Exception) else ""
            for response in responses
        ]

    def _parse_decisions(
        self,
        responses: List[str],
        ambiguous_blocks: List[AmbiguousBlock],
        inferred_knowledge: List[str],
    ) -> List[LLMDecision]:
        """
        Parse LLM responses and extract decisions

        Args:
            responses: LLM responses
            ambiguous_blocks: Original ambiguous blocks
            inferred_knowledge: Step descriptions

        Returns:
            List of decisions
        """
        decisions = []

        for i, (response, block) in enumerate(zip(responses, ambiguous_blocks)):
            decision = self._parse_single_decision(response, block, inferred_knowledge)
            decisions.append(decision)

            if decision.chosen_step_index >= 0:
                self.logger.debug(
                    "Block %d: LLM chose step %d (confidence: %.3f)",
                    block.block_id,
                    decision.chosen_step_index + 1,
                    decision.confidence,
                )
            else:
                self.logger.warning(
                    "Block %d: LLM failed to make decision", block.block_id
                )

        return decisions

    def _parse_single_decision(
        self, response: str, block: AmbiguousBlock, inferred_knowledge: List[str]
    ) -> LLMDecision:
        """
        Parse a single LLM response into a decision

        Args:
            response: LLM response
            block: Original ambiguous block
            inferred_knowledge: Step descriptions

        Returns:
            Parsed decision
        """
        if not response.strip():
            return LLMDecision(
                block_id=block.block_id,
                chosen_step_index=-1,
                confidence=0.0,
                reasoning="Empty LLM response",
                llm_used=True,
            )

        # Try to parse JSON response
        import json

        try:
            # Clean the response to extract JSON
            json_match = response.strip()
            if "```json" in json_match:
                json_match = json_match.split("```json")[1].split("```")[0]
            elif "{" in json_match and "}" in json_match:
                start = json_match.find("{")
                end = json_match.rfind("}") + 1
                json_match = json_match[start:end]

            parsed = json.loads(json_match)

            # Extract decision components
            chosen_step = parsed.get("chosen_step", 1) - 1  # Convert to 0-based index
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", "No reasoning provided"))

            # Validate step index
            if chosen_step < 0 or chosen_step >= len(inferred_knowledge):
                self.logger.warning(
                    "Invalid step index %d, defaulting to 0", chosen_step
                )
                chosen_step = 0

            return LLMDecision(
                block_id=block.block_id,
                chosen_step_index=chosen_step,
                confidence=confidence,
                reasoning=reasoning,
                llm_used=True,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(
                "Failed to parse LLM response for block %d: %s", block.block_id, e
            )

            # Fallback: choose the highest similarity score option
            if block.top_alternatives:
                fallback_step = block.top_alternatives[0][0]
                fallback_confidence = 0.3  # Lower confidence for fallback

                return LLMDecision(
                    block_id=block.block_id,
                    chosen_step_index=fallback_step,
                    confidence=fallback_confidence,
                    reasoning=f"Fallback due to parse error: {str(e)}",
                    llm_used=True,
                )
            else:
                return LLMDecision(
                    block_id=block.block_id,
                    chosen_step_index=-1,
                    confidence=0.0,
                    reasoning=f"No valid alternatives: {str(e)}",
                    llm_used=True,
                )

    def get_cost_estimate(self, num_tokens: int) -> float:
        """
        Estimate cost for LLM usage

        Args:
            num_tokens: Number of tokens in the request

        Returns:
            Cost estimate in USD
        """
        # Rough cost estimation for GPT-4o (adjust as needed)
        cost_per_token = 0.00001  # $0.01 per 1K tokens
        estimated_cost = num_tokens * cost_per_token

        self.total_cost_estimate += estimated_cost
        return estimated_cost

    def get_handling_statistics(self, result: LLMHandlingResult) -> Dict[str, Any]:
        """
        Get statistics about LLM handling

        Args:
            result: Result from resolve_ambiguous_blocks

        Returns:
            Dictionary with statistics
        """
        if not result.decisions:
            return {"error": "No decisions to analyze"}

        confidences = [d.confidence for d in result.decisions if d.confidence > 0]

        stats = {
            "total_ambiguous_blocks": len(result.decisions),
            "llm_calls_made": result.total_llm_calls,
            "successful_decisions": result.success_count,
            "failed_decisions": result.failure_count,
            "success_rate": (
                (result.success_count / len(result.decisions)) * 100
                if result.decisions
                else 0
            ),
            "average_confidence": (
                float(sum(confidences) / len(confidences)) if confidences else 0.0
            ),
            "estimated_total_cost": result.total_cost_estimate,
            "cost_per_block": (
                result.total_cost_estimate / len(result.decisions)
                if result.decisions
                else 0.0
            ),
        }

        return stats

    def _prepare_bidirectional_contexts(
        self,
        conflicts: List[BidirectionalConflict],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
        correctly_assigned_blocks: Dict[int, Dict[str, Any]],
    ) -> List[str]:
        """
        Prepare LLM prompts for bidirectional conflicts

        Args:
            conflicts: Bidirectional conflicts to resolve
            filtered_blocks: All filtered blocks
            inferred_knowledge: Step descriptions
            correctly_assigned_blocks: Non-conflicting assignments

        Returns:
            List of formatted prompts for LLM
        """
        contexts = []

        for conflict in conflicts:
            context = self._format_bidirectional_conflict_context(
                conflict, filtered_blocks, inferred_knowledge, correctly_assigned_blocks
            )
            contexts.append(context)

        return contexts

    def _format_bidirectional_conflict_context(
        self,
        conflict: BidirectionalConflict,
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
        correctly_assigned_blocks: Dict[int, Dict[str, Any]],
    ) -> str:
        """
        Format comprehensive context for a bidirectional conflict

        Args:
            conflict: The conflict to format
            filtered_blocks: All blocks
            inferred_knowledge: Step descriptions
            correctly_assigned_blocks: Non-conflicting assignments

        Returns:
            Formatted prompt for LLM
        """
        # Extract conflict block content
        block_text = self._extract_block_text(conflict.block_data)

        # Format step descriptions
        steps_text = "\n".join(
            [f"  {i+1}. {step}" for i, step in enumerate(inferred_knowledge)]
        )

        # Format forward and backward assignments
        forward_step = conflict.forward_assignment["step_index"] + 1
        forward_conf = conflict.forward_assignment["confidence"]
        backward_step = conflict.backward_assignment["step_index"] + 1
        backward_conf = conflict.backward_assignment["confidence"]

        assignments_text = f"""
FORWARD PASS ASSIGNMENT: Step {forward_step} (confidence: {forward_conf:.3f})
BACKWARD PASS ASSIGNMENT: Step {backward_step} (confidence: {backward_conf:.3f})
CONFLICT REASON: {conflict.conflict_reason}
"""

        # Format correctly assigned blocks as examples
        examples_text = ""
        if correctly_assigned_blocks:
            examples = []
            for block_idx, assignment in list(correctly_assigned_blocks.items())[
                :5
            ]:  # Show up to 5 examples
                if block_idx < len(filtered_blocks):
                    example_block = filtered_blocks[block_idx]
                    example_text = self._extract_block_text(example_block)
                    step_num = assignment["step_index"] + 1
                    examples.append(
                        f"Block {block_idx}: '{example_text}' -> Step {step_num}"
                    )

            if examples:
                examples_text = (
                    "\nEXAMPLES OF CORRECTLY ASSIGNED BLOCKS:\n" + "\n".join(examples)
                )

        # Format the comprehensive prompt
        prompt = f"""You are resolving conflicts in bidirectional DST span construction for conversational AI.

TASK OVERVIEW:
Bidirectional DST span construction uses forward and backward passes to determine step boundaries:
- Forward pass: Starts from first block, assigns to steps moving forward
- Backward pass: Starts from last block, assigns to steps moving backward
- Conflicts occur when passes disagree on step assignment or both have low confidence

STEP DESCRIPTIONS:
{steps_text}

CONFLICTING BLOCK:
Block ID: {conflict.block_id}
Content: "{block_text}"
{assignments_text.strip()}{examples_text}

INSTRUCTIONS:
1. Consider the bidirectional algorithm context and temporal flow
2. Look at correctly assigned blocks as reference examples
3. Choose the most appropriate step considering semantic meaning and conversation flow
4. The forward/backward assignments provide directional context but may conflict

RESPOND with JSON:
{{
    "chosen_step": <step_number>,
    "confidence": <0-1>,
    "reasoning": "<brief explanation considering bidirectional context>"
}}

Choose the step number (1, 2, 3, etc.) that best fits this block in the overall conversation flow."""

        return prompt

    def _parse_bidirectional_decisions(
        self,
        responses: List[str],
        conflicts: List[BidirectionalConflict],
        inferred_knowledge: List[str],
    ) -> List[LLMDecision]:
        """
        Parse LLM responses for bidirectional conflicts

        Args:
            responses: LLM responses
            conflicts: Original conflicts
            inferred_knowledge: Step descriptions

        Returns:
            List of decisions
        """
        decisions = []

        for i, (response, conflict) in enumerate(zip(responses, conflicts)):
            decision = self._parse_single_bidirectional_decision(
                response, conflict, inferred_knowledge
            )
            decisions.append(decision)

            if decision.chosen_step_index >= 0:
                self.logger.debug(
                    "Conflict Block %d: LLM chose step %d (confidence: %.3f)",
                    conflict.block_id,
                    decision.chosen_step_index + 1,
                    decision.confidence,
                )
            else:
                self.logger.warning(
                    "Conflict Block %d: LLM failed to make decision", conflict.block_id
                )

        return decisions

    def _parse_single_bidirectional_decision(
        self,
        response: str,
        conflict: BidirectionalConflict,
        inferred_knowledge: List[str],
    ) -> LLMDecision:
        """
        Parse a single LLM response for bidirectional conflict

        Args:
            response: LLM response
            conflict: Original conflict
            inferred_knowledge: Step descriptions

        Returns:
            Parsed decision
        """
        if not response.strip():
            return LLMDecision(
                block_id=conflict.block_id,
                chosen_step_index=-1,
                confidence=0.0,
                reasoning="Empty LLM response",
                llm_used=True,
            )

        # Try to parse JSON response
        import json

        try:
            # Clean the response to extract JSON
            json_match = response.strip()
            if "```json" in json_match:
                json_match = json_match.split("```json")[1].split("```")[0]
            elif "{" in json_match and "}" in json_match:
                start = json_match.find("{")
                end = json_match.rfind("}") + 1
                json_match = json_match[start:end]

            parsed = json.loads(json_match)

            # Extract decision components
            chosen_step = parsed.get("chosen_step", 1) - 1  # Convert to 0-based index
            confidence = float(parsed.get("confidence", 0.5))
            reasoning = str(parsed.get("reasoning", "No reasoning provided"))

            # Validate step index
            if chosen_step < 0 or chosen_step >= len(inferred_knowledge):
                self.logger.warning(
                    "Invalid step index %d for conflict block %d, defaulting to 0",
                    chosen_step,
                    conflict.block_id,
                )
                chosen_step = 0

            return LLMDecision(
                block_id=conflict.block_id,
                chosen_step_index=chosen_step,
                confidence=confidence,
                reasoning=reasoning,
                llm_used=True,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(
                "Failed to parse LLM response for conflict block %d: %s",
                conflict.block_id,
                e,
            )

            # Fallback: choose the higher confidence assignment
            forward_conf = conflict.forward_assignment["confidence"]
            backward_conf = conflict.backward_assignment["confidence"]

            if forward_conf >= backward_conf:
                fallback_step = conflict.forward_assignment["step_index"]
                fallback_reason = (
                    f"Fallback due to parse error: chose forward assignment"
                )
            else:
                fallback_step = conflict.backward_assignment["step_index"]
                fallback_reason = (
                    f"Fallback due to parse error: chose backward assignment"
                )

            return LLMDecision(
                block_id=conflict.block_id,
                chosen_step_index=fallback_step,
                confidence=0.3,  # Lower confidence for fallback
                reasoning=fallback_reason,
                llm_used=True,
            )
