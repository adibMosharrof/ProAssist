"""
Bidirectional DST Span Constructor Module

This module implements the Bidirectional DST Span Constructor that uses
forward and backward passes to determine step boundaries, with LLM fallback
for conflicting assignments.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from omegaconf import DictConfig

from dst_data_builder.hybrid_dst.span_constructors.base_span_constructor import (
    BaseSpanConstructor,
)
from dst_data_builder.hybrid_dst.span_constructors.global_similarity_calculator import (
    GlobalSimilarityCalculator,
    ClassificationResult,
    SimilarityResult,
)
from dst_data_builder.hybrid_dst.span_constructors.llm_ambiguous_handler import (
    LLMAmbiguousHandler,
    LLMDecision,
    LLMHandlingResult,
    BidirectionalConflict,
)


@dataclass
class BidirectionalSpanConstructionResult:
    """Result of bidirectional span construction"""

    dst_spans: List[Dict[str, Any]]
    total_blocks_processed: int
    construction_statistics: Dict[str, Any]


@dataclass
class DirectionalAssignment:
    """Assignment result from forward or backward pass"""

    block_assignments: Dict[
        int, Dict[str, Any]
    ]  # block_idx -> {step_index, confidence, source}
    confidence_scores: Dict[int, float]  # block_idx -> confidence


class BidirectionalSpanConstructor(BaseSpanConstructor):
    """
    Bidirectional DST Span Constructor

    Uses forward and backward passes to determine step boundaries:
    1. Forward pass: Start from first block, assign to steps going forward
    2. Backward pass: Start from last block, assign to steps going backward
    3. Conflict detection: Identify blocks where forward/backward disagree
    4. LLM resolution: Use LLM for conflicting assignments
    5. Span construction: Build final DST spans from resolved assignments
    """

    def __init__(self, config: DictConfig, model_config: DictConfig):
        super().__init__(config)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        self.similarity_calculator = GlobalSimilarityCalculator(config)
        self.llm_handler = LLMAmbiguousHandler(model_config)

        # Bidirectional algorithm parameters
        self.high_confidence_threshold = config.similarity.high_confidence_threshold
        self.semantic_weight = config.similarity.semantic_weight
        self.nli_weight = config.similarity.nli_weight

    def construct_spans(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> BidirectionalSpanConstructionResult:
        """
        Execute bidirectional span construction algorithm

        Args:
            filtered_blocks: Block list from Stage 1
            inferred_knowledge: Step descriptions

        Returns:
            BidirectionalSpanConstructionResult with constructed spans
        """
        total_blocks = len(filtered_blocks)
        total_steps = len(inferred_knowledge)

        self.logger.info("🔄 Starting bidirectional span construction")
        self.logger.info(f"📊 Processing {total_blocks} blocks for {total_steps} steps")

        # Phase 1: Forward Pass
        forward_result = self._execute_forward_pass(filtered_blocks, inferred_knowledge)

        # Phase 2: Backward Pass
        backward_result = self._execute_backward_pass(
            filtered_blocks, inferred_knowledge
        )

        # Phase 3: Conflict Detection
        conflicts = self._detect_conflicts(forward_result, backward_result)

        # Phase 4: LLM Conflict Resolution
        resolved_assignments = self._resolve_conflicts_with_llm(
            conflicts,
            filtered_blocks,
            inferred_knowledge,
            forward_result,
            backward_result,
        )

        # Phase 5: Span Construction
        dst_spans = self._construct_spans_from_assignments(
            resolved_assignments, filtered_blocks, inferred_knowledge
        )

        # Phase 6: Temporal Ordering
        dst_spans = self._sort_spans_temporally(dst_spans)

        # Create statistics
        construction_stats = self._create_construction_statistics(
            total_blocks,
            total_steps,
            forward_result,
            backward_result,
            conflicts,
            dst_spans,
        )

        self.logger.info(
            f"✅ Bidirectional construction complete: {len(dst_spans)} final DST spans"
        )

        return BidirectionalSpanConstructionResult(
            dst_spans=dst_spans,
            total_blocks_processed=total_blocks,
            construction_statistics=construction_stats,
        )

    def _execute_forward_pass(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> DirectionalAssignment:
        """
        Execute forward pass: Start from first block, assign going forward

        Rules:
        - First block always assigned to first step (confidence = 1.0)
        - For each subsequent block, decide: stay in current step or transition to next?
        - Cannot skip steps or go backward
        """
        self.logger.debug("➡️ Executing forward pass")

        block_assignments = {}
        confidence_scores = {}

        current_step_idx = 0
        total_steps = len(inferred_knowledge)

        # First block always assigned to first step
        block_assignments[0] = {
            "step_index": 0,
            "confidence": 1.0,
            "source": "anchored",
        }
        confidence_scores[0] = 1.0

        # Process remaining blocks
        for block_idx in range(1, len(filtered_blocks)):
            current_block = filtered_blocks[block_idx]
            current_step = inferred_knowledge[current_step_idx]

            # Can stay in current step or move to next step (if available)
            possible_steps = []
            if current_step_idx < total_steps:
                possible_steps.append((current_step_idx, current_step))

            next_step_idx = current_step_idx + 1
            if next_step_idx < total_steps:
                possible_steps.append(
                    (next_step_idx, inferred_knowledge[next_step_idx])
                )

            # Score against possible steps using batch computation
            best_step_idx, best_confidence = self._score_block_against_steps(
                current_block, possible_steps
            )

            # Update current step if transitioning
            if best_step_idx != current_step_idx:
                current_step_idx = best_step_idx

            block_assignments[block_idx] = {
                "step_index": best_step_idx,
                "confidence": best_confidence,
                "source": "forward_pass",
            }
            confidence_scores[block_idx] = best_confidence

        return DirectionalAssignment(
            block_assignments=block_assignments,
            confidence_scores=confidence_scores,
        )

    def _execute_backward_pass(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> DirectionalAssignment:
        """
        Execute backward pass: Start from last block, assign going backward

        Rules:
        - Last block always assigned to last step (confidence = 1.0)
        - For each previous block, decide: stay in current step or transition to previous?
        - Cannot skip steps or go forward
        """
        self.logger.debug("⬅️ Executing backward pass")

        block_assignments = {}
        confidence_scores = {}

        current_step_idx = len(inferred_knowledge) - 1
        total_steps = len(inferred_knowledge)
        total_blocks = len(filtered_blocks)

        # Last block always assigned to last step
        last_block_idx = total_blocks - 1
        block_assignments[last_block_idx] = {
            "step_index": current_step_idx,
            "confidence": 1.0,
            "source": "anchored",
        }
        confidence_scores[last_block_idx] = 1.0

        # Process blocks from second-to-last backward
        for block_idx in range(total_blocks - 2, -1, -1):
            current_block = filtered_blocks[block_idx]
            current_step = inferred_knowledge[current_step_idx]

            # Can stay in current step or move to previous step (if available)
            possible_steps = []
            if current_step_idx >= 0:
                possible_steps.append((current_step_idx, current_step))

            prev_step_idx = current_step_idx - 1
            if prev_step_idx >= 0:
                possible_steps.append(
                    (prev_step_idx, inferred_knowledge[prev_step_idx])
                )

            # Score against possible steps using batch computation
            best_step_idx, best_confidence = self._score_block_against_steps(
                current_block, possible_steps
            )

            # Update current step if transitioning
            if best_step_idx != current_step_idx:
                current_step_idx = best_step_idx

            block_assignments[block_idx] = {
                "step_index": best_step_idx,
                "confidence": best_confidence,
                "source": "backward_pass",
            }
            confidence_scores[block_idx] = best_confidence

        return DirectionalAssignment(
            block_assignments=block_assignments,
            confidence_scores=confidence_scores,
        )

    def _score_block_against_steps(
        self, block: Dict[str, Any], possible_steps: List[Tuple[int, str]]
    ) -> Tuple[int, float]:
        """
        Score a block against multiple possible steps using GlobalSimilarityCalculator methods

        Args:
            block: Block dictionary
            possible_steps: List of (step_idx, step_text) tuples

        Returns:
            Tuple of (best_step_idx, best_confidence)
        """
        if not possible_steps:
            return 0, 0.0

        # Use GlobalSimilarityCalculator's score_blocks method directly
        mini_blocks = [block]
        mini_steps = [step_text for _, step_text in possible_steps]

        classification_result = self.similarity_calculator.score_blocks(
            mini_blocks, mini_steps
        )

        # The result will have one block (our single block) with confidence and scores
        if classification_result.clear_blocks:
            # Block was classified as clear
            block_result = classification_result.clear_blocks[0]
            # Find the step with highest combined score
            best_idx = np.argmax(block_result.combined_scores)
            # Use the actual combined score as confidence (more meaningful for 1-2 steps)
            best_confidence = float(block_result.combined_scores[best_idx])
        elif classification_result.ambiguous_blocks:
            # Block was classified as ambiguous
            block_result = classification_result.ambiguous_blocks[0]
            # Find the step with highest combined score
            best_idx = np.argmax(block_result.combined_scores)
            # Use the actual combined score as confidence
            best_confidence = float(block_result.combined_scores[best_idx])

        best_step_idx = possible_steps[best_idx][0]

        return best_step_idx, best_confidence

    def _detect_conflicts(
        self,
        forward_result: DirectionalAssignment,
        backward_result: DirectionalAssignment,
    ) -> List[int]:
        """
        Detect conflicts between forward and backward passes

        Only checks blocks that are NOT anchored (first and last blocks are guaranteed correct).
        A conflict occurs when:
        1. Forward and backward assign block to different steps, OR
        2. Both passes have low confidence (< threshold)
        """
        conflicts = []

        # Skip anchored assignments (first and last blocks are guaranteed correct)
        anchored_blocks = {0, len(forward_result.block_assignments) - 1}

        for block_idx in forward_result.block_assignments:
            # Skip anchored blocks - they are guaranteed to be correct
            if block_idx in anchored_blocks:
                continue

            forward_assignment = forward_result.block_assignments[block_idx]
            backward_assignment = backward_result.block_assignments[block_idx]

            # Skip if either assignment is anchored (shouldn't happen but safety check)
            if (
                forward_assignment.get("source") == "anchored"
                or backward_assignment.get("source") == "anchored"
            ):
                continue

            # Check for step disagreement
            step_conflict = (
                forward_assignment["step_index"] != backward_assignment["step_index"]
            )

            # Check for low confidence (both must be below threshold)
            confidence_conflict = (
                forward_assignment["confidence"] < self.high_confidence_threshold
                and backward_assignment["confidence"] < self.high_confidence_threshold
            )

            if step_conflict or confidence_conflict:
                conflicts.append(block_idx)

        self.logger.info(
            f"⚠️ Detected {len(conflicts)} conflicts in middle blocks: {conflicts}"
        )
        return conflicts

    def _resolve_conflicts_with_llm(
        self,
        conflicts: List[int],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
        forward_result: DirectionalAssignment,
        backward_result: DirectionalAssignment,
    ) -> Dict[int, Dict[str, Any]]:
        """
        Resolve bidirectional conflicts using LLM with full context

        For non-conflicting blocks, choose the directional assignment with highest confidence.
        For conflicts, use LLM to resolve with comprehensive bidirectional context.
        Anchored assignments are included as-is since they are guaranteed correct.
        """
        resolved_assignments = {}

        # Include anchored assignments first (they are guaranteed correct)
        anchored_blocks = {0, len(forward_result.block_assignments) - 1}
        for block_idx in anchored_blocks:
            if block_idx in forward_result.block_assignments:
                # Use forward assignment for anchored blocks (both should be the same)
                assignment = forward_result.block_assignments[block_idx].copy()
                resolved_assignments[block_idx] = assignment
                self.logger.info(
                    f"✅ Included anchored assignment for block {block_idx}: step {assignment['step_index']}, source={assignment['source']}, confidence={assignment['confidence']}"
                )

        # First, handle non-conflicting blocks with confidence-based selection
        correctly_assigned_blocks = {}
        for block_idx in forward_result.block_assignments:
            if block_idx not in conflicts:
                forward_assignment = forward_result.block_assignments[block_idx]
                backward_assignment = backward_result.block_assignments[block_idx]

                # Choose the assignment with higher confidence
                if (
                    forward_assignment["confidence"]
                    >= backward_assignment["confidence"]
                ):
                    resolved_assignments[block_idx] = forward_assignment.copy()
                    resolved_assignments[block_idx][
                        "source"
                    ] = "forward_higher_confidence"
                else:
                    resolved_assignments[block_idx] = backward_assignment.copy()
                    resolved_assignments[block_idx][
                        "source"
                    ] = "backward_higher_confidence"

                # Track correctly assigned blocks for LLM context
                correctly_assigned_blocks[block_idx] = resolved_assignments[block_idx]

        # For conflicts, prepare BidirectionalConflict objects and resolve with LLM
        if conflicts:
            bidirectional_conflicts = []
            for block_idx in conflicts:
                forward_assignment = forward_result.block_assignments[block_idx]
                backward_assignment = backward_result.block_assignments[block_idx]

                # Determine conflict reason
                step_conflict = (
                    forward_assignment["step_index"]
                    != backward_assignment["step_index"]
                )
                confidence_conflict = (
                    forward_assignment["confidence"] < self.high_confidence_threshold
                    and backward_assignment["confidence"]
                    < self.high_confidence_threshold
                )

                conflict_reason = (
                    "step_disagreement" if step_conflict else "low_confidence"
                )

                conflict = BidirectionalConflict(
                    block_id=block_idx,
                    block_data=filtered_blocks[block_idx],
                    forward_assignment=forward_assignment,
                    backward_assignment=backward_assignment,
                    conflict_reason=conflict_reason,
                )
                bidirectional_conflicts.append(conflict)

            # Resolve conflicts using LLM
            llm_result = self.llm_handler.resolve_bidirectional_conflicts(
                bidirectional_conflicts,
                filtered_blocks,
                inferred_knowledge,
                correctly_assigned_blocks,
            )

            # Apply LLM decisions to resolved assignments
            for decision in llm_result.decisions:
                if decision.chosen_step_index >= 0:
                    resolved_assignments[decision.block_id] = {
                        "step_index": decision.chosen_step_index,
                        "confidence": decision.confidence,
                        "source": "llm_resolution",
                        "reasoning": decision.reasoning,
                    }
                else:
                    # Fallback: use higher confidence assignment
                    forward_assignment = forward_result.block_assignments[
                        decision.block_id
                    ]
                    backward_assignment = backward_result.block_assignments[
                        decision.block_id
                    ]

                    if (
                        forward_assignment["confidence"]
                        >= backward_assignment["confidence"]
                    ):
                        resolved_assignments[decision.block_id] = (
                            forward_assignment.copy()
                        )
                        resolved_assignments[decision.block_id][
                            "source"
                        ] = "llm_fallback_forward"
                    else:
                        resolved_assignments[decision.block_id] = (
                            backward_assignment.copy()
                        )
                        resolved_assignments[decision.block_id][
                            "source"
                        ] = "llm_fallback_backward"

            self.logger.info(
                "🤖 LLM resolved %d conflicts: %d successful, %d fallback",
                len(conflicts),
                llm_result.success_count,
                llm_result.failure_count,
            )

        return resolved_assignments

    def _construct_spans_from_assignments(
        self,
        resolved_assignments: Dict[int, Dict[str, Any]],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Construct DST spans from resolved block assignments

        Group blocks by assigned step and create spans
        """
        step_blocks = {}  # step_index -> list of block indices

        for block_idx, assignment in resolved_assignments.items():
            step_idx = assignment["step_index"]
            if step_idx not in step_blocks:
                step_blocks[step_idx] = []
            step_blocks[step_idx].append(block_idx)

        dst_spans = []

        for step_idx in sorted(step_blocks.keys()):
            block_indices = step_blocks[step_idx]
            step_blocks_data = [filtered_blocks[idx] for idx in block_indices]

            # Calculate span boundaries
            start_times = [
                self._extract_start_time(block) for block in step_blocks_data
            ]
            end_times = [self._extract_end_time(block) for block in step_blocks_data]

            # Validate and fix block-level timestamps before aggregation
            validated_times = []
            for i, (start_time, end_time) in enumerate(zip(start_times, end_times)):
                if start_time is None or end_time is None:
                    self.logger.warning(
                        f"Block {block_indices[i]} has missing timestamps, skipping"
                    )
                    continue

                # Validate block timestamps
                start_time, end_time = self._validate_and_fix_timestamps(
                    start_time, end_time
                )
                start_time, end_time = self._validate_and_fix_span_duration(
                    start_time, end_time
                )

                validated_times.append((start_time, end_time))

            if not validated_times:
                self.logger.warning(
                    f"No valid blocks for step {step_idx + 1}, skipping span"
                )
                continue

            start_times, end_times = zip(*validated_times)

            # Sort blocks by start time to ensure proper temporal ordering
            block_times = sorted(zip(start_times, end_times))
            sorted_starts, sorted_ends = zip(*block_times)

            span_start = sorted_starts[0]  # Earliest start time
            span_end = sorted_ends[-1]  # Latest end time

            # CRITICAL VALIDATION: Ensure start < end before any processing
            if span_start >= span_end:
                self.logger.error(
                    f"❌ CRITICAL: Span {step_idx + 1} has invalid timestamps from block aggregation "
                    f"(start={span_start:.2f} >= end={span_end:.2f}). "
                    f"This indicates fundamental data corruption in block timestamps. "
                    f"Block times: starts={sorted_starts}, ends={sorted_ends}. "
                    f"Skipping this corrupted span."
                )
                # Skip this corrupted span rather than creating invalid data
                continue

            # Validate and fix span boundaries (should not change start >= end case)
            span_start, span_end = self._validate_and_fix_timestamps(
                span_start, span_end
            )
            span_start, span_end = self._validate_and_fix_span_duration(
                span_start, span_end
            )

            # Calculate average confidence
            confidences = [
                resolved_assignments[block_idx]["confidence"]
                for block_idx in block_indices
            ]
            avg_confidence = sum(confidences) / len(confidences)

            # Create span
            span = self._create_common_span_structure(
                span_id=step_idx + 1,
                name=inferred_knowledge[step_idx],
                start_time=span_start,
                end_time=span_end,
                confidence=avg_confidence,
                source="bidirectional",
                source_blocks=len(block_indices),
                block_indices=block_indices,
            )

            dst_spans.append(span)

        return dst_spans

    def _create_construction_statistics(
        self,
        total_blocks: int,
        total_steps: int,
        forward_result: DirectionalAssignment,
        backward_result: DirectionalAssignment,
        conflicts: List[int],
        dst_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Create comprehensive construction statistics"""

        return {
            "construction_type": "bidirectional_span_constructor",
            "total_blocks_input": total_blocks,
            "total_steps": total_steps,
            "total_spans_created": len(dst_spans),
            "forward_pass_assignments": len(forward_result.block_assignments),
            "backward_pass_assignments": len(backward_result.block_assignments),
            "conflicts_detected": len(conflicts),
            "conflict_resolution_method": "heuristic",  # For POC
            "high_confidence_threshold": self.high_confidence_threshold,
            "semantic_weight": self.semantic_weight,
            "nli_weight": self.nli_weight,
        }
