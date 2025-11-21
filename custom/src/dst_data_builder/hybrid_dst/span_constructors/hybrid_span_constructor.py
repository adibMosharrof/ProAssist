"""
Hybrid Span Constructor Module

This module implements the hybrid DST span construction orchestrator that coordinates
the two-phase boundary detection process, combining global similarity scoring with
LLM fallback for ambiguous cases to create final DST spans.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from omegaconf import DictConfig
import numpy as np

# Import hybrid DST components
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
    AmbiguousBlock,
    LLMDecision,
    LLMHandlingResult,
)


@dataclass
class SpanConstructionResult:
    """Result of hybrid span construction"""

    dst_spans: List[Dict[str, Any]]
    clear_blocks_used: int
    ambiguous_blocks_resolved: int
    total_blocks_processed: int
    construction_statistics: Dict[str, Any]


class HybridSpanConstructor(BaseSpanConstructor):
    """
    Hybrid DST Span Construction: Two-phase boundary detection

    This class orchestrates the hybrid approach that uses global similarity scoring
    for high-confidence cases and LLM fallback for ambiguous cases, then constructs
    final DST spans with proper temporal boundaries.
    """

    def __init__(self, hybrid_config: DictConfig, model_config: DictConfig):
        # Initialize base class with span construction config
        span_config = hybrid_config.span_construction
        super().__init__(span_config)

        self.hybrid_config = hybrid_config
        self.model_config = model_config

        # Initialize hybrid DST components
        self.similarity_calculator = GlobalSimilarityCalculator(
            hybrid_config.similarity
        )
        self.llm_handler = LLMAmbiguousHandler(model_config)

    def construct_spans(
        self, filtered_blocks: List[Dict[str, Any]], inferred_knowledge: List[str]
    ) -> SpanConstructionResult:
        """
        Construct DST spans using hybrid two-phase approach

        Args:
            filtered_blocks: Blocks from overlap-aware reduction phase
            inferred_knowledge: Step descriptions for context

        Returns:
            SpanConstructionResult with final DST spans and statistics
        """
        self._validate_construct_spans_inputs(filtered_blocks, inferred_knowledge)

        total_blocks = len(filtered_blocks)
        self.logger.info(
            "🔍 Processing %d filtered blocks for DST span construction", total_blocks
        )

        # Phase 1: High-Confidence Global Similarity Scoring
        classification_result = self.similarity_calculator.score_blocks(
            filtered_blocks, inferred_knowledge
        )

        clear_count = len(classification_result.clear_blocks)
        ambiguous_count = len(classification_result.ambiguous_blocks)

        self.logger.info(
            "📊 Matrix scoring: %d/%d clear blocks (%.1f%%), %d/%d ambiguous blocks (%.1f%%)",
            clear_count,
            total_blocks,
            (clear_count / total_blocks) * 100,
            ambiguous_count,
            total_blocks,
            (ambiguous_count / total_blocks) * 100,
        )

        # Phase 2: LLM Fallback for Ambiguous Cases
        llm_decisions = []
        if classification_result.ambiguous_blocks:
            self.logger.info(
                "🤖 Triggering LLM fallback for %d ambiguous blocks", ambiguous_count
            )

            # Convert ambiguous blocks for LLM handling
            ambiguous_block_objects = self._convert_to_ambiguous_blocks(
                classification_result.ambiguous_blocks,
                filtered_blocks,
                inferred_knowledge,
            )

            # Get LLM decisions
            llm_result = self.llm_handler.resolve_ambiguous_blocks(
                ambiguous_block_objects, inferred_knowledge
            )
            llm_decisions = llm_result.decisions
            self.logger.info(
                "✅ LLM resolved %d ambiguous blocks into %d successful decisions",
                ambiguous_count,
                llm_result.success_count,
            )
        else:
            self.logger.info("✅ No ambiguous blocks found - using matrix scoring only")

        # Phase 3: Combine decisions and construct final spans
        final_spans = self._combine_decisions_and_construct_spans(
            classification_result.clear_blocks,
            classification_result.ambiguous_blocks,
            llm_decisions,
            filtered_blocks,
            inferred_knowledge,
        )

        # Log final results
        self.logger.info(
            "🎯 Constructed %d final DST spans from %d processed blocks",
            len(final_spans),
            total_blocks,
        )

        # Create statistics
        construction_stats = self._create_construction_statistics(
            classification_result, llm_decisions, final_spans
        )

        return SpanConstructionResult(
            dst_spans=final_spans,
            clear_blocks_used=clear_count,
            ambiguous_blocks_resolved=len(llm_decisions),
            total_blocks_processed=total_blocks,
            construction_statistics=construction_stats,
        )

    def _convert_to_ambiguous_blocks(
        self,
        ambiguous_results: List[SimilarityResult],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
    ) -> List[AmbiguousBlock]:
        """
        Convert SimilarityResult objects to AmbiguousBlock objects for LLM handling

        Args:
            ambiguous_results: Results from similarity calculator
            filtered_blocks: Original block data
            inferred_knowledge: Step descriptions

        Returns:
            List of AmbiguousBlock objects
        """
        ambiguous_blocks = []

        for result in ambiguous_results:
            if result.block_id < len(filtered_blocks):
                block_data = filtered_blocks[result.block_id]

                # Get top alternatives (highest scoring steps)
                scores = result.combined_scores
                top_alternatives = []

                # Sort by score and get top 3 alternatives
                score_indices = list(enumerate(scores))
                score_indices.sort(key=lambda x: x[1], reverse=True)

                for step_idx, score in score_indices[:3]:
                    if step_idx < len(inferred_knowledge):
                        top_alternatives.append((step_idx, score))

                ambiguous_block = AmbiguousBlock(
                    block_id=result.block_id,
                    block_data=block_data,
                    similarity_scores=scores,
                    confidence=result.confidence,
                    top_alternatives=top_alternatives,
                )
                ambiguous_blocks.append(ambiguous_block)

        return ambiguous_blocks

    def _combine_decisions_and_construct_spans(
        self,
        clear_results: List[SimilarityResult],
        ambiguous_results: List[SimilarityResult],
        llm_decisions: List[LLMDecision],
        filtered_blocks: List[Dict[str, Any]],
        inferred_knowledge: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Combine decisions from both phases and construct final DST spans

        Args:
            clear_results: Clear block results from similarity calculator
            ambiguous_results: Ambiguous block results
            llm_decisions: LLM decisions for ambiguous blocks
            filtered_blocks: Original block data
            inferred_knowledge: Step descriptions

        Returns:
            List of final DST spans
        """
        # Create a mapping of block_id to assigned step
        block_assignments = {}

        # Process clear blocks (use best match from similarity scores)
        for result in clear_results:
            if result.block_id < len(filtered_blocks) and result.combined_scores:
                best_step_idx = int(np.argmax(result.combined_scores))
                confidence = result.confidence
                block_assignments[result.block_id] = {
                    "step_index": best_step_idx,
                    "confidence": confidence,
                    "source": "similarity",
                    "scores": result.combined_scores,
                }

        # Process ambiguous blocks (use LLM decisions)
        for decision in llm_decisions:
            if decision.block_id in block_assignments:
                # Update with LLM decision if successful
                if decision.chosen_step_index >= 0:
                    block_assignments[decision.block_id] = {
                        "step_index": decision.chosen_step_index,
                        "confidence": decision.confidence,
                        "source": "llm",
                        "reasoning": decision.reasoning,
                    }

        # Validate temporal consistency of block assignments
        block_assignments = self._validate_temporal_consistency(
            block_assignments, filtered_blocks
        )

        # Group blocks by assigned step
        step_blocks = self._group_blocks_by_step(block_assignments, filtered_blocks)

        # Construct final spans from grouped blocks
        final_spans = self._construct_spans_from_groups(step_blocks, inferred_knowledge)

        return final_spans

    def _group_blocks_by_step(
        self,
        block_assignments: Dict[int, Dict[str, Any]],
        filtered_blocks: List[Dict[str, Any]],
    ) -> Dict[int, List[Dict[str, Any]]]:
        """
        Group blocks by their assigned step index

        Args:
            block_assignments: Mapping of block_id to step assignment
            filtered_blocks: Original block data

        Returns:
            Dictionary mapping step_index to list of blocks
        """
        step_groups = {}

        for block_id, assignment in block_assignments.items():
            step_index = assignment["step_index"]

            if step_index not in step_groups:
                step_groups[step_index] = []

            if block_id < len(filtered_blocks):
                block = filtered_blocks[block_id].copy()
                block["assignment_info"] = assignment
                step_groups[step_index].append(block)

        return step_groups

    def _construct_spans_from_groups(
        self,
        step_groups: Dict[int, List[Dict[str, Any]]],
        inferred_knowledge: List[str],
    ) -> List[Dict[str, Any]]:
        """
        Construct final DST spans from grouped blocks

        Args:
            step_groups: Blocks grouped by step index
            inferred_knowledge: Step descriptions

        Returns:
            List of final DST spans
        """
        final_spans = []

        for step_index, blocks in step_groups.items():
            if step_index < len(inferred_knowledge):
                # Sort blocks by start time
                sorted_blocks = sorted(
                    blocks, key=lambda x: self._extract_start_time(x)
                )

                # Calculate overall span boundaries from valid block timestamps
                valid_times = []
                for block in sorted_blocks:
                    start_time = self._extract_start_time(block)
                    end_time = self._extract_end_time(block)

                    # Only use blocks with valid timestamps
                    if (
                        start_time is not None
                        and end_time is not None
                        and start_time < end_time
                    ):
                        valid_times.append((start_time, end_time))

                if valid_times:
                    # Calculate span boundaries from valid block ranges
                    all_starts = [start for start, end in valid_times]
                    all_ends = [end for start, end in valid_times]
                    span_start = min(all_starts)
                    span_end = max(all_ends)

                    # Validate span duration
                    duration = span_end - span_start
                    if (
                        duration >= self.min_span_duration
                        and duration <= self.max_span_duration
                    ):
                        # Calculate average confidence
                        confidences = [
                            block["assignment_info"]["confidence"]
                            for block in blocks
                            if "assignment_info" in block
                        ]
                        avg_confidence = (
                            sum(confidences) / len(confidences) if confidences else 0.5
                        )

                        # CRITICAL VALIDATION: Ensure start < end before any processing
                        if span_start >= span_end:
                            self.logger.error(
                                f"❌ CRITICAL: Step {step_index + 1} span has invalid timestamps from block aggregation "
                                f"(start={span_start:.2f} >= end={span_end:.2f}). "
                                f"This indicates fundamental data corruption in block timestamps. "
                                f"Block times: starts={all_starts}, ends={all_ends}. "
                                f"Skipping this corrupted span."
                            )
                            continue

                        # Validate and fix span duration and timestamps
                        span_start, span_end = self._validate_and_fix_span_duration(
                            span_start, span_end, step_index + 1
                        )
                        span_start, span_end = self._validate_and_fix_timestamps(
                            span_start, span_end, step_index + 1
                        )

                        # Create span using base class method
                        span = self._create_common_span_structure(
                            span_id=step_index + 1,
                            name=inferred_knowledge[step_index],
                            start_time=span_start,
                            end_time=span_end,
                            confidence=avg_confidence,
                            source="hybrid",
                            source_blocks=len(blocks),
                        )
                        final_spans.append(span)
                    else:
                        self.logger.warning(
                            "Step %d span duration %.2fs is outside allowed range [%.1f, %.1f]",
                            step_index + 1,
                            duration,
                            self.min_span_duration,
                            self.max_span_duration,
                        )

        # Sort spans by start time and reassign sequential IDs
        final_spans = self._sort_spans_temporally(final_spans)

        return final_spans

    def _validate_temporal_consistency(
        self,
        block_assignments: Dict[int, Dict[str, Any]],
        filtered_blocks: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Validate individual blocks for temporal validity

        Since block merging should have already ensured valid timestamps,
        this mainly validates that blocks have proper start/end times.

        Args:
            block_assignments: Current block-to-step assignments
            filtered_blocks: Original block data

        Returns:
            Validated block assignments with invalid blocks removed
        """
        validated_assignments = {}

        for block_id, assignment in block_assignments.items():
            if block_id < len(filtered_blocks):
                block = filtered_blocks[block_id]
                start_time = self._extract_start_time(block)
                end_time = self._extract_end_time(block)

                # Check if block has valid timestamps
                if (
                    start_time is not None
                    and end_time is not None
                    and start_time < end_time
                ):
                    validated_assignments[block_id] = assignment
                else:
                    self.logger.debug(
                        f"Rejecting block {block_id} for step {assignment['step_index'] + 1}: "
                        f"invalid timestamps (start={start_time}, end={end_time})"
                    )

        filtered_count = len(block_assignments) - len(validated_assignments)
        if filtered_count > 0:
            self.logger.info(
                f"Temporal validation filtered {filtered_count} blocks with invalid timestamps: "
                f"{len(block_assignments)} -> {len(validated_assignments)} assignments"
            )

        return validated_assignments

    def _create_construction_statistics(
        self,
        classification_result: ClassificationResult,
        llm_decisions: List[LLMDecision],
        final_spans: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Create comprehensive statistics about the construction process

        Args:
            classification_result: Results from similarity calculator
            llm_decisions: LLM decisions
            final_spans: Final constructed spans

        Returns:
            Dictionary with construction statistics
        """
        # Get similarity statistics
        similarity_stats = self.similarity_calculator.get_similarity_statistics(
            classification_result
        )

        # Get LLM statistics if available
        llm_stats = {}
        if llm_decisions:
            # Create dummy LLMHandlingResult for statistics
            llm_result = LLMHandlingResult(
                decisions=llm_decisions,
                total_llm_calls=len(llm_decisions),
                success_count=len(
                    [d for d in llm_decisions if d.chosen_step_index >= 0]
                ),
                failure_count=len(
                    [d for d in llm_decisions if d.chosen_step_index < 0]
                ),
                total_cost_estimate=0.0,
            )
            llm_stats = self.llm_handler.get_handling_statistics(llm_result)

        # Calculate span statistics
        span_stats = self._calculate_span_statistics(final_spans)

        construction_stats = {
            "similarity_phase": similarity_stats,
            "llm_phase": llm_stats,
            "span_construction": span_stats,
            "hybrid_processing": {
                "total_blocks_input": classification_result.total_blocks,
                "clear_blocks": classification_result.clear_count,
                "ambiguous_blocks": classification_result.ambiguous_count,
                "llm_decisions_made": len(llm_decisions),
                "final_spans_created": len(final_spans),
            },
        }

        return construction_stats

    def _calculate_span_statistics(self, spans: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate statistics about the final spans

        Args:
            spans: Final constructed spans

        Returns:
            Dictionary with span statistics
        """
        if not spans:
            return {"error": "No spans to analyze"}

        # Get common temporal statistics from base class
        temporal_stats = self._calculate_temporal_statistics(spans)

        # Add confidence statistics specific to hybrid constructor
        confidences = [span["conf"] for span in spans]

        return {
            "total_spans": len(spans),
            "duration_stats": temporal_stats["duration_stats"],
            "confidence_stats": {
                "mean": float(np.mean(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences)),
                "std": float(np.std(confidences)),
            },
            "temporal_range": temporal_stats["temporal_range"],
        }
