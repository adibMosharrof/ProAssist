"""
Hybrid DST Label Generator

This module provides a hybrid DST label generation that combines
overlap-aware block reduction with bidirectional span construction
(forward/backward passes + conflict resolution) for robust DST span creation.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator
from dst_data_builder.hybrid_dst.overlap_aware_reducer import OverlapAwareBlockReducer
from dst_data_builder.hybrid_dst.span_constructors import (
    SimpleSpanConstructor,
    HybridSpanConstructor,
    BidirectionalSpanConstructor,
    LLMSpanConstructor,
)
from dst_data_builder.hybrid_dst.temporal_validator import TemporalOrderingValidator
from dst_data_builder.hybrid_dst.utils import (
    parse_blocks,
    extract_steps,
    convert_spans_to_rows,
    create_audit_log,
)


class HybridDSTLabelGenerator(BaseGPTGenerator):
    """
    Hybrid DST Label Generator

    Orchestrates three-phase hybrid DST processing:
    1. Overlap-aware block reduction
    2. Bidirectional span construction (forward/backward passes + conflict resolution)
    3. Temporal validation
    """

    def __init__(
        self,
        generator_type: str = "hybrid_dst",
        model_name: str = "gpt-4o",
        temperature: float = 0.1,
        max_tokens: int = 4000,
        max_retries: int = 1,
        generator_cfg: Optional[Dict[str, Any]] = None,
        model_cfg: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            generator_type=generator_type,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
        )
        self.logger = logging.getLogger(self.__class__.__name__)
        self.generator_type = generator_type

        # Create model config if not provided
        if model_cfg is None:
            model_cfg = {
                "name": model_name,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

        self._initialize_components(generator_cfg or {}, model_cfg)

    def _initialize_components(
        self, config: Dict[str, Any], model_cfg: Dict[str, Any]
    ) -> None:
        """Initialize hybrid DST components using provided config"""
        self.overlap_reducer = OverlapAwareBlockReducer(config)
        self.simple_span_constructor = SimpleSpanConstructor(config)
        # self.hybrid_span_constructor = BidirectionalSpanConstructor(config, model_cfg)
        self.hybrid_span_constructor = LLMSpanConstructor(config, model_cfg)
        self.temporal_validator = TemporalOrderingValidator(config)

    async def _try_generate_and_validate(
        self,
        inferred_knowledge: str,
        all_step_descriptions: str,
        previous_failure_reason: str = "",
        dst_output_dir: Optional[Any] = None,
    ) -> Tuple[bool, Any, str, Any]:
        """Generate DST labels using hybrid approach"""
        try:
            result = self._generate_dst_from_input_data(
                {
                    "inferred_knowledge": inferred_knowledge,
                    "all_step_descriptions": all_step_descriptions,
                }
            )
            return (True, result, "", None)
        except Exception as e:
            self.logger.exception("Hybrid DST generation failed: %s", e)
            return (False, None, str(e), None)

    def _generate_dst_from_input_data(
        self, input_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Execute three-phase hybrid DST processing"""
        inferred_knowledge = input_data.get("inferred_knowledge", "")
        all_step_descriptions = input_data.get("all_step_descriptions", "")

        self.logger.debug("Starting hybrid DST processing")

        # Parse input
        blocks = parse_blocks(all_step_descriptions)
        steps = extract_steps(inferred_knowledge)

        if not blocks or not steps:
            raise ValueError("No blocks or steps found in input data")

        # Phase 1: Block reduction
        reduction_result = self.overlap_reducer.reduce_blocks(blocks)
        filtered_blocks = reduction_result.filtered_blocks

        # Phase 2: Decision Tree - Simple Span vs Hybrid Span Constructor
        if len(filtered_blocks) == len(steps):
            self.logger.debug(
                "🔄 Using Simple Span Constructor (equal counts: %d == %d)",
                len(filtered_blocks),
                len(steps),
            )
            construction_result = self.simple_span_constructor.construct_spans(
                filtered_blocks, steps
            )
        else:
            self.logger.debug(
                "🔄 Using Bidirectional Span Constructor (unequal counts: %d != %d)",
                len(filtered_blocks),
                len(steps),
            )
            construction_result = self.hybrid_span_constructor.construct_spans(
                filtered_blocks, steps
            )
        spans = construction_result.dst_spans

        # Phase 3: Temporal validation
        validation_result = self.temporal_validator.validate_temporal_ordering(spans)

        # Check for temporal violations that should cause failure
        error_violations = [v for v in validation_result.violations if v.severity == "error"]
        if error_violations:
            error_messages = [v.description for v in error_violations[:3]]  # Show first 3 errors
            raise ValueError(
                f"Temporal validation failed with {len(error_violations)} errors: "
                f"{'; '.join(error_messages)}"
                f"{f' (+{len(error_violations) - 3} more)' if len(error_violations) > 3 else ''}"
            )

        validated_spans = validation_result.sorted_spans

        # Phase 4: Post-validation span integrity check
        validated_spans = self._validate_span_integrity(validated_spans)

        # Log statistics
        self._log_statistics(
            reduction_result, construction_result, validation_result, validated_spans
        )

        # Convert to output format
        rows = convert_spans_to_rows(validated_spans)

        # Store audit log
        self._audit_log = create_audit_log(
            reduction_result, construction_result, validation_result
        )

        return rows

    def _log_statistics(
        self,
        reduction_result: Any,
        construction_result: Any,
        validation_result: Any,
        final_spans: List[Dict[str, Any]],
    ) -> None:
        """Log processing statistics"""
        self.logger.debug(
            f"Block reduction: {reduction_result.original_count} -> {len(reduction_result.filtered_blocks)}"
        )
        self.logger.debug(
            f"Span construction: {len(construction_result.dst_spans)} spans"
        )
        self.logger.debug(
            f"Temporal validation: {len(validation_result.sorted_spans)} validated spans"
        )
        self.logger.debug(f"Span integrity check: {len(final_spans)} final spans")

    def _validate_span_integrity(
        self, spans: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Validate and fix span temporal integrity

        Ensures start_ts <= end_ts for all spans, fixing any invalid spans.
        """
        fixed_spans = []

        for span in spans:
            start_ts = span.get("start_ts", 0)
            end_ts = span.get("end_ts", 0)
            span_id = span.get("id", "unknown")

            if start_ts > end_ts:
                raise ValueError(
                    f"Invalid span {span_id}: start_ts ({start_ts}) > end_ts ({end_ts}). "
                    f"Data corruption detected in span construction. "
                    f"Span data: {span}"
                )

            fixed_spans.append(span)

        return fixed_spans

    async def _execute_generation_round(
        self,
        items: List[Tuple[str, str, str]],
        attempt_idx: int,
        failure_reasons: Dict[str, str],
    ) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[str, str, str, str, Any]]]:
        """Execute generation for items"""
        successes: Dict[str, Any] = {}
        failures: List[Tuple[str, str, str, str, Any]] = []

        for input_file, inferred_knowledge, all_step_descriptions in items:
            try:
                input_data = {
                    "inferred_knowledge": inferred_knowledge,
                    "all_step_descriptions": all_step_descriptions,
                }

                dst_rows = self._generate_dst_from_input_data(input_data)

                if dst_rows:
                    dst_structure = self._convert_to_dst_structure(dst_rows)
                    dst_structure["audit_log"] = self._audit_log
                    successes[input_file] = dst_structure
                else:
                    reason = "No spans generated"
                    failures.append(
                        (
                            input_file,
                            inferred_knowledge,
                            all_step_descriptions,
                            reason,
                            None,
                        )
                    )

            except Exception as e:
                reason = f"Exception: {str(e)}"
                failures.append(
                    (
                        input_file,
                        inferred_knowledge,
                        all_step_descriptions,
                        reason,
                        None,
                    )
                )

        return successes, failures

    def _convert_to_dst_structure(
        self, tsv_rows: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Convert TSV rows to DST structure format"""
        step_spans = []

        for row in tsv_rows:
            if row.get("type") == "step":
                step_id = int(row.get("id", "0").replace("S", ""))
                step_spans.append(
                    {
                        "id": step_id,
                        "name": row.get("name", ""),
                        "start_ts": row.get("start_ts", 0.0),
                        "end_ts": row.get("end_ts", 0.0),
                        "conf": 1.0,
                    }
                )

        # Sort by start time and reassign incremental IDs
        step_spans.sort(key=lambda x: x["t0"])
        for i, span in enumerate(step_spans, 1):
            span["id"] = i

        return {
            "video_uid": "",
            "steps": step_spans,
        }

    def _ensure_models_loaded(self):
        """
        Ensure ML models are loaded for processing (used by DSTDataProcessor)

        This method is called by the DSTDataProcessor to ensure models are loaded
        before processing videos, especially in multiprocessing scenarios.
        """
        self.logger.debug("🔧 Ensuring ML models are loaded for hybrid DST generation")

        # Ensure models are loaded in the hybrid span constructor (only needed for complex cases)
        if hasattr(self.hybrid_span_constructor, "similarity_calculator"):
            self.hybrid_span_constructor.similarity_calculator._ensure_models_loaded()

        self.logger.debug("✅ ML models loaded successfully")
