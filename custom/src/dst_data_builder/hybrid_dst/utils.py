"""
Hybrid DST Utilities

Utility functions for parsing, extraction, and conversion operations.
"""

import re
from typing import List, Dict, Any


def parse_blocks(all_step_descriptions: str) -> List[Dict[str, Any]]:
    """Parse raw annotation lines into normalized blocks"""
    lines = all_step_descriptions.split("\n")
    blocks = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse timestamp pattern: [start-end] or [start] (with 's' suffix)
        # Handles: [94.4s-105.2s] or [111.0s]
        match = re.search(r"\[(\d+\.?\d*)s?(?:-(\d+\.?\d*)s?)?\]", line)
        if not match:
            continue

        start_time = float(match.group(1))
        end_time = float(match.group(2)) if match.group(2) else None

        # Extract text (remove only the timestamp bracket we just matched)
        text = line[match.end() :].strip()

        # Remove leading dash/bullet if present (for sub-actions)
        text = re.sub(r"^[-•]\s*", "", text).strip()

        if not text:
            continue

        blocks.append(
            {
                "text": text,
                "start_time": start_time,
                "end_time": end_time,  # None for point blocks (single timestamp)
                "has_end_time": end_time is not None,
                "original_line": line,
            }
        )

    return blocks


def extract_steps(inferred_knowledge: str) -> List[str]:
    """Extract high-level step names from inferred knowledge"""
    lines = inferred_knowledge.split("\n")
    steps = []

    for line in lines:
        # Match numbered steps (e.g., "1. Step 1", "2. Assemble components")
        match = re.match(r"^\d+\.\s+(.+)$", line.strip())
        if match:
            step_text = match.group(1).strip()
            if step_text:
                steps.append(step_text)

    return steps


def convert_spans_to_rows(step_spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert step spans to TSV row format for compatibility"""
    rows = []
    for span in step_spans:
        rows.append(
            {
                "type": "step",
                "id": f'S{span["id"]}',
                "start_ts": span["start_ts"],
                "end_ts": span["end_ts"],
                "name": span["name"],
            }
        )
    return rows


def create_audit_log(
    reduction_result: Any, construction_result: Any, validation_result: Any
) -> Dict[str, Any]:
    """Create detailed audit log for debugging and analysis"""
    return {
        "hybrid_dst_algorithm": {
            "phases_completed": [
                "overlap_aware_block_reduction",
                "hybrid_span_construction",
                "temporal_ordering_validation",
            ],
            "processing_statistics": {
                "block_reduction": {
                    "original_blocks": reduction_result.original_count,
                    "filtered_blocks": len(reduction_result.filtered_blocks),
                    "merged_blocks": reduction_result.merged_count,
                },
                "span_construction": construction_result.construction_statistics,
                "temporal_validation": {
                    "is_valid": validation_result.is_valid,
                    "violations_count": validation_result.violation_count,
                    "sorted_spans": validation_result.span_count,
                },
            },
            "notes": "Hybrid DST label generation using global similarity + LLM fallback",
        }
    }
