"""Data classes for DST generation outputs"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class DSTOutput:
    """Complete DST output structure"""

    video_uid: str
    inferred_goal: str
    inferred_knowledge: str
    all_step_descriptions: str
    dst: Dict[str, Any]
    dialog: List[Any]
    state_rules: Dict[str, str]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "video_uid": self.video_uid,
            "inferred_goal": self.inferred_goal,
            "inferred_knowledge": self.inferred_knowledge,
            "all_step_descriptions": self.all_step_descriptions,
            "dst": self.dst,
            "dialog": self.dialog,
            "state_rules": self.state_rules,
            "metadata": self.metadata,
        }

    @classmethod
    def from_data_and_dst(
        cls,
        data: Dict[str, Any],
        dst_structure: Dict[str, Any],
        generator_name: str,
        generation_date: str = None,
    ) -> "DSTOutput":
        """Construct DSTOutput from raw input data and a DST structure.

        This centralizes the output assembly logic so callers (like SimpleDSTGenerator)
        don't need to duplicate metadata and counting logic.
        """
        parsed_anns = data.get("parsed_video_anns", {})
        all_step_descriptions = parsed_anns.get("all_step_descriptions", "")

        if generation_date is None:
            # Use a simple placeholder — callers may pass a timestamp if desired.
            generation_date = "2025-10-20"

        counts = {
            "num_steps": len(dst_structure.get("steps", [])),
            "num_substeps": sum(
                len(step.get("substeps", [])) for step in dst_structure.get("steps", [])
            ),
            "num_actions": sum(
                len(substep.get("actions", []))
                for step in dst_structure.get("steps", [])
                for substep in step.get("substeps", [])
            ),
        }

        return cls(
            video_uid=data.get("video_uid", ""),
            inferred_goal=data.get("inferred_goal", ""),
            inferred_knowledge=data.get("inferred_knowledge", ""),
            all_step_descriptions=all_step_descriptions,
            dst=dst_structure,
            dialog=data.get("dialog", []),
            state_rules={
                "definition": (
                    "For any time τ: C if end_ts ≤ τ; IP if start_ts ≤ τ < end_ts; "
                    "NS otherwise. Substep state aggregates from actions; Step state "
                    "aggregates from substeps with the same rule."
                )
            },
            metadata={
                "source": "inferred_knowledge + parsed_video_anns.all_step_descriptions",
                "generator": generator_name,
                "generation_date": generation_date,
                "counts": counts,
            },
        )
