from typing import Any, Dict, Tuple
from dst_data_builder.validators.base_validator import BaseValidator


class StructureValidator(BaseValidator):
    """Validate basic DST structure fields and types."""
    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Top-level validation entrypoint.

        Breaks the original monolithic validate into small helpers that each
        validate a single level of the DST tree. This keeps error messages
        precise and makes testing easier.
        """
        if not isinstance(dst_structure, dict):
            return False, "DST must be a dict"

        if "steps" not in dst_structure:
            return False, "Missing 'steps'"

        steps = dst_structure["steps"]
        return self._validate_steps(steps)

    def _validate_steps(self, steps: Any) -> Tuple[bool, str]:
        if not isinstance(steps, list):
            return False, "'steps' must be a list"

        for i, step in enumerate(steps):
            ok, msg = self._validate_step(i, step)
            if not ok:
                return False, msg

        return True, ""

    def _validate_step(self, i: int, step: Any) -> Tuple[bool, str]:
        if not isinstance(step, dict):
            return False, f"Step {i} must be a dict"

        if "step_id" not in step or "name" not in step:
            return False, f"Step {i} missing 'step_id' or 'name'"

        substeps = step.get("substeps", [])
        if substeps is None:
            return True, ""

        return self._validate_substeps(i, substeps)

    def _validate_substeps(self, i: int, substeps: Any) -> Tuple[bool, str]:
        if not isinstance(substeps, list):
            return False, f"Step {i} 'substeps' must be a list"

        for j, sub in enumerate(substeps):
            ok, msg = self._validate_substep(i, j, sub)
            if not ok:
                return False, msg

        return True, ""

    def _validate_substep(self, i: int, j: int, sub: Any) -> Tuple[bool, str]:
        if not isinstance(sub, dict):
            return False, f"Substep {i}.{j} must be a dict"

        if "sub_id" not in sub or "name" not in sub:
            return False, f"Substep {i}.{j} missing 'sub_id' or 'name'"

        actions = sub.get("actions", [])
        if actions is None:
            return True, ""

        return self._validate_actions(i, j, actions)

    def _validate_actions(self, i: int, j: int, actions: Any) -> Tuple[bool, str]:
        if not isinstance(actions, list):
            return False, f"Substep {i}.{j} 'actions' must be a list"

        for k, act in enumerate(actions):
            ok, msg = self._validate_action(i, j, k, act)
            if not ok:
                return False, msg

        return True, ""

    def _validate_action(self, i: int, j: int, k: int, act: Any) -> Tuple[bool, str]:
        if not isinstance(act, dict):
            return False, f"Action {i}.{j}.{k} must be a dict"

        if "act_id" not in act or "name" not in act:
            return False, f"Action {i}.{j}.{k} missing 'act_id' or 'name'"

        return True, ""
