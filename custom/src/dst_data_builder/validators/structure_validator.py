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
            return False, "DST structure must be a JSON object/dictionary, not a " + type(dst_structure).__name__

        if "steps" not in dst_structure:
            return False, "DST structure is missing required 'steps' field. The structure should have a 'steps' array containing step objects."

        steps = dst_structure["steps"]
        return self._validate_steps(steps)

    def _validate_steps(self, steps: Any) -> Tuple[bool, str]:
        if not isinstance(steps, list):
            return False, f"'steps' field must be an array/list, not {type(steps).__name__}. Each step should be an object with step_id, name, timestamps, and optionally substeps."

        if len(steps) == 0:
            return False, "'steps' array is empty. At least one step is required in the DST structure."

        for i, step in enumerate(steps):
            ok, msg = self._validate_step(i, step)
            if not ok:
                return False, msg

        return True, ""

    def _validate_step(self, i: int, step: Any) -> Tuple[bool, str]:
        if not isinstance(step, dict):
            return False, f"steps[{i}] must be an object/dictionary, not {type(step).__name__}. Each step should have step_id, name, timestamps, and optionally substeps."

        required_fields = ["step_id", "name"]
        missing_fields = [field for field in required_fields if field not in step]
        if missing_fields:
            return False, f"steps[{i}] is missing required fields: {missing_fields}. Each step must have 'step_id' (string like 'S1') and 'name' (string describing the step)."

        # Check field types
        if not isinstance(step.get("step_id"), str):
            return False, f"steps[{i}].step_id must be a string, not {type(step.get('step_id')).__name__}. Use format like 'S1', 'S2', etc."

        if not isinstance(step.get("name"), str):
            return False, f"steps[{i}].name must be a string, not {type(step.get('name')).__name__}. Provide a descriptive name for the step."

        substeps = step.get("substeps", [])
        if substeps is None:
            return True, ""

        return self._validate_substeps(i, substeps)

    def _validate_substeps(self, step_idx: int, substeps: Any) -> Tuple[bool, str]:
        if not isinstance(substeps, list):
            return False, f"steps[{step_idx}].substeps must be an array/list, not {type(substeps).__name__}. If no substeps, use empty array [] or omit the field."

        for j, substep in enumerate(substeps):
            is_valid, error_msg = self._validate_substep(step_idx, j, substep)
            if not is_valid:
                return False, error_msg

        return True, ""

    def _validate_substep(self, step_idx: int, substep_idx: int, substep: Any) -> Tuple[bool, str]:
        if not isinstance(substep, dict):
            return False, f"steps[{step_idx}].substeps[{substep_idx}] must be an object/dictionary, not {type(substep).__name__}. Each substep should have substep_id, name, timestamps, and optionally actions."

        required_fields = ["sub_id", "name"]
        missing_fields = [field for field in required_fields if field not in substep]
        if missing_fields:
            return False, f"steps[{step_idx}].substeps[{substep_idx}] is missing required fields: {missing_fields}. Each substep must have 'sub_id' (string like 'S1.1') and 'name' (string describing the substep)."

        # Check field types
        if not isinstance(substep.get("sub_id"), str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].sub_id must be a string, not {type(substep.get('sub_id')).__name__}. Use format like 'S1.1', 'S1.2', etc."

        if not isinstance(substep.get("name"), str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].name must be a string, not {type(substep.get('name')).__name__}. Provide a descriptive name for the substep."

        actions = substep.get("actions", [])
        if actions is None:
            return True, ""

        return self._validate_actions(step_idx, substep_idx, actions)

    def _validate_actions(self, step_idx: int, substep_idx: int, actions: Any) -> Tuple[bool, str]:
        if not isinstance(actions, list):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions must be an array/list, not {type(actions).__name__}. If no actions, use empty array [] or omit the field."

        for k, action in enumerate(actions):
            is_valid, error_msg = self._validate_action(step_idx, substep_idx, k, action)
            if not is_valid:
                return False, error_msg

        return True, ""

    def _validate_action(self, step_idx: int, substep_idx: int, action_idx: int, action: Any) -> Tuple[bool, str]:
        if not isinstance(action, dict):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] must be an object/dictionary, not {type(action).__name__}. Each action should have action_id, name, and timestamps."

        required_fields = ["act_id", "name"]
        missing_fields = [field for field in required_fields if field not in action]
        if missing_fields:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] is missing required fields: {missing_fields}. Each action must have 'act_id' (string like 'S1.1.a') and 'name' (string describing the action)."

        # Check field types
        if not isinstance(action.get("act_id"), str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}].act_id must be a string, not {type(action.get('act_id')).__name__}. Use format like 'S1.1.a', 'S1.1.b', etc."

        if not isinstance(action.get("name"), str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}].name must be a string, not {type(action.get('name')).__name__}. Provide a descriptive name for the action."

        return True, ""
