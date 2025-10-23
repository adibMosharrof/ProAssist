from typing import Any, Dict, Tuple
from dst_data_builder.validators.base_validator import BaseValidator


class TimestampsValidator(BaseValidator):
    """Validate timestamps in steps/substeps/actions when present.

     This validator enforces strict timestamp rules across the DST tree. It
     performs the following checks for each step, substep, and action:

     1. Presence check: `timestamps` must be present and non-null on every
         step, substep, and action. If missing or null validation fails with
         an explanatory message (e.g., "timestamps required").

     2. Type check: `timestamps` must be a dictionary/object. If it's not a
         dict the validator fails with: "timestamps must be a dict".

     3. Numeric check: `start_ts` and `end_ts` must be present and numeric
         (convertible to float). If not, validation fails with:
         "start_ts and end_ts must be numeric".

     4. Local ordering check: For each object, `start_ts <= end_ts` must hold
         (with optional epsilon tolerance). If not, validation fails with:
         "start_ts must be <= end_ts".

     5. Global ordering and containment (with epsilon tolerance):
         - Steps must be non-overlapping and sequential: each step's
            `start_ts` must be >= previous step's `end_ts`.
         - Substeps within a step must be sequential and lie within the
            parent step's time span (sub_start >= step_start and sub_end <= step_end),
            and each substep's start must be >= previous substep's end.
         - Actions within a substep must be sequential and lie within the
            parent substep's time span; each action's start must be >= previous
            action's end.

     Notes:
     - An optional epsilon tolerance can be set to account for floating-point
        precision or annotation noise in timestamp comparisons.
     - Validator returns (True, "") on success, or (False, message) where
        `message` explains the reason for failure and indicates the failing
        location (Step/Substep/Action) when possible.
    """

    def _check_ts(self, ts_obj) -> Tuple[bool, float, float, str]:
        """Check and parse timestamps, returning (ok, start_ts, end_ts, msg)."""
        return self._parse_ts(ts_obj)

    def __init__(self, epsilon: float = 0.0):
        """Create a TimestampsValidator.

        epsilon: optional tolerance (in same time units as timestamps). When
        non-zero, comparisons that require >= or <= will allow a small leeway
        of `epsilon` to account for floating point or annotation noise.
        """
        self.epsilon = float(epsilon)

    def _parse_ts(self, ts_obj) -> Tuple[bool, float, float, str]:
        """Parse and validate timestamps object, returning (ok, start_ts, end_ts, msg).

        This centralizes presence, type, numeric, and local-order checks so
        other methods can reuse parsed numeric values and avoid duplication.
        """
        # Check if timestamps object is present
        if ts_obj is None:
            return False, 0.0, 0.0, "timestamps required"

        # Check if timestamps is a dictionary
        if not isinstance(ts_obj, dict):
            return False, 0.0, 0.0, "timestamps must be a dict"

        # Extract start_ts and end_ts
        start = ts_obj.get("start_ts")
        end = ts_obj.get("end_ts")
        if start is None or end is None:
            return False, 0.0, 0.0, "timestamps required"

        # Convert to float and check if numeric
        try:
            start_ts = float(start)
            end_ts = float(end)
        except Exception:
            return False, 0.0, 0.0, "start_ts and end_ts must be numeric"

        # Check local ordering: start_ts <= end_ts (with epsilon tolerance)
        if start_ts > end_ts + self.epsilon:
            return False, start_ts, end_ts, f"start_ts must be <= end_ts (epsilon={self.epsilon})"

        return True, start_ts, end_ts, ""

    # Removed smaller helpers in favor of _parse_ts to avoid duplication.

    def _validate_actions(self, sub: Dict[str, Any], sub_start: float, sub_end: float) -> Tuple[bool, Any, str]:
        """Validate actions within a substep. Returns (ok, last_act_end, msg)."""
        prev_act_end = None
        actions = sub.get("actions", [])
        # If actions is not a list, avoid duplicating structure validation here.
        if actions is None:
            return True, None, ""
        if not isinstance(actions, list):
            return False, None, "Invalid structure for actions; run StructureValidator first"

        for aj, act in enumerate(actions):
            act_id = act.get("act_id", f"{sub.get('sub_id', 'sub')}.{aj}")
            ts_act = act.get("timestamps")
            ok, act_start, act_end, msg = self._check_ts(ts_act)
            if not ok:
                return False, None, f"Action {act_id} timestamps invalid: {msg}"

            # Check if action is within parent substep (with epsilon tolerance)
            if act_start + self.epsilon < sub_start or act_end > sub_end + self.epsilon:
                return (
                    False,
                    None,
                    f"Action {act_id} time span [{act_start}, {act_end}] not within parent substep {sub.get('sub_id', '')} [{sub_start}, {sub_end}]",
                )

            # Check sequential ordering with previous action
            if prev_act_end is not None and act_start + self.epsilon < prev_act_end:
                return (
                    False,
                    None,
                    f"Action {act_id} starts before previous action ended ({act_start} < {prev_act_end})",
                )

            prev_act_end = act_end

        return True, prev_act_end, ""

    def _validate_substeps(self, step: Dict[str, Any], s_start: float, s_end: float) -> Tuple[bool, Any, str]:
        """Validate substeps within a step. Returns (ok, last_sub_end, msg)."""
        prev_sub_end = None
        substeps = step.get("substeps", [])
        if substeps is None:
            return True, None, ""
        if not isinstance(substeps, list):
            return False, None, "Invalid structure for substeps; run StructureValidator first"

        for sj, sub in enumerate(substeps):
            sub_id = sub.get("sub_id", f"{step.get('step_id', '')}.{sj}")
            ts_sub = sub.get("timestamps")
            ok, sub_start, sub_end, msg = self._check_ts(ts_sub)
            if not ok:
                return False, None, f"Substep {sub_id} timestamps invalid: {msg}"

            # Check if substep is within parent step (with epsilon tolerance)
            if sub_start + self.epsilon < s_start or sub_end > s_end + self.epsilon:
                return (
                    False,
                    None,
                    f"Substep {sub_id} time span [{sub_start}, {sub_end}] not within parent step {step.get('step_id', '')} [{s_start}, {s_end}]",
                )

            # Check sequential ordering with previous substep
            if prev_sub_end is not None and sub_start + self.epsilon < prev_sub_end:
                return (
                    False,
                    None,
                    f"Substep {sub_id} starts before previous substep ended ({sub_start} < {prev_sub_end})",
                )

            # Validate actions inside substep
            ok, last_act_end, msg = self._validate_actions(sub, sub_start, sub_end)
            if not ok:
                return False, None, msg

            prev_sub_end = sub_end

        return True, prev_sub_end, ""

    def _validate_step(self, step: Dict[str, Any], prev_step_end: float) -> Tuple[bool, Any, str]:
        """Validate a single step, including its substeps and actions.

        Returns (ok, step_end, msg).
        """
        step_id = step.get("step_id", "")
        ts = step.get("timestamps")
        ok, s_start, s_end, msg = self._check_ts(ts)
        if not ok:
            return False, None, f"Step {step_id} timestamps invalid: {msg}"

        # Check sequential ordering with previous step (with epsilon tolerance)
        if prev_step_end is not None and s_start + self.epsilon < prev_step_end:
            return (
                False,
                None,
                f"Step {step_id} starts before previous step ended ({s_start} < {prev_step_end})",
            )

        # Validate substeps (and their actions)
        ok, last_sub_end, msg = self._validate_substeps(step, s_start, s_end)
        if not ok:
            return False, None, msg

        return True, s_end, ""

    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        steps = dst_structure.get("steps", [])

        prev_step_end = None
        for si, step in enumerate(steps):
            ok, step_end, msg = self._validate_step(step, prev_step_end)
            if not ok:
                return False, msg

            prev_step_end = step_end

        return True, ""
