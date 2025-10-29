from typing import Any, Dict, Tuple
import logging
from dst_data_builder.validators.base_validator import BaseValidator

logger = logging.getLogger(__name__)


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

     Post-processing:
     - If validation fails due to temporal ordering violations, the validator
        will attempt to automatically fix timestamps while preserving durations
        and semantic content.
     - Fixed structures are re-validated to ensure correctness.

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

    def __init__(self, epsilon: float = 0.0, enable_post_processing: bool = True):
        """Create a TimestampsValidator.

        epsilon: optional tolerance (in same time units as timestamps). When
        non-zero, comparisons that require >= or <= will allow a small leeway
        of `epsilon` to account for floating point or annotation noise.
        enable_post_processing: if True, attempt to fix temporal violations automatically
        """
        self.epsilon = float(epsilon)
        self.enable_post_processing = enable_post_processing
        self.post_processing_fixes_count = 0  # Track how many times post-processing fixed issues

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

    def _validate_actions(self, step_idx: int, substep_idx: int, sub: Dict[str, Any], sub_start: float, sub_end: float) -> Tuple[bool, Any, str]:
        """Validate actions within a substep. Returns (ok, last_act_end, msg)."""
        prev_act_end = None
        actions = sub.get("actions", [])
        # If actions is not a list, avoid duplicating structure validation here.
        if actions is None:
            return True, None, ""
        if not isinstance(actions, list):
            return False, None, f"steps[{step_idx}].substeps[{substep_idx}].actions must be a list; run StructureValidator first"

        for aj, act in enumerate(actions):
            act_id = act.get("act_id", f"{sub.get('sub_id', 'sub')}.{aj}")
            ts_act = act.get("timestamps")
            ok, act_start, act_end, msg = self._check_ts(ts_act)
            if not ok:
                return False, None, f"steps[{step_idx}].substeps[{substep_idx}].actions[{aj}] ({act_id}) timestamps invalid: {msg}"

            # Check if action is within parent substep (with epsilon tolerance)
            if act_start + self.epsilon < sub_start or act_end > sub_end + self.epsilon:
                return (
                    False,
                    None,
                    f"steps[{step_idx}].substeps[{substep_idx}].actions[{aj}] ({act_id}) time span [{act_start}, {act_end}] not within parent substep {sub.get('substep_id', '')} [{sub_start}, {sub_end}]",
                )

            # Check sequential ordering with previous action
            if prev_act_end is not None and act_start + self.epsilon < prev_act_end:
                return (
                    False,
                    None,
                    f"steps[{step_idx}].substeps[{substep_idx}].actions[{aj}] ({act_id}) starts before previous action ended ({act_start} < {prev_act_end})",
                )

            prev_act_end = act_end

        return True, prev_act_end, ""

    def _validate_substeps(self, step_idx: int, step: Dict[str, Any], s_start: float, s_end: float) -> Tuple[bool, Any, str]:
        """Validate substeps within a step. Returns (ok, last_sub_end, msg)."""
        prev_sub_end = None
        substeps = step.get("substeps", [])
        if substeps is None:
            return True, None, ""
        if not isinstance(substeps, list):
            return False, None, f"steps[{step_idx}].substeps must be a list; run StructureValidator first"

        for sj, sub in enumerate(substeps):
            sub_id = sub.get("sub_id", f"{step.get('step_id', '')}.{sj}")
            ts_sub = sub.get("timestamps")
            ok, sub_start, sub_end, msg = self._check_ts(ts_sub)
            if not ok:
                return False, None, f"steps[{step_idx}].substeps[{sj}] ({sub_id}) timestamps invalid: {msg}"

            # Check if substep is within parent step (only if step has explicit timestamps)
            if s_start is not None and s_end is not None:
                if sub_start + self.epsilon < s_start or sub_end > s_end + self.epsilon:
                    return (
                        False,
                        None,
                        f"steps[{step_idx}].substeps[{sj}] ({sub_id}) time span [{sub_start}, {sub_end}] not within parent step {step.get('step_id', '')} [{s_start}, {s_end}]",
                    )

            # Check sequential ordering with previous substep
            if prev_sub_end is not None and sub_start + self.epsilon < prev_sub_end:
                return (
                    False,
                    None,
                    f"steps[{step_idx}].substeps[{sj}] ({sub_id}) starts before previous substep ended ({sub_start} < {prev_sub_end})",
                )

            # Validate actions inside substep
            ok, last_act_end, msg = self._validate_actions(step_idx, sj, sub, sub_start, sub_end)
            if not ok:
                return False, None, msg

            prev_sub_end = sub_end

        return True, prev_sub_end, ""

    def _validate_step(self, step_idx: int, step: Dict[str, Any], prev_step_end: float) -> Tuple[bool, Any, str]:
        """Validate a single step, including its substeps and actions.

        Returns (ok, step_end, msg).
        """
        step_id = step.get("step_id", "")
        ts = step.get("timestamps")
        if ts is None:
            return False, None, f"steps[{step_idx}] ({step_id}) timestamps required"

        ok, s_start, s_end, msg = self._check_ts(ts)
        if not ok:
            return False, None, f"steps[{step_idx}] ({step_id}) timestamps invalid: {msg}"

        # Check sequential ordering with previous step (with epsilon tolerance)
        if prev_step_end is not None and s_start + self.epsilon < prev_step_end:
            return (
                False,
                None,
                f"steps[{step_idx}] ({step_id}) starts before previous step ended ({s_start} < {prev_step_end})",
            )

        # Validate substeps (and their actions)
        ok, last_sub_end, msg = self._validate_substeps(step_idx, step, s_start, s_end)
        if not ok:
            return False, None, msg

        # Return the step's end time (or None if derived from substeps)
        return True, s_end if s_end is not None else last_sub_end, ""

    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        steps = dst_structure.get("steps", [])

        prev_step_end = None
        for si, step in enumerate(steps):
            ok, step_end, msg = self._validate_step(si, step, prev_step_end)
            if not ok:
                # If validation failed and post-processing is enabled, try to fix
                if self.enable_post_processing:
                    logger.info(f"Timestamp validation failed: {msg}. Attempting post-processing fix...")
                    fixed_structure = self._fix_temporal_violations(dst_structure)
                    
                    # Re-validate the fixed structure
                    ok_after_fix, msg_after_fix = self._validate_without_postprocessing(fixed_structure)
                    if ok_after_fix:
                        self.post_processing_fixes_count += 1
                        logger.info(f"✓ Post-processing successfully fixed timestamp violations (total fixes: {self.post_processing_fixes_count})")
                        # Update the original structure with fixed timestamps
                        dst_structure.update(fixed_structure)
                        return True, ""
                    else:
                        logger.warning(f"✗ Post-processing attempted but validation still failed: {msg_after_fix}")
                        return False, f"Post-processing attempted but failed: {msg_after_fix}"
                
                return False, msg

            # Only update prev_step_end if this step has explicit timestamps
            if step_end is not None:
                prev_step_end = step_end

        return True, ""

    def _validate_without_postprocessing(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate without triggering post-processing (used internally after fixing)."""
        steps = dst_structure.get("steps", [])

        prev_step_end = None
        for si, step in enumerate(steps):
            ok, step_end, msg = self._validate_step(si, step, prev_step_end)
            if not ok:
                return False, msg

            if step_end is not None:
                prev_step_end = step_end

        return True, ""

    def _fix_temporal_violations(self, dst_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix temporal violations by adjusting timestamps.
        
        Strategy:
        1. Preserve durations of all elements where possible
        2. Adjust start times to ensure proper sequential ordering
        3. Ensure child elements fit within parent time bounds
        4. Maintain the semantic content unchanged
        
        Returns a modified copy of the structure with fixed timestamps.
        """
        import copy
        fixed = copy.deepcopy(dst_structure)
        
        steps = fixed.get("steps", [])
        if not steps:
            return fixed
        
        current_step_start = None
        
        for step in steps:
            step_ts = step.get("timestamps")
            if step_ts is None:
                continue
                
            # Get original step times
            try:
                orig_step_start = float(step_ts.get("start_ts", 0))
                orig_step_end = float(step_ts.get("end_ts", 0))
            except (ValueError, TypeError):
                continue
            
            # Adjust step start if it overlaps with previous step
            if current_step_start is not None and orig_step_start < current_step_start:
                step_duration = orig_step_end - orig_step_start
                step_ts["start_ts"] = current_step_start
                step_ts["end_ts"] = current_step_start + step_duration
                orig_step_start = current_step_start
                orig_step_end = current_step_start + step_duration
            
            # Fix substeps within this step
            substeps = step.get("substeps", [])
            current_substep_end = orig_step_start
            
            for substep in substeps:
                sub_ts = substep.get("timestamps")
                if sub_ts is None:
                    continue
                
                try:
                    orig_sub_start = float(sub_ts.get("start_ts", 0))
                    orig_sub_end = float(sub_ts.get("end_ts", 0))
                except (ValueError, TypeError):
                    continue
                
                # Adjust substep start to be after previous substep
                if orig_sub_start < current_substep_end:
                    sub_duration = orig_sub_end - orig_sub_start
                    sub_ts["start_ts"] = current_substep_end
                    sub_ts["end_ts"] = current_substep_end + sub_duration
                    orig_sub_start = current_substep_end
                    orig_sub_end = current_substep_end + sub_duration
                
                # Ensure substep starts at or after step start
                if orig_sub_start < orig_step_start:
                    sub_duration = orig_sub_end - orig_sub_start
                    sub_ts["start_ts"] = orig_step_start
                    sub_ts["end_ts"] = orig_step_start + sub_duration
                    orig_sub_start = orig_step_start
                    orig_sub_end = orig_step_start + sub_duration
                
                # Fix actions within this substep
                actions = substep.get("actions", [])
                current_action_end = orig_sub_start
                
                for action in actions:
                    act_ts = action.get("timestamps")
                    if act_ts is None:
                        continue
                    
                    try:
                        orig_act_start = float(act_ts.get("start_ts", 0))
                        orig_act_end = float(act_ts.get("end_ts", 0))
                    except (ValueError, TypeError):
                        continue
                    
                    # Adjust action start to be after previous action
                    if orig_act_start < current_action_end:
                        act_duration = orig_act_end - orig_act_start
                        act_ts["start_ts"] = current_action_end
                        act_ts["end_ts"] = current_action_end + act_duration
                        orig_act_start = current_action_end
                        orig_act_end = current_action_end + act_duration
                    
                    # Ensure action starts at or after substep start
                    if orig_act_start < orig_sub_start:
                        act_duration = orig_act_end - orig_act_start
                        act_ts["start_ts"] = orig_sub_start
                        act_ts["end_ts"] = orig_sub_start + act_duration
                        orig_act_start = orig_sub_start
                        orig_act_end = orig_sub_start + act_duration
                    
                    current_action_end = orig_act_end
                
                # Update substep end to accommodate all actions if needed
                if current_action_end > orig_sub_end:
                    sub_ts["end_ts"] = current_action_end
                    orig_sub_end = current_action_end
                
                current_substep_end = orig_sub_end
            
            # Update step end to accommodate all substeps if needed
            if current_substep_end > orig_step_end:
                step_ts["end_ts"] = current_substep_end
                orig_step_end = current_substep_end
            
            current_step_start = orig_step_end
        
        return fixed

    def get_post_processing_stats(self) -> Dict[str, int]:
        """Return statistics about post-processing fixes."""
        return {
            "total_fixes": self.post_processing_fixes_count
        }

    def reset_post_processing_stats(self):
        """Reset the post-processing statistics counter."""
        self.post_processing_fixes_count = 0
