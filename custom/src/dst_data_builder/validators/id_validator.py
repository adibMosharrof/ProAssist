from typing import Any, Dict, Tuple, List, Union
import re
from dst_data_builder.validators.base_validator import BaseValidator


def _letters_to_number(s: str) -> int:
    """Convert a lowercase letter sequence to a 1-based number (a=1, z=26, aa=27)."""
    s = s.lower()
    n = 0
    for ch in s:
        if not ("a" <= ch <= "z"):
            return -1
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n


class IdValidator(BaseValidator):
    """Validate identifier formats and ordering.

    This validator follows the project's validator pattern: a single
    `validate(dst_structure) -> (bool, msg)` entrypoint and small helper
    methods for clarity.
    """

    STEP_RE = re.compile(r"^S(\d+)$")
    SUB_RE = re.compile(r"^S(\d+)\.(\d+)$")
    ACT_RE = re.compile(r"^S(\d+)\.(\d+)\.([a-z]+)$")

    def __init__(self, epsilon: float = 0.0):
        self.epsilon = float(epsilon)

    def _extract_start(self, node: Dict[str, Any]) -> Tuple[bool, float]:
        ts = node.get("timestamps")
        if ts is None:
            return False, 0.0
        try:
            s = float(ts.get("start_ts"))
        except Exception:
            return False, 0.0
        return True, s

    def _check_sibling_order(self, items: List[Tuple[int, float]]) -> Tuple[bool, str]:
        """Ensure numeric ids increase as start_ts increases (with epsilon)."""
        n = len(items)
        for i in range(n):
            id_i, s_i = items[i]
            for j in range(i + 1, n):
                id_j, s_j = items[j]
                if s_i + self.epsilon < s_j and id_i >= id_j:
                    return (
                        False,
                        f"ID ordering mismatch: id {id_i} (start={s_i}) should be < id {id_j} (start={s_j})",
                    )
        return True, ""

    def _validate_action(self, step_idx: int, substep_idx: int, action_idx: int, step_num: int, sub_num: int, act: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(act, dict):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] must be a JSON object/dictionary with action_id, name, and timestamps fields."

        act_id = act.get("act_id")
        if not act_id or not isinstance(act_id, str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] is missing 'act_id' field. Each action must have an 'act_id' string like 'S{step_num}.{sub_num}.a' or 'S{step_num}.{sub_num}.b'."

        m = self.ACT_RE.match(act_id)
        if not m:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] has invalid action_id format: '{act_id}'. Action IDs must follow the pattern 'S<number>.<number>.<letter>' like 'S1.1.a', 'S1.1.b', 'S2.3.c', etc."

        act_step_num = int(m.group(1))
        act_sub_num = int(m.group(2))
        act_suffix = m.group(3)

        if act_step_num != step_num or act_sub_num != sub_num:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] ('{act_id}') does not belong to its parent substep. The action ID should match the parent substep ID 'S{step_num}.{sub_num}'."

        if act_id in ids_seen:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] has duplicate action ID: '{act_id}'. Each action must have a unique ID within the entire DST structure."
        ids_seen.add(act_id)

        ok_ts, start = self._extract_start(act)
        if not ok_ts:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] is missing valid timestamps for ordering check. Each action must have 'timestamps' with 'start_ts' and 'end_ts' numeric values."

        suffix_num = _letters_to_number(act_suffix)
        if suffix_num <= 0:
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions[{action_idx}] has invalid action suffix in action_id '{act_id}'. The suffix must be lowercase letters like 'a', 'b', 'c', etc."

        return True, (suffix_num, start)

    def _validate_substep(self, step_idx: int, substep_idx: int, step_num: int, sub: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(sub, dict):
            return False, f"steps[{step_idx}].substeps[{substep_idx}] must be a JSON object/dictionary with substep_id, name, timestamps, and optionally actions."

        sub_id = sub.get("sub_id")
        if not sub_id or not isinstance(sub_id, str):
            return False, f"steps[{step_idx}].substeps[{substep_idx}] is missing 'sub_id' field. Each substep must have a 'sub_id' string like 'S{step_num}.1', 'S{step_num}.2', etc."

        m = self.SUB_RE.match(sub_id)
        if not m:
            return False, f"steps[{step_idx}].substeps[{substep_idx}] has invalid substep_id format: '{sub_id}'. Substep IDs must follow the pattern 'S<number>.<number>' like 'S1.1', 'S1.2', 'S2.1', etc."

        parent_step = int(m.group(1))
        sub_num = int(m.group(2))
        if parent_step != step_num:
            return False, f"steps[{step_idx}].substeps[{substep_idx}] ('{sub_id}') does not belong to its parent step. The substep ID should start with 'S{step_num}.' to match the parent step."

        if sub_id in ids_seen:
            return False, f"steps[{step_idx}].substeps[{substep_idx}] has duplicate substep ID: '{sub_id}'. Each substep must have a unique ID within the entire DST structure."
        ids_seen.add(sub_id)

        ok_ts, start = self._extract_start(sub)
        if not ok_ts:
            return False, f"steps[{step_idx}].substeps[{substep_idx}] is missing valid timestamps for ordering check. Each substep must have 'timestamps' with 'start_ts' and 'end_ts' numeric values."

        # Validate actions and collect ordering tuples
        actions = sub.get("actions", [])
        if actions is None:
            return True, (sub_num, start)
        if not isinstance(actions, list):
            return False, f"steps[{step_idx}].substeps[{substep_idx}].actions must be an array/list of action objects."

        act_items: List[Tuple[int, float]] = []
        for k, act in enumerate(actions):
            ok, res = self._validate_action(step_idx, substep_idx, k, step_num, sub_num, act, ids_seen)
            if not ok:
                return False, res
            act_items.append(res)

        ok, msg = self._check_sibling_order(act_items)
        if not ok:
            return False, f"In steps[{step_idx}].substeps[{substep_idx}] ({sub_id}): {msg}"

        return True, (sub_num, start)

    def _validate_step(self, step_idx: int, step: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(step, dict):
            return False, f"steps[{step_idx}] must be a JSON object/dictionary with step_id, name, timestamps, and optionally substeps."

        sid = step.get("step_id")
        if not sid or not isinstance(sid, str):
            return False, f"steps[{step_idx}] is missing 'step_id' field. Each step must have a 'step_id' string like 'S1', 'S2', 'S3', etc."

        m = self.STEP_RE.match(sid)
        if not m:
            return False, f"steps[{step_idx}] has invalid step_id format: '{sid}'. Step IDs must follow the pattern 'S<number>' like 'S1', 'S2', 'S3', etc."

        step_num = int(m.group(1))
        if sid in ids_seen:
            return False, f"steps[{step_idx}] has duplicate step ID: '{sid}'. Each step must have a unique ID within the entire DST structure."
        ids_seen.add(sid)

        ok_ts, start = self._extract_start(step)
        if not ok_ts:
            # Steps may not have timestamps if they derive state from substeps
            start = 0.0

        substeps = step.get("substeps", [])
        if substeps is None:
            return True, (step_num, start)
        if not isinstance(substeps, list):
            return False, f"steps[{step_idx}].substeps must be an array/list of substep objects."

        sub_items: List[Tuple[int, float]] = []
        for j, sub in enumerate(substeps):
            ok, res = self._validate_substep(step_idx, j, step_num, sub, ids_seen)
            if not ok:
                return False, res
            sub_items.append(res)

        ok, msg = self._check_sibling_order(sub_items)
        if not ok:
            return False, f"In steps[{step_idx}] ({sid}): {msg}"

        return True, (step_num, start)

    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Public entrypoint; returns (True, "") on success or (False, msg)."""
        if not isinstance(dst_structure, dict):
            return False, "DST structure must be a JSON object/dictionary, not a " + type(dst_structure).__name__

        ids_seen = set()
        steps = dst_structure.get("steps", [])
        if steps is None:
            return False, "DST structure is missing required 'steps' field. The structure should have a 'steps' array."
        if not isinstance(steps, list):
            return False, "'steps' field must be an array/list of step objects."

        if len(steps) == 0:
            return False, "'steps' array is empty. At least one step is required in the DST structure."

        step_items: List[Tuple[int, float]] = []
        for i, step in enumerate(steps):
            ok, res = self._validate_step(i, step, ids_seen)
            if not ok:
                return False, res
            step_items.append(res)

        ok, msg = self._check_sibling_order(step_items)
        if not ok:
            return False, msg

        return True, ""

