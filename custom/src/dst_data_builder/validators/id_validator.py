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

    def _validate_action(self, step_num: int, sub_num: int, act: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(act, dict):
            return False, "Action must be a dict"

        act_id = act.get("act_id")
        if not act_id or not isinstance(act_id, str):
            return False, "Action missing act_id"

        m = self.ACT_RE.match(act_id)
        if not m:
            return False, f"Invalid act_id format: {act_id}"

        act_step_num = int(m.group(1))
        act_sub_num = int(m.group(2))
        act_suffix = m.group(3)

        if act_step_num != step_num or act_sub_num != sub_num:
            return False, f"Action {act_id} does not belong to parent substep"

        if act_id in ids_seen:
            return False, f"Duplicate id: {act_id}"
        ids_seen.add(act_id)

        ok_ts, start = self._extract_start(act)
        if not ok_ts:
            return False, "Missing timestamps for ordering check; run TimestampsValidator first"

        suffix_num = _letters_to_number(act_suffix)
        if suffix_num <= 0:
            return False, f"Invalid act suffix in act_id: {act_id}"

        return True, (suffix_num, start)

    def _validate_substep(self, step_num: int, sub: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(sub, dict):
            return False, "Substep must be a dict"

        sub_id = sub.get("sub_id")
        if not sub_id or not isinstance(sub_id, str):
            return False, "Substep missing sub_id"

        m = self.SUB_RE.match(sub_id)
        if not m:
            return False, f"Invalid sub_id format: {sub_id}"

        parent_step = int(m.group(1))
        sub_num = int(m.group(2))
        if parent_step != step_num:
            return False, f"Substep {sub_id} does not belong to parent step S{step_num}"

        if sub_id in ids_seen:
            return False, f"Duplicate id: {sub_id}"
        ids_seen.add(sub_id)

        ok_ts, start = self._extract_start(sub)
        if not ok_ts:
            return False, "Missing timestamps for ordering check; run TimestampsValidator first"

        # Validate actions and collect ordering tuples
        actions = sub.get("actions", [])
        if actions is None:
            return True, (sub_num, start)
        if not isinstance(actions, list):
            return False, "Substep 'actions' must be a list"

        act_items: List[Tuple[int, float]] = []
        for act in actions:
            ok, res = self._validate_action(step_num, sub_num, act, ids_seen)
            if not ok:
                return False, res
            act_items.append(res)

        ok, msg = self._check_sibling_order(act_items)
        if not ok:
            return False, f"In substep {sub_id}: {msg}"

        return True, (sub_num, start)

    def _validate_step(self, step: Dict[str, Any], ids_seen: set) -> Tuple[bool, Union[Tuple[int, float], str]]:
        if not isinstance(step, dict):
            return False, "Step must be a dict"

        sid = step.get("step_id")
        if not sid or not isinstance(sid, str):
            return False, "Step missing step_id"

        m = self.STEP_RE.match(sid)
        if not m:
            return False, f"Invalid step_id format: {sid}"

        step_num = int(m.group(1))
        if sid in ids_seen:
            return False, f"Duplicate id: {sid}"
        ids_seen.add(sid)

        ok_ts, start = self._extract_start(step)
        if not ok_ts:
            return False, "Missing timestamps for ordering check; run TimestampsValidator first"

        substeps = step.get("substeps", [])
        if substeps is None:
            return True, (step_num, start)
        if not isinstance(substeps, list):
            return False, "Step 'substeps' must be a list"

        sub_items: List[Tuple[int, float]] = []
        for sub in substeps:
            ok, res = self._validate_substep(step_num, sub, ids_seen)
            if not ok:
                return False, res
            sub_items.append(res)

        ok, msg = self._check_sibling_order(sub_items)
        if not ok:
            return False, f"In step {sid}: {msg}"

        return True, (step_num, start)

    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        """Public entrypoint; returns (True, "") on success or (False, msg)."""
        if not isinstance(dst_structure, dict):
            return False, "DST must be a dict"

        ids_seen = set()
        steps = dst_structure.get("steps", [])
        if steps is None:
            return False, "Missing 'steps'"
        if not isinstance(steps, list):
            return False, "'steps' must be a list"

        step_items: List[Tuple[int, float]] = []
        for step in steps:
            ok, res = self._validate_step(step, ids_seen)
            if not ok:
                return False, res
            step_items.append(res)

        ok, msg = self._check_sibling_order(step_items)
        if not ok:
            return False, msg

        return True, ""

