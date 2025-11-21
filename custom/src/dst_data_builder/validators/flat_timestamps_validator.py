from typing import Any, Dict, Tuple
import logging
from dst_data_builder.validators.base_validator import BaseValidator

logger = logging.getLogger(__name__)


class FlatTimestampsValidator(BaseValidator):
    """Validate timestamps in flat DST structures.

    This validator enforces timestamp rules for flat DST structures where each
    item has start_ts and end_ts fields. It performs the following checks:

    1. Presence check: `start_ts` and `end_ts` must be present and non-null
    2. Type check: `start_ts` and `end_ts` must be numeric (convertible to float)
    3. Local ordering check: `start_ts <= end_ts` must hold (with optional epsilon tolerance)
    4. Global ordering check: Items should be in chronological order by start_ts

    Post-processing:
    - If validation fails due to temporal ordering violations, the validator
      will attempt to automatically fix timestamps while preserving durations
      and semantic content.
    - Fixed structures are re-validated to ensure correctness.
    """

    def __init__(self, epsilon: float = 0.0, enable_post_processing: bool = True):
        """Create a FlatTimestampsValidator.

        epsilon: optional tolerance (in same time units as timestamps). When
        non-zero, comparisons that require >= or <= will allow a small leeway
        of `epsilon` to account for floating point or annotation noise.
        enable_post_processing: if True, attempt to fix temporal violations automatically
        """
        self.epsilon = float(epsilon)
        self.enable_post_processing = enable_post_processing
        self.post_processing_fixes_count = 0  # Track how many times post-processing fixed issues

    def _parse_ts(self, dst_item: Dict[str, Any]) -> Tuple[bool, float, float, str]:
        """Parse and validate timestamps for a single DST item.

        Returns (ok, start_ts, end_ts, msg).
        """
        # Extract start_ts and end_ts
        start = dst_item.get("start_ts")
        end = dst_item.get("end_ts")

        if start is None or end is None:
            return False, 0.0, 0.0, "start_ts and end_ts required"

        # Convert to float and check if numeric
        try:
            start_ts = float(start)
            end_ts = float(end)
        except (ValueError, TypeError):
            return False, 0.0, 0.0, "start_ts and end_ts must be numeric"

        # Check local ordering: start_ts <= end_ts (with epsilon tolerance)
        if start_ts > end_ts + self.epsilon:
            return False, start_ts, end_ts, f"start_ts must be <= end_ts (start={start_ts}, end={end_ts}, epsilon={self.epsilon})"

        return True, start_ts, end_ts, ""

    def validate(self, training_sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate timestamps in the DST structure of a training sample."""
        dst_items = training_sample.get("dst", [])

        if not dst_items:
            return True, ""  # No DST items to validate

        # Validate each DST item
        for i, dst_item in enumerate(dst_items):
            item_id = dst_item.get("id", f"item_{i}")
            ok, start_ts, end_ts, msg = self._parse_ts(dst_item)
            if not ok:
                # If validation failed and post-processing is enabled, try to fix
                if self.enable_post_processing:
                    logger.info(f"DST timestamp validation failed for {item_id}: {msg}. Attempting post-processing fix...")
                    fixed_sample = self._fix_temporal_violations(training_sample)

                    # Re-validate the fixed structure
                    ok_after_fix, msg_after_fix = self._validate_without_postprocessing(fixed_sample)
                    if ok_after_fix:
                        self.post_processing_fixes_count += 1
                        logger.info(f"✓ Post-processing successfully fixed DST timestamp violations (total fixes: {self.post_processing_fixes_count})")
                        # Update the original sample with fixed timestamps
                        training_sample["dst"] = fixed_sample["dst"]
                        return True, ""
                    else:
                        logger.warning(f"✗ Post-processing attempted but DST validation still failed: {msg_after_fix}")
                        return False, f"DST post-processing attempted but failed: {msg_after_fix}"

                return False, f"DST item {item_id} ({i}): {msg}"

        # Check global ordering (items should be roughly in chronological order)
        prev_end_ts = None
        for i, dst_item in enumerate(dst_items):
            ok, start_ts, end_ts, _ = self._parse_ts(dst_item)
            if not ok:
                continue  # Shouldn't happen since we validated above

            # Allow some tolerance for non-sequential items, but flag major violations
            if prev_end_ts is not None and start_ts < prev_end_ts - 10.0:  # 10 second tolerance
                item_id = dst_item.get("id", f"item_{i}")
                logger.warning(f"DST item {item_id} starts significantly before previous item ended (start={start_ts}, prev_end={prev_end_ts})")

            prev_end_ts = max(prev_end_ts or 0, end_ts)

        return True, ""

    def _validate_without_postprocessing(self, training_sample: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate DST timestamps without triggering post-processing (used internally after fixing)."""
        dst_items = training_sample.get("dst", [])

        for i, dst_item in enumerate(dst_items):
            ok, _, _, msg = self._parse_ts(dst_item)
            if not ok:
                item_id = dst_item.get("id", f"item_{i}")
                return False, f"DST item {item_id} ({i}): {msg}"

        return True, ""

    def _fix_temporal_violations(self, training_sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Attempt to fix temporal violations in DST items by adjusting timestamps.

        Strategy:
        1. Preserve durations of all DST items where possible
        2. Adjust start times to ensure start_ts <= end_ts for each item
        3. Sort items chronologically by start_ts if needed
        4. Maintain the semantic content unchanged

        Returns a modified copy of the training sample with fixed DST timestamps.
        """
        import copy
        fixed = copy.deepcopy(training_sample)

        dst_items = fixed.get("dst", [])
        if not dst_items:
            return fixed

        # First pass: Fix individual item violations (start_ts > end_ts)
        for dst_item in dst_items:
            start_ts = dst_item.get("start_ts")
            end_ts = dst_item.get("end_ts")

            if start_ts is not None and end_ts is not None:
                try:
                    start_val = float(start_ts)
                    end_val = float(end_ts)

                    # If start > end, swap them (preserve the time span but fix ordering)
                    if start_val > end_val:
                        dst_item["start_ts"] = end_val
                        dst_item["end_ts"] = start_val
                        logger.debug(f"Swapped timestamps for DST item {dst_item.get('id', 'unknown')}: {start_val} <-> {end_val}")

                except (ValueError, TypeError):
                    continue  # Skip invalid numeric values

        # Second pass: Ensure chronological ordering
        # Sort by start_ts to maintain temporal flow
        try:
            dst_items.sort(key=lambda x: float(x.get("start_ts", 0)))
        except (ValueError, TypeError, AttributeError):
            pass  # Skip sorting if timestamps are invalid

        # Third pass: Adjust overlapping items to be sequential
        prev_end_ts = None
        for dst_item in dst_items:
            try:
                start_ts = float(dst_item.get("start_ts", 0))
                end_ts = float(dst_item.get("end_ts", 0))

                # If this item starts before the previous one ended, adjust it
                if prev_end_ts is not None and start_ts < prev_end_ts:
                    duration = end_ts - start_ts
                    dst_item["start_ts"] = prev_end_ts
                    dst_item["end_ts"] = prev_end_ts + duration
                    logger.debug(f"Adjusted DST item {dst_item.get('id', 'unknown')} to start after previous: {start_ts} -> {prev_end_ts}")

                prev_end_ts = max(prev_end_ts or 0, float(dst_item.get("end_ts", 0)))

            except (ValueError, TypeError):
                continue

        fixed["dst"] = dst_items
        return fixed

    def get_post_processing_stats(self) -> Dict[str, int]:
        """Return statistics about post-processing fixes."""
        return {
            "total_dst_fixes": self.post_processing_fixes_count
        }

    def reset_post_processing_stats(self):
        """Reset the post-processing statistics counter."""
        self.post_processing_fixes_count = 0