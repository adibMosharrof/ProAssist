from typing import Dict, List, Any, Tuple, Optional
import logging

from custom.src.prospect.metrics.base_metric import BaseMetric

logger = logging.getLogger(__name__)

class DSTContentMetrics(BaseMetric):
    """
    Metrics for DST content generation: Step ID and Transition accuracy.
    Uses exact matching.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self.total_updates = 0
        self.step_matches = 0
        self.transition_matches = 0
        self.exact_matches = 0
        
    def update(self, prediction: List[Any], reference: Dict[str, Any]) -> None:
        """
        Update metrics with frame-level predictions and references.
        
        Args:
            prediction: List of FrameOutput objects
            reference: Sample dictionary containing ground truth
        """
        # Extract ground truth updates
        # We need to find turns with role="DST_UPDATE" and match them with predictions
        # This is tricky because predictions are frame-by-frame.
        # We'll use a simple heuristic:
        # 1. Collect all predicted DST updates (non-None dst_update fields)
        # 2. Collect all reference DST updates
        # 3. Match them sequentially (assuming order is preserved)
        # Note: This is a simplification. A more robust approach would match by timestamp/frame window.
        # Given the "clumped" nature of ProAssist data, sequential matching is reasonable if we filter by proximity.
        
        pred_updates = []
        for frame_out in prediction:
            if frame_out.dst_update:
                pred_updates.append(frame_out.dst_update)
                
        ref_updates = []
        for turn in reference["conversation"]:
            if turn["role"] == "DST_UPDATE":
                # Extract the update string "S1->start"
                # The dataset provides this via get_dst_update_str, but here we have the raw dict
                # We need to replicate the formatting logic or assume it's available.
                # Let's assume we can parse the content dict.
                content = turn["content"]
                if isinstance(content, list) and len(content) > 0:
                    update = content[0]
                    ref_str = f"{update['id']}->{update['transition']}"
                    ref_updates.append(ref_str)
        
        # Match predictions to references
        # For now, we'll just compare the lists. If lengths differ, we might miss some or have extras.
        # We'll iterate up to the minimum length.
        # Ideally, we should penalize missing/extra, but "accuracy" usually implies "of the matched ones" 
        # or "over the union".
        # Let's stick to the plan's metric definitions which imply we evaluate the *generated* updates against *expected* ones.
        # Actually, standard DST metrics often look at "Joint Goal Accuracy" (state at end of turn).
        # Here we are evaluating the *update operation*.
        
        # Let's align by index for simplicity as a starting point.
        # A better approach would be to align by time, but let's start simple.
        
        min_len = min(len(pred_updates), len(ref_updates))
        
        for i in range(min_len):
            self._evaluate_single_update(pred_updates[i], ref_updates[i])
            
        # Log warning if counts differ significantly
        if len(pred_updates) != len(ref_updates):
            logger.debug(f"DST update count mismatch: Pred={len(pred_updates)}, Ref={len(ref_updates)}")
            
    def _evaluate_single_update(self, pred_text: str, ref_text: str) -> None:
        self.total_updates += 1
        
        pred_parsed = self._parse_dst_update(pred_text)
        ref_parsed = self._parse_dst_update(ref_text)
        
        if pred_parsed is None or ref_parsed is None:
            return
            
        step_match = pred_parsed[0] == ref_parsed[0]
        transition_match = pred_parsed[1] == ref_parsed[1]
        exact_match = step_match and transition_match
        
        if step_match:
            self.step_matches += 1
        if transition_match:
            self.transition_matches += 1
        if exact_match:
            self.exact_matches += 1
            
    def _parse_dst_update(self, text: str) -> Optional[Tuple[str, str]]:
        """Parse "S1->complete" or "S1 -> complete" into (step_id, transition)."""
        if not text:
            return None
        text = text.strip()
        if "->" not in text:
            return None
        parts = text.split("->", 1)
        if len(parts) != 2:
            return None
        step_id = parts[0].strip()
        transition = parts[1].strip()
        return (step_id, transition)
        
    def compute(self) -> Dict[str, float]:
        if self.total_updates == 0:
            return {
                "dst_step_accuracy": 0.0,
                "dst_transition_accuracy": 0.0,
                "dst_exact_match": 0.0,
            }
            
        return {
            "dst_step_accuracy": self.step_matches / self.total_updates,
            "dst_transition_accuracy": self.transition_matches / self.total_updates,
            "dst_exact_match": self.exact_matches / self.total_updates,
        }
