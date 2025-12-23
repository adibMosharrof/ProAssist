import torch
import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, accuracy_score

from custom.src.prospect.metrics.base_metric import BaseMetric

class DSTBinaryMetrics(BaseMetric):
    """
    Metrics for binary decisions: Speaking and DST Update.
    Calculates Accuracy, Balanced Accuracy, Precision, Recall, F1.
    """
    
    def __init__(self):
        self.reset()
        
    def reset(self) -> None:
        self.speaking_preds = []
        self.speaking_refs = []
        self.dst_update_preds = []
        self.dst_update_refs = []
        
    def update(self, prediction: List[Any], reference: Dict[str, Any]) -> None:
        """
        Update metrics with frame-level predictions and references.
        
        Args:
            prediction: List of FrameOutput objects
            reference: Sample dictionary containing ground truth
        """
        # We need to align predictions with reference frames
        # Assuming FrameOutput has frame_idx_in_stream
        
        # Extract ground truth from reference
        # We need to reconstruct the frame-level labels from the conversation
        # This logic mimics the data collator's label generation
        
        num_frames = len(prediction)
        
        # Initialize frame-level labels
        speaking_labels = np.zeros(num_frames, dtype=int)
        dst_update_labels = np.zeros(num_frames, dtype=int)
        
        # Populate labels from conversation
        # Note: This assumes reference["conversation"] has start_frame/end_frame
        # and we are processing the whole clip.
        # If prediction is a subset, we need to handle offsets.
        
        clip_start_frame = reference.get("start_frame_idx", 0)
        
        for turn in reference["conversation"]:
            role = turn["role"]
            # Use start_frame if available, otherwise skip
            if "start_frame" not in turn:
                continue
                
            # Map video frame index to stream index
            # stream_idx = video_frame_idx - clip_start_frame
            start_idx = turn["start_frame"] - clip_start_frame
            
            if 0 <= start_idx < num_frames:
                if role == "assistant":
                    speaking_labels[start_idx] = 1
                elif role == "DST_UPDATE":
                    dst_update_labels[start_idx] = 1
        
        # Extract predictions from binary heads
        speaking_preds_frame = []
        dst_update_preds_frame = []
        
        for frame_out in prediction:
            # Use binary head predictions, not whether text was generated
            # If the output doesn't have these fields, fall back to text generation check
            if hasattr(frame_out, 'speaking'):
                speaking_preds_frame.append(frame_out.speaking)
            else:
                speaking_preds_frame.append(1 if frame_out.gen else 0)
            
            if hasattr(frame_out, 'dst_update_binary'):
                dst_update_preds_frame.append(frame_out.dst_update_binary)
            else:
                dst_update_preds_frame.append(1 if frame_out.dst_update else 0)
            
        self.speaking_preds.extend(speaking_preds_frame)
        self.speaking_refs.extend(speaking_labels)
        self.dst_update_preds.extend(dst_update_preds_frame)
        self.dst_update_refs.extend(dst_update_labels)
        
    def compute(self) -> Dict[str, float]:
        metrics = {}
        
        # Speaking Metrics
        if self.speaking_refs:
            metrics.update(self._compute_binary_metrics(
                self.speaking_refs, self.speaking_preds, "speaking"
            ))
            
        # DST Update Metrics
        if self.dst_update_refs:
            metrics.update(self._compute_binary_metrics(
                self.dst_update_refs, self.dst_update_preds, "dst_update"
            ))
            
        return metrics
        
    def _compute_binary_metrics(self, y_true, y_pred, prefix) -> Dict[str, float]:
        # Convert to numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        
        # zero_division=0 to handle cases with no positive samples
        p, r, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0
        )
        
        return {
            f"{prefix}_balanced_acc": float(balanced_acc),
            f"{prefix}_precision": float(p),
            f"{prefix}_recall": float(r),
            f"{prefix}_f1": float(f1),
        }
