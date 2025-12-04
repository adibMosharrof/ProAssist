import math
from typing import List, Dict, Any, Optional
from .base_style import BaseInputStyle

class ProAssistInputStyle(BaseInputStyle):
    """
    ProAssist Style: Continuous frame assignment.
    - Frames are assigned from the end of the previous turn to the current turn's timestamp.
    - No gaps between turns.
    - Enforces DST_UPDATE before assistant at the same timestamp.
    """
    
    def calculate_frame_indices(self, conversation: List[Dict[str, Any]], total_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        # 1. Sort conversation to ensure DST_UPDATE comes before assistant at same timestamp
        # We assume 'time' is present. If not, we keep original order.
        # Stable sort: first by time, then by role priority (DST_UPDATE < assistant)
        
        def sort_key(turn):
            time_val = turn.get("time", 0.0)
            role = turn.get("role", "")
            # Priority: DST_UPDATE (0) < assistant (1) < others (2)
            role_priority = 2
            if role == "DST_UPDATE":
                role_priority = 0
            elif role == "assistant":
                role_priority = 1
            return (time_val, role_priority)

        # Only sort if 'time' exists in turns
        if conversation and "time" in conversation[0]:
             conversation = sorted(conversation, key=sort_key)

        updated_conversation = []
        current_start_frame = 0
        
        for turn in conversation:
            new_turn = turn.copy()
            time_sec = turn.get("time")
            
            if time_sec is None:
                # Fallback: keep original or skip
                updated_conversation.append(new_turn)
                continue
                
            # Calculate end frame
            target_end_frame = math.floor(time_sec * self.fps)
            
            # Clamp to total frames
            if total_frames is not None and target_end_frame > total_frames:
                target_end_frame = total_frames
                
            # Ensure monotonicity
            if target_end_frame < current_start_frame:
                target_end_frame = current_start_frame
                
            new_turn["start_frame"] = current_start_frame
            new_turn["end_frame"] = target_end_frame
            
            updated_conversation.append(new_turn)
            current_start_frame = target_end_frame
            
        return updated_conversation
