import math
from typing import List, Dict, Any, Optional
from .base_style import BaseInputStyle

class WindowedInputStyle(BaseInputStyle):
    """
    Windowed Style: Fixed window around timestamp.
    - Frames are assigned based on a fixed window size (e.g., 4 frames) ending at the timestamp.
    - Gaps are allowed between turns.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.window_size = config.get("window_size", 4)

    def calculate_frame_indices(self, conversation: List[Dict[str, Any]], total_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        updated_conversation = []
        
        for turn in conversation:
            new_turn = turn.copy()
            time_sec = turn.get("time")
            
            if time_sec is None:
                updated_conversation.append(new_turn)
                continue
                
            # Calculate end frame (inclusive)
            end_frame = math.floor(time_sec * self.fps)
            
            # Calculate start frame based on window size
            # If window_size is 4, and end is 100, start is 97 (97, 98, 99, 100 = 4 frames)
            start_frame = end_frame - self.window_size + 1
            
            # Clamp to 0
            if start_frame < 0:
                start_frame = 0
                
            # Clamp to total frames
            if total_frames is not None:
                if end_frame >= total_frames:
                    end_frame = total_frames - 1
                if start_frame >= total_frames:
                    start_frame = total_frames - 1 # Or handle as empty
            
            # Ensure start <= end
            if start_frame > end_frame:
                 start_frame = end_frame

            new_turn["start_frame"] = start_frame
            new_turn["end_frame"] = end_frame
            
            updated_conversation.append(new_turn)
            
        return updated_conversation
