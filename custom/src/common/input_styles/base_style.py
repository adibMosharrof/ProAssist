from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class BaseInputStyle(ABC):
    """
    Abstract base class for Input Styles.
    Defines how frame indices are calculated for a conversation.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.fps = config.get("fps", 30)

    @abstractmethod
    def calculate_frame_indices(self, conversation: List[Dict[str, Any]], total_frames: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Calculate start_frame and end_frame for each turn in the conversation.
        
        Args:
            conversation: List of conversation turns.
            total_frames: Total number of frames in the video (optional limit).
            
        Returns:
            Updated conversation list with 'start_frame' and 'end_frame' set.
        """
        pass
