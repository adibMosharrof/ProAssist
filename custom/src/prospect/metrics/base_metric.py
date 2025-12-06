from abc import ABC, abstractmethod
from typing import Dict, List, Any

class BaseMetric(ABC):
    """
    Abstract base class for all DST inference metrics.
    """
    
    @abstractmethod
    def update(self, prediction: Any, reference: Any) -> None:
        """
        Update metric state with a single sample's prediction and reference.
        
        Args:
            prediction: The model's output for a sample (e.g., list of FrameOutput)
            reference: The ground truth for a sample (e.g., sample dict)
        """
        pass
        
    @abstractmethod
    def compute(self) -> Dict[str, float]:
        """
        Compute the final metric values based on accumulated state.
        
        Returns:
            Dictionary of metric names and their values.
        """
        pass
        
    @abstractmethod
    def reset(self) -> None:
        """
        Reset the metric state.
        """
        pass
