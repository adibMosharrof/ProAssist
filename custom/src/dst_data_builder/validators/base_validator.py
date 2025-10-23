from abc import ABC, abstractmethod
from typing import Any, Tuple, Dict


class BaseValidator(ABC):
    """Abstract base class for DST validators.

    Implementations should provide a validate(dst_structure) method that
    returns (is_valid: bool, message: str).
    """

    @abstractmethod
    def validate(self, dst_structure: Dict[str, Any]) -> Tuple[bool, str]:
        pass
