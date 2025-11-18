"""Validators package for DST structures

This package exposes validator classes that conform to the BaseValidator
interface. Validators implement a .validate(dst_structure) -> (bool, str)
method where the tuple is (is_valid, message). The message should be empty
on success or a short description of the failure.
"""

from .base_validator import BaseValidator
from .structure_validator import StructureValidator
from .timestamps_validator import TimestampsValidator
from .id_validator import IdValidator
from .training_format_validator import TrainingFormatValidator

__all__ = [
    "BaseValidator",
    "StructureValidator",
    "TimestampsValidator",
    "IdValidator",
    "TrainingFormatValidator",
]
