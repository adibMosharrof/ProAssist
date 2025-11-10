"""ProAssist DST Dataset - Lean dataset class that receives data in __init__"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Union, Optional

from dst_data_builder.data_sources.base_dst_dataset import BaseDSTDataset


class ProAssistDSTDataset(BaseDSTDataset):
    """Simple, lean dataset for ProAssist data.
    
    This dataset receives data as a list in __init__ and provides basic data access.
    All data loading and processing is handled by DSTDataModule.
    """
    
    def __init__(self, data: List[Dict[str, Any]], num_rows: Optional[int] = None):
        """Initialize dataset with pre-loaded data.
        
        Args:
            data: List of video objects (already loaded from JSON)
            num_rows: Optional truncation of dataset size
        """
        super().__init__()
        self._data = data
        self._num_rows = None if (num_rows is None or num_rows == -1) else int(num_rows)
        self.logger = logging.getLogger(self.__class__.__name__)

    def __len__(self) -> int:
        """Return dataset size"""
        data = self._data if self._num_rows is None else self._data[:self._num_rows]
        return len(data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get item by index"""
        data = self._data if self._num_rows is None else self._data[:self._num_rows]
        if idx >= len(data):
            raise IndexError("Index out of range")
        return data[idx]

    def get_dataset_size(self) -> int:
        """Get dataset size (compatibility method)"""
        return len(self)

    def get_data(self) -> List[Dict[str, Any]]:
        """Get the underlying data list"""
        return self._data if self._num_rows is None else self._data[:self._num_rows]
