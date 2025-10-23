import json
from pathlib import Path
from typing import Dict, Any, List, Union, Optional


class BaseDSTDataset:
    """Base dataset class for DST datasets.

    This class intentionally keeps a very small implementation surface so
    concrete dataset classes can implement file listing/loading themselves
    without requiring a separate DataSource object.
    """

    def __init__(self):
        # Cache the items list for performance; concrete classes should set
        # self._items (list of Path-like objects) during initialization.
        self._items: Optional[List[Union[str, Path]]] = None
        # Optional truncation applied after listing (None or -1 => no truncation)
        self._num_rows: Optional[int] = None

    def _get_items(self) -> List[Union[str, Path]]:
        """Return cached items or raise if not initialized by concrete class."""
        if self._items is None:
            raise RuntimeError(
                "Dataset items not initialized. Concrete dataset must populate self._items."
            )

        if self._num_rows is None or int(self._num_rows) == -1:
            return list(self._items)
        return list(self._items)[: int(self._num_rows)]

    def __len__(self) -> int:
        return len(self._get_items())

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        items = self._get_items()
        item = items[idx]
        # Default behavior: treat item as a Path to a JSON file and load it.
        path = Path(item)
        with open(path, "r") as f:
            return json.load(f)

    def get_dataset_size(self) -> int:
        """Compatibility helper: total number of items after truncation"""
        return len(self._get_items())

    def get_file_paths(self) -> List[Path]:
        """Return a list of Path objects for all items (after truncation)."""
        items = self._get_items()
        file_paths: List[Path] = []
        for item in items:
            if isinstance(item, Path):
                file_paths.append(item)
            else:
                file_paths.append(Path(str(item)))
        return file_paths
