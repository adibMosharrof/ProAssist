from pathlib import Path
from typing import Optional, Union, List

from dst_data_builder.data_sources.base_dst_dataset import BaseDSTDataset


class ManualDSTDataset(BaseDSTDataset):
    """Dataset for manual JSON files in a single directory.

    This dataset lists JSON files under `data_path` and loads them on demand.
    It intentionally does not depend on a separate `ManualDataSource` class.
    """

    def __init__(self, data_path: Union[str, Path], num_rows: Optional[int] = None):
        super().__init__()
        self.base_dir = Path(data_path)
        # Ensure base_dir exists; if not, dataset will have zero items
        if not self.base_dir.exists():
            self._items = []
        else:
            self._items = sorted(self.base_dir.glob("*.json"))

        self._num_rows = None if (num_rows is None or num_rows == -1) else int(num_rows)
