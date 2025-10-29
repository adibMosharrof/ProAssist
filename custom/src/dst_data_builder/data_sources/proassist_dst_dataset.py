from pathlib import Path
from typing import Optional, Union, List

from dst_data_builder.data_sources.base_dst_dataset import BaseDSTDataset


class ProAssistDSTDataset(BaseDSTDataset):
    """Dataset for ProAssist data with nested JSON files.

    - If a `datasets` list is provided, it looks under
      `base_dir/processed_data/<dataset>/generated_dialogs/<split>` for JSON files
      where `<split>` is one of `train`, `val`, `test`.
    - Otherwise it falls back to a recursive search of all JSON files under
      `base_dir`.

    The `num_rows` argument truncates the results (None or -1 => no truncation).
    """

    def __init__(
        self,
        data_path: Union[str, Path],
        num_rows: Optional[int] = None,
        datasets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.base_dir = Path(data_path)
        self.datasets = datasets or []

        items: List[Path] = []

        if not self.base_dir.exists():
            items = []
        else:
            if self.datasets:
                # Look into <dataset>/generated_dialogs/<split>
                # (assumes base_dir already points to processed_data)
                for dataset_name in self.datasets:
                    dataset_path = self.base_dir / dataset_name / "generated_dialogs"
                    if dataset_path.exists():
                        for split in ["train", "val", "test"]:
                            split_path = dataset_path / split
                            if split_path.exists():
                                items.extend(sorted(split_path.rglob("*.json")))
            else:
                # Fallback: include all JSON files
                items = list(self.base_dir.rglob("*.json"))

        # Apply truncation if requested
        if num_rows is not None and num_rows != -1:
            items = items[: int(num_rows)]

        self._items = sorted(items)
        self._num_rows = None if (num_rows is None or num_rows == -1) else int(num_rows)
