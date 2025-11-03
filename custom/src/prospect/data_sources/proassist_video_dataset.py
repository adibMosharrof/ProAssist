"""ProAssist video dataset for PROSPECT evaluation"""

import io
import json
import logging
import base64
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
import pyarrow as pa
from PIL import Image
from torch.utils.data import Dataset


logger = logging.getLogger(__name__)


@dataclass
class VideoSample:
    """Single video sample for PROSPECT evaluation"""

    video_id: str
    frames: List[Image.Image]
    frame_indices: List[int]
    timestamps: List[float]
    dst_annotations: pd.DataFrame
    ground_truth_dialogues: List[Dict[str, Any]]
    dataset_name: str = "assembly101"


class ProAssistVideoDataset(Dataset):
    """Dataset for loading ProAssist videos with DST annotations

    This dataset is compatible with ProAssist's evaluation framework.
    Each sample represents a full video with frames, DST annotations,
    and ground truth dialogues.
    """

    def __init__(
        self,
        data_path: str,
        dst_annotation_path: str,
        dialogue_path: Optional[str] = None,
        video_ids: Optional[List[str]] = None,
        frame_dir: str = "frames",
        fps: int = 2,
        **kwargs,
    ):
        """
        Initialize ProAssist video dataset

        Args:
            data_path: Path to processed data (contains frames/)
            dst_annotation_path: Path to DST TSV annotations
            dialogue_path: Path to ground truth dialogues (optional)
            video_ids: List of video IDs to load (if None, discover all)
            frame_dir: Subdirectory containing frame arrow files
            fps: Frames per second
        """
        self.data_path = Path(data_path)
        self.dst_annotation_path = Path(dst_annotation_path)
        self.dialogue_path = Path(dialogue_path) if dialogue_path else None
        self.frame_dir = frame_dir
        self.fps = fps

        # Discover videos
        if video_ids:
            self.video_ids = video_ids
            logger.info(f"Loading {len(video_ids)} specified videos")
        else:
            self.video_ids = self._discover_videos()
            logger.info(f"Discovered {len(self.video_ids)} videos")

        # Load all samples
        self.samples = self._load_all_samples()
        logger.info(f"Successfully loaded {len(self.samples)} video samples")

    def _discover_videos(self) -> List[str]:
        """Discover available videos from TSV files"""
        tsv_files = list(self.dst_annotation_path.glob("*.tsv"))
        video_ids = []

        for tsv in tsv_files:
            # Extract video_id from filename
            # E.g., assembly_nusar-2021_...9011-c03f...tsv -> 9011-c03f
            filename = tsv.stem
            if "_" in filename:
                parts = filename.split("_")
                for part in parts:
                    # Look for patterns like "9011-c03f", "P01_11", "T48"
                    if "-" in part and len(part) < 20:
                        video_ids.append(part)
                        break
                    elif part.startswith("P") and len(part) < 10:
                        video_ids.append(part)
                        break
                    elif part.startswith("T") and part[1:].isdigit():
                        video_ids.append(part)
                        break

        return video_ids

    def _load_all_samples(self) -> List[VideoSample]:
        """Load all video samples"""
        samples = []
        for video_id in self.video_ids:
            try:
                sample = self._load_single_video(video_id)
                samples.append(sample)
                logger.debug(f"Loaded video {video_id}: {len(sample.frames)} frames")
            except Exception as e:
                logger.error(f"Failed to load {video_id}: {e}")

        if not samples:
            raise ValueError("No videos were successfully loaded!")

        return samples

    def _load_single_video(self, video_id: str) -> VideoSample:
        """Load a single video sample"""
        # Load DST annotations
        dst_df = self._load_dst_tsv(video_id)

        # Load frames
        frames, frame_indices, timestamps = self._load_frames(video_id)

        # Load ground truth dialogues (optional)
        gt_dialogues = self._load_dialogues(video_id)

        return VideoSample(
            video_id=video_id,
            frames=frames,
            frame_indices=frame_indices,
            timestamps=timestamps,
            dst_annotations=dst_df,
            ground_truth_dialogues=gt_dialogues,
        )

    def _load_dst_tsv(self, video_id: str) -> pd.DataFrame:
        """Load DST annotations from TSV"""
        tsv_files = list(self.dst_annotation_path.glob(f"*{video_id}*.tsv"))

        if not tsv_files:
            raise FileNotFoundError(
                f"No DST TSV file found for video {video_id} in {self.dst_annotation_path}"
            )

        if len(tsv_files) > 1:
            logger.warning(
                f"Multiple TSV files found for {video_id}, using first: {tsv_files[0]}"
            )

        df = pd.read_csv(tsv_files[0], sep="\t")
        logger.debug(f"Loaded DST annotations: {len(df)} rows")
        return df

    def _load_frames(self, video_id: str) -> tuple:
        """Load frames from Arrow file"""
        frame_path = self.data_path / self.frame_dir
        arrow_files = list(frame_path.glob(f"*{video_id}*.arrow"))

        if not arrow_files:
            raise FileNotFoundError(
                f"No frame arrow file found for video {video_id} in {frame_path}"
            )

        if len(arrow_files) > 1:
            logger.warning(
                f"Multiple arrow files found for {video_id}, using first: {arrow_files[0]}"
            )

        # Read arrow file (using Arrow IPC Stream format)
        with pa.memory_map(str(arrow_files[0]), "r") as source:
            reader = pa.ipc.open_stream(source)
            table = reader.read_all()

        # Extract images (column name is 'frame', data is base64-encoded strings)
        image_column = "frame" if "frame" in table.column_names else "image"
        images = []
        for frame_data in table[image_column]:
            frame_str = frame_data.as_py()
            # Decode base64 string to image
            img_bytes = base64.b64decode(frame_str)
            img = Image.open(io.BytesIO(img_bytes))
            images.append(img)

        frame_indices = list(range(len(images)))
        timestamps = [i / self.fps for i in frame_indices]

        logger.debug(f"Loaded {len(images)} frames")
        return images, frame_indices, timestamps

    def _load_dialogues(self, video_id: str) -> List[Dict]:
        """Load ground truth dialogues (optional for evaluation)"""
        if not self.dialogue_path or not self.dialogue_path.exists():
            logger.debug("No dialogue path provided, skipping ground truth dialogues")
            return []

        dialogue_files = list(self.dialogue_path.glob(f"*{video_id}*.json"))

        if not dialogue_files:
            logger.debug(f"No ground truth dialogue file found for {video_id}")
            return []

        if len(dialogue_files) > 1:
            logger.warning(
                f"Multiple dialogue files found for {video_id}, using first: {dialogue_files[0]}"
            )

        with open(dialogue_files[0]) as f:
            data = json.load(f)

        # Extract assistant dialogues with timestamps
        dialogues = []
        conversation = data.get("conversation", [])

        for turn in conversation:
            if turn.get("from") == "assistant":
                dialogues.append(
                    {
                        "timestamp": turn.get("timestamp", 0),
                        "content": turn.get("value", ""),
                    }
                )

        logger.debug(f"Loaded {len(dialogues)} ground truth dialogues")
        return dialogues

    def __len__(self) -> int:
        """Return number of videos in dataset"""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a video sample in format compatible with ProAssist evaluators

        Returns:
            Dict with keys:
                - video_id: str
                - dataset: str
                - frames: List[Image.Image]
                - conversation: List[Dict] (ground truth dialogues)
                - dst_annotations: pd.DataFrame
                - fps: int
        """
        sample = self.samples[idx]

        # Convert ground truth dialogues to conversation format
        conversation = [
            {"from": "assistant", "value": d["content"], "timestamp": d["timestamp"]}
            for d in sample.ground_truth_dialogues
        ]

        return {
            "video_id": sample.video_id,
            "dataset": sample.dataset_name,
            "frames": sample.frames,
            "conversation": conversation,
            "dst_annotations": sample.dst_annotations,
            "fps": self.fps,
        }

    @property
    def dataset_name(self) -> str:
        """Dataset name for evaluation"""
        return "prospect/proassist_dst"
