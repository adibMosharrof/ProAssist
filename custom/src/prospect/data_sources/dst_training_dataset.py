"""DST Training Dataset - Embeddings Only

This dataset loads precomputed vision embeddings from pickle files.
Each clip is one training sample with turn-by-turn conversation data.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
from torch.utils.data import Dataset

from custom.src.prospect.data_sources.dst_chat_formatter import DSTMultimodalChat

logger = logging.getLogger(__name__)


class DSTTrainingDataset(Dataset):
    """
    Dataset for DST-enhanced training with precomputed embeddings.

    Key Features:
    - Load clips from JSON files (each clip is one training sample)
    - Load precomputed vision embeddings from pickle files (no raw frame loading)
    - DST state context in system prompt
    - Turn-by-turn processing with conversation history
    - Embeddings are extracted offline using SmolVLM2 vision encoder
    """

    def __init__(
        self,
        data_path: str,
        step_name: str,
        dataset_name: str,
        max_seq_len: int = 4096,
        neg_frame_sampling_rate: float = 0.5,
    ):
        """
        Initialize DST Training Dataset with precomputed embeddings.

        Args:
            data_path: Path to data directory containing dataset folders with training JSON files
                      (e.g., 'custom/outputs/dst_generated/hybrid_dst/2025-11-26/16-38-41_gpt-4o_proassist_50rows')
            step_name: Step name (train, val, test)
            dataset_name: Dataset name (e.g., 'assembly101')
            max_seq_len: Maximum sequence length
            neg_frame_sampling_rate: Rate to sample negative (silent) frames (0.5 = 50%)
        """
        self.data_path = Path(data_path)
        self.step_name = step_name
        self.dataset_name = dataset_name
        self.max_seq_len = max_seq_len
        self.neg_frame_sampling_rate = neg_frame_sampling_rate

        # Lazy loading: only count clips without loading data into memory
        self._num_clips = self._get_num_clips()
        self._clips_cache = {}  # Keyed by data_file path

        logger.info(
            f"Initialized DST dataset with {self._num_clips} clips from {data_path} "
            f"(embeddings from {self.data_path}/{dataset_name}/frames, precomputed embeddings only, neg_sampling_rate={neg_frame_sampling_rate})"
        )

    def _get_data_file_path(self) -> Path:
        """Get the data file path for this dataset."""
        return self.data_path / self.dataset_name / f"{self.step_name}_training.json"

    def _load_clips_from_file(self, data_file: Path) -> List[Dict[str, Any]]:
        """Load clips from a JSON file."""
        if not data_file.exists():
            raise FileNotFoundError(f"Data file not found: {data_file}")

        with open(data_file, "r") as f:
            clips = json.load(f)

        # Validate data structure
        if not isinstance(clips, list):
            clips = [clips]  # Handle single-clip format

        return clips

    def _get_num_clips(self) -> int:
        """Count clips from JSON file without loading all data into memory."""
        data_file = self._get_data_file_path()
        clips = self._load_clips_from_file(data_file)
        return len(clips)

    def _load_clips(self) -> List[Dict[str, Any]]:
        """Load all clips from JSON file with caching for lazy loading."""
        data_file = self._get_data_file_path()
        data_file_str = str(data_file)
        
        # Return cached clips if already loaded
        if data_file_str in self._clips_cache:
            return self._clips_cache[data_file_str]
        
        clips = self._load_clips_from_file(data_file)

        # Cache the loaded clips for future access (keyed by data_file path)
        self._clips_cache[data_file_str] = clips

        return clips

    def __len__(self) -> int:
        """Return number of clips (training samples)."""
        return self._num_clips

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get one training sample (one clip with all its turns).

        For efficiency, loads PRECOMPUTED EMBEDDINGS instead of raw frames.
        The embeddings are expected to be stored as numpy arrays in pickle files
        extracted from video using SmolVLM2 vision encoder.

        Returns:
            {
                'video_uid': str,
                'clip_idx': int,
                'conversation': List[Dict],  # All turns in this clip
                'dst': List[Dict],           # Global DST steps
                'embeddings': torch.Tensor,  # Precomputed [num_frames, 2048]
                'neg_frame_sampling_rate': float,
                'sample_idx': int,
            }
        """
        import pickle
        
        # Lazy load clips on first access
        clips = self._load_clips()
        clip = clips[idx]

        # Use 'id' field which corresponds to the embeddings filename
        clip_id = clip["id"]
        
        # Load precomputed embeddings instead of raw frames
        # Embeddings are at: {data_path}/{dataset_name}/frames/{id}_embeddings.pkl
        embeddings_dir = self.data_path / self.dataset_name / "frames"
        embeddings_file = embeddings_dir / f"{clip_id}_embeddings.pkl"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(
                f"Embeddings file not found for clip {idx} (id={clip_id}). "
                f"Expected: {embeddings_file}\n"
                f"Ensure precomputed embeddings exist at: {embeddings_dir}/{clip_id}_embeddings.pkl"
            )
        
        try:
            with open(embeddings_file, "rb") as f:
                embeddings_np = pickle.load(f)
            embeddings = torch.from_numpy(embeddings_np).float()  # [num_frames, 2048]
        except Exception as e:
            logger.error(f"Failed to load embeddings from {embeddings_file}: {e}")
            raise
        
        # Filter valid turns (those with frames within embeddings bounds)
        # IMPORTANT: Frame indices in turns are VIDEO indices, need to convert to pickle indices
        valid_turns = []
        num_frames = embeddings.shape[0]
        clip_start_frame = clip.get("start_frame_idx", 0)
        
        for turn in clip["conversation"]:
            if "start_frame" in turn and "end_frame" in turn:
                start_frame = turn["start_frame"]
                end_frame = turn["end_frame"]
                
                # Convert video indices to pickle indices
                start_pickle = start_frame - clip_start_frame
                end_pickle = end_frame - clip_start_frame
                
                # Only keep turns where pickle indices are within bounds
                # end_pickle must be < num_frames for inclusive slicing [start:end+1]
                if start_pickle >= 0 and end_pickle < num_frames and end_pickle >= start_pickle:
                    valid_turns.append(turn)
            else:
                # Turns without frame info (like some system messages) are always valid
                valid_turns.append(turn)
        
        if not valid_turns:
            raise ValueError(
                f"No valid turns found for clip {idx} (embeddings_file={embeddings_file}). "
                f"Embeddings has {num_frames} frames. "
                "Ensure turn frame indices are within valid bounds."
            )
        
        sample = {
            "video_uid": clip["video_uid"],
            "clip_idx": clip.get("clip_idx", 0),
            "conversation": valid_turns,  # Use filtered turns
            "dst": clip["dst"],
            "initial_dst_state": clip.get("initial_dst_state", None),
            "embeddings": embeddings,  # Precomputed [num_frames, 2048]
            "neg_frame_sampling_rate": self.neg_frame_sampling_rate,
            "dataset": clip.get("dataset", self.dataset_name),
            "sample_idx": idx,
            "start_frame_idx": clip.get("start_frame_idx", 0),  # For frame index mapping
        }

        return sample

    def build_system_prompt(
        self, dst_steps: List[Dict], initial_dst_state: Optional[Dict] = None
    ) -> str:
        """
        Build system prompt with DST steps (replaces task knowledge).

        Format:
            You are a proactive assistant helping with task completion.

            Task Steps:
            - S1: Assemble the chassis by attaching and screwing the chassis parts together
            - S2: Attach wheels to the chassis
            - S3: Assemble the arm and attach it to the chassis
            ...

            [For split conversations, add current state:]
            Current Progress: S1:completed, S2:in_progress

        Args:
            dst_steps: Global DST array from JSON
            initial_dst_state: For split conversations (clip_idx > 0)

        Returns:
            System prompt string
        """
        prompt_parts = [
            "You are a proactive assistant helping with task completion.",
            "",
            "Task Steps:",
        ]

        # Add DST steps (id + name, omit timestamps)
        for step in dst_steps:
            prompt_parts.append(f"- {step['id']}: {step['name']}")

        # For split conversations, add current progress
        if initial_dst_state:
            state_str = ", ".join([f"{k}:{v}" for k, v in initial_dst_state.items()])
            prompt_parts.append("")
            prompt_parts.append(f"Current Progress: {state_str}")

        return "\n".join(prompt_parts)

    def is_silent_frame(self, turn: Dict) -> bool:
        """
        Identify silent frames (no assistant or DST_UPDATE).

        Silent frames: Model should not speak or update DST
        Active frames: assistant or DST_UPDATE turns

        Args:
            turn: Conversation turn dict

        Returns:
            True if silent (user turn with no model action)
        """
        return turn["role"] not in ["assistant", "DST_UPDATE"]

    def should_include_turn(self, turn: Dict, neg_sampling_rate: float) -> bool:
        """
        Determine if turn should be included in training (with negative sampling).

        Args:
            turn: Conversation turn dict
            neg_sampling_rate: Negative sampling rate (0.5 = 50% of silent frames)

        Returns:
            True if turn should be included
        """
        # Always include active frames (assistant, DST_UPDATE)
        if not self.is_silent_frame(turn):
            return True

        # For silent frames, apply negative sampling
        return random.random() < neg_sampling_rate

    def get_turn_type(self, turn: Dict) -> str:
        """
        Get turn type for loss computation.

        Returns:
            'assistant', 'DST_UPDATE', 'user', 'system'
        """
        return turn["role"]

    def get_dst_state_str(self, dst_state: Optional[Dict]) -> str:
        """
        Convert DST state dict to structured text format.

        Format: "S1:completed, S2:in_progress, S3:not_started"

        Args:
            dst_state: Dictionary mapping step IDs to states

        Returns:
            Formatted DST state string
        """
        if not dst_state:
            return ""

        return ", ".join([f"{k}:{v}" for k, v in dst_state.items()])

    def get_dst_update_str(self, dst_update_content: List[Dict]) -> str:
        """
        Convert DST update content to structured text format.

        Format: "S1->start" or "S2->complete"

        Args:
            dst_update_content: List of DST update dicts (usually single item)

        Returns:
            Formatted DST update string
        """
        if not dst_update_content:
            return ""

        # Handle single or multiple updates
        # User confirmed multiple updates are generated separately
        update = dst_update_content[0]  # Take first update
        return f"{update['id']}->{update['transition']}"
