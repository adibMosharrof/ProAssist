"""DST Data Collator with multimodal frame processing.

This collator processes DST training samples with conversation + video frames + DST data
in a multimodal training setup.
"""

import logging
from typing import Dict, Any, List, Tuple
import torch

logger = logging.getLogger(__name__)


class DSTDataCollator:
    """Data collator for multimodal DST training with video frames."""

    def __init__(
        self,
        max_seq_len: int = 4096,
        num_dst_states: int = 3,
        frame_size: Tuple[int, int] = (224, 224),
        normalize_frames: bool = True,
        tokenizer=None,
    ):
        self.max_seq_len = max_seq_len
        self.num_dst_states = num_dst_states
        self.frame_size = frame_size
        self.normalize_frames = normalize_frames
        self.tokenizer = tokenizer

        logger.info("Initialized DSTDataCollator with stateless multimodal processing")

    def __call__(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Process a batch of multimodal DST training samples."""
        return self._process_multimodal_batch(samples)

    def _process_multimodal_batch(self, samples: list[dict]) -> dict[str, torch.Tensor]:
        """Process multimodal batch with conversation + frames + DST data."""
        batch_size = len(samples)

        # Extract basic mmassist fields
        video_uids = [s["video_uid"] for s in samples]
        conversations = [s["conversation"] for s in samples]
        sample_indices = [s["sample_idx"] for s in samples]

        # Extract multimodal data
        frame_lists = [s.get("frames", []) for s in samples]
        frame_metadata = [s.get("frame_metadata", {}) for s in samples]
        dst_data_list = [s.get("dst", []) for s in samples]
        dst_labels_list = [s.get("dst_labels", {}) for s in samples]
        speaking_labels_list = [s.get("speaking_labels", {}) for s in samples]

        # Process conversation tokens with real tokenization
        input_ids, attention_mask, labels = self._process_conversations(
            conversations, batch_size, self.tokenizer
        )

        # Process video frames
        frame_tensors = self._process_video_frames(frame_lists, batch_size)

        # Prepare DST tensors
        dst_tensors = self._prepare_dst_tensors(dst_labels_list, dst_data_list)

        # Prepare speaking labels
        speaking_tensors = self._prepare_speaking_tensors(speaking_labels_list)

        # Prepare frame metadata
        frame_meta_tensors = self._prepare_frame_metadata(frame_metadata, batch_size)

        batch = {
            # Basic mmassist tensors
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # Multimodal video frame data
            "video_frames": frame_tensors,
            "frame_mask": frame_meta_tensors["frame_mask"],
            "frame_count": frame_meta_tensors["frame_count"],
            # mmassist metadata
            "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
            "video_uid": video_uids,
            "conversation": conversations,  # Keep raw conversations for processing
            # DST-specific tensors
            "dst_update_labels": dst_tensors["dst_update_labels"],
            "dst_state_labels": dst_tensors["dst_state_labels"],
            "temporal_dst_update_labels": dst_tensors["temporal_dst_update_labels"],  # NEW: For BCE loss
            "event_dst_targets": dst_tensors["event_dst_targets"],  # NEW: Ground truth transitions
            "temporal_speaking_labels": speaking_tensors["temporal_speaking_labels"],  # For BCE loss
            "event_speaking_targets": speaking_tensors["event_speaking_targets"],  # Ground truth text + metadata
            "max_events": speaking_tensors["max_events"],
            # Combined metadata
            "user_types": [s.get("user_type", "unknown") for s in samples],
            "frame_sampling_strategy": [
                meta.get("frame_sampling_strategy", "uniform")
                for meta in frame_metadata
            ],
            "num_dst_steps": dst_tensors["num_steps"],
            "video_duration": dst_tensors["video_duration"],
            "fps": [meta.get("fps", 2) for meta in frame_metadata],
        }

        return batch

    def _process_conversations(
        self, conversations: List[List[Dict]], batch_size: int, tokenizer
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process conversations into input tensors with REAL tokenization."""

        if tokenizer is None:
            # Fallback to fake tokenization if no tokenizer provided
            input_ids = torch.randint(
                1000, 30000, (batch_size, self.max_seq_len), dtype=torch.long
            )
            attention_mask = torch.ones(batch_size, self.max_seq_len, dtype=torch.long)
            labels = torch.roll(input_ids, shifts=-1, dims=1)
            labels[:, -1] = -100  # Padding token for labels
            return input_ids, attention_mask, labels

        # REAL tokenization of conversations
        input_ids_batch = []
        attention_mask_batch = []

        for conversation in conversations:
            # Convert conversation to text
            conversation_text = ""
            for turn in conversation:
                # Handle both enhanced format (type) and legacy format (role)
                event_type = turn.get("type", turn.get("role", ""))
                content = turn.get("content", "")
                
                if content:
                    if event_type == "SPEAK":
                        conversation_text += f"Assistant: {content}\n"
                    elif event_type == "DST_UPDATE":
                        # Include DST transition information
                        transitions = turn.get("content", [])
                        if isinstance(transitions, list):
                            trans_text = ", ".join([f"{t.get('id', '?')}:{t.get('transition', '?')}" for t in transitions])
                            conversation_text += f"DST Update: {trans_text}\n"
                        else:
                            conversation_text += f"DST Update: {transitions}\n"
                    elif event_type:
                        conversation_text += f"{event_type}: {content}\n"
                    else:
                        conversation_text += f"{content}\n"

            # Tokenize the conversation text
            encoded = tokenizer(
                conversation_text,
                truncation=True,
                padding="max_length",
                max_length=self.max_seq_len,
                return_tensors="pt",
            )

            input_ids_batch.append(encoded["input_ids"].squeeze(0))
            attention_mask_batch.append(encoded["attention_mask"].squeeze(0))

        # Stack into batch tensors
        input_ids = torch.stack(input_ids_batch)
        attention_mask = torch.stack(attention_mask_batch)

        # Create labels (shifted for language modeling)
        labels = input_ids.clone()
        # For language modeling, we shift everything by 1 position
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = -100  # Padding token for labels

        return input_ids, attention_mask, labels

    def _process_video_frames(
        self, frame_lists: List[List[torch.Tensor]], batch_size: int
    ) -> torch.Tensor:
        """Process video frame lists into padded tensors."""
        # if not frame_lists or not any(frame_lists):
        #     # Return empty frames
        #     return torch.zeros(batch_size, 0, 3, *self.frame_size, dtype=torch.float32)

        # Find max number of frames in batch
        max_frames = max(len(frames) for frames in frame_lists) if frame_lists else 0
        if max_frames == 0:
            return torch.zeros(batch_size, 0, 3, *self.frame_size, dtype=torch.float32)

        # Prepare padded frame tensor (batch_size, max_frames, channels, height, width)
        frame_tensor = torch.zeros(
            batch_size, max_frames, 3, *self.frame_size, dtype=torch.float32
        )

        for batch_idx, frames in enumerate(frame_lists):
            for frame_idx, frame in enumerate(frames):
                if frame_idx < max_frames:
                    # Resize frame to target size and normalize if needed
                    processed_frame = self._resize_and_normalize_frame(frame)
                    # Ensure we have the right number of channels
                    if processed_frame.dim() == 3:
                        frame_tensor[batch_idx, frame_idx] = processed_frame
                    elif processed_frame.dim() == 2:
                        # Grayscale to RGB
                        frame_tensor[batch_idx, frame_idx, 0] = processed_frame
                        frame_tensor[batch_idx, frame_idx, 1] = processed_frame
                        frame_tensor[batch_idx, frame_idx, 2] = processed_frame

        return frame_tensor

    def _resize_and_normalize_frame(self, frame: torch.Tensor) -> torch.Tensor:
        """Resize frame to target size and normalize if needed."""
        # Handle different input formats
        if frame.dim() == 2:
            # Grayscale, expand to RGB
            frame = frame.unsqueeze(0).repeat(3, 1, 1)
        elif frame.dim() == 3:
            # (H, W, C) -> (C, H, W)
            frame = frame.permute(2, 0, 1)
        elif frame.dim() == 4:
            # Remove batch dimension if present
            frame = frame.squeeze(0)

        # Resize to target size
        from torchvision.transforms.functional import resize

        frame = resize(frame, self.frame_size)

        # Normalize frames if needed
        if self.normalize_frames:
            # Normalize to [0, 1] range
            frame = frame.float() / 255.0 if frame.max() > 1.0 else frame.float()

        return frame

    def _prepare_dst_tensors(
        self, dst_labels_list: List[Dict], dst_data_list: List[List]
    ) -> Dict[str, torch.Tensor]:
        """Prepare DST-specific tensors for training with temporal structure."""
        batch_size = len(dst_labels_list)
        max_steps = (
            max(len(labels.get("step_ids", [])) for labels in dst_labels_list)
            if dst_labels_list
            else 0
        )

        if max_steps == 0:
            return {
                "dst_update_labels": torch.zeros((batch_size, 1), dtype=torch.long),
                "dst_state_labels": torch.zeros((batch_size, 1), dtype=torch.long),
                "temporal_dst_update_labels": torch.zeros((batch_size, 1), dtype=torch.long),  # NEW
                "event_dst_targets": [],  # NEW
                "num_steps": torch.zeros((batch_size,), dtype=torch.long),
                "video_duration": torch.zeros((batch_size,), dtype=torch.float),
            }

        # Prepare padded tensors
        dst_update_labels = torch.zeros((batch_size, max_steps), dtype=torch.long)
        dst_state_labels = torch.zeros((batch_size, max_steps), dtype=torch.long)
        
        # NEW: Temporal DST update labels (similar to temporal speaking labels)
        temporal_dst_update_labels = []
        event_dst_targets = []
        max_events = 0
        
        num_steps = torch.zeros((batch_size,), dtype=torch.long)
        video_duration = torch.zeros((batch_size,), dtype=torch.float)

        for i, (dst_labels, dst_data) in enumerate(zip(dst_labels_list, dst_data_list)):
            # Get DST labels
            updates = dst_labels.get("dst_update_labels", [])
            states = dst_labels.get("dst_state_labels", [])

            # Pad to max_steps
            padded_updates = updates + [0] * (max_steps - len(updates))
            padded_states = states + [1] * (max_steps - len(states))

            dst_update_labels[i, :] = torch.tensor(
                padded_updates[:max_steps], dtype=torch.long
            )
            dst_state_labels[i, :] = torch.tensor(
                padded_states[:max_steps], dtype=torch.long
            )

            # NEW: Handle temporal DST labels
            temporal_labels = dst_labels.get("temporal_dst_update_labels", [])
            temporal_dst_update_labels.append(temporal_labels)
            max_events = max(max_events, len(temporal_labels))
            
            event_targets = dst_labels.get("event_dst_targets", [])
            event_dst_targets.append(event_targets)

            num_steps[i] = len(dst_data)
            video_duration[i] = dst_labels.get("video_duration", 0.0)
        
        # Pad temporal DST update labels
        padded_temporal_dst_labels = torch.zeros(
            (batch_size, max_events), dtype=torch.long
        )
        
        for i, labels in enumerate(temporal_dst_update_labels):
            padded_labels = labels + [0] * (max_events - len(labels))
            padded_temporal_dst_labels[i, :] = torch.tensor(padded_labels[:max_events], dtype=torch.long)

        return {
            "dst_update_labels": dst_update_labels,
            "dst_state_labels": dst_state_labels,
            "temporal_dst_update_labels": padded_temporal_dst_labels,  # NEW: For BCE loss
            "event_dst_targets": event_dst_targets,  # NEW: Ground truth transitions
            "num_steps": num_steps,
            "video_duration": video_duration,
        }

    def _prepare_speaking_tensors(
        self, speaking_labels_list: List[Dict]
    ) -> Dict[str, torch.Tensor]:
        """Prepare speaking labels as tensors for multi-task training."""
        
        # Temporal speaking labels for BCE loss
        temporal_speaking_labels = []
        
        # Event-level speaking targets for ground truth text comparison
        event_speaking_targets = []
        
        max_events = 0
        
        for speaking_labels in speaking_labels_list:
            # Temporal speaking decisions (1=speak, 0=silent)
            temporal_labels = speaking_labels.get("temporal_speaking_labels", [])
            temporal_speaking_labels.append(temporal_labels)
            
            # Ground truth text and metadata for evaluation
            event_targets = speaking_labels.get("event_speaking_targets", [])
            event_speaking_targets.append(event_targets)
            
            max_events = max(max_events, len(temporal_labels))
        
        # Pad temporal speaking labels for batching
        padded_temporal_labels = torch.zeros(
            (len(speaking_labels_list), max_events), dtype=torch.long
        )
        
        for i, labels in enumerate(temporal_speaking_labels):
            padded_labels = labels + [0] * (max_events - len(labels))
            padded_temporal_labels[i, :] = torch.tensor(padded_labels[:max_events], dtype=torch.long)
        
        return {
            "temporal_speaking_labels": padded_temporal_labels,  # For BCE loss
            "event_speaking_targets": event_speaking_targets,    # Ground truth text + metadata
            "max_events": max_events,
        }

    def _prepare_frame_metadata(
        self, frame_metadata: List[Dict], batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """Prepare frame metadata as tensors."""
        frame_counts = []
        frame_masks = []

        max_frames = 0
        for meta in frame_metadata:
            frame_counts.append(meta.get("frame_count", 0))
            max_frames = max(max_frames, meta.get("frame_count", 0))

        # Create frame masks (1 for real frames, 0 for padding)
        for i, count in enumerate(frame_counts):
            mask = torch.zeros(max_frames, dtype=torch.long)
            mask[:count] = 1
            frame_masks.append(mask)

        return {
            "frame_count": torch.tensor(frame_counts, dtype=torch.long),
            "frame_mask": (
                torch.stack(frame_masks)
                if frame_masks
                else torch.zeros((batch_size, 0), dtype=torch.long)
            ),
        }
