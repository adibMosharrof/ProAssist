"""DST ProAssist Data Collator for DST Training.

This collator processes event data and builds continuous sequences with:
- <image> tokens for each frame
- [DST] prefix for DST updates
- [ASST] prefix for assistant responses
- Binary labels for speaking/DST decision heads at <image> positions
"""

import logging
from typing import Dict, Any, List
import torch
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
import datasets as hf_datasets

logger = logging.getLogger(__name__)


class DSTProAssistCollator:
    """Data collator for DST ProAssist DST training.
    
    Processes event data where only frames with events (speaking or DST updates)
    are stored. Builds continuous sequences with role tokens and binary labels.
    
    Input format (from sparse_sequence_generator.py):
    {
        "video_uid": str,
        "clip_idx": int,
        "start_frame": int,
        "end_frame": int,
        "events": [
            {
                "frame_idx": int,
                "speaking": 0/1,
                "dst_update": 0/1,
                "dst_updates": ["S1->start", ...],
                "response": str or null
            },
            ...
        ],
        "siglip_features_path": str  # Path to .arrow file with embeddings
    }
    
    Output format:
    {
        "input_ids": [batch_size, seq_len],
        "speaking_gen_labels": [batch_size, seq_len],  # LM labels for assistant responses
        "dst_gen_labels": [batch_size, seq_len],  # LM labels for DST updates
        "speaking_labels": [batch_size, seq_len],  # Binary labels at <image> positions
        "dst_update_labels": [batch_size, seq_len],  # Binary labels at <image> positions
        "image_embeds": List[Tensor],  # List of [num_frames, 1152] tensors
    }
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_len: int = 4096,
        siglip_features_dir: Path = None,
        negative_sampling_rate: float = 1.0,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.siglip_features_dir = Path(siglip_features_dir) if siglip_features_dir else None
        self.negative_sampling_rate = negative_sampling_rate
        
        # Get token IDs from tokenizer
        self.img_token_id = tokenizer.convert_tokens_to_ids("<image>")
        self.dst_token_id = tokenizer.convert_tokens_to_ids("[DST]")
        self.asst_token_id = tokenizer.convert_tokens_to_ids("[ASST]")
        self.eos_token_id = tokenizer.eos_token_id
        
        # Set pad_token_id (use eos_token_id if not set)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of DST ProAssist samples."""
        all_input_ids = []
        all_speaking_gen_labels = []
        all_dst_gen_labels = []
        all_speaking_labels = []
        all_dst_labels = []
        all_image_embeds = []
        
        for sample in samples:
            # Build continuous sequence from events
            input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels = self._build_sequence(sample)
                
            # Load SigLIP embeddings
            embeddings = self._load_embeddings(sample)
            
            all_input_ids.append(torch.tensor(input_ids, dtype=torch.long))
            all_speaking_gen_labels.append(torch.tensor(speaking_gen_labels, dtype=torch.long))
            all_dst_gen_labels.append(torch.tensor(dst_gen_labels, dtype=torch.long))
            all_speaking_labels.append(torch.tensor(speaking_labels, dtype=torch.long))
            all_dst_labels.append(torch.tensor(dst_labels, dtype=torch.long))
            all_image_embeds.append(embeddings)
        
        # Pad sequences
        input_ids = pad_sequence(all_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        speaking_gen_labels = pad_sequence(all_speaking_gen_labels, batch_first=True, padding_value=-100)
        dst_gen_labels = pad_sequence(all_dst_gen_labels, batch_first=True, padding_value=-100)
        speaking_labels = pad_sequence(all_speaking_labels, batch_first=True, padding_value=-100)
        dst_labels = pad_sequence(all_dst_labels, batch_first=True, padding_value=-100)
        
        return {
            "input_ids": input_ids,
            "labels": speaking_gen_labels,  # Required for Trainer to detect labels and extract loss
            "speaking_gen_labels": speaking_gen_labels,
            "dst_gen_labels": dst_gen_labels,
            "speaking_labels": speaking_labels,
            "dst_update_labels": dst_labels,
            "image_embeds": all_image_embeds,  # List of tensors
        }
    
    def _build_sequence(self, sample: Dict[str, Any]):
        """Build continuous sequence from conversation.
        
        For each frame in [start_frame, end_frame):
        1. Add <image> token
        2. Set binary labels (speaking=0/1, dst_update=0/1)
        3. If frame has events, add [DST] + DST text or [ASST] + response
        
        Returns:
            input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels
        """
        input_ids = []
        speaking_gen_labels = []
        dst_gen_labels = []
        speaking_labels = []
        dst_labels = []
        
        start_frame = sample["start_frame"]
        end_frame = sample["end_frame"]
        conversation = sample.get("conversation", [])
        
        # Parse conversation into events by frame
        events_by_frame = {}
        
        for turn in conversation:
            role = turn.get("role", "")
            turn_start = turn.get("start_frame", 0)
            
            # Initialize event for this frame if needed
            if turn_start not in events_by_frame:
                events_by_frame[turn_start] = {
                    "speaking": 0,
                    "dst_update": 0,
                    "dst_updates": [],
                    "responses": [],
                    "system_instruction": None
                }
            
            event = events_by_frame[turn_start]
            
            # Use explicit flags if present, otherwise derive from role
            if "speaking" in turn:
                event["speaking"] = turn["speaking"]
            if "dst_update" in turn:
                event["dst_update"] = turn["dst_update"]
            
            if role == "DST_UPDATE":
                if "dst_update" not in turn:
                    event["dst_update"] = 1
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            step_id = item.get("id", "")
                            transition = item.get("transition", "")
                            dst_text = f"{step_id}->{transition}"
                            event["dst_updates"].append(dst_text)
                elif isinstance(content, str):
                    event["dst_updates"].append(content)
            
            elif role == "assistant":
                if "speaking" not in turn:
                    event["speaking"] = 1
                content = turn.get("content", "")
                if content:
                    event["responses"].append(content)
            
            elif role == "system":
                event["system_instruction"] = turn.get("content", "")
        
        # Negative frame subsampling (following ProAssist's approach)
        all_frames = set(range(start_frame, end_frame))
        positive_frames = set(events_by_frame.keys())
        
        # Negative frames = all - positive
        negative_frames = sorted(list(all_frames - positive_frames))
        
        # Sample negative frames based on sampling_rate
        sampled_negative_frames = set()
        if negative_frames and self.negative_sampling_rate > 0:
            if self.negative_sampling_rate >= 1.0:
                sampled_negative_frames = set(negative_frames)
            else:
                import random
                num_sample = max(int(self.negative_sampling_rate * len(negative_frames)), 1)
                sampled_negative_frames = set(random.sample(negative_frames, num_sample))
        
        # Process ALL frames in the clip
        for frame_idx in range(start_frame, end_frame):
            # Add <image> token
            input_ids.append(self.img_token_id)
            speaking_gen_labels.append(-100)  # Don't predict <image>
            dst_gen_labels.append(-100)  # Don't predict <image>
            
            # Get event data for this frame (if any)
            event = events_by_frame.get(frame_idx, {})
            speaking = event.get("speaking", 0)
            dst_update = event.get("dst_update", 0)
            
            # Binary labels at <image> position
            if frame_idx in positive_frames:
                speaking_labels.append(speaking)
                dst_labels.append(dst_update)
            elif frame_idx in sampled_negative_frames:
                speaking_labels.append(0)
                dst_labels.append(0)
            else:
                speaking_labels.append(-100)
                dst_labels.append(-100)
            
            # Add system instruction (if any)
            system_instruction = event.get("system_instruction")
            if system_instruction:
                sys_tokens = self.tokenizer.encode(system_instruction, add_special_tokens=False)
                input_ids.extend(sys_tokens)
                speaking_gen_labels.extend([-100] * len(sys_tokens))
                dst_gen_labels.extend([-100] * len(sys_tokens))
                speaking_labels.extend([-100] * len(sys_tokens))
                dst_labels.extend([-100] * len(sys_tokens))
            
            # Add DST updates (if any)
            for dst_text in event.get("dst_updates", []):
                dst_tokens = [self.dst_token_id] + self.tokenizer.encode(
                    dst_text, add_special_tokens=False
                )
                dst_tokens.append(self.eos_token_id)
                
                input_ids.extend(dst_tokens)
                speaking_gen_labels.extend([-100] * len(dst_tokens))
                dst_gen_labels.extend(dst_tokens)
                speaking_labels.extend([-100] * len(dst_tokens))
                dst_labels.extend([-100] * len(dst_tokens))
            
            # Add assistant responses (if any)
            for response in event.get("responses", []):
                resp_tokens = [self.asst_token_id] + self.tokenizer.encode(
                    response, add_special_tokens=False
                )
                resp_tokens.append(self.eos_token_id)
                
                input_ids.extend(resp_tokens)
                speaking_gen_labels.extend(resp_tokens)
                dst_gen_labels.extend([-100] * len(resp_tokens))
                speaking_labels.extend([-100] * len(resp_tokens))
                dst_labels.extend([-100] * len(resp_tokens))
        
        return input_ids, speaking_gen_labels, dst_gen_labels, speaking_labels, dst_labels
    
    def _load_embeddings(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Load SigLIP embeddings from .arrow file.
        
        Returns:
            Tensor of shape [num_frames, 1152]
        """
        if not self.siglip_features_dir:
            raise ValueError("siglip_features_dir must be provided to load embeddings")
        
        # Get clip ID from sample
        clip_id = sample.get("id") or sample.get("video_uid")
        
        # Construct path to .arrow file
        # Format: {siglip_features_dir}/{dataset_name}/siglip_features/{clip_id}.arrow
        # We need to infer dataset_name from the path or sample
        dataset_name = sample.get("dataset_name", "assembly101")  # Default to assembly101
        
        arrow_path = self.siglip_features_dir / dataset_name / "siglip_features" / f"{clip_id}.arrow"
        
        if not arrow_path.exists():
            raise FileNotFoundError(f"SigLIP features not found: {arrow_path}")
        
        
        # Load from arrow file
        dataset = hf_datasets.Dataset.from_file(str(arrow_path))
        
        # Extract CLS embeddings
        embeddings = []
        for i in range(len(dataset)):
            cls_embed = dataset[i]["cls"]  # [1152]
            embeddings.append(torch.tensor(cls_embed, dtype=torch.bfloat16))
        
        # Stack into [num_frames, 1152]
        return torch.stack(embeddings, dim=0)
