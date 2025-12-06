#!/usr/bin/env python3
"""
SigLIP Vision Embeddings Saver for DST Generated Data

Extract and save vision embeddings using ProAssist's SigLIP encoder.
Produces Arrow files with CLS token embeddings for efficient training.

Usage:
    python -m custom.src.dst_data_builder.siglip_embeddings_saver
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import torch
import numpy as np
import datasets as hf_datasets
from datasets.arrow_writer import ArrowWriter
from PIL import Image
import base64
import io
from tqdm import tqdm

from mmassist.model.vision import VisualEncoder
from mmassist.model.configuration_proact import ProActConfig
from mmassist.data.utils import img_base64_to_tensor

logger = logging.getLogger(__name__)


class SigLIPEmbeddingsSaver:
    """
    Extract and save vision embeddings using ProAssist's SigLIP encoder.
    
    Following ProAssist pattern:
    - Load raw frames from arrow files  
    - Extract [CLS] token from SigLIP vision encoder
    - Save to Arrow files (ProAssist format)
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize SigLIP embeddings saver.
        
        Args:
            cfg: Hydra configuration object
        """
        self.vision_pretrained = cfg.model.get("vision_pretrained", "google/siglip-so400m-patch14-384")
        self.batch_size = cfg.processing.batch_size
        self.vision_hidden_size = cfg.model.get("vision_hidden_size", 1152)
        
        # Build full frames_root path
        project_root = Path(cfg.project_root)
        frames_root_relative = cfg.processing.frames_root
        self.frames_root = project_root / frames_root_relative
        
        # Determine device
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.device = f"cuda:{local_rank}"
        
        # Load SigLIP encoder
        self._load_vision_encoder()
        
        logger.info(f"✓ Initialized SigLIPEmbeddingsSaver on {self.device} with batch_size={self.batch_size}")
    
    def _load_vision_encoder(self):
        """Load SigLIP vision encoder using ProAssist's VisualEncoder."""
        logger.info(f"Loading SigLIP encoder from {self.vision_pretrained}...")
        
        config = ProActConfig(
            vision_pretrained=self.vision_pretrained,
            use_img_cls_token=True,
            img_patch_token_size=0,  # CLS token only
        )
        
        self.encoder = VisualEncoder.from_config(config)
        self.encoder = self.encoder.to(self.device, torch.float16)
        self.encoder.eval()
        self.encoder.requires_grad_(False)
        
        logger.info(f"✓ SigLIP encoder loaded ({self.vision_hidden_size}-dim output)")
    
    def validate_clip(self, clip: Dict[str, Any], dataset_name: str) -> bool:
        """Validate if a clip's frame indices are within bounds of the Arrow file."""
        video_uid = clip.get("video_uid")
        if not video_uid:
            return False
        
        # Build frame file path (Assembly101 special case)
        if dataset_name == "assembly101":
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        else:
            frame_file = f"frames/{video_uid}.arrow"
        
        arrow_path = self.frames_root / dataset_name / frame_file
        
        if not arrow_path.exists():
            logger.warning(f"Arrow file not found: {arrow_path}")
            return False
        
        try:
            dataset = hf_datasets.Dataset.from_file(str(arrow_path))
            dataset_len = len(dataset)
        except Exception as e:
            logger.warning(f"Could not read Arrow file {arrow_path}: {e}")
            return False
        
        start_frame = clip.get("start_frame_idx")
        end_frame = clip.get("end_frame_idx")
        
        if start_frame is None or end_frame is None:
            logger.warning(f"Missing frame indices for {video_uid}")
            return False
        
        if start_frame < 0 or end_frame > dataset_len or start_frame >= end_frame:
            logger.warning(f"Invalid frame range [{start_frame}, {end_frame}) for {video_uid}")
            return False
        
        return True
    
    def load_frames_from_arrow(
        self,
        arrow_file: Path,
        frame_indices: List[int],
    ) -> List[torch.Tensor]:
        """Load frames from Arrow file and convert to tensors."""
        if not arrow_file.exists():
            raise FileNotFoundError(f"Arrow file not found: {arrow_file}")
        
        dataset = hf_datasets.Dataset.from_file(str(arrow_file))
        frames = []
        
        for idx in frame_indices:
            if idx >= len(dataset):
                raise IndexError(f"Frame index {idx} exceeds dataset size {len(dataset)}")
            
            item = dataset[idx]
            
            # Extract frame data
            if "frame" in item:
                frame_data = item["frame"]
            elif "image" in item:
                frame_data = item["image"]
            else:
                raise KeyError(f"No 'frame' or 'image' key in Arrow file")
            
            # Convert to tensor
            if isinstance(frame_data, str):
                # Base64-encoded string -> tensor
                tensor = img_base64_to_tensor(frame_data)
            elif isinstance(frame_data, Image.Image):
                # PIL Image -> tensor (normalize to 0-1)
                tensor = torch.from_numpy(np.array(frame_data)).float() / 255.0
                tensor = tensor.permute(2, 0, 1)  # HWC -> CHW
            elif isinstance(frame_data, np.ndarray):
                tensor = torch.from_numpy(frame_data).float() / 255.0
                if tensor.dim() == 3 and tensor.shape[-1] == 3:
                    tensor = tensor.permute(2, 0, 1)
            else:
                raise TypeError(f"Unexpected frame type: {type(frame_data)}")
            
            frames.append(tensor)
        
        return frames
    
    def extract_embeddings(self, frames: List[torch.Tensor]) -> torch.Tensor:
        """Extract CLS token embeddings from frames using SigLIP encoder."""
        all_embeddings = []
        
        for batch_start in range(0, len(frames), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            
            # Stack frames into batch tensor
            batch_tensor = torch.stack(batch_frames).to(self.device, torch.float16)
            
            with torch.no_grad():
                # Encode with SigLIP -> [batch, seq_len, hidden_dim]
                embeddings = self.encoder.encode(batch_tensor)
                # Extract CLS token (first token) -> [batch, hidden_dim]
                cls_embeddings = embeddings[:, 0, :]
                all_embeddings.append(cls_embeddings.cpu().half())
        
        return torch.cat(all_embeddings, dim=0)
    
    def save_embeddings_for_clip(
        self,
        clip: Dict[str, Any],
        dataset_name: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """Extract and save SigLIP embeddings for a single clip."""
        video_uid = clip.get("video_uid")
        clip_id = clip.get("id", video_uid)
        
        # Build frame file path
        if dataset_name == "assembly101":
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        else:
            frame_file = f"frames/{video_uid}.arrow"
        
        arrow_path = self.frames_root / dataset_name / frame_file
        
        # Get frame indices
        start_frame = clip["start_frame_idx"]
        end_frame = clip["end_frame_idx"]
        frame_indices = list(range(start_frame, end_frame))
        
        # Check if output already exists
        output_file = output_dir / f"{clip_id}.arrow"
        if output_file.exists():
            logger.info(f"Skipping {clip_id}: Embeddings already exist")
            return {"clip_id": clip_id, "skipped": True}
        
        # Load frames
        frames = self.load_frames_from_arrow(arrow_path, frame_indices)
        
        if not frames:
            logger.warning(f"No frames loaded for clip {video_uid}")
            return {"clip_id": clip_id, "error": "No frames loaded"}
        
        # Extract embeddings
        embeddings = self.extract_embeddings(frames)  # [num_frames, hidden_dim]
        
        # Save in ProAssist Arrow format
        writer = ArrowWriter(path=str(output_file))
        for i in range(len(embeddings)):
            writer.write({
                "cls": embeddings[i].numpy(),  # [hidden_dim]
            })
        writer.finalize()
        
        logger.info(f"✓ Saved {clip_id}: {len(embeddings)} frames -> {output_file}")
        
        return {
            "clip_id": clip_id,
            "num_frames": len(embeddings),
            "output_file": str(output_file),
        }


def save_siglip_embeddings(cfg: DictConfig) -> Dict[str, Any]:
    """Main function to save SigLIP embeddings for all datasets."""
    dataset_output_dir = Path(cfg.dataset.output_dir)
    dataset_names = cfg.dataset.names
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Process {local_rank}/{world_size} starting...")
    
    summary = {
        "dataset_output_dir": str(dataset_output_dir),
        "process_index": local_rank,
        "num_processes": world_size,
        "datasets": {},
    }
    
    # Create saver
    saver = SigLIPEmbeddingsSaver(cfg)
    
    # Process each dataset
    for dataset_name in dataset_names:
        dataset_json_dir = dataset_output_dir / dataset_name
        
        if not dataset_json_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_json_dir}")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*70}")
        
        # Create siglip_features subdirectory
        features_dir = dataset_json_dir / "siglip_features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all training files
        # Support both old format (*_training.json) and new format (train.json, val.json, test.json)
        training_files = list(dataset_json_dir.glob("*_training.json"))
        if not training_files:
            # Try new format
            potential_files = list(dataset_json_dir.glob("*.json"))
            training_files = [
                f for f in potential_files 
                if f.name in ["train.json", "val.json", "test.json"]
            ]
        
        if not training_files:
            logger.warning(f"No training files found in {dataset_json_dir}")
            continue
        
        dataset_summary = {
            "features_dir": str(features_dir),
            "splits": {},
        }
        
        for data_file in sorted(training_files):
            if "_training" in data_file.name:
                split_name = data_file.stem.replace("_training", "")
            else:
                split_name = data_file.stem
            logger.info(f"\nProcessing split: {split_name}")
            
            # Load clips
            with open(data_file) as f:
                clips = json.load(f)
            
            # Split across processes
            clips_chunk = clips[local_rank::world_size]
            logger.info(f"Process {local_rank}: {len(clips_chunk)}/{len(clips)} clips")
            
            results = []
            for clip in tqdm(clips_chunk, desc=f"Extracting {split_name}"):
                if not saver.validate_clip(clip, dataset_name):
                    continue
                try:
                    result = saver.save_embeddings_for_clip(clip, dataset_name, features_dir)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed on clip {clip.get('id')}: {e}")
            
            dataset_summary["splits"][split_name] = {
                "total_clips": len(clips_chunk),
                "processed": len(results),
            }
        
        summary["datasets"][dataset_name] = dataset_summary
    
    return summary


@hydra.main(
    version_base=None,
    config_path="../../../custom/config/dst_data_generator",
    config_name="siglip_embeddings",
)
def main(cfg: DictConfig) -> None:
    """Main function to run SigLIP embeddings extraction."""
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("Starting SigLIP Embeddings Extraction")
    logger.info(f"Vision model: {cfg.model.vision_pretrained}")
    logger.info(f"Batch size: {cfg.processing.batch_size}")
    
    try:
        summary = save_siglip_embeddings(cfg)
        
        logger.info("✓ SigLIP embeddings extraction completed!")
        
        # Save summary
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        summary_file = output_dir / "siglip_embeddings_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Summary saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
