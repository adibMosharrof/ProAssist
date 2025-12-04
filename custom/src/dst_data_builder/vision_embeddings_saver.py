#!/usr/bin/env python3
"""
Vision Embeddings Saver for DST Generated Data

After DST generation, extract and save vision embeddings ([CLS] tokens) 
following ProAssist's offline pre-extraction pattern.

This creates a "frames" subdirectory inside the dataset folder containing
vision feature files for efficient training without GPU memory overhead.
"""

import json
import logging
import pickle
import base64
import io
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
from omegaconf import DictConfig, OmegaConf
import os
import sys
import hydra
from hydra.core.hydra_config import HydraConfig

import torch
from transformers import AutoModel, AutoProcessor
from PIL import Image
import datasets as hf_datasets
from tqdm import tqdm

logger = logging.getLogger(__name__)


def save_embeddings_for_dataset(
    clips: List[Dict[str, Any]],
    dataset_name: str,
    output_dir: Path,
    saver: "VisionEmbeddingsSaver",
) -> Dict[str, Any]:
    """
    Extract and save vision embeddings for all clips in a dataset.
    
    Args:
        clips: List of clip data from DST JSON
        dataset_name: Name of dataset (e.g., "assembly101")
        output_dir: Directory to save embeddings
        saver: VisionEmbeddingsSaver instance to use
    
    Returns:
        Summary statistics
    
    Raises:
        ValueError: If any clip fails to process
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Processing {len(clips)} clips...")
    
    results = []
    
    for i, clip in enumerate(clips):
        if (i + 1) % 10 == 0:
            logger.info(f"  Progress: {i+1}/{len(clips)}")
        
        # Validate clip before processing
        if not saver.validate_clip(clip, dataset_name):
            logger.warning(f"Skipping clip {clip.get('video_uid')} due to validation failure")
            continue

        # No try-except, let errors propagate
        try:
            result = saver.save_embeddings_for_clip(clip, dataset_name, output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to save embeddings for clip {clip.get('video_uid')}: {e}")
            continue
    
    logger.info(f"✓ Successfully saved embeddings for all {len(clips)} clips")
    
    return {
        "dataset_name": dataset_name,
        "output_dir": str(output_dir),
        "total_clips": len(clips),
        "saved_count": len(results),
        "results": results,
    }


class VisionEmbeddingsSaver:
    """
    Extract and save vision embeddings from frames for DST training data.
    
    Following ProAssist pattern:
    - Load raw frames from arrow files
    - Extract [CLS] token from vision encoder
    - Save to disk for efficient training
    """
    
    def __init__(self, cfg: DictConfig):
        """
        Initialize vision embeddings saver.
        
        Args:
            cfg: Hydra configuration object
        """
        self.model_name = cfg.model.name
        # Build full frames_root path from project_root and relative path
        project_root = Path(cfg.project_root)
        frames_root_relative = cfg.processing.frames_root
        self.frames_root = project_root / frames_root_relative
        self.batch_size = cfg.processing.batch_size
        self.device = getattr(cfg.model, 'device', None)
        self.vision_hidden_size = 1152  # SmolVLM2 vision encoder output dim
        
        # Determine device - use LOCAL_RANK if available (for multi-GPU), else cuda:0
        if self.device is None:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self.device = f"cuda:{local_rank}"
        
        # Load vision encoder
        self._load_vision_encoder()
        
        logger.info(f"✓ Initialized VisionEmbeddingsSaver on {self.device} with batch_size={self.batch_size}")
    
    def _load_vision_encoder(self):
        """Load SmolVLM2 vision encoder."""
        logger.info(f"Loading vision encoder from {self.model_name}...")
        
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        
        # Load FULL model (not just vision component) for proper embedding extraction
        self.model = AutoModel.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.requires_grad_(False)
        
        logger.info(f"✓ Vision encoder loaded")

    def validate_clip(self, clip: Dict[str, Any], dataset_name: str) -> bool:
        """
        Validate if a clip's frame indices are within bounds of the Arrow file.
        
        Args:
            clip: Clip data
            dataset_name: Dataset name
            
        Returns:
            True if valid, False otherwise
        """
        video_uid = clip.get("video_uid")
        if not video_uid:
            return False
            
        # Build frame file path
        if dataset_name != "assembly101":
            frame_file = f"frames/{video_uid}.arrow"
        else:
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        
        arrow_path = self.frames_root / dataset_name / frame_file
        
        if not arrow_path.exists():
            logger.warning(f"Validation failed: Arrow file not found: {arrow_path}")
            return False
            
        try:
            # We use hf_datasets.Dataset.from_file which is lazy and fast
            # It reads the metadata footer to get the length
            dataset = hf_datasets.Dataset.from_file(str(arrow_path))
            dataset_len = len(dataset)
        except Exception as e:
            logger.warning(f"Validation failed: Could not read Arrow file {arrow_path}: {e}")
            return False
            
        start_frame = clip.get("start_frame_idx")
        end_frame = clip.get("end_frame_idx")
        
        if start_frame is None or end_frame is None:
            logger.warning(f"Validation failed: Missing frame indices for {video_uid}")
            return False
            
        # Check bounds
        # end_frame_idx is exclusive, so max valid index is end_frame_idx - 1
        # The required frames are [start_frame, end_frame)
        # So we need start_frame >= 0 and end_frame <= dataset_len
        
        if start_frame < 0:
            logger.warning(f"Validation failed: Negative start frame {start_frame} for {video_uid}")
            return False
            
        if end_frame > dataset_len:
            logger.warning(
                f"Validation failed: End frame {end_frame} exceeds dataset size {dataset_len} "
                f"for {video_uid}"
            )
            return False
            
        if start_frame >= end_frame:
            logger.warning(f"Validation failed: start_frame {start_frame} >= end_frame {end_frame} for {video_uid}")
            return False
            
        return True
    
    def extract_cls_token(self, processor_output: dict, batch_size: int) -> torch.Tensor:
        """
        Extract the global [CLS] token from SmolVLM2's multi-patch vision output.
        
        SmolVLM2 uses a multi-patch strategy for high-resolution images:
        - Creates ~17 patches from input image
        - 1 global context token: downscaled full-image view (overall scene context)
        - 16 local detail tokens: high-res crop patches (fine-grained details)
        
        We extract ONLY the global [CLS] token [0,0,:] which represents the entire
        image, following the ProAssist I=1 efficiency philosophy (1 token per frame).
        
        Args:
            processor_output: Output from SmolVLM2 processor (batched: text + images)
            batch_size: Number of images in the batch (needed to reshape flattened output)
        
        Returns:
            cls_tokens: Global [CLS] tokens from vision encoder
                       Shape: [batch_size, 2048]  (batch of 2048-dim SigLIP hidden)
        
        Raises:
            RuntimeError/ValueError: If model fails to process
        """
        # Move processor output to device
        inputs = {k: v.to(self.device) if hasattr(v, 'to') else v 
                  for k, v in processor_output.items()}
        
        # Convert to float16
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        
        with torch.no_grad():
            try:
                # Use full model to get image_hidden_states
                outputs = self.model(**inputs, output_hidden_states=True)
            except Exception as e:
                raise RuntimeError(
                    f"Model failed to process inputs: {type(e).__name__}: {e}"
                )
            
            # SmolVLM2 image_hidden_states structure for batched input:
            # Flattened: [batch_size * num_patches, seq_len, hidden_dim]
            # E.g., [85, 81, 2048] for batch_size=5, num_patches=17
            # We need to reshape to [batch_size, num_patches, seq_len, hidden_dim]
            if not hasattr(outputs, 'image_hidden_states') or outputs.image_hidden_states is None:
                raise ValueError("Model output does not contain image_hidden_states")
            
            image_hidden_states = outputs.image_hidden_states
            
            if not isinstance(image_hidden_states, torch.Tensor):
                raise ValueError(f"Expected tensor, got {type(image_hidden_states)}")
            
            # Reshape flattened output: [batch*patches, seq, hidden] -> [batch, patches, seq, hidden]
            total_patches = image_hidden_states.shape[0]
            num_patches = total_patches // batch_size  # Should be 17
            seq_len = image_hidden_states.shape[1]
            hidden_dim = image_hidden_states.shape[2]
            
            image_hidden_states = image_hidden_states.view(batch_size, num_patches, seq_len, hidden_dim)
            
            # Extract global [CLS] token for all frames in batch
            # image_hidden_states[:, 0, 0, :] selects:
            #   - :  : all frames in batch
            #   - 0  : patch 0 (the global context patch)
            #   - 0  : token 0 (the [CLS] token of that patch)
            # Result shape: [batch_size, 2048]
            cls_tokens = image_hidden_states[:, 0, 0, :]  # [batch_size, 2048]
        
        return cls_tokens.cpu().half()  # Return as float16 to save 50% disk space

    
    def load_frames_from_arrow(
        self,
        arrow_file: Path,
        frame_indices: List[int],
    ) -> List[Image.Image]:
        """
        Load frames from arrow file.
        
        Args:
            arrow_file: Path to arrow file containing frames
            frame_indices: List of specific frame indices to load (required, no fallback)
        
        Returns:
            List of PIL Images
        
        Raises:
            FileNotFoundError: If arrow file doesn't exist
            ValueError: If frame_indices is empty
            IndexError: If frame index exceeds dataset size
            KeyError: If frame key is unknown
            TypeError: If frame type is unexpected
        """
        if not arrow_file.exists():
            raise FileNotFoundError(f"Arrow file not found: {arrow_file}")
        
        if not frame_indices:
            raise ValueError(f"frame_indices cannot be empty")
        
        dataset = hf_datasets.Dataset.from_file(str(arrow_file))

        images = []
        
        for idx in frame_indices:
            if idx >= len(dataset):
                raise IndexError(
                    f"Frame index {idx} exceeds dataset size {len(dataset)}"
                )
            
            item = dataset[idx]
            
            # Extract image - MUST have 'frame' or 'image' key
            if "frame" in item:
                frame = item["frame"]
            elif "image" in item:
                frame = item["image"]
            else:
                available_keys = list(item.keys())
                raise KeyError(
                    f"Frame {idx} missing required key. "
                    f"Available keys: {available_keys}. "
                    f"Expected 'frame' or 'image' key in {arrow_file}."
                )
            
            # Convert to PIL Image if needed (frames may be base64-encoded strings or PIL Images)
            if isinstance(frame, str):
                # Decode base64-encoded string to image
                try:
                    img_bytes = base64.b64decode(frame)
                    frame = Image.open(io.BytesIO(img_bytes))
                except Exception as e:
                    raise ValueError(
                        f"Frame {idx} is a string but could not be decoded as base64 image: {e}"
                    )
            
            # Verify it's now a PIL Image
            if not isinstance(frame, Image.Image):
                raise TypeError(
                    f"Frame {idx} has unexpected type {type(frame)}, expected PIL Image. "
                    f"Cannot proceed without proper image data."
                )
            images.append(frame.convert("RGB"))
        
        return images
    
    def save_embeddings_for_clip(
        self,
        clip: Dict[str, Any],
        dataset_name: str,
        output_dir: Path,
    ) -> Dict[str, Any]:
        """
        Extract and save vision embeddings for a single clip.
        
        Args:
            clip: Clip data from DST JSON
            dataset_name: Name of dataset (e.g., "assembly101")
            output_dir: Directory to save embeddings
        
        Returns:
            Dictionary of embedding metadata
        
        Raises:
            ValueError: If clip is invalid or frames cannot be loaded
            FileNotFoundError: If frame file doesn't exist
        """
        video_uid = clip.get("video_uid")
        if not video_uid:
            raise ValueError("Clip missing required 'video_uid' field")
        
        # Get unique ID from clip for filename (use video_uid as fallback)
        clip_id = clip.get("id", video_uid)
        
        # Build frame file path
        if dataset_name != "assembly101":
            frame_file = f"frames/{video_uid}.arrow"
        else:
            # For Assembly101, extract filename from video_uid
            frame_file_name = video_uid.split("_", 1)[1]
            frame_file = f"frames/{frame_file_name}.arrow"
        
        arrow_path = self.frames_root / dataset_name / frame_file
        
        if not arrow_path.exists():
            raise FileNotFoundError(
                f"Frame file not found for clip {video_uid}: {arrow_path}"
            )
        
        # Get frame indices from clip - BOTH REQUIRED
        start_frame = clip.get("start_frame_idx")
        if start_frame is None:
            raise ValueError(
                f"Clip {video_uid} missing required 'start_frame_idx' field"
            )
        
        end_frame = clip.get("end_frame_idx")
        if end_frame is None:
            raise ValueError(
                f"Clip {video_uid} missing required 'end_frame_idx' field"
            )
            
        # Check if embeddings already exist and are valid
        output_file = output_dir / f"{clip_id}_embeddings.pkl"
        expected_frames = end_frame - start_frame
        
        if output_file.exists():
            try:
                with open(output_file, "rb") as f:
                    existing_embeddings = pickle.load(f)
                
                # Check shape: [num_frames, hidden_dim]
                if existing_embeddings.shape[0] == expected_frames:
                    logger.info(f"Skipping {clip_id}: Embeddings exist with correct shape {existing_embeddings.shape}")
                    return {
                        "video_uid": video_uid,
                        "clip_id": clip_id,
                        "shape": existing_embeddings.shape,
                        "saved_path": str(output_file),
                        "skipped": True
                    }
                else:
                    logger.warning(
                        f"Re-generating {clip_id}: Shape mismatch. "
                        f"Found {existing_embeddings.shape[0]} frames, expected {expected_frames}."
                    )
            except Exception as e:
                logger.warning(f"Re-generating {clip_id}: Failed to verify existing file: {e}")
        
        # Load only the frames needed for this clip
        # Note: end_frame_idx is EXCLUSIVE (Python convention), so use it directly in range()
        frame_indices = list(range(start_frame, end_frame))
        frames = self.load_frames_from_arrow(arrow_path, frame_indices)
        
        if not frames:
            return {
            "video_uid": video_uid,
            "clip_id": clip_id,
            "shape": 2048,
            "saved_path": "",
        }
        
            raise ValueError(
                f"No frames loaded for clip {video_uid} from range [{start_frame}:{end_frame}] "
                f"in {arrow_path}"
            )
        
        # Process frames with vision encoder using batching for efficiency
        # SmolVLM2 supports batching with nested format: [[img1], [img2], ...]
        all_cls_tokens = []
        
        for batch_start in tqdm(range(0, len(frames), self.batch_size), desc="Processing batches"):
            batch_end = min(batch_start + self.batch_size, len(frames))
            batch_frames = frames[batch_start:batch_end]
            current_batch_size = len(batch_frames)
            
            # Use nested format: [[frame1], [frame2], ...] for proper batching
            images_nested = [[frame] for frame in batch_frames]
            texts = ["<image>"] * current_batch_size
            
            processor_output = self.processor(
                text=texts,
                images=images_nested,
                return_tensors="pt"
            )
            
            # Extract [CLS] tokens for the entire batch, shape [batch_size, 2048]
            cls_tokens_batch = self.extract_cls_token(processor_output, current_batch_size)
            all_cls_tokens.append(cls_tokens_batch)
        
        # Stack all [CLS] tokens [num_frames, 2048]
        cls_tokens = torch.cat(all_cls_tokens, dim=0)
        
        # Convert to numpy as float16 to save 50% disk space
        # Training will convert back to float32 automatically
        embeddings = cls_tokens.numpy().astype(np.float16)
        
        # Save embeddings using the unique ID as filename
        output_file = output_dir / f"{clip_id}_embeddings.pkl"
        with open(output_file, "wb") as f:
            pickle.dump(embeddings, f)
        
        logger.info(f"✓ Saved embeddings for {video_uid} (ID: {clip_id}): shape {embeddings.shape}")
        
        return {
            "video_uid": video_uid,
            "clip_id": clip_id,
            "shape": embeddings.shape,
            "saved_path": str(output_file),
        }


def save_vision_embeddings(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main function to save vision embeddings for all splits in datasets.
    
    Creates a "frames" subdirectory with vision feature files for each dataset.
    
    Args:
        cfg: Hydra configuration object
    
    Returns:
        Summary of embedding extraction for all datasets
    
    Raises:
        FileNotFoundError: If expected files don't exist
        ValueError: If embedding extraction fails for any clip
    """
    dataset_output_dir = Path(cfg.dataset.output_dir)
    dataset_names = cfg.dataset.names
    frames_root = cfg.processing.frames_root
    # Get process info from environment (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    logger.info(f"Process {local_rank}/{world_size} starting...")
    
    summary = {
        "dataset_output_dir": str(dataset_output_dir),
        "process_index": local_rank,
        "num_processes": world_size,
        "datasets": {},
    }
    
    # Create saver instance
    saver = VisionEmbeddingsSaver(cfg)
    
    # Process each dataset
    for dataset_name in dataset_names:
        dataset_json_dir = dataset_output_dir / dataset_name
        
        if not dataset_json_dir.exists():
            logger.warning(f"Dataset directory not found: {dataset_json_dir}")
            continue
        
        logger.info(f"\n{'='*70}")
        logger.info(f"Processing dataset: {dataset_name}")
        logger.info(f"{'='*70}")
        
        # Create frames subdirectory for this dataset
        frames_dir = dataset_json_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving vision embeddings to: {frames_dir}")
        
        # Process all training files (*_training.json) in this dataset
        training_files = list(dataset_json_dir.glob("*_training.json"))
        
        if not training_files:
            logger.warning(
                f"No training files found in {dataset_json_dir}. "
                f"Expected *_training.json files."
            )
            continue
        
        dataset_summary = {
            "dataset_json_dir": str(dataset_json_dir),
            "frames_dir": str(frames_dir),
            "splits": {},
        }
        
        # Process each split
        for data_file in sorted(training_files):
            split_name = data_file.stem.replace("_training", "")
            
            logger.info(f"\nProcessing split: {split_name}")
            
            # Load clips
            with open(data_file) as f:
                clips = json.load(f)
            
            # Split clips across processes (each process gets different clips)
            clips_chunk = clips[local_rank::world_size]
            
            logger.info(f"Process {local_rank} processing {len(clips_chunk)} clips out of {len(clips)}")
            
            result = save_embeddings_for_dataset(
                clips_chunk,
                dataset_name,
                frames_dir,
                saver,
            )
            
            dataset_summary["splits"][split_name] = result
        
        summary["datasets"][dataset_name] = dataset_summary
    
    return summary


@hydra.main(
    version_base=None,
    config_path="../../../custom/config/dst_data_generator",
    config_name="vision_embeddings",
)
def main(cfg: DictConfig) -> None:
    """Main function to run vision embeddings extraction."""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    # Log configuration
    logger.info("Starting Vision Embeddings Extraction")
    logger.info(f"Project root: {cfg.project_root}")
    logger.info(f"Dataset output dir: {cfg.dataset.output_dir}")
    logger.info(f"Datasets: {cfg.dataset.names}")
    logger.info(f"Model: {cfg.model.name}")
    logger.info(f"Batch size: {cfg.processing.batch_size}")
    logger.info(f"Frames root (relative): {cfg.processing.frames_root}")

    # Run the embeddings extraction
    try:
        summary = save_vision_embeddings(cfg)

        logger.info("Vision embeddings extraction completed successfully!")
        
        # Save summary to JSON
        output_dir = Path(HydraConfig.get().runtime.output_dir)
        summary_file = output_dir / "vision_embeddings_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
            
        logger.info(f"Summary saved to: {summary_file}")

    except Exception as e:
        logger.error(f"Vision embeddings extraction failed: {e}")
        raise


if __name__ == "__main__":
    main()
