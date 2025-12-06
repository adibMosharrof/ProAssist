"""
Sparse Sequence Generator for Continuous DST Training

Extends SimpleDSTGenerator to produce sparse event format with:
- Token-based sequence splitting
- Only event frames stored (not silent frames)
- DST state tracking across clips

Reads from generated_dialogs/{split}_filtered.json and produces training samples.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from omegaconf import DictConfig
import hydra
from hydra.core.hydra_config import HydraConfig
from transformers import AutoTokenizer
from tqdm import tqdm
import uuid

from dst_data_builder.simple_dst_generator import SimpleDSTGenerator


@dataclass 
class EventFrame:
    """Represents a frame with an event (speaking or DST update)"""
    frame_idx: int
    speaking: int = 0
    dst_update: int = 0
    dst_updates: List[str] = field(default_factory=list)
    response: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "speaking": self.speaking,
            "dst_update": self.dst_update,
            "dst_updates": self.dst_updates,
            "response": self.response,
        }
    
    def estimate_tokens(self, tokenizer) -> int:
        """Estimate token count for this event"""
        tokens = 1  # <image> token
        if self.dst_updates:
            for update in self.dst_updates:
                tokens += len(tokenizer.encode(f"[DST] {update}", add_special_tokens=False))
        if self.response:
            tokens += len(tokenizer.encode(f"[ASST] {self.response}", add_special_tokens=False))
        return tokens


class SparseSequenceGenerator(SimpleDSTGenerator):
    """
    Generates sparse sequence training data from ProAssist dialogs.
    
    Extends SimpleDSTGenerator to:
    1. Generate DST labels via GPT (inherited)
    2. Apply training modules (inherited)
    3. Convert to sparse event format (new)
    """
    
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        
        # Additional config for sparse format
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.get("tokenizer", "meta-llama/Llama-3.2-3B-Instruct")
        )
        self.max_sequence_tokens = cfg.get("max_sequence_tokens", 4096)
        self.fps = cfg.get("fps", 2)
        
        self.logger.info(f"🔢 Max sequence tokens: {self.max_sequence_tokens}")
        self.logger.info(f"🎬 FPS: {self.fps}")
    
    def convert_to_sparse_format(self, training_sample: dict) -> dict:
        """
        Convert training sample to sparse event format.
        
        Only stores frames with events (DST_UPDATE or assistant response).
        """
        conversation = training_sample.get("conversation", [])
        
        # Extract frame range
        all_start_frames = []
        all_end_frames = []
        for turn in conversation:
            if "start_frame" in turn:
                all_start_frames.append(turn["start_frame"])
            if "end_frame" in turn:
                all_end_frames.append(turn["end_frame"])
        
        if not all_start_frames or not all_end_frames:
            training_sample.pop("conversation", None)
            training_sample["start_frame"] = 0
            training_sample["end_frame"] = 0
            training_sample["events"] = []
            return training_sample
        
        min_frame = min(all_start_frames)
        max_frame = max(all_end_frames)
        
        # Build sparse event data
        frame_events: Dict[int, EventFrame] = {}
        
        for turn in conversation:
            role = turn.get("role", "")
            start_frame = turn.get("start_frame", 0)
            
            if role == "DST_UPDATE":
                if start_frame not in frame_events:
                    frame_events[start_frame] = EventFrame(frame_idx=start_frame)
                frame_events[start_frame].dst_update = 1
                
                # Format DST content
                content = turn.get("content", [])
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            step_id = item.get("id", "")
                            transition = item.get("transition", "")
                            dst_text = f"{step_id}->{transition}"
                            frame_events[start_frame].dst_updates.append(dst_text)
                elif isinstance(content, str):
                    frame_events[start_frame].dst_updates.append(content)
            
            elif role == "assistant":
                if start_frame not in frame_events:
                    frame_events[start_frame] = EventFrame(frame_idx=start_frame)
                frame_events[start_frame].speaking = 1
                content = turn.get("content", "")
                if content:
                    if frame_events[start_frame].response:
                        frame_events[start_frame].response += " " + content
                    else:
                        frame_events[start_frame].response = content
        
        # Sort by frame index
        events_list = [frame_events[idx].to_dict() for idx in sorted(frame_events.keys())]
        
        # Remove conversation and other unused fields
        training_sample.pop("conversation", None)
        training_sample.pop("_dst_transitions", None)
        training_sample.pop("dst_state_history", None)
        training_sample.pop("parsed_video_anns", None)
        training_sample.pop("gen_args", None)
        training_sample.pop("video_labels", None)
        training_sample.pop("metadata", None)
        
        # Add sparse format fields
        training_sample["start_frame"] = min_frame
        training_sample["end_frame"] = max_frame
        training_sample["events"] = events_list
        training_sample["num_total_frames"] = max_frame - min_frame + 1
        training_sample["num_event_frames"] = len(events_list)
        training_sample["speaking_frames"] = sum(1 for e in events_list if e["speaking"])
        training_sample["dst_update_frames"] = sum(1 for e in events_list if e["dst_update"])
        
        return training_sample

    def run(self, cfg: DictConfig) -> None:
        """Run the sparse sequence generation process"""
        
        # Get configuration
        datasets = cfg.data_source.datasets
        splits = ["test", "val", "train"]
        num_rows = cfg.data_source.num_rows

        self.logger.info("🚀 Starting Sparse Sequence DST Generation")
        self.logger.info(f"📊 Datasets: {datasets}")
        self.logger.info(f"🔄 Splits: {splits}")
        self.logger.info(f"📝 Rows per dataset/split: {num_rows}")

        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        output_base_dir = Path(hydra_output_dir)

        self.logger.info("📁 Output directory: %s", output_base_dir.resolve())

        total_processed = 0
        total_failed = 0
        total_clips = 0

        for dataset_name in datasets:
            dataset_output_dir = output_base_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True, parents=True)

            for split in splits:
                self.logger.info(f"Processing {dataset_name}/{split}...")
                
                # 1. Process dataset split (generates DST via GPT)
                processed, failed = self.data_processor.process_dataset_split(
                    dataset_name, split, num_rows, dataset_output_dir
                )
                total_processed += processed
                total_failed += failed
                
                # 2. Load enhanced data (intermediate file from data_processor)
                enhanced_file = dataset_output_dir / f"{split}.json"
                if not enhanced_file.exists():
                    self.logger.error(f"Enhanced file not found: {enhanced_file}")
                    continue
                
                with open(enhanced_file, "r") as f:
                    enhanced_data = json.load(f)
                
                # 3. Create training format (applies all training modules)
                training_data = self.create_training_format(
                    enhanced_data, dataset_name, split
                )
                
                # 4. Convert each sample to sparse format
                sparse_data = []
                for sample in training_data:
                    sparse_sample = self.convert_to_sparse_format(sample)
                    sparse_data.append(sparse_sample)
                
                # 5. Delete intermediate file BEFORE saving final output
                if enhanced_file.exists():
                    enhanced_file.unlink()
                
                # 6. Save sparse format (same filename, now safe to use)
                output_file = dataset_output_dir / f"{split}.json"
                with open(output_file, "w") as f:
                    json.dump(sparse_data, f, indent=2)
                
                # Log stats
                total_events = sum(s.get("num_event_frames", 0) for s in sparse_data)
                total_speaking = sum(s.get("speaking_frames", 0) for s in sparse_data)
                total_dst = sum(s.get("dst_update_frames", 0) for s in sparse_data)
                
                self.logger.info(
                    f"✅ {dataset_name}/{split}: {len(sparse_data)} clips, "
                    f"{total_events} events ({total_speaking} speaking, {total_dst} DST)"
                )
                total_clips += len(sparse_data)

        self.logger.info("=== Summary ===")
        self.logger.info(f"✅ Processed: {total_processed} videos")
        self.logger.info(f"❌ Failed: {total_failed} videos")
        self.logger.info(f"📊 Total clips: {total_clips}")
        self.logger.info(f"📁 Output directory: {output_base_dir}")


@hydra.main(
    config_path="../../config/dst_data_generator",
    config_name="sparse_sequence_generator",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main entry point"""
    logger = logging.getLogger(__name__)
    
    # Suppress verbose logging
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    generator = SparseSequenceGenerator(cfg)
    generator.run(cfg)


if __name__ == "__main__":
    main()
