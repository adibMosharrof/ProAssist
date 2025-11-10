import json
import asyncio
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from tqdm import tqdm

from dst_data_builder.gpt_generators.proassist_label_generator import ProAssistDSTLabelGenerator


class SimpleDSTGenerator:
    """Simple DST generator that adds dst labels to existing JSON data"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Initialize DST generator directly
        self.dst_generator = ProAssistDSTLabelGenerator(
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            max_retries=cfg.max_retries,
        )

    async def run(self, cfg: DictConfig) -> None:
        """Run the DST generation process with the given configuration"""
        
        # Get configuration
        datasets = cfg.data_source.datasets
        splits = ['train', 'val', 'test']
        num_rows = cfg.data_source.get('num_rows', 2)
        batch_size = cfg.generator.get('batch_size', 20)
        
        self.logger.info("🚀 Starting Simple DST Generation")
        self.logger.info(f"📊 Datasets: {datasets}")
        self.logger.info(f"🔄 Splits: {splits}")
        self.logger.info(f"📝 Rows per dataset/split: {num_rows}")
        self.logger.info(f"🧮 Batch size: {batch_size}")
        
        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        output_base_dir = Path(hydra_output_dir)
        
        self.logger.info("📁 Output directory: %s", output_base_dir.resolve())

        total_processed = 0
        total_failed = 0

        # Process each dataset and split combination
        # TODO: surround this in tqdm
        for dataset_name in datasets:
            # Create dataset output directory
            dataset_output_dir = output_base_dir / dataset_name
            dataset_output_dir.mkdir(exist_ok=True, parents=True)
            
            for split in splits:
                try:
                    processed, failed = await self._process_dataset_split(
                        dataset_name, split, num_rows, batch_size, dataset_output_dir
                    )
                    total_processed += processed
                    total_failed += failed
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process {dataset_name}/{split}: {e}")

        self.logger.info("=== Summary ===")
        self.logger.info("✅ Processed: %d", total_processed)
        self.logger.info("❌ Failed: %d", total_failed)
        self.logger.info("📁 Output directory: %s", output_base_dir)

    async def _process_dataset_split(self, dataset_name: str, split: str, num_rows: int, batch_size: int, output_dir: Path) -> tuple[int, int]:
        """Process a specific dataset and split combination with proper batching"""
        
        # Input file path
        input_file = Path(f"data/proassist/processed_data/{dataset_name}/generated_dialogs/{split}_filtered.json")
        
        if not input_file.exists():
            self.logger.warning(f"Input file not found: {input_file}")
            return 0, 0
        
        # Output file
        output_file = output_dir / f"{split}.json"
        
        self.logger.info(f"📁 Processing {dataset_name}/{split}: {input_file} -> {output_file}")
        
        # Load and limit data
        with open(input_file, 'r') as f:
            data = json.load(f)
        
        # Handle num_rows = -1 as "process all rows"
        if num_rows == -1:
            limited_data = data
            self.logger.info(f"📊 Processing ALL {len(limited_data)} rows from {len(data)} total")
        else:
            limited_data = data[:num_rows]
            self.logger.info(f"📊 Limited to {len(limited_data)} rows from {len(data)} total")
        
        self.logger.info(f"🧮 Using batch size: {batch_size}")
        
        # Process videos in batches
        processed_data = []
        failed_count = 0
        
        # Process in batches
        for batch_start in range(0, len(limited_data), batch_size):
            batch_end = min(batch_start + batch_size, len(limited_data))
            batch = limited_data[batch_start:batch_end]
            
            self.logger.info(f"📦 Processing batch {batch_start//batch_size + 1}: videos {batch_start}-{batch_end-1}")
            
            # Process each video in the batch
            for i, video_data in enumerate(batch):
                video_index = batch_start + i
                try:
                    # Generate DST labels for this video
                    dst_labels = await self._generate_dst_labels(video_data, video_index, dataset_name)
                    
                    # Add dst key to the original data
                    enhanced_video_data = video_data.copy()
                    enhanced_video_data['dst'] = dst_labels
                    
                    processed_data.append(enhanced_video_data)
                    self.logger.debug(f"✅ Processed video {video_index}: {video_data.get('video_uid', 'unknown')}")
                    
                except Exception as e:
                    self.logger.error(f"❌ Failed to process video {video_index}: {e}")
                    failed_count += 1
                    # Still add the original data without dst
                    processed_data.append(video_data)
        
        # Save processed data
        with open(output_file, 'w') as f:
            json.dump(processed_data, f, indent=2)
        
        # Removed save info messages to reduce log verbosity
        return len(processed_data), failed_count

    async def _generate_dst_labels(self, video_data: dict, video_index: int, dataset_name: str) -> dict:
        """Generate DST labels for a single video's data"""
        
        # Extract needed data from the video
        inferred_knowledge = video_data.get('inferred_knowledge', '')
        
        # Extract dialog from conversations
        all_step_descriptions = ""
        conversations = video_data.get('conversations', [])
        if conversations and len(conversations) > 0:
            dialog_content = []
            for conv in conversations[0].get('conversation', []):
                role = conv.get('role', '')
                content = conv.get('content', '')
                time = conv.get('time', 0.0)
                if role and content:
                    dialog_content.append(f"[{time:.1f}] {role}: {content}")
            all_step_descriptions = "\n".join(dialog_content)
        
        # Prepare input data for the generator
        input_data = {
            'inferred_knowledge': inferred_knowledge,
            'all_step_descriptions': all_step_descriptions
        }
        
        # Generate DST labels using the deterministic alignment method
        return self.dst_generator._generate_dst_from_input_data(input_data)


@hydra.main(config_path="../../config/dst_data_generator", config_name="simple_dst_generator", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Simple DST Generator with Hydra configuration...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    generator = SimpleDSTGenerator(cfg)
    asyncio.run(generator.run(cfg))


if __name__ == "__main__":
    main()
