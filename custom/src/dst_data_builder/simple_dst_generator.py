import json
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from tqdm import tqdm

from dst_data_builder.dst_data_processor import DSTDataProcessor
from dst_data_builder.gpt_generators.proassist_label_generator import ProAssistDSTLabelGenerator
from dst_data_builder.gpt_generators.speak_dst_generator import SpeakDSTGenerator

class SimpleDSTGenerator:
    """Simple DST generator that adds dst labels to existing JSON data"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Initialize DST generator
        self.dst_generator = ProAssistDSTLabelGenerator(
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            max_retries=cfg.max_retries,
        )

        # Initialize SpeakDST generator for enhanced format if enabled
        self.speak_dst_enabled = cfg.get('generation', {}).get('enable_speak_dst_integration', False)
        if self.speak_dst_enabled:
            speak_dst_config = {
                'evidence_spans': cfg.get('generation', {}).get('evidence_spans', False),
                'disable_progress_bars': True,  # Disable nested progress bars
            }
            self.speak_dst_generator = SpeakDSTGenerator(speak_dst_config)
            self.logger.info("🔄 SPEAK/DST integration enabled with enhanced format")
        else:
            self.speak_dst_generator = None
            self.logger.info("📝 Using original format with basic DST annotations")

        # Initialize DST data processor with generators
        self.data_processor = DSTDataProcessor(
            dst_generator=self.dst_generator,
            speak_dst_generator=self.speak_dst_generator,
            logger=self.logger,
            is_multiprocessing=cfg.get('generation', {}).get('enable_multiprocessing', False),
            num_processes=cfg.get('generation', {}).get('multiprocessing_processes', None)
        )

    def run(self, cfg: DictConfig) -> None:
        """Run the DST generation process with the given configuration"""
        
        # Get configuration
        datasets = cfg.data_source.datasets
        splits = ['train', 'val', 'test']
        num_rows = cfg.data_source.get('num_rows', 2)
        
        self.logger.info("🚀 Starting Simple DST Generation")
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

        # Process each dataset and split combination with hierarchical progress tracking
        total_datasets = len(datasets)
        total_splits = len(splits)

        with tqdm(total=total_datasets, desc="📊 Processing datasets", unit="dataset", position=0) as dataset_pbar:
            for dataset_idx, dataset_name in enumerate(datasets):
                dataset_pbar.set_description(f"📊 Processing dataset: {dataset_name}")

                # Create dataset output directory
                dataset_output_dir = output_base_dir / dataset_name
                dataset_output_dir.mkdir(exist_ok=True, parents=True)

                # Process splits within this dataset
                with tqdm(total=total_splits, desc=f"🔄 {dataset_name} splits",
                         unit="split", position=1, leave=False) as split_pbar:
                    for split_idx, split in enumerate(splits):
                        try:
                            split_pbar.set_description(f"🔄 Processing {dataset_name}/{split}")
                            processed, failed = self.data_processor.process_dataset_split(
                                dataset_name, split, num_rows, dataset_output_dir
                            )
                            total_processed += processed
                            total_failed += failed
                            split_pbar.set_postfix({
                                '✅ Processed': total_processed,
                                '❌ Failed': total_failed
                            })

                        except Exception as e:
                            self.logger.error(f"❌ Failed to process {dataset_name}/{split}: {e}")

                        split_pbar.update(1)

                dataset_pbar.update(1)

        self.logger.info("=== Summary ===")
        self.logger.info("✅ Processed: %d", total_processed)
        self.logger.info("❌ Failed: %d", total_failed)
        self.logger.info("📁 Output directory: %s", output_base_dir)



@hydra.main(config_path="../../config/dst_data_generator", config_name="simple_dst_generator", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Simple DST Generator with Hydra configuration...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    generator = SimpleDSTGenerator(cfg)
    generator.run(cfg)


if __name__ == "__main__":
    main()
