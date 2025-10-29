import json
import asyncio
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

from dst_data_builder.gpt_generators.gpt_generator_factory import GPTGeneratorFactory
from dst_data_builder.data_sources.dst_data_module import DSTDataModule


class SimpleDSTGenerator:
    """Simple DST generator using LLM with Hydra configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        generator_type = cfg.generator.type
        self.logger.info("🔧 Using GPT generator type: %s", generator_type)

        # Initialize GPT generator using factory
        gen_cfg = cfg.get("generator", {}) or {}
        
        self.gpt_generator = GPTGeneratorFactory.create_generator(
            generator_type=generator_type,
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            max_retries=cfg.max_retries,
            generator_cfg=gen_cfg,
        )

    async def run(self, cfg: DictConfig) -> None:
        """Run the DST generation process with the given configuration"""
        # Instantiate data module from cfg
        ds_cfg = cfg.get("data_source", {}) or {}
        ds_name = ds_cfg.get("name", "manual")
        self.logger.info("📦 Using data module: %s", ds_name)

        # Create data module with configuration
        data_module = DSTDataModule(
            data_source_name=ds_name,
            data_source_cfg=ds_cfg,
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir

        self.logger.info("📁 Hydra output directory: %s", Path(hydra_output_dir).resolve())

        # Create output directory
        dst_output_dir = Path(hydra_output_dir) / "dst_outputs"
        dst_output_dir.mkdir(exist_ok=True, parents=True)

        self.logger.info("📁 DST output directory: %s", dst_output_dir.resolve())

        # Get file paths to process
        file_paths = data_module.get_file_paths()
        if not file_paths:
            self.logger.warning("No input files found by data module; nothing to process")
            raise ValueError("No input files found")

        self.logger.info("📥 Found %d input files to process", len(file_paths))

        # Let the generator handle batch processing and incremental saving
        batch_size = cfg.generator.get("batch_size", 5)
        processed, failed = await self.gpt_generator.generate_and_save_dst_outputs(
            file_paths=[str(fp) for fp in file_paths],
            dst_output_dir=dst_output_dir,
            batch_size=batch_size
        )

        # Save summary
        self._save_summary(dst_output_dir, len(file_paths), processed, failed, ds_name)

        self.logger.info("=== Summary ===")
        self.logger.info("✅ Processed: %d", processed)
        self.logger.info("❌ Failed: %d", failed)
        self.logger.info("📁 Output directory: %s", dst_output_dir)

    def _save_summary(self, dst_output_dir: Path, total_files: int, processed: int, failed: int, ds_name: str) -> None:
        """Save generation summary to file."""
        try:
            summary_file = dst_output_dir / "generation_summary.json"
            summary = {
                "total_files": total_files,
                "processed": processed,
                "failed": failed,
                "timestamp": str(datetime.now()),
                "config": {
                    "data_source": ds_name,
                    "model": self.gpt_generator.model_name,
                    "max_retries": self.gpt_generator.max_retries
                }
            }
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            self.logger.info("📊 Saved summary: %s", summary_file)
        except Exception as e:
            self.logger.exception("Failed to save summary: %s", e)


@hydra.main(config_path="../../config", config_name="simple_dst_generator", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger = logging.getLogger(__name__)
    logger.info("🚀 Starting Simple DST Generator with Hydra configuration...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    generator = SimpleDSTGenerator(cfg)

    # Use the generator's run() method which reads the data_source config (data_path)
    asyncio.run(generator.run(cfg))


if __name__ == "__main__":
    main()
