import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging

from dst_data_builder.datatypes.dst_output import DSTOutput
from dst_data_builder.gpt_generators.gpt_generator_factory import GPTGeneratorFactory
from dst_data_builder.data_sources.dst_data_module import DSTDataModule


class SimpleDSTGenerator:
    """Simple DST generator using LLM with Hydra configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg

        # Get API key from environment variable

        generator_type = cfg.generator.type

        print(f"🔧 Using GPT generator type: {generator_type}")

        # Initialize GPT generator using factory
        self.gpt_generator = GPTGeneratorFactory.create_generator(
            generator_type=generator_type,
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens
        )
        self.logger = logging.getLogger(__name__)


    def run(self, cfg: DictConfig) -> None:
        """Run the DST generation process with the given configuration"""
        # Instantiate data module from cfg (defaults to 'manual')
        ds_cfg = cfg.get("data_source", {}) or {}
        ds_name = ds_cfg.get("name", "manual")
        print(f"📦 Using data module: {ds_name}")

        # Create data module with configuration
        data_module = DSTDataModule(
            data_source_name=ds_name,
            data_source_cfg=ds_cfg,
            batch_size=1,  # Process one file at a time
            shuffle=False,
            num_workers=0,
        )

        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir

        print(f"📁 Current working directory: {Path.cwd()}")
        print(f"📁 Hydra output directory: {hydra_output_dir}")
        print(
            f"📁 Hydra output directory (absolute): {Path(hydra_output_dir).resolve()}"
        )

        # Create a subdirectory for DST outputs within the Hydra run directory
        dst_output_dir = Path(hydra_output_dir) / "dst_outputs"
        dst_output_dir.mkdir(exist_ok=True, parents=True)

        print(f"📁 DST output directory: {dst_output_dir}")
        print(f"📁 DST output directory (absolute): {dst_output_dir.resolve()}")

        # Get dataloader (lazy initialization)
        dataloader = data_module.dataloader

        print(f"🚀 Processing {len(dataloader)} batches from data module {ds_name}...")

        # Prefer to use data_module.get_file_paths() so generator implementations
        file_paths = data_module.get_file_paths()
        if not len(file_paths):
            self.logger.warning("No input files found by data module; nothing to process")
            raise ValueError("No input files found")


        print(f"📥 Found {len(file_paths)} input files to process")

        try:
            outputs = self.gpt_generator.generate_dst_outputs(list(file_paths))
        except Exception as e:
            self.logger.exception(f"Generator raised an exception during processing: {e}")
            raise

        processed = 0
        failed = 0

        for input_path, result in outputs.items():
            out_name = f"dst_{Path(input_path).name}"
            out_file = dst_output_dir / out_name

            if result is None:
                self.logger.warning(f"Generation failed for {input_path}")
                failed += 1
                continue

            # result may already be a DSTOutput dataclass or a plain dict
            try:
                if hasattr(result, "to_dict"):
                    payload = result.to_dict()
                else:
                    payload = result

                with open(out_file, "w") as f:
                    json.dump(payload, f, indent=2)

                print(f"✅ Saved: {out_file}")
                processed += 1

            except Exception as e:
                self.logger.exception(f"Failed to save output for {input_path}: {e}")
                failed += 1

        print("=== Summary ===")
        print(f"✅ Processed: {processed}")
        print(f"❌ Failed: {failed}")
        print(f"📁 Output directory: {dst_output_dir}")



@hydra.main(config_path="../../config", config_name="simple_dst_generator", version_base=None)
def main(cfg: DictConfig):
    """Main function with Hydra configuration"""
    print("🚀 Starting Simple DST Generator with Hydra configuration...")
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")

    generator = SimpleDSTGenerator(cfg)

    # Use the generator's run() method which reads the data_source config (data_path)
    generator.run(cfg)


if __name__ == "__main__":
    main()
