#!/usr/bin/env python3
"""
DST Training Setup Test
Run this script to verify DST training setup before running full training.
Usage: python test_dst_training.py --skip_model_loading
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Test DST training setup")
    parser.add_argument(
        "--skip_model_loading",
        action="store_true",
        help="Skip loading the model (faster test)",
    )
    args = parser.parse_args()

    # Initialize Hydra config
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf

    # Initialize Hydra
    with initialize(version_base=None, config_path="../../../config/prospect/train"):
        cfg = compose(config_name="dst_training")
        config = OmegaConf.to_container(cfg, resolve=True)

    logger.info(f"\n{'='*80}")
    logger.info("DST Training Setup Test")
    logger.info(f"{'='*80}\n")

    try:
        # Test imports
        logger.info("🔍 Testing imports...")
        from prospect.models.dst_smolvlm_with_strategies import (
            DSTSmolVLMWithStrategies,
            DSTSmolVLMConfig,
        )
        from prospect.data.dst_frame_loader import DSTFrameDataLoader
        from prospect.data.dst_frame_dataset import DSTFrameDataset
        from prospect.data.dst_frame_collator import DSTDataCollator

        logger.info("✅ All imports successful")

        # Test configuration
        logger.info("🔧 Testing configuration...")
        logger.info(f"  Model: {config['model']['name']}")
        logger.info(f"  Training epochs: {config['training']['num_epochs']}")
        logger.info(f"  Data path: {config['data']['data_path']}")
        logger.info(f"  DST data path: {config['data']['dst_data_path']}")
        logger.info("✅ Configuration loaded successfully")

        # Test data paths
        logger.info("📂 Testing data paths...")
        data_path = Path(config["data"]["data_path"])
        dst_data_path = Path(config["data"]["dst_data_path"])

        if data_path.exists():
            logger.info(f"✅ Data path exists: {data_path}")
        else:
            logger.warning(f"⚠️  Data path not found: {data_path}")

        if dst_data_path.exists():
            logger.info(f"✅ DST data path exists: {dst_data_path}")
        else:
            logger.warning(f"⚠️  DST data path not found: {dst_data_path}")

        # Test model loading (if not skipped)
        if not args.skip_model_loading:
            logger.info("🤖 Testing model loading...")
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(
                config["model"]["name"],
                trust_remote_code=True,
            )
            logger.info("✅ Model processor loaded successfully")
        else:
            logger.info("⏭️  Model loading skipped")

        # Test data loader setup
        logger.info("📊 Testing data loader setup...")
        data_loader = DSTFrameDataLoader(
            data_path=config["data"]["data_path"],
            dst_data_path=config["data"]["dst_data_path"],
            max_seq_len=config["data"]["max_seq_len"],
            num_dst_states=config["model"]["num_dst_states"],
            fps=config["data"]["fps"],
        )
        logger.info("✅ Data loader created successfully")

        print(f"\n{'='*80}")
        print(f"✅ DST Training Setup Test PASSED")
        print(f"{'='*80}")
        print(f"Ready to run training!")
        print(f"Use: ./custom/runner/run_dst_training.sh")
        print(f"{'='*80}\n")

        return 0

    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ DST Training Setup Test FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
