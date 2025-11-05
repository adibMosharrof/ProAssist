#!/usr/bin/env python3
"""
Single Strategy E2E Test
Run this script to test a single context strategy.
Usage: python test_single_strategy.py --strategy none|drop_all|drop_middle|summarize_and_drop|summarize_with_dst
"""

import sys
import argparse
import logging
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from main E2E test
from test_e2e_strategies import E2EStrategyTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Test a single context strategy')
    parser.add_argument('--strategy', type=str, required=True,
                       choices=['none', 'drop_all', 'drop_middle', 'summarize_and_drop', 'summarize_with_dst'],
                       help='Strategy to test')
    args = parser.parse_args()
    
    # Initialize Hydra config
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf
    
    # Initialize Hydra
    with initialize(version_base=None, config_path="../../../config/prospect"):
        cfg = compose(config_name="prospect")
        config = OmegaConf.to_container(cfg, resolve=True)
    
    # Get video IDs from config
    video_ids = config['data_source'].get('video_ids', ['9011-c03f'])
    output_dir = f"./custom/outputs/single_strategy_tests/{args.strategy}"
    
    # Initialize tester
    logger.info(f"\n{'='*80}")
    logger.info(f"Testing strategy: {args.strategy}")
    logger.info(f"{'='*80}\n")
    
    tester = E2EStrategyTester(
        video_ids=video_ids,
        data_source_config=config['data_source'],
        model_config=config['model'],
        generator_config=config['generator'],
        output_dir=output_dir,
    )
    
    # Test single strategy
    print(f"\n{'='*80}")
    print(f"Testing strategy: {args.strategy}")
    print(f"{'='*80}\n")
    
    try:
        metrics = tester._test_strategy(tester.strategy_configs[args.strategy])
        
        print(f"\n{'='*80}")
        print(f"✅ Strategy {args.strategy} completed!")
        print(f"{'='*80}")
        print(f"Results:")
        print(f"  Frames: {metrics.num_frames}")
        print(f"  Generated: {metrics.num_dialogues_generated}")
        print(f"  Matched: {metrics.num_matched}")
        print(f"  Missed: {metrics.num_missed}")
        print(f"  Redundant: {metrics.num_redundant}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall: {metrics.recall:.4f}")
        print(f"  F1: {metrics.f1:.4f}")
        print(f"  Jaccard: {metrics.jaccard:.4f}")
        print(f"  BLEU-1: {metrics.bleu_1:.4f}")
        print(f"  BLEU-2: {metrics.bleu_2:.4f}")
        print(f"  BLEU-3: {metrics.bleu_3:.4f}")
        print(f"  BLEU-4: {metrics.bleu_4:.4f}")
        print(f"  CIDEr: {metrics.cider:.4f}")
        print(f"  METEOR: {metrics.meteor:.4f}")
        print(f"  Semantic Similarity: {metrics.semantic_similarity:.4f}")
        print(f"  Time Difference: {metrics.time_difference:.4f}")
        print(f"  Total Time: {metrics.total_time:.2f}s")
        print(f"  Time/Frame: {metrics.avg_time_per_frame:.4f}s")
        print(f"  Peak Memory: {metrics.peak_memory_mb:.2f} MB")
        print(f"{'='*80}\n")
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*80}")
        print(f"❌ Strategy {args.strategy} FAILED")
        print(f"{'='*80}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
