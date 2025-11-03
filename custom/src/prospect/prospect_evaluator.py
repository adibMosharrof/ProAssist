"""PROSPECT Evaluator - Main entry point with Hydra configuration"""

import logging
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from prospect.data_sources.data_source_factory import DataSourceFactory
from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.runners.sanity_check_runner import SanityCheckRunner
from prospect.generators.generator_factory import GeneratorFactory


class ProspectEvaluator:
    """
    Main evaluator for PROSPECT (PROactive State tracking for ProCEdural Task assistance).

    This evaluator uses:
    - Hydra for configuration management
    - Custom VLM runner for dialogue generation
    - ProAssist's StreamEvaluator for evaluation

    Architecture matches dst_data_builder pattern for consistency.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize PROSPECT evaluator

        Args:
            cfg: Hydra configuration
        """
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Get Hydra output directory
        hydra_cfg = HydraConfig.get()
        self.output_dir = Path(hydra_cfg.runtime.output_dir)

        self.logger.info("=" * 60)
        self.logger.info("PROSPECT Evaluator Initialized")
        self.logger.info("=" * 60)
        self.logger.info(f"Output directory: {self.output_dir}")
        self.logger.info(f"Experiment name: {cfg.exp_name}")

    def run(self):
        """Run PROSPECT evaluation pipeline"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🚀 Starting PROSPECT Evaluation")
        self.logger.info("=" * 60)

        # Print configuration
        self.logger.info("\n📋 Configuration:")
        self.logger.info(f"\n{OmegaConf.to_yaml(self.cfg)}")

        # Step 1: Create dataset
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📦 Step 1: Loading Dataset")
        self.logger.info("=" * 60)
        dataset = DataSourceFactory.create_dataset(self.cfg.data_source)
        self.logger.info(f"✅ Loaded {len(dataset)} videos")

        # Log video IDs
        for i, video_id in enumerate(dataset.video_ids):
            num_frames = len(dataset.samples[i].frames)
            self.logger.info(f"  - Video {i+1}: {video_id} ({num_frames} frames)")

        # Step 2: Create runner
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🔧 Step 2: Creating Inference Runner")
        self.logger.info("=" * 60)

        runner_type = self.cfg.generator.runner_type
        self.logger.info(f"Runner type: {runner_type}")
        self.logger.info(f"Generator type: {self.cfg.generator.type}")

        if runner_type == "vlm_stream":
            self.logger.info(f"Model: {self.cfg.model.name}")
            
            # Prepare context strategy config
            context_strategy_type = self.cfg.context_strategy.get("type", "none")
            context_strategy_config = dict(self.cfg.context_strategy)
            context_strategy_config.pop("type", None)  # Remove type from config dict
            
            self.logger.info(f"Context strategy: {context_strategy_type}")
            
            runner = VLMStreamRunner(
                model_name=self.cfg.model.name,
                eval_name=f"{self.cfg.generator.type}_{self.cfg.model.log_name}",
                device=self.cfg.model.device,
                torch_dtype=self.cfg.model.torch_dtype,
                max_new_tokens=self.cfg.model.max_new_tokens,
                temperature=self.cfg.model.temperature,
                top_p=self.cfg.model.get("top_p", 0.9),
                do_sample=self.cfg.model.get("do_sample", True),
                transition_detection_prompt=self.cfg.generator.transition_detection_prompt,
                dialogue_generation_prompt=self.cfg.generator.dialogue_generation_prompt,
                fps=self.cfg.fps,
                not_talk_threshold=self.cfg.not_talk_threshold,
                use_gt_substeps=self.cfg.generator.get("use_gt_substeps", True),
                cache_dir=self.cfg.model.get("cache_dir", None),
                context_strategy_type=context_strategy_type,
                context_strategy_config=context_strategy_config,
                use_kv_cache=self.cfg.generator.get("use_kv_cache", False),
                max_seq_len=self.cfg.generator.get("max_seq_len", 4096),
                reserved_seq_len=self.cfg.generator.get("reserved_seq_len", 128),
            )

        elif runner_type == "sanity_check":
            self.logger.info("🔮 Sanity check runner (ground truth oracle)")
            runner = SanityCheckRunner(
                fps=self.cfg.fps,
            )

        else:
            raise ValueError(f"Unknown runner type: {runner_type}")

        self.logger.info("✅ Runner created")

        # Step 3: Create generator
        self.logger.info("\n" + "=" * 60)
        self.logger.info("🎯 Step 3: Creating Generator")
        self.logger.info("=" * 60)
        generator = GeneratorFactory.create_generator(
            generator_cfg=self.cfg.generator,
            dataset=dataset,
            runner=runner,
            output_dir=str(self.output_dir),
            main_cfg=self.cfg,
        )
        self.logger.info("✅ Generator created")

        # Step 4: Run evaluation
        self.logger.info("\n" + "=" * 60)
        self.logger.info("▶️  Step 4: Running Evaluation")
        self.logger.info("=" * 60)
        metrics = generator.run()

        # Step 5: Print results
        self._print_results(metrics)

        self.logger.info("\n" + "=" * 60)
        self.logger.info("✅ PROSPECT Evaluation Complete!")
        self.logger.info("=" * 60)
        self.logger.info(f"Results saved to: {self.output_dir}")
        self.logger.info(f"  - Predictions: {self.output_dir}/results/")
        self.logger.info(f"  - Metrics: {self.output_dir}/metrics.json")
        self.logger.info(f"  - All results: {self.output_dir}/all_results.json")

        return metrics

    def _print_results(self, metrics: dict):
        """Print evaluation results in a nice format"""
        self.logger.info("\n" + "=" * 60)
        self.logger.info("📊 PROSPECT Evaluation Results")
        self.logger.info("=" * 60)

        # Core metrics
        if "precision" in metrics:
            self.logger.info(f"\n🎯 Dialogue Generation Metrics:")
            self.logger.info(f"  Precision (AP):     {metrics['precision']:.4f}")
            self.logger.info(f"  Recall (AR):        {metrics['recall']:.4f}")
            self.logger.info(f"  F1 Score:           {metrics['F1']:.4f}")
            self.logger.info(f"  Jaccard Index (JI): {metrics['jaccard_index']:.4f}")

        # NLG metrics (check both "Bleu_4" and "dialog_Bleu_4" formats)
        has_bleu = "Bleu_4" in metrics or "dialog_Bleu_4" in metrics
        if has_bleu:
            self.logger.info(f"\n📝 NLG Quality Metrics:")
            for i in range(1, 5):
                # Try both key formats
                key = f"Bleu_{i}" if f"Bleu_{i}" in metrics else f"dialog_Bleu_{i}"
                if key in metrics:
                    self.logger.info(f"  BLEU-{i}:   {metrics[key]:.4f}")
            
            # CIDEr and METEOR
            cider_key = "CIDEr" if "CIDEr" in metrics else "dialog_CIDEr"
            if cider_key in metrics:
                self.logger.info(f"  CIDEr:    {metrics[cider_key]:.4f}")
            
            meteor_key = "METEOR" if "METEOR" in metrics else "dialog_METEOR"
            if meteor_key in metrics:
                self.logger.info(f"  METEOR:   {metrics[meteor_key]:.4f}")

        # Detailed counts
        if "num_matched" in metrics:
            self.logger.info(f"\n📈 Detailed Counts:")
            self.logger.info(f"  Matched dialogues:   {metrics['num_matched']}")
            self.logger.info(f"  Missed dialogues:    {metrics['num_missed']}")
            self.logger.info(f"  Redundant dialogues: {metrics['num_redundant']}")

        # Additional metrics
        if "semantic_score" in metrics:
            self.logger.info(f"\n🔍 Additional Metrics:")
            self.logger.info(f"  Semantic similarity: {metrics['semantic_score']:.4f}")
            if "time_diff" in metrics:
                self.logger.info(f"  Time difference:     {metrics['time_diff']:.2f}s")

        # Comparison to ProAssist baseline
        self.logger.info(f"\n📚 Comparison to ProAssist Paper:")
        self.logger.info(f"  ProAssist (trained): F1 ~0.35, BLEU-4 ~0.25")
        
        # Get BLEU-4 from either key format
        bleu4 = metrics.get('Bleu_4', metrics.get('dialog_Bleu_4', 0))
        self.logger.info(
            f"  PROSPECT (baseline): F1 {metrics.get('F1', 0):.4f}, BLEU-4 {bleu4:.4f}"
        )

        if metrics.get("F1", 0) > 0.10:
            self.logger.info(f"  ✅ Baseline shows promising results!")
        else:
            self.logger.info(f"  ⚠️  Lower than expected - may need tuning")
        
        # Explain low BLEU if applicable
        if bleu4 < 0.01 and metrics.get("num_matched", 0) > 0:
            self.logger.info(f"\n⚠️  Note: Very low BLEU-4 ({bleu4:.6f}) despite {metrics.get('num_matched', 0)} matches")
            self.logger.info(f"  Cause: VLM paraphrases instead of using exact words")
            self.logger.info(f"  Evidence: Semantic similarity = {metrics.get('semantic_score', 0):.2f} (content is relevant)")
            self.logger.info(f"  Solution: This is expected for zero-shot VLMs - fine-tuning would improve BLEU")


@hydra.main(
    config_path="../../config/prospect", config_name="prospect", version_base=None
)
def main(cfg: DictConfig):
    """Main entry point with Hydra configuration"""
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, cfg.logging.level), format=cfg.logging.format
    )

    # Create and run evaluator
    evaluator = ProspectEvaluator(cfg)
    evaluator.run()


if __name__ == "__main__":
    main()
