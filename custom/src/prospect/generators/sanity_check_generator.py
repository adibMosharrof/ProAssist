"""Sanity check generator - validates evaluation pipeline with ground truth"""

import logging
from pathlib import Path
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf

from prospect.runners.sanity_check_runner import SanityCheckRunner
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator


logger = logging.getLogger(__name__)


class SanityCheckGenerator:
    """
    Sanity check generator for pipeline validation.

    Uses ground truth dialogues as predictions to verify the evaluation
    pipeline achieves perfect (or near-perfect) metrics.

    This acts as a debugging baseline - if metrics are not perfect,
    there's likely a bug in the evaluation pipeline.
    """

    def __init__(
        self,
        dataset: ProAssistVideoDataset,
        runner: SanityCheckRunner,
        output_dir: str,
        cfg: DictConfig,
    ):
        """
        Initialize sanity check generator

        Args:
            dataset: ProAssist video dataset
            runner: Sanity check runner (ground truth passthrough)
            output_dir: Output directory for results
            cfg: Full Hydra configuration
        """
        self.dataset = dataset
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.cfg = cfg

        logger.info("Initializing SanityCheckGenerator")
        logger.info(f"Dataset: {len(dataset)} videos")
        logger.info(f"Output: {output_dir}")
        logger.info("🔮 Using ground truth as predictions (perfect oracle)")

        # Create ProAssist's StreamEvaluator
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> StreamEvaluator:
        """Build ProAssist's StreamEvaluator with sanity check runner"""
        logger.info("Creating StreamEvaluator (ProAssist evaluation framework)")

        # Convert match_window_time to tuple (Hydra returns ListConfig, not list)
        match_window_time = OmegaConf.to_container(
            self.cfg.match_window_time, resolve=True
        )
        if not isinstance(match_window_time, tuple):
            match_window_time = tuple(match_window_time)

        logger.info(
            f"match_window_time type: {type(match_window_time)}, value: {match_window_time}"
        )

        evaluator = StreamEvaluator.build(
            model_path=str(self.output_dir),
            dataset=self.dataset,
            model=None,  # Not needed, ground truth passthrough
            tokenizer=None,  # Not needed
            inference_runner=self.runner,  # Sanity check runner
            sts_model_type=self.cfg.sts_model_type,
            match_window_time=match_window_time,
            match_dist_func_factor=self.cfg.match_dist_func_factor,
            match_dist_func_power=self.cfg.match_dist_func_power,
            match_semantic_score_threshold=self.cfg.match_semantic_score_threshold,
            nlg_metrics=list(self.cfg.nlg_metrics),
            fps=self.cfg.fps,
            not_talk_threshold=self.cfg.not_talk_threshold,
            eval_max_seq_len_str=self.cfg.eval_max_seq_len_str,
            context_handling_method=self.cfg.generator.context_handling_method,
            use_gt_context=self.cfg.generator.use_gt_context,
        )

        logger.info("✅ StreamEvaluator created")
        return evaluator

    def run(self) -> Dict[str, Any]:
        """
        Run sanity check evaluation on all videos.

        Expected Results:
        - Precision (AP): ≥ 0.95
        - Recall (AR): ≥ 0.95
        - F1 Score: ≥ 0.95
        - Jaccard Index (JI): ≥ 0.95
        - BLEU-4: ≥ 0.95
        - CIDEr: High (>3.0)
        - METEOR: ≥ 0.95

        Returns:
            Dict of metrics
        """
        logger.info("=" * 60)
        logger.info("Starting PROSPECT Sanity Check Evaluation")
        logger.info("🔮 Ground Truth Oracle Mode")
        logger.info("=" * 60)
        logger.info(f"Videos to evaluate: {len(self.dataset)}")

        # Run predictions on all videos
        sample_indices = list(range(len(self.dataset)))

        logger.info("Running sanity check inference on all videos...")
        self.evaluator.run_all_predictions(sample_indices, progress_bar=True)

        # Compute metrics using ProAssist's evaluation
        logger.info("Computing metrics...")
        metrics = self.evaluator.compute_metrics(must_complete=True)

        logger.info("=" * 60)
        logger.info("📊 PROSPECT Sanity Check Results")
        logger.info("=" * 60)

        # Log key metrics
        if "precision" in metrics:
            logger.info(
                f"Precision (AP):     {metrics['precision']:.4f} (expect ≥ 0.95)"
            )
        if "recall" in metrics:
            logger.info(f"Recall (AR):        {metrics['recall']:.4f} (expect ≥ 0.95)")
        if "F1" in metrics:
            logger.info(f"F1 Score:           {metrics['F1']:.4f} (expect ≥ 0.95)")
        if "jaccard_index" in metrics:
            logger.info(
                f"Jaccard Index (JI): {metrics['jaccard_index']:.4f} (expect ≥ 0.95)"
            )
        if "Bleu_4" in metrics:
            logger.info(f"BLEU-4:             {metrics['Bleu_4']:.4f} (expect ≥ 0.95)")

        # Validate results
        passed = True
        min_threshold = 0.95

        for key, threshold in [
            ("precision", min_threshold),
            ("recall", min_threshold),
            ("F1", min_threshold),
            ("jaccard_index", min_threshold),
            ("Bleu_4", min_threshold),
        ]:
            if key in metrics and metrics[key] < threshold:
                logger.warning(
                    f"⚠️  {key}: {metrics[key]:.4f} is below threshold {threshold}"
                )
                passed = False

        if passed:
            logger.info("\n✅ SANITY CHECK PASSED - Pipeline works correctly!")
        else:
            logger.warning("\n⚠️ SANITY CHECK FAILED - Pipeline may have issues")

        logger.info("=" * 60)

        return metrics
