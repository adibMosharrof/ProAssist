"""Baseline generator for PROSPECT evaluation"""

import logging
from pathlib import Path
from typing import Dict, Any

from omegaconf import DictConfig, OmegaConf

from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
from mmassist.eval.evaluators.stream_evaluator import StreamEvaluator


logger = logging.getLogger(__name__)


class BaselineGenerator:
    """
    Baseline generator: VLM-based dialogue generation at transitions.

    This class wraps:
    - VLMStreamRunner: Custom VLM inference
    - StreamEvaluator: ProAssist's evaluation framework

    By using ProAssist's StreamEvaluator, we get identical metrics
    (AP, AR, F1, BLEU, JI) without duplicating code.
    """

    def __init__(
        self,
        dataset: ProAssistVideoDataset,
        runner: VLMStreamRunner,
        output_dir: str,
        cfg: DictConfig,
    ):
        """
        Initialize baseline generator

        Args:
            dataset: ProAssist video dataset
            runner: VLM stream runner
            output_dir: Output directory for results
            cfg: Full Hydra configuration
        """
        self.dataset = dataset
        self.runner = runner
        self.output_dir = Path(output_dir)
        self.cfg = cfg

        logger.info("Initializing BaselineGenerator")
        logger.info(f"Dataset: {len(dataset)} videos")
        logger.info(f"Output: {output_dir}")

        # Create ProAssist's StreamEvaluator
        # This gives us identical evaluation to ProAssist paper
        self.evaluator = self._build_evaluator()

    def _build_evaluator(self) -> StreamEvaluator:
        """Build ProAssist's StreamEvaluator with our custom runner"""
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
            model=None,  # Not needed, we handle inference in runner
            tokenizer=None,  # Not needed
            inference_runner=self.runner,  # Our custom VLM runner
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
        Run baseline evaluation on all videos.

        Returns:
            Dict of metrics (AP, AR, F1, BLEU, JI, etc.)
        """
        logger.info("=" * 60)
        logger.info("Starting PROSPECT Baseline Evaluation")
        logger.info("=" * 60)
        logger.info(f"Videos to evaluate: {len(self.dataset)}")

        # Run predictions on all videos
        sample_indices = list(range(len(self.dataset)))

        logger.info("Running inference on all videos...")
        self.evaluator.run_all_predictions(sample_indices, progress_bar=True)

        # Compute metrics using ProAssist's evaluation
        logger.info("Computing metrics...")
        metrics = self.evaluator.compute_metrics(must_complete=True)

        logger.info("=" * 60)
        logger.info("Baseline Evaluation Complete!")
        logger.info("=" * 60)

        return metrics
