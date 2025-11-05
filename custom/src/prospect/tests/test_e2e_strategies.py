"""
End-to-end testing of all context strategies on real video

Tests all strategies:
1. none - No KV cache (stateless baseline)
2. drop_all - Drop all cache on overflow
3. drop_middle - Keep start + end, drop middle
4. summarize_and_drop - Generate summary then drop

Metrics collected (same format as PROSPECT evaluator):
- Dialogue generation quality (AP, AR, F1, Jaccard)
- NLG metrics (BLEU, CIDEr, METEOR)
- Semantic similarity and time difference
- KV cache statistics
- Generation time and memory
"""

# Source bash profile FIRST to set correct HOME directory
# This fixes disk space issues with HuggingFace cache
import os
import subprocess

bash_profile = os.path.expanduser("~/.bash_profile")
if os.path.exists(bash_profile):
    try:
        # Get environment from sourced bash profile
        result = subprocess.run(
            f"source {bash_profile} && env",
            shell=True,
            executable="/bin/bash",
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            # Update os.environ with values from bash profile
            for line in result.stdout.strip().split("\n"):
                if "=" in line:
                    key, _, value = line.partition("=")
                    os.environ[key] = value
            new_home = os.environ.get("HOME", "unknown")
            print(f"✅ Sourced {bash_profile} (HOME now: {new_home})")
        else:
            print(
                f"⚠️  Warning: bash profile source failed with code {result.returncode}"
            )
    except Exception as e:
        print(f"⚠️  Warning: Could not source bash profile: {e}")
else:
    print(f"⚠️  Warning: {bash_profile} not found")

import sys
import json
import time
import logging
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict
import pandas as pd
import torch

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prospect.data_sources.data_source_factory import DataSourceFactory
from prospect.generators.baseline_generator import BaselineGenerator
from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.visualization import create_trace
from prospect.visualization.html_timeline_generator import HTMLTimelineGenerator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class StrategyMetrics:
    """Metrics for a single strategy run (matching PROSPECT evaluator format)"""

    strategy_name: str
    num_frames: int
    num_dialogues_generated: int
    num_dialogues_reference: int
    num_matched: int
    num_missed: int
    num_redundant: int
    precision: float  # AP
    recall: float  # AR
    f1: float
    jaccard: float  # JI
    bleu_1: float
    bleu_2: float
    bleu_3: float
    bleu_4: float
    cider: float
    meteor: float
    semantic_similarity: float
    time_difference: float
    total_time: float
    avg_time_per_frame: float
    peak_memory_mb: float


class E2EStrategyTester:
    """End-to-end tester for all context strategies"""

    def __init__(
        self,
        video_ids: List[str],
        data_source_config: Dict,
        model_config: Dict,
        generator_config: Dict,
        output_dir: str = "./custom/outputs/e2e_strategy_comparison",
    ):
        """
        Initialize E2E tester

        Args:
            video_ids: List of video IDs to test
            data_source_config: Data source configuration
            model_config: Model configuration
            generator_config: Generator configuration
            output_dir: Where to save results
        """
        self.video_ids = video_ids
        self.data_source_config = data_source_config
        self.model_config = model_config
        self.generator_config = generator_config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Strategies to test
        self.strategies = [
            {"name": "none", "use_kv_cache": False},
            {"name": "drop_all", "use_kv_cache": True},
            {"name": "drop_middle", "use_kv_cache": True},
            self._load_strategy_config("summarize_and_drop"),
            {
                "name": "summarize_with_dst",
                "use_kv_cache": True,
                "dst_file": "data/proassist_dst_manual_data/assembly101/assembly_nusar-2021_action_both_9011-c03f_9011_user_id_2021-02-01_160239__HMC_84355350_mono10bit.tsv",
            },
        ]

        # Create strategy configs dict for easy lookup
        self.strategy_configs = {s["name"]: s for s in self.strategies}

    def _load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """
        Load strategy configuration from YAML file
        
        Args:
            strategy_name: Name of the strategy (e.g., 'summarize_and_drop')
            
        Returns:
            Strategy configuration dict
        """
        import yaml
        from pathlib import Path
        
        config_path = Path(__file__).parent.parent.parent.parent.parent / "custom" / "config" / "prospect" / "context_strategy" / f"{strategy_name}.yaml"
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Add required fields
            config["name"] = strategy_name
            config["use_kv_cache"] = True
            
            return config
            
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_path}, using default config")
            return {"name": strategy_name, "use_kv_cache": True}
        except Exception as e:
            logger.error(f"Error loading config {config_path}: {e}, using default config")
            return {"name": strategy_name, "use_kv_cache": True}

    def run_all_strategies(self) -> Dict[str, StrategyMetrics]:
        """
        Run all strategies and collect metrics

        Returns:
            Dict mapping strategy name to metrics
        """
        results = {}

        for strategy_config in self.strategies:
            strategy_name = strategy_config["name"]

            logger.info("=" * 80)
            logger.info(f"Testing strategy: {strategy_name}")
            logger.info("=" * 80)

            try:
                metrics = self._test_strategy(strategy_config)
                results[strategy_name] = metrics

                logger.info(f"✅ Strategy {strategy_name} completed")
                logger.info(f"   Generated: {metrics.num_dialogues_generated}")
                logger.info(f"   Matched: {metrics.num_matched}")
                logger.info(f"   F1: {metrics.f1:.4f}")
                logger.info(f"   BLEU-4: {metrics.bleu_4:.4f}")

            except Exception as e:
                logger.error(f"❌ Strategy {strategy_name} failed: {e}")
                import traceback

                traceback.print_exc()

        return results

    def _test_strategy(self, strategy_config: Dict) -> StrategyMetrics:
        """
        Test a single strategy using BaselineGenerator

        Args:
            strategy_config: Strategy configuration

        Returns:
            StrategyMetrics with results
        """
        strategy_name = strategy_config["name"]
        use_kv_cache = strategy_config["use_kv_cache"]

        # Create timestamped run directory (Hydra-style)
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.output_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging to file (Hydra-style)
        log_file = run_dir / f"{strategy_name}.log"
        file_handler = logging.FileHandler(log_file, mode="w")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

        # Add file handler to root logger so all loggers write to file
        root_logger = logging.getLogger()
        root_logger.addHandler(file_handler)

        logger.info(f"Output directory: {run_dir}")
        logger.info(f"Log file: {log_file}")

        # Load dataset
        data_source_config = dict(self.data_source_config)
        data_source_config["video_ids"] = self.video_ids

        # Convert to DictConfig for DataSourceFactory
        from omegaconf import DictConfig, OmegaConf

        data_source_cfg = DictConfig(data_source_config)
        dataset = DataSourceFactory.create_dataset(data_source_cfg)

        logger.info(f"Loaded {len(dataset)} videos for {strategy_name}")

        # Update generator config with strategy
        gen_config = dict(self.generator_config)
        gen_config["context_handling_method"] = (
            strategy_name if use_kv_cache else "none"
        )

        # Create full config for BaselineGenerator
        full_config = {
            "data_source": data_source_config,
            "model": self.model_config,
            "generator": gen_config,
            "fps": self.data_source_config["fps"],
            "not_talk_threshold": 0.5,
            "eval_max_seq_len_str": "4k",
            "match_window_time": [-15, 15],
            "match_dist_func_factor": 0.3,
            "match_dist_func_power": 1.5,
            "match_semantic_score_threshold": 0.5,
            "sts_model_type": "sentence-transformers/all-mpnet-base-v2",
            "nlg_metrics": ["Bleu", "CIDEr", "METEOR"],
        }
        cfg = DictConfig(full_config)

        # Create runner
        from prospect.context_strategies.context_strategy_factory import (
            ContextStrategyFactory,
        )

        # Build context strategy config
        context_strategy_config = {}
        if strategy_name == "summarize_with_dst":
            dst_file = strategy_config.get("dst_file")
            if dst_file:
                context_strategy_config["dst_file"] = dst_file
        elif strategy_name == "summarize_and_drop":
            # Pass the loaded YAML config for summarize_and_drop
            context_strategy_config = strategy_config.copy()
            # Remove strategy metadata that's not needed by the factory
            context_strategy_config.pop("name", None)
            context_strategy_config.pop("use_kv_cache", None)

        # Create trace for timeline visualization
        # Get video info
        video_id = self.video_ids[0] if self.video_ids else "unknown"
        num_frames = sum(len(dataset[i]["frames"]) for i in range(len(dataset)))

        # Create frames directory for saving frame images
        frames_dir = run_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        trace_kwargs = {}
        if strategy_name == "drop_middle":
            trace_kwargs["last_keep_len"] = 512
        elif strategy_name == "summarize_with_dst":
            trace_kwargs["dst_file"] = strategy_config.get("dst_file")

        trace = create_trace(
            strategy_name=strategy_name,
            video_id=video_id,
            total_frames=num_frames,
            frames_dir=str(frames_dir),
            **trace_kwargs,
        )

        logger.info(
            f"Created trace for {strategy_name} timeline visualization (frames: {frames_dir})"
        )

        runner = VLMStreamRunner(
            model_name=self.model_config["name"],
            eval_name=f"e2e_{strategy_name}",
            device=self.model_config.get("device", "cuda"),
            torch_dtype=self.model_config.get("torch_dtype", "bfloat16"),
            max_new_tokens=self.model_config.get("max_new_tokens", 100),
            temperature=self.model_config.get("temperature", 0.7),
            top_p=self.model_config.get("top_p", 0.9),
            do_sample=self.model_config.get("do_sample", True),
            transition_detection_prompt=gen_config.get(
                "transition_detection_prompt", ""
            ),
            dialogue_generation_prompt=gen_config.get("dialogue_generation_prompt", ""),
            fps=self.data_source_config["fps"],
            use_gt_substeps=gen_config.get("use_gt_substeps", True),
            use_kv_cache=use_kv_cache,
            context_strategy_type=strategy_name if use_kv_cache else "none",
            context_strategy_config=(
                context_strategy_config if context_strategy_config else None
            ),
            cache_dir=self.model_config.get("cache_dir"),
            trace=trace,  # Pass trace to runner
        )

        # Create generator (uses BaselineGenerator like PROSPECT evaluator)
        start_time = time.time()
        torch.cuda.reset_peak_memory_stats()

        generator = BaselineGenerator(
            dataset=dataset,
            runner=runner,
            output_dir=str(run_dir / "eval"),  # Add eval subdirectory
            cfg=cfg,
            trace=trace,
        )

        # Run evaluation
        logger.info(f"Running evaluation for {strategy_name}...")
        metrics_data = generator.run()  # Returns metrics dict directly, not nested

        end_time = time.time()
        total_time = end_time - start_time
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB

        # Update trace with final metrics
        trace.total_time = total_time
        trace.peak_memory_mb = peak_memory
        trace.set_metrics(metrics_data)

        # Generate HTML timeline visualization in the run directory
        html_path = run_dir / f"{strategy_name}_timeline.html"
        try:
            logger.info(f"Generating HTML timeline: {html_path}")
            logger.info(f"Trace has {len(trace.compression_events)} compression events")
            logger.info(f"Trace has {len(trace.generation_events)} generation events")
            logger.info(
                f"Trace has {len(trace.ground_truth_dialogues)} ground truth dialogues"
            )

            # Save trace to JSON for debugging
            trace_json_path = run_dir / f"{strategy_name}_trace.json"
            trace.save_json(trace_json_path)
            logger.info(f"Trace data saved to: {trace_json_path}")

            generator_html = HTMLTimelineGenerator(trace)
            generator_html.generate_html(html_path, include_frames=True)
            logger.info(f"✅ HTML timeline saved: {html_path}")
        except Exception as e:
            logger.error(f"❌ Failed to generate HTML timeline: {e}")
            import traceback

            traceback.print_exc()

        # Calculate metrics matching your previous format
        num_frames = sum(len(dataset[i]["frames"]) for i in range(len(dataset)))

        # StreamEvaluator.compute_metrics() returns metrics directly, with these keys:
        # - precision, recall, F1 (dialogue quality)
        # - num_matched, num_mismatched, num_missed, num_redundant (counts)
        # - Bleu_1, Bleu_2, Bleu_3, Bleu_4, CIDEr, METEOR (NLG metrics)
        # - semantic_score, time_diff (semantic/temporal)
        # - jaccard_index, missing_rate, redundant_rate

        metrics = StrategyMetrics(
            strategy_name=strategy_name,
            num_frames=num_frames,
            # num_dialogues_generated = matched + mismatched + redundant
            # (all dialogues that were actually generated, regardless of quality)
            num_dialogues_generated=metrics_data.get("num_matched", 0)
            + metrics_data.get("num_mismatched", 0)
            + metrics_data.get("num_redundant", 0),
            num_dialogues_reference=metrics_data.get("num_matched", 0)
            + metrics_data.get("num_missed", 0),
            num_matched=metrics_data.get("num_matched", 0),
            num_missed=metrics_data.get("num_missed", 0),
            num_redundant=metrics_data.get("num_redundant", 0),
            precision=metrics_data.get("precision", 0.0),
            recall=metrics_data.get("recall", 0.0),
            f1=metrics_data.get("F1", 0.0),
            jaccard=metrics_data.get("jaccard_index", 0.0),
            bleu_1=metrics_data.get("Bleu_1", 0.0),
            bleu_2=metrics_data.get("Bleu_2", 0.0),
            bleu_3=metrics_data.get("Bleu_3", 0.0),
            bleu_4=metrics_data.get("Bleu_4", 0.0),
            cider=metrics_data.get("CIDEr", 0.0),
            meteor=metrics_data.get("METEOR", 0.0),
            semantic_similarity=metrics_data.get("semantic_score", 0.0),
            time_difference=metrics_data.get("time_diff", 0.0),
            total_time=total_time,
            avg_time_per_frame=total_time / num_frames if num_frames > 0 else 0.0,
            peak_memory_mb=peak_memory,
        )

        # Clean up GPU memory before next strategy
        # Explicitly delete model from runner
        if hasattr(runner, "model"):
            del runner.model
        if hasattr(runner, "processor"):
            del runner.processor
        del runner, generator, dataset

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.info(
                f"GPU memory freed. Current allocated: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB"
            )

        # Remove file handler to avoid duplicate logging in next test
        root_logger.removeHandler(file_handler)
        file_handler.close()

        return metrics

    def generate_summary_report(
        self, results: Dict[str, StrategyMetrics]
    ) -> pd.DataFrame:
        """
        Generate summary comparison of all strategies

        Args:
            results: Dict mapping strategy name to metrics

        Returns:
            DataFrame with comparison
        """
        # Convert to DataFrame
        rows = []
        for strategy_name, metrics in results.items():
            rows.append(
                {
                    "Strategy": strategy_name,
                    "Use KV Cache": "Yes" if strategy_name != "none" else "No",
                    "Frames": metrics.num_frames,
                    "Generated": metrics.num_dialogues_generated,
                    "Reference": metrics.num_dialogues_reference,
                    "Matched": metrics.num_matched,
                    "Missed": metrics.num_missed,
                    "Redundant": metrics.num_redundant,
                    "Precision (AP)": f"{metrics.precision:.4f}",
                    "Recall (AR)": f"{metrics.recall:.4f}",
                    "F1 Score": f"{metrics.f1:.4f}",
                    "Jaccard (JI)": f"{metrics.jaccard:.4f}",
                    "BLEU-1": f"{metrics.bleu_1:.4f}",
                    "BLEU-2": f"{metrics.bleu_2:.4f}",
                    "BLEU-3": f"{metrics.bleu_3:.4f}",
                    "BLEU-4": f"{metrics.bleu_4:.4f}",
                    "CIDEr": f"{metrics.cider:.4f}",
                    "METEOR": f"{metrics.meteor:.4f}",
                    "Semantic Sim": f"{metrics.semantic_similarity:.4f}",
                    "Time Diff (s)": f"{metrics.time_difference:.2f}",
                    "Total Time (s)": f"{metrics.total_time:.2f}",
                    "Time/Frame (s)": f"{metrics.avg_time_per_frame:.3f}",
                    "Peak Memory (MB)": f"{metrics.peak_memory_mb:.1f}",
                }
            )

        df = pd.DataFrame(rows)

        # Save to CSV
        csv_path = self.output_dir / "strategy_comparison.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved comparison to {csv_path}")

        # Save detailed JSON
        json_path = self.output_dir / "strategy_metrics_detailed.json"
        detailed = {strategy: asdict(metrics) for strategy, metrics in results.items()}
        with open(json_path, "w") as f:
            json.dump(detailed, f, indent=2)
        logger.info(f"Saved detailed metrics to {json_path}")

        return df

    def print_summary(self, df: pd.DataFrame):
        """Print summary table"""
        print("\n" + "=" * 150)
        print("STRATEGY COMPARISON SUMMARY")
        print("=" * 150)

        # Print core metrics
        core_cols = [
            "Strategy",
            "Use KV Cache",
            "Generated",
            "Matched",
            "F1 Score",
            "BLEU-4",
            "CIDEr",
            "METEOR",
        ]
        print("\n📊 Core Metrics:")
        print(df[core_cols].to_string(index=False))

        # Print detailed quality metrics
        quality_cols = [
            "Strategy",
            "Precision (AP)",
            "Recall (AR)",
            "Jaccard (JI)",
            "Semantic Sim",
            "Time Diff (s)",
        ]
        print("\n📈 Quality Metrics:")
        print(df[quality_cols].to_string(index=False))

        # Print performance metrics
        perf_cols = ["Strategy", "Total Time (s)", "Time/Frame (s)", "Peak Memory (MB)"]
        print("\n⚡ Performance Metrics:")
        print(df[perf_cols].to_string(index=False))

        print("=" * 150)


def main():
    """Main test runner"""
    import hydra
    from hydra import compose, initialize
    from omegaconf import DictConfig, OmegaConf

    # Initialize Hydra - config_path is relative to this file
    # This file is in: custom/src/prospect/tests/
    # Config is in: custom/config/prospect/
    # So we need to go up 3 levels: ../../../config/prospect
    with initialize(version_base=None, config_path="../../../config/prospect"):
        # Compose config with defaults
        cfg = compose(config_name="prospect")

        # Convert to regular dict for easier manipulation
        config = OmegaConf.to_container(cfg, resolve=True)

    # Get video IDs from data_source config or use default
    video_ids = config["data_source"].get("video_ids", ["9011-c03f"])
    output_dir = "./custom/outputs/e2e_strategy_comparison"

    # Create tester
    logger.info("\n" + "=" * 100)
    logger.info("Starting E2E testing of all context strategies")
    logger.info("=" * 100)
    logger.info(f"Video IDs: {video_ids}")
    logger.info(f"Output: {output_dir}")

    tester = E2EStrategyTester(
        video_ids=video_ids,
        data_source_config=config["data_source"],
        model_config=config["model"],
        generator_config=config["generator"],
        output_dir=output_dir,
    )

    # Run all strategies
    results = tester.run_all_strategies()

    # Generate and print summary
    df = tester.generate_summary_report(results)
    tester.print_summary(df)

    logger.info("\n✅ E2E testing complete!")
    logger.info(f"Results saved to: {tester.output_dir}")


if __name__ == "__main__":
    main()
