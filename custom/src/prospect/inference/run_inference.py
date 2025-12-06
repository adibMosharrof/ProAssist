import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging
import os
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

from custom.src.prospect.inference.dst_evaluator import DSTEvaluator
from custom.src.prospect.metrics.dst_binary_metrics import DSTBinaryMetrics
from custom.src.prospect.metrics.dst_content_metrics import DSTContentMetrics
from custom.src.prospect.data_sources.dst_training_dataset import DSTTrainingDataset
from custom.src.prospect.metrics.proassist_metrics import ProAssistMetrics
from custom.src.prospect.utils.logging_utils import Tee

# Register resolver for ${project_root}
if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: os.getcwd())

class InferencePipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self._setup_output_dir()
        self._setup_logging()
        
    def _setup_output_dir(self):
        # Use HydraConfig to get the actual output dir (which respects the config due to hydra.run.dir)
        hydra_cfg = HydraConfig.get()
        self.output_dir = Path(hydra_cfg.runtime.output_dir)
        self.logger.info(f"Output directory set to: {self.output_dir}")

    def _setup_logging(self):
        # Hydra already sets up logging to ${hydra.runtime.output_dir}/${hydra.job.name}.log
        # Ensure our logger uses the same level
        self.logger.setLevel(logging.INFO)
        
        # Redirect stdout and stderr to files using Tee class
        stdout_file = self.output_dir / "stdout.log"
        stderr_file = self.output_dir / "stderr.log"
        
        sys.stdout = Tee(open(stdout_file, 'w'), sys.stdout)
        sys.stderr = Tee(open(stderr_file, 'w'), sys.stderr)
        

    def load_dataset(self):
        self.logger.info("Loading dataset...")
        # We reuse DSTTrainingDataset but for evaluation (test set)
        # Support multiple datasets by concatenating them
        datasets = []
        
        for dataset_name in self.cfg.data.datasets:
            dataset = DSTTrainingDataset(
                data_path=self.cfg.data.data_path,
                step_name=self.cfg.data.step_name, # e.g. "test"
                dataset_name=dataset_name,
                max_seq_len=self.cfg.inference.max_seq_len,
                neg_frame_sampling_rate=0.0, # No negative sampling for inference
                input_style=self.cfg.data.get("input_style", "proassist")
            )
            datasets.append(dataset)
        
        # Concatenate all datasets
        if len(datasets) == 1:
            self.dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            self.dataset = ConcatDataset(datasets)
        
        self.logger.info(f"Loaded {len(self.dataset)} samples from {len(datasets)} dataset(s)")
        
        # Optionally limit samples for testing
        if getattr(self.cfg.inference, "limit_samples", None):
            from torch.utils.data import Subset
            limit = self.cfg.inference.limit_samples
            indices = range(min(len(self.dataset), limit))
            self.dataset = Subset(self.dataset, indices)
            self.logger.info(f"Limiting dataset to {limit} samples")



    def initialize_metrics(self) -> List[Any]:
        metrics = []
        if self.cfg.metrics.binary:
            metrics.append(DSTBinaryMetrics())
        if self.cfg.metrics.content:
            metrics.append(DSTContentMetrics())
        if self.cfg.metrics.get("proassist", False):
            metrics.append(ProAssistMetrics())
        return metrics

    def save_metrics_to_csv(self, metrics: Dict[str, float]):
        csv_file = self.output_dir / "metrics.csv"
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Metric", "Value"])
            for key, value in metrics.items():
                writer.writerow([key, value])
        self.logger.info(f"Metrics saved to {csv_file}")

    def run(self):
        self.load_dataset()
        metrics = self.initialize_metrics()
        
        evaluator = DSTEvaluator(
            model_config=self.cfg.model,
            dataset=self.dataset,
            metrics=metrics,
            output_dir=str(self.output_dir),
            num_gpus=self.cfg.inference.num_gpus,
            fps=self.cfg.inference.fps
        )
        
        results = evaluator.evaluate()
        self.save_metrics_to_csv(results)

@hydra.main(config_path="../../../config/inference", config_name="dst_inference", version_base=None)
def main(cfg: DictConfig):
    # Get the actual output directory that Hydra created
    # Note: We need to handle the case where output_dir is overridden in the config
    # But for consistency with training script, let's use the one established by the pipeline
    # or just use the hydra one.
    
    # However, InferencePipeline sets up its own output dir based on cfg.
    # Let's instantiate pipeline first to get the output dir, or just use the logic from training script.
    
    # Actually, let's do it inside main before pipeline runs.
    # We need to know where to save the logs.
    # The pipeline calculates output_dir.
    
    # Let's use a temporary pipeline init to get the config/output dir? 
    # Or just replicate the output dir logic.
    
    # Better yet, let's wrap the pipeline run.
    
    pipeline = InferencePipeline(cfg)
    output_dir = pipeline.output_dir
    
    import sys
    
    # Redirect stdout and stderr to separate log files
    stdout_log_file = output_dir / "inference_stdout.log"
    stderr_log_file = output_dir / "inference_stderr.log"
    
    pipeline.run()

if __name__ == "__main__":
    main()
