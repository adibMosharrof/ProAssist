import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig
import logging
import os
import csv
import sys
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer
from peft import PeftModel
from accelerate import Accelerator

from custom.src.prospect.inference.dst_evaluator import DSTEvaluator
from custom.src.prospect.metrics.dst_binary_metrics import DSTBinaryMetrics
from custom.src.prospect.metrics.dst_content_metrics import DSTContentMetrics
from custom.src.prospect.data_sources.dst_proassist_dataset import DSTProAssistDataset
from custom.src.prospect.models.dst_proact import DSTProActLlamaConfig, DSTProActLlamaForCausalLM
from custom.src.prospect.metrics.proassist_metrics import ProAssistMetrics
from custom.src.prospect.utils.logging_utils import Tee
import datasets as hf_datasets

# Register resolver for ${project_root}
if not OmegaConf.has_resolver("project_root"):
    OmegaConf.register_new_resolver("project_root", lambda: os.getcwd())

class DSTInferenceDataset(DSTProAssistDataset):
    """Extended dataset for inference with embedding loading."""
    def __init__(self, *args, siglip_features_dir=None, **kwargs):
        # Pass siglip_features_dir to parent class so it can filter during init
        kwargs['siglip_features_dir'] = siglip_features_dir
        super().__init__(*args, **kwargs)
        self.siglip_features_dir = Path(siglip_features_dir) if siglip_features_dir else None

    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        sample["embeddings"] = self._load_embeddings(sample)
        sample["conversation"] = self._build_conversation(sample)
        return sample

    def _load_embeddings(self, sample):
        if not self.siglip_features_dir:
            raise ValueError("siglip_features_dir must be provided")
        
        # The clip_id is the "id" field from the JSON
        clip_id = sample.get("id")
        if not clip_id:
            raise ValueError(f"Sample missing 'id' field: {sample.keys()}")

        dataset_name = sample.get("dataset_name", "assembly101")
        # Path to SigLIP features: {siglip_features_dir}/{dataset_name}/siglip_features/{clip_id}.arrow
        arrow_path = self.siglip_features_dir / dataset_name / "siglip_features" / f"{clip_id}.arrow"
        
        if not arrow_path.exists():
            raise FileNotFoundError(f"SigLIP features not found at {arrow_path}")

        dataset = hf_datasets.Dataset.from_file(str(arrow_path))
        # Load as float16/bfloat16 to match model
        embeddings = [torch.tensor(dataset[i]["cls"], dtype=torch.float16) for i in range(len(dataset))]
        return torch.stack(embeddings, dim=0)

    def _build_conversation(self, sample):
        """
        Extract conversation from sample for evaluation.
        
        The data now uses the conversation format (same as training):
        - Each turn has: role, content, start_frame, (optional) speaking, dst_update
        """
        # Return conversation directly if it exists in the expected format
        conversation = sample.get("conversation", [])
        
        if conversation:
            return conversation
        
        # Fallback: convert old events format if present (for backward compatibility)
        events = sample.get("events", [])
        if not events:
            return []
        
        converted_conversation = []
        for event in events:
            frame_idx = event["frame_idx"]
            
            # Assistant response (singular, can be None)
            response = event.get("response")
            if response:
                converted_conversation.append({
                    "role": "assistant", 
                    "start_frame": frame_idx,
                    "content": response
                })
            
            # DST updates
            for update in event.get("dst_updates", []):
                # Format: "ID->Transition"
                if "->" in update:
                    step_id, transition = update.split("->")
                    converted_conversation.append({
                        "role": "DST_UPDATE",
                        "start_frame": frame_idx,
                        "content": [{"id": step_id.strip(), "transition": transition.strip()}]
                    })
        return converted_conversation

class InferencePipeline:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self._setup_output_dir()
        self._setup_logging()

    def _setup_output_dir(self):
        # Use HydraConfig to get the actual output dir
        if HydraConfig.initialized():
            hydra_cfg = HydraConfig.get()
            self.output_dir = Path(hydra_cfg.runtime.output_dir)
        else:
            # Fallback for direct execution
            self.output_dir = Path("outputs") / "inference"
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
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
        print("DEBUG: Entering load_dataset")
        self.logger.info("Loading dataset...")
        
        datasets = []
        data_root = Path(self.cfg.project_root) / self.cfg.data.data_path
        
        for dataset_name in self.cfg.data.datasets:
            # Path to specific JSON file: {root}/{dataset_name}/{step_name}.json
            json_path = data_root / dataset_name / f"{self.cfg.data.step_name}.json"
            
            if not json_path.exists():
                self.logger.warning(f"Dataset file not found: {json_path}")
                continue
                
            dataset = DSTInferenceDataset(
                data_path=str(json_path),
                dataset_name=dataset_name,
                siglip_features_dir=data_root
            )
            datasets.append(dataset)
        
        if not datasets:
            raise ValueError(f"No datasets loaded from {data_root}")

        # Concatenate all datasets
        if len(datasets) == 1:
            self.dataset = datasets[0]
        else:
            from torch.utils.data import ConcatDataset
            self.dataset = ConcatDataset(datasets)
        
        self.logger.info(f"Loaded {len(self.dataset)} samples from {len(datasets)} dataset(s)")
        
        # Limit samples
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

    def load_model(self):
        """Load model and tokenizer from checkpoint with LoRA adapters."""
        self.logger.info(f"Loading model from {self.cfg.model.llm_pretrained}")
        
        # Get device from accelerate process index
        accelerator = Accelerator()
        
        # Create config
        config = DSTProActLlamaConfig.from_pretrained_llama(
            self.cfg.model.llm_pretrained,
            vision_hidden_size=self.cfg.model.vision_hidden_size,
            max_seq_len=self.cfg.inference.max_seq_len,
        )
        
        # Load base model with bfloat16 (matches training dtype)
        # Pass accelerator.device to place model on correct device
        self.model = DSTProActLlamaForCausalLM.from_pretrained(
            self.cfg.model.llm_pretrained,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=accelerator.device
        )
        
        # Initialize multimodal modules
        self.model.init_multimodal_modules()
        self.logger.info("✓ Initialized multimodal modules")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.llm_pretrained)
        
        # Add special tokens BEFORE loading LoRA (critical for vocab alignment)
        # These were added during training and need to be present for checkpoint compatibility
        tokens_to_add = ["<image>", "[DST]", "[ASST]"]
        self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
        # Resize model embeddings to match new vocab size
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.logger.info(f"✓ Added special tokens (vocab size: {len(self.tokenizer)})")
        
        # Store token IDs in config
        config.img_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        config.dst_gen_token_id = self.tokenizer.convert_tokens_to_ids("[DST]")
        config.asst_gen_token_id = self.tokenizer.convert_tokens_to_ids("[ASST]")
        self.model.config.img_token_id = config.img_token_id
        self.model.config.dst_gen_token_id = config.dst_gen_token_id
        self.model.config.asst_gen_token_id = config.asst_gen_token_id
        
        # Load LoRA adapters from checkpoint AFTER resizing embeddings
        checkpoint_path = Path(self.cfg.model.checkpoint_path)
        if checkpoint_path.exists():
            self.logger.info(f"Loading LoRA adapters from {checkpoint_path}")
            try:
                self.model = PeftModel.from_pretrained(self.model, str(checkpoint_path))
                self.logger.info("✓ LoRA adapters loaded successfully")
            except RuntimeError as e:
                self.logger.error(f"Failed to load LoRA adapters: {e}")
                self.logger.warning("Using model without LoRA adapters")
        else:
            self.logger.warning(f"Checkpoint path does not exist: {checkpoint_path}")
            self.logger.info("Using model without LoRA adapters")
        
        # Set to evaluation mode
        self.model.eval()
        self.logger.info("✓ Model loaded and set to eval mode")

    def run(self):
        self.load_dataset()
        self.load_model()
        metrics = self.initialize_metrics()
        
        evaluator = DSTEvaluator(
            model=self.model,
            tokenizer=self.tokenizer,
            dataset=self.dataset,
            metrics=metrics,
            output_dir=str(self.output_dir),
            num_gpus=self.cfg.inference.num_gpus,
            fps=self.cfg.inference.fps,
            speaking_threshold=self.cfg.inference.speaking_threshold,
            dst_threshold=self.cfg.inference.dst_threshold
        )
        
        results = evaluator.evaluate()
        self.save_metrics_to_csv(results)

@hydra.main(config_path="../../../config/inference", config_name="dst_inference", version_base=None)
def main(cfg: DictConfig):
    pipeline = InferencePipeline(cfg)
    pipeline.run()

if __name__ == "__main__":
    main()
