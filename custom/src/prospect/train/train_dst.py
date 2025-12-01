"""
DST Training Script for Multi-Task SmolVLM Training

This script handles the complete training pipeline for DST-enhanced SmolVLM models,
including data loading, multi-task training, evaluation, and checkpointing.
Using Hydra configuration for clean parameter management.
"""

import os
import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Setup logging before imports
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Handle TensorFlow import issues gracefully by wrapping imports in try-catch
TRANSFORMERS_AVAILABLE = False

try:
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow warnings
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # Avoid warning
    
    # Import transformers components
    from transformers import (
        AutoProcessor,
        AutoModelForCausalLM,
        TrainingArguments,
        Trainer,
        get_linear_schedule_with_warmup,
    )
    from transformers.trainer_utils import get_last_checkpoint
    
    TRANSFORMERS_AVAILABLE = True
    logger.info("✅ Transformers imported successfully with TensorFlow")
    
except ImportError as e:
    print(f"⚠️ Warning: Could not import transformers properly: {e}")
    print("This might be due to TensorFlow/Keras version compatibility issues.")
    print("Creating dummy classes to avoid immediate errors...")
    
    # Create dummy classes to avoid immediate import errors
    class AutoProcessor:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("Transformers not available due to dependency conflicts")
    
    class AutoModelForCausalLM:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise ImportError("Transformers not available due to dependency conflicts")
    
    class TrainingArguments:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class Trainer:
        def __init__(self, *args, **kwargs):
            raise ImportError("Transformers not available due to dependency conflicts")
        
        def train(self, *args, **kwargs):
            raise ImportError("Transformers not available due to dependency conflicts")
        
        def save_model(self, *args, **kwargs):
            raise ImportError("Transformers not available due to dependency conflicts")
    
    def get_linear_schedule_with_warmup(*args, **kwargs):
        raise ImportError("Transformers not available due to dependency conflicts")
    
    def get_last_checkpoint(*args, **kwargs):
        return None
    
    # Make sure our dummy classes are in the global namespace
    import sys
    sys.modules['transformers'] = type('Module', (), {
        'AutoProcessor': AutoProcessor,
        'AutoModelForCausalLM': AutoModelForCausalLM,
        'TrainingArguments': TrainingArguments,
        'Trainer': Trainer,
        'get_linear_schedule_with_warmup': get_linear_schedule_with_warmup,
        'trainer_utils': type('Module', (), {'get_last_checkpoint': get_last_checkpoint})()
    })()
    
    # Assign to global namespace
    globals().update({
        'AutoProcessor': AutoProcessor,
        'AutoModelForCausalLM': AutoModelForCausalLM,
        'TrainingArguments': TrainingArguments,
        'Trainer': Trainer,
        'get_linear_schedule_with_warmup': get_linear_schedule_with_warmup,
        'get_last_checkpoint': get_last_checkpoint
    })

# Add the project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

from prospect.models.dst_smolvlm_with_strategies import (
    DSTSmolVLMWithStrategies,
    DSTSmolVLMConfig,
)
from prospect.data.dst_frame_loader import DSTFrameDataLoader
from prospect.data.dst_frame_dataset import DSTFrameDataset
from prospect.data.dst_frame_collator import DSTDataCollator


@dataclass
class ModelConfig:
    """Model configuration"""

    name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
    model_type: str = "dst_smolvlm_with_strategies"
    num_dst_states: int = 3
    dst_update_loss_weight: float = 1.0
    dst_state_loss_weight: float = 1.0
    speaking_loss_weight: float = 1.0
    max_new_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True


@dataclass
class TrainingConfig:
    """Training configuration"""

    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_epochs: int = 10
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    eval_steps: int = 500
    save_steps: int = 1000
    logging_steps: int = 100
    max_grad_norm: float = 1.0

    # Multi-GPU training for 24GB RTX setup
    dataloader_num_workers: int = 4
    dataloader_pin_memory: bool = True
    dataloader_drop_last: bool = True

    # Mixed precision training
    fp16: bool = True
    bf16: bool = False

    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8

    # Evaluation
    evaluation_strategy: str = "steps"
    eval_accumulation_steps: int = 1
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


@dataclass
class DataConfig:
    """Data configuration"""

    train_dataset: str = "dst_assembly101_train"
    eval_dataset: str = "dst_assembly101_val"
    max_seq_len: int = 4096
    reserved_seq_len: int = 128
    batch_size: int = 2
    num_workers: int = 4
    fps: int = 2

    # Data paths (configured at runtime)
    data_path: str = "data/proassist/processed_data/assembly101"
    dst_data_path: str = (
        "custom/outputs/dst_generated/proassist_label/2025-11-06/17-02-11_gpt-4o_proassist_50rows"
    )
    dialogue_path: str = "data/proassist/processed_data/assembly101/generated_dialogs"

    # Training videos (subset for quick testing)
    train_videos: List[str] = None
    eval_videos: List[str] = None

    def __post_init__(self):
        if self.train_videos is None:
            self.train_videos = ["9011-c03f"]  # Quick test subset
        if self.eval_videos is None:
            self.eval_videos = ["9011-c03f"]  # Quick test subset


@dataclass
class OutputConfig:
    """Output configuration"""

    output_dir: str = "./outputs/dst_training"
    run_name: str = "dst-smolvlm2-training"
    experiment_name: str = "dst-multi-task-v1"

    # Logging
    report_to: List[str] = None  # ["wandb"] if needed
    logging_dir: str = "./logs/dst_training"

    # Checkpointing
    save_total_limit: int = 3
    save_safetensors: bool = True

    def __post_init__(self):
        if self.report_to is None:
            self.report_to = []  # Disable wandb by default


@dataclass
class OptimizationConfig:
    """GPU Memory optimization for 24GB RTX"""

    gradient_checkpointing: bool = True
    max_memory_MB: int = 24000
    low_cpu_mem_usage: bool = True

    # Batch size adjustments
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 8  # Effective batch size: 8

    # Memory efficiency
    dataloader_pin_memory: bool = True
    remove_unused_columns: bool = False


@dataclass
class DSTTrainingConfig:
    """DST Training Configuration - main config that combines all sub-configs"""

    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    data: DataConfig = DataConfig()
    output: OutputConfig = OutputConfig()
    optimization: OptimizationConfig = OptimizationConfig()


class DSTTrainer:
    """Main DST training class with Hydra configuration"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        self.trainer = None

    def setup_model_and_processor(self):
        """Setup model and processor from configuration"""
        self.logger.info(f"Loading model: {self.cfg.model.name}")

        # Load processor with N=1 config to reduce token count
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.name,
            trust_remote_code=True,
            size={"longest_edge": 384}  # Force N=1 configuration (1 patch → 81 tokens per image)
        )

        # Create model config
        model_config = {
            "hidden_size": 2816,  # SmolVLM2 hidden size
            "max_seq_len": self.cfg.data.max_seq_len,
        }

        dst_config = DSTSmolVLMConfig(
            model_config,
            num_dst_states=self.cfg.model.num_dst_states,
            dst_update_loss_weight=self.cfg.model.dst_update_loss_weight,
            dst_state_loss_weight=self.cfg.model.dst_state_loss_weight,
            speaking_loss_weight=self.cfg.model.speaking_loss_weight,
        )

        # Create model
        self.model = DSTSmolVLMWithStrategies.from_pretrained(
            self.cfg.model.name,
            config=dst_config,
            torch_dtype=torch.bfloat16 if self.cfg.training.bf16 else torch.float16,
            trust_remote_code=True,
        )

        self.logger.info("Model and processor loaded successfully")

    def load_datasets(self):
        """Load training and evaluation datasets"""
        # Create data loaders
        train_data_loader = DSTFrameDataLoader(
            data_path=self.cfg.data.data_path,
            dst_data_path=self.cfg.data.dst_data_path,
            max_seq_len=self.cfg.data.max_seq_len,
            num_dst_states=self.cfg.model.num_dst_states,
            fps=self.cfg.data.fps,
        )

        # Filter to training videos
        train_data_loader.video_ids = self.cfg.data.train_videos
        train_data_loader.dst_data = {
            vid: train_data_loader.dst_data.get(vid, {"tasks": {}, "dialogue": []})
            for vid in self.cfg.data.train_videos
        }

        # Create datasets
        train_dataset = DSTFrameDataset(
            data_loader=train_data_loader,
            processor=self.processor,
            max_seq_len=self.cfg.data.max_seq_len,
            reserved_seq_len=self.cfg.data.reserved_seq_len,
        )

        # Create evaluation dataset
        eval_data_loader = DSTFrameDataLoader(
            data_path=self.cfg.data.data_path,
            dst_data_path=self.cfg.data.dst_data_path,
            max_seq_len=self.cfg.data.max_seq_len,
            num_dst_states=self.cfg.model.num_dst_states,
            fps=self.cfg.data.fps,
        )

        # Filter to evaluation videos
        eval_data_loader.video_ids = self.cfg.data.eval_videos
        eval_data_loader.dst_data = {
            vid: eval_data_loader.dst_data.get(vid, {"tasks": {}, "dialogue": []})
            for vid in self.cfg.data.eval_videos
        }

        eval_dataset = DSTFrameDataset(
            data_loader=eval_data_loader,
            processor=self.processor,
            max_seq_len=self.cfg.data.max_seq_len,
            reserved_seq_len=self.cfg.data.reserved_seq_len,
        )

        self.logger.info(
            f"Loaded datasets: train={len(train_dataset)} samples, eval={len(eval_dataset)} samples"
        )

        return train_dataset, eval_dataset

    def create_training_args(self, output_dir: str) -> TrainingArguments:
        """Create training arguments from configuration"""
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.training.num_epochs,
            per_device_train_batch_size=self.cfg.optimization.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.optimization.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            weight_decay=self.cfg.training.weight_decay,
            warmup_steps=self.cfg.training.warmup_steps,
            max_grad_norm=self.cfg.training.max_grad_norm,
            # Evaluation and logging
            evaluation_strategy=self.cfg.training.evaluation_strategy,
            eval_steps=self.cfg.training.eval_steps,
            logging_steps=self.cfg.training.logging_steps,
            save_steps=self.cfg.training.save_steps,
            save_total_limit=self.cfg.output.save_total_limit,
            load_best_model_at_end=self.cfg.training.load_best_model_at_end,
            metric_for_best_model=self.cfg.training.metric_for_best_model,
            greater_is_better=self.cfg.training.greater_is_better,
            # Mixed precision
            fp16=self.cfg.training.fp16,
            bf16=self.cfg.training.bf16,
            # Data loading
            dataloader_num_workers=self.cfg.training.dataloader_num_workers,
            dataloader_pin_memory=self.cfg.training.dataloader_pin_memory,
            dataloader_drop_last=self.cfg.training.dataloader_drop_last,
            # Other
            report_to=self.cfg.output.report_to,
            remove_unused_columns=self.cfg.optimization.remove_unused_columns,
            run_name=self.cfg.output.run_name,
            logging_dir=self.cfg.output.logging_dir,
        )

        return training_args

    def create_data_collators(self):
        """Create data collators for training and evaluation"""
        train_collator = DSTDataCollator(
            max_seq_len=self.cfg.data.max_seq_len,
        )

        eval_collator = DSTDataCollator(
            max_seq_len=self.cfg.data.max_seq_len,
        )

        return train_collator, eval_collator

    def run(self, cfg: DictConfig) -> None:
        """Main training pipeline with Hydra configuration"""
        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        output_base_dir = Path(hydra_output_dir)

        self.logger.info("🚀 Starting DST Training")
        self.logger.info(f"📁 Output directory: {output_base_dir.resolve()}")
        self.logger.info(f"📊 Configuration:\n{OmegaConf.to_yaml(cfg)}")

        # Setup model and processor
        self.setup_model_and_processor()

        # Load datasets
        train_dataset, eval_dataset = self.load_datasets()

        # Create data collators
        train_collator, eval_collator = self.create_data_collators()

        # Create training arguments
        training_args = self.create_training_args(str(output_base_dir / "checkpoints"))

        # Create custom trainer
        trainer = DSTCustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=train_collator,
            train_config=cfg,
        )

        # Resume from checkpoint if available
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")

        # Start training
        self.logger.info("Starting DST training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)

        # Save final model
        self.logger.info("Training completed. Saving final model...")
        trainer.save_model()
        self.processor.save_pretrained(training_args.output_dir)

        # Save training config
        config_path = training_args.output_dir / "training_config.yaml"
        with open(config_path, "w") as f:
            f.write(OmegaConf.to_yaml(cfg))

        self.logger.info(
            f"Training completed. Results saved to: {training_args.output_dir}"
        )


class DSTCustomTrainer(Trainer):
    """Custom trainer for DST multi-task training"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Initialize focal loss criterion (using kornia)
        try:
            from kornia.losses import FocalLoss as KorniaFocalLoss

            self.focal_criterion = KorniaFocalLoss(
                alpha=0.25, gamma=2.0, reduction="mean"
            )
        except ImportError:
            self.logger = logging.getLogger(__name__)
            self.logger.warning(
                "Kornia focal loss not available, using standard cross-entropy"
            )
            self.focal_criterion = nn.CrossEntropyLoss(reduction="mean")

        # Store training config
        self.train_config = kwargs.get("train_config", {})
        self.logger = logging.getLogger(__name__)

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute multi-task loss with focal loss for class imbalance"""
        # Get model outputs
        outputs = model(**inputs)

        # Get base language modeling loss
        language_loss = 0.0

        if "logits" in outputs and "labels" in inputs:
            # Standard language modeling loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = inputs["labels"][..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss(reduction="mean")
            language_loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

        total_loss = language_loss

        # Add DST-specific losses with focal loss for class imbalance
        if inputs.get("speaking_labels") is not None:
            # Kornia FocalLoss expects probabilities, not raw logits
            speaking_probs = torch.softmax(outputs.speaking_logits.view(-1, 2), dim=-1)
            speaking_loss = self.focal_criterion(
                speaking_probs, inputs["speaking_labels"].view(-1)
            )
            total_loss += speaking_loss * self.train_config.model.speaking_loss_weight

        if inputs.get("dst_update_labels") is not None:
            dst_update_probs = torch.softmax(
                outputs.dst_update_logits.view(-1, 2), dim=-1
            )
            dst_update_loss = self.focal_criterion(
                dst_update_probs, inputs["dst_update_labels"].view(-1)
            )
            total_loss += (
                dst_update_loss * self.train_config.model.dst_update_loss_weight
            )

        if inputs.get("dst_state_labels") is not None:
            dst_state_probs = torch.softmax(
                outputs.dst_state_logits.view(
                    -1, self.train_config.model.num_dst_states
                ),
                dim=-1,
            )
            dst_state_loss = self.focal_criterion(
                dst_state_probs, inputs["dst_state_labels"].view(-1)
            )
            total_loss += dst_state_loss * self.train_config.model.dst_state_loss_weight

        # Log losses
        if (
            hasattr(self, "state")
            and self.state.global_step % self.train_config.training.logging_steps == 0
        ):
            self.logger.info(
                f"Step {self.state.global_step}: "
                f"Total Loss: {total_loss.item():.4f}, "
                f"Language Loss: {language_loss.item():.4f}, "
                f"Speaking Loss: {speaking_loss.item() if inputs.get('speaking_labels') is not None else 0:.4f}, "
                f"DST Update Loss: {dst_update_loss.item() if inputs.get('dst_update_labels') is not None else 0:.4f}, "
                f"DST State Loss: {dst_state_loss.item() if inputs.get('dst_state_labels') is not None else 0:.4f}"
            )

        return (total_loss, outputs) if return_outputs else total_loss


# Register configuration with Hydra
cs = ConfigStore.instance()
cs.store(name="dst_training", node=DSTTrainingConfig)


@hydra.main(
    config_path="../../../config/prospect/train",
    config_name="dst_training",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger.info("🚀 Starting DST Training with Hydra configuration...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))

    trainer = DSTTrainer(cfg)
    trainer.run(cfg)


if __name__ == "__main__":
    main()
