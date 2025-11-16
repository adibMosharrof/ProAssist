#!/usr/bin/env python3
"""
Simple DST Training Script using existing DSTDataModule with Context Strategy Integration

Following the pattern of simple_dst_generator.py - clean, simple, direct config access.
Extends to support context strategies like summarize_with_dst.
"""

import json
import asyncio
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    AutoConfig,
    get_linear_schedule_with_warmup,
)
from transformers.trainer_utils import get_last_checkpoint

# Import existing DST components from dst_data_builder
sys.path.append(str(project_root / "custom" / "src"))

# Import new DST training data module
from prospect.data_sources.dst_training_datamodule import DSTTrainingDataModule

# Import PROSPECT context strategy components
from prospect.context_strategies.context_strategy_factory import ContextStrategyFactory
from prospect.context_strategies import BaseContextStrategy
from prospect.utils.chat_formatter import ChatFormatter
from prospect.timeline_trace.timeline_trace import BaseTrace

# Import PROSPECT components
from prospect.models.dst_smolvlm_with_strategies import DSTSmolVLMWithStrategies
from prospect.train.dst_custom_trainer import DSTCustomTrainer

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SimpleDSTTrainer:
    """Simple DST trainer following simple_dst_generator.py pattern with context strategy support"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.processor = None
        self.chat_formatter = None
        self.context_strategy = None
        self.trace = None

        # Setup context strategy (following VLM stream runner pattern)
        self._setup_context_strategy()

        # Setup model using direct config access (like simple_dst_generator.py)
        self._setup_model()

    def _setup_context_strategy(self):
        """Initialize context strategy for DST training"""
        # Context strategy for KV cache management (following VLM stream runner pattern)
        context_strategy_type = getattr(self.cfg, "context_strategy_type", "none")
        if context_strategy_type and context_strategy_type != "none":
            strategy_config = getattr(self.cfg, "context_strategy_config", {}) or {}
            self.context_strategy = ContextStrategyFactory.create_strategy(
                strategy_type=context_strategy_type,
                max_seq_len=getattr(self.cfg, "max_seq_len", 4096),
                reserved_seq_len=getattr(self.cfg, "reserved_seq_len", 128),
                **strategy_config,
            )
            self.logger.info(f"Context strategy enabled: {self.context_strategy.name}")
        else:
            self.logger.info("No context strategy (stateless processing)")

    def _setup_model(self):
        """Initialize DST-enhanced SmolVLM model using direct config access"""
        self.logger.info(f"Setting up DST SmolVLM model: {self.cfg.model.name}")

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.model.name,
            trust_remote_code=True,
        )

        # Initialize chat formatter (following VLM stream runner pattern)
        self.chat_formatter = ChatFormatter(self.processor.tokenizer)
        self.logger.info("✅ Chat formatter initialized")

        # Create base model config
        config = AutoConfig.from_pretrained(self.cfg.model.name)

        config.num_dst_states = self.cfg.model.num_dst_states
        config.dst_update_loss_weight = self.cfg.model.dst_update_loss_weight
        config.dst_state_loss_weight = self.cfg.model.dst_state_loss_weight
        config.speaking_loss_weight = self.cfg.model.speaking_loss_weight
        config.hidden_size = self.cfg.model.hidden_size
        config.max_seq_len = self.cfg.model.max_seq_len
        # Create model - try without torch_dtype first
        self.model = DSTSmolVLMWithStrategies.from_pretrained(
            self.cfg.model.name,
            config=config,
            trust_remote_code=True,
        )

        self.logger.info("DST SmolVLM model loaded successfully")
        self.logger.info(
            f"   Context strategy: {self.context_strategy.name if self.context_strategy else 'none'}"
        )

    def run(self, cfg: DictConfig) -> None:
        """Run the DST training process"""

        self.logger.info("🚀 Starting Simple DST Training")
        self.logger.info(f"📊 Model: {cfg.model.name}")
        self.logger.info(f"📁 Data source: {cfg.data_source.name}")
        self.logger.info(f"📋 DST config: {cfg.model.num_dst_states} states")
        self.logger.info(f"📊 Training epochs: {cfg.training.num_epochs}")

        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        output_base_dir = Path(hydra_output_dir)

        self.logger.info("📁 Output directory: %s", output_base_dir.resolve())

        # Load datasets using existing DSTDataModule
        train_dataset, eval_dataset, data_module = self._load_datasets()

        # Create trainer
        model_save_path = str(output_base_dir / "checkpoints")
        trainer = self._create_trainer(
            model_save_path, train_dataset, eval_dataset, data_module
        )

        # Resume from checkpoint if available
        last_checkpoint = get_last_checkpoint(trainer.args.output_dir)
        if last_checkpoint:
            self.logger.info(f"Resuming from checkpoint: {last_checkpoint}")
            trainer.train(resume_from_checkpoint=last_checkpoint)
        else:
            trainer.train()

        # Save final model
        trainer.save_model()
        self.processor.save_pretrained(trainer.args.output_dir)

        self.logger.info("🎉 DST Training completed successfully!")

    def _load_datasets(self):
        """Load training and evaluation datasets using new DSTTrainingDataModule"""
        self.logger.info("Loading datasets using DSTTrainingDataModule...")

        # Use the new DSTTrainingDataModule for stateless training
        data_module = DSTTrainingDataModule(
            dst_data_path=self.cfg.data_source.dst_data_path,
            raw_data_path=self.cfg.data_source.raw_data_path,
            datasets=self.cfg.data_source.datasets,
            fps=self.cfg.data_source.fps,
            max_seq_len=self.cfg.data_source.max_seq_len,
            reserved_seq_len=self.cfg.data_source.reserved_seq_len,
            video_ids=None,
        )

        # Setup the data module
        data_module.setup()

        # Get train and val datasets
        train_dataset = data_module.get_train_dataset()
        eval_dataset = data_module.get_val_dataset()

        if train_dataset is None or eval_dataset is None:
            raise ValueError(
                "Failed to load train/eval datasets from DSTTrainingDataModule"
            )

        self.logger.info(
            f"Datasets loaded: train={len(train_dataset)}, eval={len(eval_dataset)}"
        )
        return train_dataset, eval_dataset, data_module

    def _create_trainer(
        self, output_dir: str, train_dataset, eval_dataset, data_module
    ):
        """Create HF Trainer with DST-specific loss computation"""

        # Create training arguments from configuration
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=self.cfg.training.num_epochs,
            per_device_train_batch_size=1,  # Will use gradient accumulation
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=self.cfg.model.gradient_accumulation_steps,
            learning_rate=self.cfg.model.learning_rate,
            weight_decay=0.01,
            warmup_steps=self.cfg.model.warmup_steps,
            max_grad_norm=1.0,
            # Evaluation and logging
            eval_strategy="steps",  # Changed from evaluation_strategy
            eval_steps=self.cfg.training.eval_steps,
            logging_steps=self.cfg.training.logging_steps,
            save_steps=self.cfg.training.save_steps,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            # Mixed precision
            fp16=self.cfg.model.get("fp16", False),
            bf16=self.cfg.model.get("bf16", False),
            # Data loading
            dataloader_num_workers=self.cfg.model.num_workers,
            dataloader_pin_memory=True,
            dataloader_drop_last=True,
            report_to=["wandb"],
            # Other
            remove_unused_columns=False,
            run_name=self.cfg.exp_name,
        )

        # Create custom trainer with DST loss computation
        trainer = DSTCustomTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            train_config=self.cfg,
            data_collator=data_module.get_data_collator(),
        )
        trainer.processor = self.processor  # Store processor as instance variable

        return trainer


@hydra.main(
    config_path="../../../config/prospect",
    config_name="dst_training",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger.info("🚀 Starting Simple DST Training with Hydra configuration...")
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(cfg))
    run = wandb.init(
        # Set the wandb entity where your project will be logged (generally your team name).
        entity="adibm",
        # Set the wandb project where this run will be logged.
        project="prospect",
        # Track hyperparameters and run metadata.
        config={
            "learning_rate": 0.02,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": 10,
        },
    )
    # Create training pipeline (following simple_dst_generator.py pattern)
    trainer = SimpleDSTTrainer(cfg)
    trainer.run(cfg)

    logger.info("🎉 Simple DST Training completed successfully!")
    run.finish()


if __name__ == "__main__":
    main()
