#!/usr/bin/env python3
"""
DST Training Script for PROSPECT using Hydra configuration.

Implements DST-enhanced training with:
- Turn-by-turn processing
- Multi-task loss (LM + DST binary + DST generation)
- Negative frame sampling
- Precomputed embeddings from pickle files
- Multi-GPU training with Accelerate
"""

import logging
import os
import sys
from pathlib import Path
from torch.utils.data import ConcatDataset

import hydra
import torch
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoProcessor, TrainingArguments, AutoConfig
from transformers.trainer_callback import EarlyStoppingCallback

from custom.src.prospect.data_sources.dst_training_dataset import (
    DSTMultimodalChat,
    DSTTrainingDataset,
)
from custom.src.prospect.data_sources.dst_data_collator import DSTDataCollator
from custom.src.prospect.models.dst_smolvlm_with_strategies import (
    DSTSmolVLMWithStrategies,
)
from custom.src.prospect.train.dst_custom_trainer import DSTCustomTrainer

logger = logging.getLogger(__name__)


class DSTTrainingProspect:
    """DST Training class for PROSPECT with Hydra configuration."""

    def __init__(self, cfg: DictConfig):
        """Initialize the DST training class with configuration."""
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)
        
        # Suppress verbose logging from third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

    def _load_model_and_processor(self):
        """Load processor and model."""
        self.logger.info("Loading processor...")
        # Load processor with default config or custom processor_size from config
        processor_size = self.cfg.model.get("processor_size")
        if processor_size:
            self.processor = AutoProcessor.from_pretrained(
                self.cfg.model.name,
                size={"longest_edge": processor_size}
            )
            self.logger.info(f"✓ Processor configured with longest_edge={processor_size}")
        else:
            self.processor = AutoProcessor.from_pretrained(self.cfg.model.name)
            self.logger.info(f"✓ Processor using default configuration")
        
        self.logger.info("Loading model configuration...")
        self.config = AutoConfig.from_pretrained(self.cfg.model.name)
        
        # Add vision_hidden_size from config (SigLIP output dimension)
        if hasattr(self.cfg.model, 'vision_hidden_size'):
            self.config.vision_hidden_size = self.cfg.model.vision_hidden_size
            self.logger.info(f"✓ Vision hidden size set to {self.cfg.model.vision_hidden_size}")
        
        # Add use_img_cls_token setting
        if hasattr(self.cfg.model, 'use_img_cls_token'):
            self.config.use_img_cls_token = self.cfg.model.use_img_cls_token
            self.logger.info(f"✓ Using [CLS] token strategy: {self.cfg.model.use_img_cls_token}")
        
        # Update vision config to match processor size if custom size specified
        if processor_size and hasattr(self.config, 'vision_config'):
            self.config.vision_config.max_image_size = {'longest_edge': processor_size}
            self.logger.info(f"✓ Vision config updated to longest_edge={processor_size}")
        
        # Add 4 loss weight configurations
        self.config.speaking_gen_weight = self.cfg.model.speaking_gen_weight
        self.config.speaking_binary_weight = self.cfg.model.speaking_binary_weight
        self.config.dst_gen_weight = self.cfg.model.dst_gen_weight
        self.config.dst_binary_weight = self.cfg.model.dst_binary_weight
        
        self.logger.info("Loading model...")
        self.model = DSTSmolVLMWithStrategies.from_pretrained(
            self.cfg.model.name,
            config=self.config,
            torch_dtype=torch.bfloat16 if self.cfg.model.bf16 else torch.float32,
        )

        # Freeze everything except trainable heads (following ProAssist "frozen" mode strategy)
        # Since vision embeddings are pre-computed, we only train:
        # - vision_projector: projects vision embeddings to LLM space
        # - speaking_decision_head: binary classification for turn-taking
        # - dst_update_head: binary classification for DST updates
        self.model.requires_grad_(False)  # Freeze entire model
        
        # Unfreeze only the trainable projection and decision heads
        if hasattr(self.model, 'vision_projector'):
            self.model.vision_projector.requires_grad_(True)
        if hasattr(self.model, 'speaking_decision_head'):
            self.model.speaking_decision_head.requires_grad_(True)
        if hasattr(self.model, 'dst_update_head'):
            self.model.dst_update_head.requires_grad_(True)
        
        self.logger.info("✓ Froze LLM, unfroze projection and decision heads (frozen mode)")

        # Move frozen vision_model to CPU to save GPU memory (1.6 GB saved)
        # Since we use precomputed embeddings, the vision model is never called during training
        # This is safe because: (1) vision_encoder has requires_grad=False, (2) forward pass checks for image_embeds first
        self.model.model.vision_model.to('cpu')
        self.logger.info("✓ Moved frozen vision_model to CPU (saves ~1.6 GB GPU memory)")

    def _setup_datasets(self):
        """Setup train and validation datasets."""
        datasets_list = list(self.cfg.data_source.datasets)
        project_root = Path(self.cfg.project_root)
        
        train_datasets = []
        val_datasets = []
        
        for dataset_name in datasets_list:
            self.logger.info(f"Loading dataset: {dataset_name}")
            
           
            # Load train dataset
            train_ds = DSTTrainingDataset(
                data_path=self.cfg.data_source.data_path,
                step_name="train",
                dataset_name=dataset_name,
                max_seq_len=self.cfg.data_source.max_seq_len,
                neg_frame_sampling_rate=self.cfg.data_source.neg_frame_sampling_rate,
            )
            train_datasets.append(train_ds)
            
            # Load val dataset (same pattern as train)
            val_ds = DSTTrainingDataset(
                data_path=self.cfg.data_source.data_path,
                step_name="val",
                dataset_name=dataset_name,
                max_seq_len=self.cfg.data_source.max_seq_len,
                neg_frame_sampling_rate=1.0,  # No negative sampling for validation
            )
            val_datasets.append(val_ds)
        
        # Concatenate datasets
        self.train_dataset = ConcatDataset(train_datasets) if train_datasets else DSTTrainingDataset(
            data_path="",
            dataset_name="empty",
            max_seq_len=self.cfg.data_source.max_seq_len,
            neg_frame_sampling_rate=0,
        )
        self.val_dataset = ConcatDataset(val_datasets) if val_datasets else DSTTrainingDataset(
            data_path="",
            dataset_name="empty",
            max_seq_len=self.cfg.data_source.max_seq_len,
            neg_frame_sampling_rate=1.0,
        )

    def _setup_data_collator(self):
        """Setup chat formatter and data collator."""
        tokenizer = self.processor.tokenizer
        img_token = "<image>"

        # Add special tokens if needed
        if img_token not in tokenizer.get_vocab():
            tokenizer.add_special_tokens({"additional_special_tokens": [img_token]})
            self.model.resize_token_embeddings(len(tokenizer))
            self.logger.info(f"✓ Added special token: {img_token}")

        tokenizer.img_token_id = tokenizer.convert_tokens_to_ids(img_token)
        tokenizer.ignore_id = -100
        
        # CRITICAL: Also set img_token_id on model config so forward() can find image tokens
        self.model.config.img_token_id = tokenizer.img_token_id
        self.logger.info(f"✓ Set img_token_id on model config: {tokenizer.img_token_id}")

        # Create DST-aware chat formatter
        chat_formatter = DSTMultimodalChat(
            img_token=img_token,
            num_tokens_per_img=1,
            sep_token=None,
        )

        # Create data collator (works for both train and val batches)
        # Frame limiting is now handled in the conversation splitter during data prep
        self.data_collator = DSTDataCollator(
            tokenizer=tokenizer,
            chat_formatter=chat_formatter,
            max_seq_len=self.cfg.data_source.max_seq_len,
        )

    def _setup_training_arguments(self):
        """Configure training arguments from Hydra config."""
        return TrainingArguments(
            output_dir=self.cfg.output_dir,
            num_train_epochs=self.cfg.model.num_train_epochs,
            per_device_train_batch_size=self.cfg.model.per_device_train_batch_size,
            per_device_eval_batch_size=self.cfg.model.per_device_eval_batch_size,
            gradient_accumulation_steps=self.cfg.model.gradient_accumulation_steps,
            learning_rate=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
            warmup_steps=self.cfg.model.warmup_steps,
            bf16=self.cfg.model.bf16,
            gradient_checkpointing=self.cfg.model.gradient_checkpointing,
            eval_strategy=self.cfg.model.eval_strategy,
            eval_steps=self.cfg.model.eval_steps,
            save_steps=self.cfg.model.save_steps,
            save_total_limit=self.cfg.model.save_total_limit,
            logging_steps=self.cfg.model.logging_steps,
            report_to=self.cfg.model.report_to,
            remove_unused_columns=self.cfg.model.remove_unused_columns,
            ddp_find_unused_parameters=self.cfg.model.ddp_find_unused_parameters,
            max_steps=self.cfg.model.max_steps,
            # Early stopping parameters
            load_best_model_at_end=self.cfg.model.get("load_best_model_at_end", False),
            metric_for_best_model=self.cfg.model.get("metric_for_best_model", "eval_loss"),
            greater_is_better=self.cfg.model.get("greater_is_better", False),
        )

    def _create_trainer(self):
        """Create and return the DST custom trainer."""
        self.training_args = self._setup_training_arguments()
        
        trainer = DSTCustomTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            data_collator=self.data_collator,
            train_config=self.cfg,  # Pass config for loss weights
            processor=self.processor,
        )
        
        # Add early stopping callback if configured
        if (
            hasattr(self.cfg.model, "early_stopping_patience")
            and self.cfg.model.early_stopping_patience > 0
        ):
            early_stop_callback = EarlyStoppingCallback(
                early_stopping_patience=self.cfg.model.early_stopping_patience,
                early_stopping_threshold=0.0,
            )
            trainer.add_callback(early_stop_callback)
            self.logger.info(
                f"✓ Early stopping enabled with patience={self.cfg.model.early_stopping_patience}"
            )
        
        return trainer

    def run(self, cfg: DictConfig) -> None:
        """Run the DST training process."""
        # Initialize Accelerator for multi-GPU training
        accelerator = Accelerator()
        
        # Log GPU info from main process only
        if accelerator.is_main_process:
            self.logger.info("🚀 Starting DST Training for PROSPECT")
            self.logger.info(f"Multi-GPU setup: {accelerator.num_processes} GPUs")

        # Setup all components
        self._load_model_and_processor()
        self._setup_datasets()
        self._setup_data_collator()
        
        # Let Trainer handle model and dataset preparation with Accelerate
        # Create trainer
        if accelerator.is_main_process:
            self.logger.info("Creating trainer...")
        trainer = self._create_trainer()

        # Train
        if accelerator.is_main_process:
            self.logger.info("=" * 80)
            self.logger.info("Starting training...")
            self.logger.info("=" * 80)

        trainer.train()

        # Save final model (only from main process)
        if accelerator.is_main_process:
            self.logger.info("Training complete! Saving model...")
            trainer.save_model()
            self.processor.save_pretrained(self.training_args.output_dir)
            self.logger.info("✓ Model saved to: %s", self.training_args.output_dir)
        
        # Synchronize all processes before exiting
        accelerator.wait_for_everyone()


@hydra.main(version_base=None, config_path="../../../config/prospect", config_name="dst_training")
def main(cfg: DictConfig) -> None:
    """Main training function with Hydra configuration."""
    
    # Get the actual output directory that Hydra created
    hydra_cfg = HydraConfig.get()
    hydra_output_dir = hydra_cfg.runtime.output_dir
    
    # Redirect stdout and stderr to separate log files
    # stdout captures print statements and info logs
    stdout_log_file = os.path.join(hydra_output_dir, "training_stdout.log")
    # stderr captures error messages, warnings, and exceptions
    stderr_log_file = os.path.join(hydra_output_dir, "training_stderr.log")
    
    # with open(stdout_log_file, "w") as stdout_f, open(stderr_log_file, "w") as stderr_f:
    #     # Create a tee-like object that writes to both file and console
    #     class Tee:
    #         def __init__(self, file_obj, console_obj):
    #             self.file = file_obj
    #             self.console = console_obj
    #         
    #         def write(self, message):
    #             self.file.write(message)
    #             self.file.flush()
    #             self.console.write(message)
    #         
    #         def flush(self):
    #             self.file.flush()
    #             self.console.flush()
    #         
    #         def isatty(self):
    #             return self.console.isatty()
    #     
    #     # Store original stdout/stderr
    #     original_stdout = sys.stdout
    #     original_stderr = sys.stderr
    #     
    #     # Redirect stdout and stderr to separate files while still showing on console
    #     sys.stdout = Tee(stdout_f, original_stdout)
    #     sys.stderr = Tee(stderr_f, original_stderr)
    #     
    #     try:
    #         trainer = DSTTrainingProspect(cfg)
    #         trainer.run(cfg)
    #     finally:
    #         # Restore original stdout/stderr
    #         sys.stdout = original_stdout
    #         sys.stderr = original_stderr
    with open(stdout_log_file, "w") as stdout_f, open(stderr_log_file, "w") as stderr_f:
        # Create a tee-like object that writes to both file and console
        class Tee:
            def __init__(self, file_obj, console_obj):
                self.file = file_obj
                self.console = console_obj
            
            def write(self, message):
                self.file.write(message)
                self.file.flush()
                self.console.write(message)
            
            def flush(self):
                self.file.flush()
                self.console.flush()
            
            def isatty(self):
                return self.console.isatty()
        
        # Store original stdout/stderr
        original_stdout = sys.stdout
        original_stderr = sys.stderr
        
        # Redirect stdout and stderr to separate files while still showing on console
        sys.stdout = Tee(stdout_f, original_stdout)
        sys.stderr = Tee(stderr_f, original_stderr)
        
        try:
            trainer = DSTTrainingProspect(cfg)
            trainer.run(cfg)
        finally:
            # Restore original stdout/stderr
            sys.stdout = original_stdout
            sys.stderr = original_stderr


if __name__ == "__main__":
    main()
