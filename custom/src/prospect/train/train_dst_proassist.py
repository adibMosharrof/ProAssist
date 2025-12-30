#!/usr/bin/env python3
"""
DST ProAssist Training Script

Trains DST ProAct model with:
- Binary heads for speaking/DST decisions
- Separate losses for DST and assistant responses
- SigLIP vision embeddings
- LoRA efficient fine-tuning
- Role tokens [DST] and [ASST]
"""

import logging
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List

import torch
import hydra
from accelerate import Accelerator
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from transformers import AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from transformers.integrations import TensorBoardCallback
from transformers.trainer_callback import EarlyStoppingCallback
from peft import prepare_model_for_kbit_training
import wandb

from prospect.models.dst_proact import DSTProActLlamaConfig, DSTProActLlamaForCausalLM
from prospect.train.dst_proassist_trainer import DSTProAssistTrainer
from prospect.utils.lora_utils import (
    get_lora_config,
    apply_lora_to_model,
    print_trainable_parameters,
)
from prospect.utils.logging_utils import Tee

# Direct imports to avoid circular dependency
import importlib.util

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Model configuration."""

    llm_pretrained: str = "meta-llama/Llama-3.2-3B-Instruct"
    vision_hidden_size: int = 1152  # SigLIP output dimension
    max_seq_len: int = 4096
    # Binary decision heads (always enabled)
    binary_decision_head_type: str = "linear"
    binary_loss_weight: float = 1.0
    binary_threshold: float = 0.5  # Threshold for binary classification
    # Generation head architecture
    # If True: use lm_head for speaking + dst_generation_head for DST
    # If False: use single lm_head for both
    use_separate_generation_heads: bool = False
    # Quantization settings
    use_int4_quantization: bool = False
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_use_double_quant: bool = True


@dataclass
class LoRAConfig:
    """LoRA configuration."""

    use_lora: bool = True
    lora_r: int = 128
    lora_alpha: int = 256
    lora_dropout: float = 0.05
    lora_target_modules: list = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class DataConfig:
    """Data configuration."""

    data_dir: str = (
        "custom/outputs/dst_generated/sparse_format/2025-12-06/05-27-35_gpt-4o_proassist_sparse"
    )
    dataset_name: str = "assembly101"
    siglip_features_dir: str = None  # Will be set to data_dir if None
    negative_sampling_rate: float = (
        1.0  # Rate for negative frame subsampling (1.0 = all, 0.5 = 50%)
    )


class DSTProAssistTraining:
    """DST ProAssist Training class with Hydra configuration."""

    def __init__(self, cfg: DictConfig):
        """Initialize the DST ProAssist training class with configuration."""
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Suppress verbose logging from third-party libraries
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("openai").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)

        # Load dataset and collator modules
        self._load_data_modules()

        hydra_cfg = HydraConfig.get()
        self.output_dir = Path(hydra_cfg.runtime.output_dir)
        # Setup logging to files
        # self._setup_logging()

    def _setup_logging(self):
        """Setup stdout/stderr logging to files using Tee class."""
        # Get Hydra output directory

        # Redirect stdout and stderr to files using Tee class
        stdout_file = self.output_dir / "training_stdout.log"
        stderr_file = self.output_dir / "training_stderr.log"

        sys.stdout = Tee(open(stdout_file, "w"), sys.stdout)
        sys.stderr = Tee(open(stderr_file, "w"), sys.stderr)

        self.logger.info(f"✓ Output directory: {self.output_dir}")
        self.logger.info(f"✓ Logging stdout to: {stdout_file}")
        self.logger.info(f"✓ Logging stderr to: {stderr_file}")

    def _load_data_modules(self):
        """Load dataset and collator modules to avoid circular imports."""
        # Load dst_proassist_dataset
        spec = importlib.util.spec_from_file_location(
            "dst_proassist_dataset",
            Path(__file__).parent.parent / "data_sources" / "dst_proassist_dataset.py",
        )
        dst_dataset_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dst_dataset_module)
        self.DSTProAssistDataset = dst_dataset_module.DSTProAssistDataset

        # Load dst_proassist_collator
        spec = importlib.util.spec_from_file_location(
            "dst_proassist_collator",
            Path(__file__).parent.parent / "data_sources" / "dst_proassist_collator.py",
        )
        dst_collator_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(dst_collator_module)
        self.DSTProAssistCollator = dst_collator_module.DSTProAssistCollator

    def _load_model_and_tokenizer(self, model_cfg: ModelConfig, lora_cfg: LoRAConfig):
        """Load model and tokenizer with optional int4 quantization and LoRA."""
        self.logger.info(f"Loading model from {model_cfg.llm_pretrained}")

        # Create config
        config = DSTProActLlamaConfig.from_pretrained_llama(
            model_cfg.llm_pretrained,
            vision_hidden_size=model_cfg.vision_hidden_size,
            max_seq_len=model_cfg.max_seq_len,
            use_separate_generation_heads=model_cfg.use_separate_generation_heads,
            binary_decision_head_type=model_cfg.binary_decision_head_type,
            binary_loss_weight=model_cfg.binary_loss_weight,
            binary_threshold=model_cfg.binary_threshold,
        )

        # Setup quantization config if enabled
        quantization_config = None
        if model_cfg.use_int4_quantization:
            self.logger.info("🔧 Enabling 4-bit quantization (NF4)")

            # Map string dtype to torch dtype
            compute_dtype_map = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }
            compute_dtype = compute_dtype_map.get(
                model_cfg.bnb_4bit_compute_dtype, torch.bfloat16
            )

            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_quant_type=model_cfg.bnb_4bit_quant_type,
                bnb_4bit_use_double_quant=model_cfg.bnb_4bit_use_double_quant,
                llm_int8_enable_fp32_cpu_offload=True,  # Allow CPU offload for non-quantized modules
            )
            self.logger.info(f"  ├─ Compute dtype: {model_cfg.bnb_4bit_compute_dtype}")
            self.logger.info(f"  ├─ Quant type: {model_cfg.bnb_4bit_quant_type}")
            self.logger.info(
                f"  └─ Double quant: {model_cfg.bnb_4bit_use_double_quant}"
            )

        # Load model with optional quantization
        load_kwargs = {
            "config": config,
        }

        # Get accelerator for device management
        from accelerate import Accelerator

        accelerator = Accelerator()

        if quantization_config is not None:
            load_kwargs["quantization_config"] = quantization_config
            load_kwargs["torch_dtype"] = (
                torch.bfloat16
            )  # Force BF16 for non-quantized layers
            # Map entire model to the current process's device
            load_kwargs["device_map"] = {"": accelerator.device}
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = {"": accelerator.device}

        self.model = DSTProActLlamaForCausalLM.from_pretrained(
            model_cfg.llm_pretrained, **load_kwargs
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_cfg.llm_pretrained)

        # Add special tokens
        tokens_to_add = []
        if "<image>" not in self.tokenizer.get_vocab():
            tokens_to_add.append("<image>")
        if "[DST]" not in self.tokenizer.get_vocab():
            tokens_to_add.append("[DST]")
        if "[ASST]" not in self.tokenizer.get_vocab():
            tokens_to_add.append("[ASST]")

        if tokens_to_add:
            self.tokenizer.add_tokens(tokens_to_add, special_tokens=True)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.logger.info(f"✓ Added tokens: {tokens_to_add}")

        # Store token IDs in config
        config.img_token_id = self.tokenizer.convert_tokens_to_ids("<image>")
        config.dst_gen_token_id = self.tokenizer.convert_tokens_to_ids("[DST]")
        config.asst_gen_token_id = self.tokenizer.convert_tokens_to_ids("[ASST]")
        self.model.config.img_token_id = config.img_token_id
        self.model.config.dst_gen_token_id = config.dst_gen_token_id
        self.model.config.asst_gen_token_id = config.asst_gen_token_id

        # Initialize multimodal modules AFTER tokenizer resize so generation heads (if enabled)
        # use the final vocab size.
        self.model.init_multimodal_modules()
        self.logger.info("✓ Initialized multimodal modules")

        # Set pad token
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        # Prepare for kbit training if enabled (Critical for QLoRA stability)
        if model_cfg.use_int4_quantization:
            self.model = prepare_model_for_kbit_training(
                self.model,
                use_gradient_checkpointing=self.cfg.training.get(
                    "gradient_checkpointing", False
                ),
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
            self.logger.info("✓ Prepared model for k-bit training")

        # Apply LoRA if enabled
        if lora_cfg.use_lora:
            self.logger.info("Applying LoRA to model...")
            # Convert ListConfig to list to avoid JSON serialization issues
            target_modules = list(lora_cfg.lora_target_modules)

            # Build modules_to_save list
            modules_to_save = [
                "mm_projector",
                "speaking_decision_head",
                "dst_update_head",
            ]

            # lm_head is always saved (used for speaking in both modes)
            modules_to_save.append("lm_head")

            # Add dst_generation_head only when separate heads are enabled
            use_separate_heads = getattr(
                self.model.config, "use_separate_generation_heads", False
            )
            if use_separate_heads:
                modules_to_save.append("dst_generation_head")
                self.logger.info(
                    "✓ Using lm_head for speaking, dst_generation_head for DST"
                )
            else:
                self.logger.info("✓ Using single lm_head for both speaking and DST")

            self.logger.info(f"  modules_to_save: {modules_to_save}")

            lora_config = get_lora_config(
                lora_r=lora_cfg.lora_r,
                lora_alpha=lora_cfg.lora_alpha,
                lora_dropout=lora_cfg.lora_dropout,
                target_modules=target_modules,
                modules_to_save=modules_to_save,
            )
            self.model = apply_lora_to_model(self.model, lora_config)

            # Verify trainable parameters
            self.logger.info("✓ LoRA applied successfully")
            print_trainable_parameters(self.model)
        else:
            # Freeze base model, train only multimodal modules
            self.logger.info(
                "🔒 Freezing base LLM, training only multimodal modules..."
            )
            self.model.requires_grad_(False)

            # Enable gradients for custom modules
            if (
                hasattr(self.model, "mm_projector")
                and self.model.mm_projector is not None
            ):
                self.model.mm_projector.requires_grad_(True)
                self.logger.info("  ├─ mm_projector: trainable")

            if (
                hasattr(self.model, "speaking_decision_head")
                and self.model.speaking_decision_head is not None
            ):
                self.model.speaking_decision_head.requires_grad_(True)
                self.logger.info("  ├─ speaking_decision_head: trainable")

            if (
                hasattr(self.model, "dst_update_head")
                and self.model.dst_update_head is not None
            ):
                self.model.dst_update_head.requires_grad_(True)
                self.logger.info("  └─ dst_update_head: trainable")

            print_trainable_parameters(self.model)

    def _setup_datasets(self, data_cfg: DataConfig):
        """Setup train/val/test datasets."""
        data_dir = Path(data_cfg.data_dir)
        dataset_dir = data_dir / data_cfg.dataset_name

        self.datasets = {}
        for split in ["train", "val", "test"]:
            json_path = dataset_dir / f"{split}.json"
            if json_path.exists():
                self.logger.info(f"Loading {split} dataset from {json_path}")
                self.datasets[split] = self.DSTProAssistDataset(
                    data_path=str(json_path),
                    dataset_name=data_cfg.dataset_name,
                    siglip_features_dir=data_cfg.siglip_features_dir,
                )
                self.logger.info(
                    f"✓ Loaded {len(self.datasets[split])} {split} samples"
                )
            else:
                self.logger.warning(f"⚠ {split} dataset not found: {json_path}")

    def _setup_data_collator(self, data_cfg: DataConfig, model_cfg: ModelConfig):
        """Setup data collator."""
        self.data_collator = self.DSTProAssistCollator(
            tokenizer=self.tokenizer,
            max_seq_len=model_cfg.max_seq_len,
            siglip_features_dir=Path(data_cfg.siglip_features_dir),
            negative_sampling_rate=data_cfg.negative_sampling_rate,
        )
        self.logger.info("✓ Created collator")

    def _create_training_args(self):
        """Create training arguments."""
        # Use Hydra output directory + checkpoints subdirectory
        checkpoint_dir = self.output_dir / self.cfg.training.output_dir

        # Get learning rate scheduler settings
        lr_scheduler_type = self.cfg.training.get("lr_scheduler_type", "linear")
        lr_scheduler_kwargs = self.cfg.training.get("lr_scheduler_kwargs", {})
        weight_decay = self.cfg.training.get("weight_decay", 0.0)

        return TrainingArguments(
            output_dir=str(checkpoint_dir),
            num_train_epochs=self.cfg.training.num_epochs,
            per_device_train_batch_size=self.cfg.training.batch_size,
            per_device_eval_batch_size=self.cfg.training.eval_batch_size,
            gradient_accumulation_steps=self.cfg.training.gradient_accumulation_steps,
            learning_rate=self.cfg.training.learning_rate,
            warmup_steps=self.cfg.training.warmup_steps,
            logging_steps=self.cfg.training.logging_steps,
            save_steps=self.cfg.training.save_steps,
            eval_steps=self.cfg.training.eval_steps,
            eval_strategy="steps" if "val" in self.datasets else "no",
            save_strategy="steps",
            save_total_limit=self.cfg.training.save_total_limit,
            load_best_model_at_end=False,
            metric_for_best_model="eval_loss" if "val" in self.datasets else None,
            bf16=self.cfg.training.bf16,
            fp16=self.cfg.training.fp16,
            dataloader_num_workers=self.cfg.training.num_workers,
            remove_unused_columns=False,
            report_to=self.cfg.training.get("report_to", []),
            disable_tqdm=True,  # Disable progress bar for clean dictionary-style logging
            greater_is_better=False,  # eval_loss: lower is better
            weight_decay=weight_decay,
            lr_scheduler_type=lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
        )

    def _create_trainer(self):
        """Create trainer."""
        training_args = self._create_training_args()

        trainer = DSTProAssistTrainer(
            model=self.model,
            args=training_args,
            train_dataset=self.datasets.get("train"),
            eval_dataset=self.datasets.get("val"),
            data_collator=self.data_collator,
            processor=self.tokenizer,
        )

        # Add early stopping callback
        early_stop_callback = EarlyStoppingCallback(
            early_stopping_patience=self.cfg.training.get("early_stopping_patience", 3),
            early_stopping_threshold=0.0,
        )
        trainer.add_callback(early_stop_callback)
        self.logger.info(
            f"✓ Early stopping enabled: patience={self.cfg.training.get('early_stopping_patience', 3)}, "
            f"metric=eval_loss"
        )

        self.logger.info("✓ Created trainer")
        return trainer

    def run(self):
        """Run the DST ProAssist training process."""
        # Initialize Accelerator for multi-GPU training
        accelerator = Accelerator()

        # Log GPU info from main process only
        if accelerator.is_main_process:
            self.logger.info("=" * 80)
            self.logger.info("DST ProAssist Training")
            self.logger.info("=" * 80)
            self.logger.info(f"🚀 Multi-GPU setup: {accelerator.num_processes} GPUs")

        # Create configs
        model_cfg = ModelConfig(**self.cfg.model)
        lora_cfg = LoRAConfig(**self.cfg.lora)
        data_cfg = DataConfig(**self.cfg.data)

        # Set siglip_features_dir if not specified
        if data_cfg.siglip_features_dir is None:
            data_cfg.siglip_features_dir = data_cfg.data_dir

        # Setup all components
        self._load_model_and_tokenizer(model_cfg, lora_cfg)
        self._setup_datasets(data_cfg)
        self._setup_data_collator(data_cfg, model_cfg)

        # Create trainer (Trainer handles Accelerate integration automatically)
        trainer = self._create_trainer()

        if accelerator.is_main_process:
            self.logger.info(f"Training samples: {len(self.datasets.get('train', []))}")
            self.logger.info(f"Validation samples: {len(self.datasets.get('val', []))}")

        # Train
        if accelerator.is_main_process:
            self.logger.info("Starting training...")
        trainer.train()

        # Save final model to checkpoint directory (only from main process)
        if accelerator.is_main_process:
            self.logger.info(f"Saving model to {trainer.args.output_dir}")
            trainer.save_model()
            self.tokenizer.save_pretrained(trainer.args.output_dir)
            self.logger.info("✅ Training complete!")

        # Synchronize all processes before exiting
        accelerator.wait_for_everyone()


# Calculate absolute config path
_script_dir = Path(__file__).parent
_config_dir = _script_dir.parent.parent.parent / "config" / "training"


@hydra.main(
    version_base=None,
    config_path=str(_config_dir),
    config_name="dst_proassist_training",
)
def main(cfg: DictConfig):
    """Main training function."""
    try:
        trainer = DSTProAssistTraining(cfg)
        trainer.run()
    finally:
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
