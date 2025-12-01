import json
from datetime import datetime
from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.core.hydra_config import HydraConfig
import logging
from tqdm import tqdm
import uuid

from dst_data_builder.dst_data_processor import DSTDataProcessor
from dst_data_builder.hybrid_dst.hybrid_dst_generator import (
    HybridDSTLabelGenerator,
)
from dst_data_builder.validators.training_format_validator import (
    TrainingFormatValidator,
)
from dst_data_builder.validators.flat_timestamps_validator import (
    FlatTimestampsValidator,
)

# Training modules
from dst_data_builder.training_modules import SpeakDSTGenerator
from dst_data_builder.training_modules.dst_event_grounding import DSTEventGrounding
from dst_data_builder.training_modules.conversation_splitter import ConversationSplitter
from dst_data_builder.training_modules.dataset_metadata_generator import (
    DatasetMetadataGenerator,
)
from dst_data_builder.training_modules.sequence_length_calculator import (
    SequenceLengthCalculator,
)
from dst_data_builder.training_modules.frame_integration import FrameIntegration
from dst_data_builder.training_modules.dst_state_tracker import DSTStateTracker


class SimpleDSTGenerator:
    """Simple Dialog State Tracing (DST) generator that adds DST labels to existing JSON data"""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.logger = logging.getLogger(__name__)

        # Initialize Hybrid DST generator
        generator_cfg = getattr(cfg, "generator", {})
        model_cfg = {
            "name": cfg.model.name,
            "temperature": cfg.model.temperature,
            "max_tokens": cfg.model.max_tokens,
        }
        self.dst_generator = HybridDSTLabelGenerator(
            model_name=cfg.model.name,
            temperature=cfg.model.temperature,
            max_tokens=cfg.model.max_tokens,
            max_retries=cfg.max_retries,
            generator_cfg=generator_cfg,
            model_cfg=model_cfg,
        )

        # Initialize training data creation modules first
        self._initialize_training_modules()

        # Initialize DST data processor with generators
        self.data_processor = DSTDataProcessor(
            dst_generator=self.dst_generator,
            speak_dst_generator=self.speak_dst_generator,
            logger=self.logger,
            is_multiprocessing=cfg.get("generation", {}).get(
                "enable_multiprocessing", False
            ),
            num_processes=cfg.get("generation", {}).get(
                "multiprocessing_processes", None
            ),
        )

    def _initialize_training_modules(self):
        """Initialize all training data creation modules"""
        self.logger.info("🔄 Initializing training data creation modules...")

        # Initialize all training modules
        training_config = self.cfg.get("training_creation", {})

        self.frame_integration = FrameIntegration(self.cfg)
        self.sequence_calculator = SequenceLengthCalculator(training_config)
        self.conversation_splitter = ConversationSplitter(self.cfg)
        self.dst_state_tracker = DSTStateTracker(training_config)
        self.speak_dst_generator = SpeakDSTGenerator(training_config)
        self.dst_grounding = DSTEventGrounding(training_config)
        self.metadata_generator = DatasetMetadataGenerator(training_config)

        self.training_modules = {
            "frame_integration": self.frame_integration,
            "sequence_calculator": self.sequence_calculator,
            "conversation_splitter": self.conversation_splitter,
            "dst_state_tracker": self.dst_state_tracker,
            "enhanced_speak_dst": self.speak_dst_generator,
            "dst_grounding": self.dst_grounding,
            "metadata_generator": self.metadata_generator,
        }

        # Initialize training format validators
        self.training_validators = [
            TrainingFormatValidator(),
            FlatTimestampsValidator(epsilon=0.1, enable_post_processing=True)
        ]
        self.logger.info("🔍 Training format validators initialized")

        self.logger.info("✅ All training modules initialized successfully")

    def create_training_format(
        self, enhanced_data: list, dataset_name: str, split: str
    ) -> list:
        """
        Create training data directly from enhanced DST data

        Args:
            enhanced_data: List of enhanced DST data items
            dataset_name: Name of the dataset
            split: Data split (train/val/test)

        Returns:
            List of training data samples
        """
        if not self.training_modules:
            self.logger.warning(
                "Training modules not initialized, returning enhanced data"
            )
            return enhanced_data

        training_samples = []
        validation_stats = {"valid": 0, "invalid": 0, "errors": []}
        sample_counter = 0  # Global counter for absolute uniqueness

        for video_data in enhanced_data:
            # 1. Create training conversation structure
            video_data = self.speak_dst_generator.create_training_conversation(
                video_data
            )

            # 2. Add frame information to the training conversation
            video_data = self.frame_integration.add_frame_metadata(
                video_data, dataset_name
            )

            # 3. Track DST state throughout conversation for accurate splitting
            video_data = self.dst_state_tracker.track_dst_transitions(video_data)

            # 4. Split long conversations into multiple training samples
            conversation_segments = self.conversation_splitter.split_conversations(
                video_data
            )

            # 5. Process each conversation segment
            for segment_idx, segment_data in enumerate(conversation_segments):
                # Inject correct initial DST state for this segment
                segment_data = self.dst_state_tracker.inject_initial_dst_state(
                    segment_data, segment_idx
                )

                # Add frame grounding and labels
                segment_data = self.dst_grounding.add_frames_and_labels(segment_data)

                # Calculate sequence lengths for this specific segment
                segment_data = self.sequence_calculator.add_sequence_metadata(
                    segment_data
                )

                # Add dataset metadata with proper clip_idx
                segment_data = self.metadata_generator.add_training_metadata(
                    segment_data, dataset_name, split, clip_idx=segment_idx
                )
                
                # Add unique ID based on video_uid, clip_idx, segment_idx, and sample counter
                video_uid = segment_data.get('video_uid', 'unknown')
                clip_idx = segment_data.get('clip_idx', 0)
                segment_data['id'] = str(uuid.uuid5(
                    uuid.NAMESPACE_DNS, 
                    f"{video_uid}_{clip_idx}_{segment_idx}_{sample_counter}"
                ))
                sample_counter += 1

                # Validate training format before adding to samples
                if self.training_validators:
                    is_valid = True
                    error_msgs = []

                    # Run all validators
                    for validator in self.training_validators:
                        validator_valid, validator_error = validator.validate(
                            segment_data
                        )
                        if not validator_valid:
                            is_valid = False
                            error_msgs.append(validator_error)

                    if is_valid:
                        training_samples.append(segment_data)
                        validation_stats["valid"] += 1
                        self.logger.debug(f"✅ Valid training sample {segment_idx}")
                    else:
                        validation_stats["invalid"] += 1
                        combined_error = "; ".join(error_msgs)
                        validation_stats["errors"].append(combined_error)
                        self.logger.warning(
                            f"❌ Invalid training sample {segment_idx}: {combined_error}"
                        )
                else:
                    # If validators not available, include without validation
                    training_samples.append(segment_data)

        # Log validation summary
        total_samples = validation_stats["valid"] + validation_stats["invalid"]
        if total_samples > 0:
            validation_rate = validation_stats["valid"] / total_samples
            self.logger.info(
                f"📊 Training format validation: {validation_stats['valid']}/{total_samples} valid "
                f"({validation_rate:.1%})"
            )

            if validation_stats["errors"]:
                unique_errors = list(set(validation_stats["errors"]))
                self.logger.warning(
                    f"🔍 Validation errors found: {len(unique_errors)} unique issues"
                )
                for error in unique_errors[:3]:  # Show first 3 unique errors
                    self.logger.warning(f"  • {error}")

        self.logger.info(
            f"Created {len(training_samples)} training samples from {len(enhanced_data)} enhanced items"
        )
        return training_samples

    def run(self, cfg: DictConfig) -> None:
        """Run the DST generation process with the given configuration"""

        # Get configuration
        datasets = cfg.data_source.datasets
        # splits = ["train", "val", "test"]
        splits = ["test", "val", "train"]
        # splits = ["test"]
        num_rows = cfg.data_source.num_rows

        self.logger.info("🚀 Starting Simple DST Generation")
        self.logger.info(f"📊 Datasets: {datasets}")
        self.logger.info(f"🔄 Splits: {splits}")
        self.logger.info(f"📝 Rows per dataset/split: {num_rows}")

        # Get Hydra's runtime output directory
        hydra_cfg = HydraConfig.get()
        hydra_output_dir = hydra_cfg.runtime.output_dir
        output_base_dir = Path(hydra_output_dir)

        self.logger.info("📁 Output directory: %s", output_base_dir.resolve())

        total_processed = 0
        total_failed = 0

        # Process each dataset and split combination with hierarchical progress tracking
        total_datasets = len(datasets)
        total_splits = len(splits)

        with tqdm(
            total=total_datasets,
            desc="📊 Processing datasets",
            unit="dataset",
            position=0,
        ) as dataset_pbar:
            for dataset_idx, dataset_name in enumerate(datasets):
                dataset_pbar.set_description(f"📊 Processing dataset: {dataset_name}")

                # Create dataset output directory
                dataset_output_dir = output_base_dir / dataset_name
                dataset_output_dir.mkdir(exist_ok=True, parents=True)

                # Process splits within this dataset
                with tqdm(
                    total=total_splits,
                    desc=f"🔄 {dataset_name} splits",
                    unit="split",
                    position=1,
                    leave=False,
                ) as split_pbar:
                    for split_idx, split in enumerate(splits):
                        split_pbar.set_description(
                            f"🔄 Processing {dataset_name}/{split}"
                        )
                        # Process and save intermediate enhanced data (always needed for now)
                        processed, failed = self.data_processor.process_dataset_split(
                            dataset_name, split, num_rows, dataset_output_dir
                        )
                        total_processed += processed
                        total_failed += failed

                        self.logger.info(
                            f"Creating training format for {dataset_name}/{split}"
                        )

                        # Load enhanced data
                        enhanced_file = dataset_output_dir / f"{split}.json"
                        if enhanced_file.exists():
                            with open(enhanced_file, "r") as f:
                                enhanced_data = json.load(f)

                            # Create training format
                            training_data = self.create_training_format(
                                enhanced_data, dataset_name, split
                            )

                            # Save training format
                            training_file = (
                                dataset_output_dir / f"{split}_training.json"
                            )
                            with open(training_file, "w") as f:
                                json.dump(training_data, f, indent=2)

                            self.logger.info(
                                f"✅ Created training format: {training_file}"
                            )

                            # Clean up intermediate files if not needed
                            save_intermediate = cfg.get("output", {}).get("save_intermediate", False)
                            if not save_intermediate:
                                if enhanced_file.exists():
                                    enhanced_file.unlink()
                                    self.logger.debug(f"🗑️ Removed intermediate file: {enhanced_file}")
                        else:
                            self.logger.error(f"Expected intermediate file not found: {enhanced_file}")

                        split_pbar.update(1)

                dataset_pbar.update(1)

        self.logger.info("=== Summary ===")
        self.logger.info("✅ Processed: %d", total_processed)
        self.logger.info("❌ Failed: %d", total_failed)
        self.logger.info("📁 Output directory: %s", output_base_dir)


@hydra.main(
    config_path="../../config/dst_data_generator",
    config_name="simple_dst_generator",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration"""
    logger = logging.getLogger(__name__)
    
    # Suppress verbose logging from third-party libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    logger.info("🚀 Starting Simple DST Generator with Hydra configuration...")

    generator = SimpleDSTGenerator(cfg)
    generator.run(cfg)


if __name__ == "__main__":
    main()
