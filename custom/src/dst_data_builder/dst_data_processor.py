"""
DSTDataProcessor - Handles data manipulation and processing logic for DST generation

This class encapsulates all data processing responsibilities, allowing SimpleDSTGenerator
to focus on orchestration while maintaining separation of concerns.
"""

import json
import logging
import os
import gc
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from dst_data_builder.gpt_generators.base_gpt_generator import BaseGPTGenerator
from dst_data_builder.training_modules.speak_dst_generator import SpeakDSTGenerator


class DSTDataProcessor:
    """
    Handles data manipulation and DST processing logic

    Responsibilities:
    - Load and limit input JSON data
    - Process videos into individual conversation training examples
    - Apply SPEAK/DST transformation (if enabled)
    - Save processed data to output files
    """

    def __init__(
        self,
        dst_generator: BaseGPTGenerator,
        speak_dst_generator: Optional[SpeakDSTGenerator] = None,
        logger: Optional[logging.Logger] = None,
        is_multiprocessing: bool = False,
        num_processes: Optional[int] = None,
    ):
        self.dst_generator = dst_generator
        self.speak_dst_generator = speak_dst_generator
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.is_multiprocessing = is_multiprocessing
        self.num_processes = num_processes

    def process_single_video(
        self, video_data: dict, video_index: int, dataset_name: str
    ) -> tuple[list, int]:
        """
        Process a single video (for parallelization)
        Returns: (processed_conversations, failed_count)
        """
        try:
            # Step 1: Ensure models are loaded for this worker process
            self.dst_generator._ensure_models_loaded()

            # Step 2: Generate DST labels for this video
            dst_labels = self._generate_dst_labels(
                video_data, video_index, dataset_name
            )

            # Step 3: Extract conversations and process them
            conversations = video_data.get("conversations", [])
            if not conversations:
                self.logger.warning(f"⚠️ No conversations found for video {video_index}")
                return [], 0

            processed_conversations = []

            for conv_idx, conversation in enumerate(conversations):
                conversation_example = self._create_conversation_example(
                    video_data, conversation, dst_labels, video_index, conv_idx
                )

                # Apply SPEAK/DST transformation if enabled
                if self.speak_dst_generator:
                    conversation_example = self._apply_speak_dst_transformation(
                        conversation_example
                    )

                processed_conversations.append(conversation_example)

            return processed_conversations, 0

        except Exception as e:
            self.logger.error(
                f"❌ Failed to process video {video_index} ({dataset_name}): {e}"
            )
            # Return empty list and count as 1 failure
            return [], 1

    def process_dataset_split(
        self, dataset_name: str, split: str, num_rows: int, output_dir: Path, suffix: str = ""
    ) -> Tuple[int, int]:
        """
        Process a specific dataset and split combination

        Returns:
            Tuple of (processed_conversations_count, failed_videos_count)
        """

        # Input file path
        input_file = Path(
            f"data/proassist/processed_data/{dataset_name}/generated_dialogs/{split}{suffix}.json"
        )

        if not input_file.exists():
            self.logger.warning(f"Input file not found: {input_file}")
            return 0, 0

        # Output file
        output_file = output_dir / f"{split}.json"

        self.logger.info(
            f"📁 Processing {dataset_name}/{split}: {input_file} -> {output_file}"
        )

        # Load and limit data
        limited_data = self._load_and_limit_data(input_file, num_rows)

        if not limited_data:
            self.logger.warning(f"No data loaded from {input_file}")
            return 0, 0

        # Create video index pairs for processing
        video_data_pairs = [
            (video_data, video_index, dataset_name)
            for video_index, video_data in enumerate(limited_data)
        ]

        if self.is_multiprocessing:
            # Get generator configs for worker processes
            dst_config = self._extract_dst_generator_config()
            speak_config = (
                self._extract_speak_dst_generator_config()
                if self.speak_dst_generator
                else None
            )

            # Extend video_data_pairs with configs for each worker call
            extended_data_pairs = [
                (video_data, video_index, dataset_name, dst_config, speak_config)
                for video_data, video_index, dataset_name in video_data_pairs
            ]

            # Parallel processing using ProcessPoolExecutor
            self.logger.info(
                f"🚀 Starting parallel processing with {len(limited_data)} videos using {self.num_processes} processes"
            )

            with tqdm(
                total=len(limited_data),
                desc=f"🎬 {dataset_name}/{split} videos",
                unit="video",
            ) as video_pbar:

                with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
                    results = []
                    for result in executor.map(
                        _unpack_and_process_video_worker, extended_data_pairs
                    ):
                        results.append(result)
                        video_pbar.update(1)

            # Aggregate results
            all_processed_conversations = []
            total_failed = 0

            for processed_conversations, failed_count in results:
                all_processed_conversations.extend(processed_conversations)
                total_failed += failed_count

        else:
            # Sequential processing
            all_processed_conversations = []
            total_failed = 0

            with tqdm(
                total=len(limited_data),
                desc=f"🎬 {dataset_name}/{split} videos",
                unit="video",
            ) as video_pbar:

                for video_data, video_index, dataset_name in video_data_pairs:
                    processed_conversations, failed_count = self.process_single_video(
                        video_data, video_index, dataset_name
                    )
                    all_processed_conversations.extend(processed_conversations)
                    total_failed += failed_count
                    video_pbar.update(1)

        # Save processed data - each conversation is a separate training example
        self._save_processed_data(all_processed_conversations, output_file)

        return len(all_processed_conversations), total_failed

    def _load_and_limit_data(
        self, input_file: Path, num_rows: int
    ) -> List[Dict[str, Any]]:
        """Load JSON data and limit to specified number of rows"""

        with open(input_file, "r") as f:
            data = json.load(f)

        # Handle num_rows = -1 as "process all rows"
        if num_rows == -1:
            limited_data = data
            self.logger.info(
                f"📊 Processing ALL {len(limited_data)} rows from {len(data)} total"
            )
        else:
            limited_data = data[:num_rows]
            self.logger.info(
                f"📊 Limited to {len(limited_data)} rows from {len(data)} total"
            )

        return limited_data

    def _generate_dst_labels(
        self, video_data: dict, video_index: int, dataset_name: str
    ) -> dict:
        """Generate DST labels for a single video's data"""

        # Extract needed data from the video
        inferred_knowledge = video_data["inferred_knowledge"]

        # Use the raw procedural annotations from all_step_descriptions field
        all_step_descriptions = video_data["parsed_video_anns"]["all_step_descriptions"]

        # Prepare input data for the generator
        input_data = {
            "inferred_knowledge": inferred_knowledge,
            "all_step_descriptions": all_step_descriptions,
        }

        # Generate DST labels using the deterministic alignment method
        return self.dst_generator._generate_dst_from_input_data(input_data)

    def _create_conversation_example(
        self,
        video_data: dict,
        conversation: dict,
        dst_labels: dict,
        video_index: int,
        conv_idx: int,
    ) -> dict:
        """Create a single conversation training example"""

        # Get the conversation turns (already have progress fields from ProAssist)
        conversation_turns = conversation.get("conversation", [])

        # Create a separate training example for each conversation
        conversation_example = {
            "video_uid": video_data.get("video_uid", ""),
            "inferred_goal": video_data.get("inferred_goal", ""),
            "inferred_knowledge": video_data.get("inferred_knowledge", ""),
            "video_labels": video_data.get("video_labels", {}),
            "conversation": conversation_turns,
            "dst": dst_labels,  # Add DST labels to each conversation
            "metadata": {
                "original_video_index": video_index,
                "conversation_index": conv_idx,
                "user_type": conversation.get("user_type", f"conversation_{conv_idx}"),
            },
        }

        return conversation_example

    def _apply_speak_dst_transformation(self, conversation_example: dict) -> dict:
        """Apply SPEAK/DST transformation if the generator is available"""

        if not self.speak_dst_generator:
            return conversation_example

        try:
            # Transform for SPEAK/DST training format
            # Create a temporary video_data structure for the training generator
            video_data = {
                "conversation": conversation_example["conversation"],
                "dst": conversation_example["dst"],
                "metadata": conversation_example["metadata"],
            }

            # Apply training conversation transformation
            video_data = self.speak_dst_generator.create_training_conversation(
                video_data
            )

            # Update the conversation example with enhanced format
            conversation_example["conversation"] = video_data["conversation"]
            conversation_example["metadata"]["user_type"] += "_dst_enhanced"

            return conversation_example

        except Exception as e:
            self.logger.warning(
                f"Failed to apply SPEAK/DST transformation: {e}. Using original format."
            )
            return conversation_example

    def _save_processed_data(
        self, processed_conversations: List[dict], output_file: Path
    ) -> None:
        """Save processed conversations to output file"""

        with open(output_file, "w") as f:
            json.dump(processed_conversations, f, indent=2)

        self.logger.debug(
            f"💾 Saved {len(processed_conversations)} conversations to {output_file}"
        )

    def _extract_dst_generator_config(self) -> Dict[str, Any]:
        """Extract config needed to recreate dst generator for worker processes"""
        return {
            "generator_type": getattr(
                self.dst_generator, "generator_type", "proassist_label"
            ),
            "model_name": getattr(self.dst_generator, "model_name", "gpt-4o"),
            "temperature": getattr(self.dst_generator, "temperature", 0.1),
            "max_tokens": getattr(self.dst_generator, "max_tokens", 4000),
            "max_retries": getattr(self.dst_generator, "max_retries", 1),
            "generator_cfg": getattr(self.dst_generator, "generator_cfg", {}),
        }

    def _extract_speak_dst_generator_config(self) -> Optional[Dict[str, Any]]:
        """Extract config needed to recreate speak dst generator for worker processes"""
        if not self.speak_dst_generator:
            return None

        return {"cfg": self.speak_dst_generator.cfg}


def _process_video_worker(
    video_data: dict,
    video_index: int,
    dataset_name: str,
    dst_generator_config: Dict[str, Any],
    speak_dst_config: Optional[Dict[str, Any]] = None,
) -> tuple[list, int]:
    """
    Module-level function for ProcessPoolExecutor - processes single video
    Recreates generators in each worker process to avoid pickle issues
    """
    import logging
    import torch

    logger = logging.getLogger("DSTDataProcessor")

    try:
        # Recreate generators from config (instead of passing instance)
        from dst_data_builder.gpt_generators.gpt_generator_factory import (
            GPTGeneratorFactory,
        )

        dst_generator = GPTGeneratorFactory.create_generator(**dst_generator_config)

        speak_dst_generator = None
        if speak_dst_config:
            speak_dst_generator = SpeakDSTGenerator(**speak_dst_config)

        # Create processor instance locally
        processor = DSTDataProcessor(dst_generator, speak_dst_generator, logger)

        # Load models in worker
        processor.dst_generator._ensure_models_loaded()

        result = processor.process_single_video(video_data, video_index, dataset_name)

        # Memory cleanup after processing each video
        _cleanup_worker_memory(processor)

        return result

    except Exception as e:
        logger.error(f"❌ Error processing video {video_index}: {e}")
        # Ensure cleanup even on error
        _cleanup_worker_memory(processor)
        # Return empty results and count as failure instead of raising
        return [], 1


def _cleanup_worker_memory(processor: DSTDataProcessor):
    """Clean up GPU memory and objects after processing a video"""
    import torch

    # Clear GPU cache
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()  # Ensure all operations complete
        torch.cuda.empty_cache()

    # Clear any references to generators to allow garbage collection
    if hasattr(processor, "dst_generator"):
        # Explicitly delete model objects
        if (
            hasattr(processor.dst_generator, "encoder")
            and processor.dst_generator.encoder is not None
        ):
            del processor.dst_generator.encoder
        if (
            hasattr(processor.dst_generator, "nli_model")
            and processor.dst_generator.nli_model is not None
        ):
            del processor.dst_generator.nli_model

        # Clear cached data
        if hasattr(processor.dst_generator, "_audit_log"):
            processor.dst_generator._audit_log = None
        if hasattr(processor.dst_generator, "_step_span_confidences"):
            processor.dst_generator._step_span_confidences = None

        processor.dst_generator.encoder = None
        processor.dst_generator.nli_model = None

    if hasattr(processor, "speak_dst_generator"):
        processor.speak_dst_generator = None

    # Force garbage collection
    gc.collect()

    # Clear any large objects
    if hasattr(processor, "logger"):
        # Remove handler references to prevent memory retention
        processor.logger.handlers.clear()

    return processor


def _unpack_and_process_video_worker(args: tuple) -> tuple[list, int]:
    """
    Wrapper function to unpack tuple arguments for ProcessPoolExecutor
    """
    return _process_video_worker(*args)
