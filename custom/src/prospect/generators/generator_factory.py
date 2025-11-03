"""Factory for creating generators"""

import logging
from omegaconf import DictConfig

from prospect.generators.baseline_generator import BaselineGenerator
from prospect.generators.sanity_check_generator import SanityCheckGenerator
from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset
from prospect.runners.vlm_stream_runner import VLMStreamRunner
from prospect.runners.sanity_check_runner import SanityCheckRunner


logger = logging.getLogger(__name__)


class GeneratorFactory:
    """Factory for creating evaluation generators"""

    @staticmethod
    def create_generator(
        generator_cfg: DictConfig,
        dataset: ProAssistVideoDataset,
        runner,
        output_dir: str,
        main_cfg: DictConfig,
    ):
        """
        Create generator from Hydra configuration

        Args:
            generator_cfg: Generator config from Hydra
            dataset: Dataset instance
            runner: Inference runner instance (VLMStreamRunner or SanityCheckRunner)
            output_dir: Output directory
            main_cfg: Full Hydra configuration

        Returns:
            Generator instance
        """
        generator_type = generator_cfg.type
        logger.info(f"Creating generator: {generator_type}")

        if generator_type == "baseline":
            return BaselineGenerator(
                dataset=dataset,
                runner=runner,
                output_dir=output_dir,
                cfg=main_cfg,
            )

        elif generator_type == "sanity_check":
            return SanityCheckGenerator(
                dataset=dataset,
                runner=runner,
                output_dir=output_dir,
                cfg=main_cfg,
            )

        else:
            raise ValueError(f"Unknown generator type: {generator_type}")
