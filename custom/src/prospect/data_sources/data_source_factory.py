"""Factory for creating data sources"""

import logging
from typing import Any
from omegaconf import DictConfig, OmegaConf

from prospect.data_sources.proassist_video_dataset import ProAssistVideoDataset


logger = logging.getLogger(__name__)


class DataSourceFactory:
    """Factory for creating data source datasets"""
    
    @staticmethod
    def create_dataset(data_source_cfg: DictConfig) -> ProAssistVideoDataset:
        """
        Create dataset from Hydra configuration
        
        Args:
            data_source_cfg: Hydra config for data source
            
        Returns:
            Dataset instance
        """
        # Convert DictConfig to dict
        cfg_dict = OmegaConf.to_container(data_source_cfg, resolve=True)
        
        data_source_name = cfg_dict.get("name", "proassist_dst")
        logger.info(f"Creating data source: {data_source_name}")
        
        if data_source_name == "proassist_dst":
            # Remove 'name' from kwargs before passing to constructor
            cfg_dict.pop("name", None)
            return ProAssistVideoDataset(**cfg_dict)
        else:
            raise ValueError(f"Unknown data source: {data_source_name}")
