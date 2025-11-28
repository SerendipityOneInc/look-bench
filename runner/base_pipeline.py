"""
Base Pipeline class for LookBench
All pipelines should inherit from this class
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from utils.logging import get_logger

logger = get_logger(__name__)


class BasePipeline(ABC):
    """Base class for all pipelines in LookBench"""

    def __init__(self, config_manager, model_manager, data_manager):
        """
        Initialize pipeline

        Args:
            config_manager: Configuration manager instance
            model_manager: Model manager instance
            data_manager: Data manager instance
        """
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_manager = data_manager

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the pipeline

        Args:
            **kwargs: Pipeline-specific arguments

        Returns:
            Dictionary containing pipeline results
        """
        pass

    @abstractmethod
    def get_pipeline_name(self) -> str:
        """Get the name of this pipeline"""
        pass

    def validate_inputs(self, **kwargs) -> bool:
        """
        Validate pipeline inputs

        Args:
            **kwargs: Input arguments to validate

        Returns:
            True if inputs are valid, False otherwise
        """
        return True

