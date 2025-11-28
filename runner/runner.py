"""
Runner class for LookBench
Builds and executes pipelines from configuration
"""

import logging
from typing import Dict, Any, Optional
import yaml

from manager import ConfigManager, ModelManager, DataManager
from runner import create_pipeline
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


class Runner:
    """Main runner class that builds and executes pipelines from config"""

    def __init__(
        self,
        config_manager: ConfigManager,
        model_manager: ModelManager,
        data_manager: DataManager,
        pipeline_name: Optional[str] = None
    ):
        """
        Initialize runner

        Args:
            config_manager: Configuration manager
            model_manager: Model manager
            data_manager: Data manager
            pipeline_name: Optional pipeline name (overrides config)
        """
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_manager = data_manager
        
        # Get pipeline name from config or argument
        self.pipeline_name = pipeline_name or self._get_pipeline_from_config()
        
        log_structured(logger, logging.INFO, "Runner initialized",
                     pipeline_name=self.pipeline_name)

    def _get_pipeline_from_config(self) -> str:
        """Get pipeline name from config"""
        pipeline_config = self.config_manager.config.get('pipeline', {})
        return pipeline_config.get('name', 'evaluation')

    @classmethod
    def build_from_config(cls, config_path: str, pipeline_name: Optional[str] = None) -> 'Runner':
        """
        Build runner from config file path

        Args:
            config_path: Path to configuration file
            pipeline_name: Optional pipeline name (overrides config)

        Returns:
            Runner instance
        """
        config_manager = ConfigManager(config_path)
        
        # Configure logging from config file
        from utils.logging import configure_logging
        configure_logging(config_manager.config)
        
        # Initialize managers
        model_manager = ModelManager(config_manager)
        data_manager = DataManager(config_manager)
        
        return cls(config_manager, model_manager, data_manager, pipeline_name)

    def run(self) -> Dict[str, Any]:
        """
        Run the pipeline with configuration

        Returns:
            Dictionary containing pipeline results
        """
        log_structured(logger, logging.INFO, "Starting pipeline execution",
                     pipeline_name=self.pipeline_name)

        # Get pipeline arguments from config
        pipeline_config = self.config_manager.config.get('pipeline', {})
        pipeline_kwargs = pipeline_config.get('args', {})
        
        # Get model and dataset from config if not in pipeline args
        if 'model_name' not in pipeline_kwargs:
            pipeline_kwargs['model_name'] = pipeline_config.get('model', 'clip')
        if 'dataset_type' not in pipeline_kwargs:
            pipeline_kwargs['dataset_type'] = pipeline_config.get('dataset', 'fashion200k')

        # Create and run pipeline
        pipeline = create_pipeline(
            self.pipeline_name,
            self.config_manager,
            self.model_manager,
            self.data_manager
        )

        # Run pipeline
        results = pipeline.run(**pipeline_kwargs)

        log_structured(logger, logging.INFO, "Pipeline execution completed",
                     pipeline_name=self.pipeline_name)

        return results

