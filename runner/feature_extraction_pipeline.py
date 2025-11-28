"""
Feature Extraction Pipeline for LookBench
Extracts features without running evaluation
"""

import logging
from typing import Dict, Any
import os

from .base_pipeline import BasePipeline
from .evaluator import Evaluator
from .pipeline import register_pipeline
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


@register_pipeline("feature_extraction")
class FeatureExtractionPipeline(BasePipeline):
    """Pipeline for extracting features without evaluation"""

    def __init__(self, config_manager, model_manager, data_manager):
        """Initialize feature extraction pipeline"""
        super().__init__(config_manager, model_manager, data_manager)
        self.evaluator = Evaluator(config_manager, model_manager, data_manager)

    def get_pipeline_name(self) -> str:
        """Get pipeline name"""
        return "feature_extraction"

    def run(
        self,
        model_name: str = "clip",
        dataset_type: str = "fashion200k",
        save_path: str = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run feature extraction pipeline

        Args:
            model_name: Name of the model to use
            dataset_type: Type of dataset to use
            save_path: Optional path to save extracted features
            **kwargs: Additional arguments (batch_size, num_workers, etc.)

        Returns:
            Dictionary containing extracted features
        """
        log_structured(logger, logging.INFO, "Starting feature extraction pipeline",
                     model_name=model_name, dataset_type=dataset_type)

        # Step 1: Load model and prepare data
        model_info = self.evaluator.load_model_and_data(model_name, dataset_type)

        # Step 2: Extract features
        features_info = self.evaluator.extract_features(
            model=model_info['model'],
            datasets=model_info['datasets'],
            batch_size=kwargs.get('batch_size'),
            num_workers=kwargs.get('num_workers')
        )

        # Step 3: Save features if path provided
        if save_path:
            self._save_features(features_info, save_path)

        log_structured(logger, logging.INFO, "Feature extraction pipeline completed",
                     model_name=model_name, dataset_type=dataset_type)

        return {
            'features_info': features_info,
            'model_name': model_name,
            'dataset_type': dataset_type,
            'save_path': save_path
        }

    def _save_features(self, features_info: Dict[str, Any], save_path: str):
        """Save extracted features to disk"""
        import torch
        
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        save_dict = {
            'query_features': features_info['query_features'],
            'query_labels': features_info['query_labels'],
            'gallery_features': features_info['gallery_features'],
            'gallery_labels': features_info['gallery_labels']
        }
        
        torch.save(save_dict, save_path)
        log_structured(logger, logging.INFO, "Features saved",
                     save_path=save_path)

