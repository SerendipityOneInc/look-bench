"""
Evaluation Pipeline for LookBench
Standard evaluation pipeline that runs full evaluation
"""

import logging
from typing import Dict, Any

from .base_pipeline import BasePipeline
from .evaluator import Evaluator
from .pipeline import register_pipeline
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


@register_pipeline("evaluation")
class EvaluationPipeline(BasePipeline):
    """Standard evaluation pipeline for fashion image retrieval"""

    def __init__(self, config_manager, model_manager, data_manager):
        """Initialize evaluation pipeline"""
        super().__init__(config_manager, model_manager, data_manager)
        self.evaluator = Evaluator(config_manager, model_manager, data_manager)

    def get_pipeline_name(self) -> str:
        """Get pipeline name"""
        return "evaluation"

    def run(
        self,
        model_name: str = "clip",
        dataset_type: str = "fashion200k",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run complete evaluation pipeline

        Args:
            model_name: Name of the model to evaluate
            dataset_type: Type of dataset to use
            **kwargs: Additional arguments (metric, top_k, l2norm, etc.)

        Returns:
            Dictionary containing evaluation metrics
        """
        log_structured(logger, logging.INFO, "Starting evaluation pipeline",
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

        # Step 3: Run evaluation
        metrics = self.evaluator.evaluate(
            features_info=features_info,
            model_name=model_name,
            metric=kwargs.get('metric'),
            top_k=kwargs.get('top_k'),
            l2norm=kwargs.get('l2norm')
        )

        log_structured(logger, logging.INFO, "Evaluation pipeline completed",
                     model_name=model_name, dataset_type=dataset_type, metrics=metrics)

        return {
            'metrics': metrics,
            'model_name': model_name,
            'dataset_type': dataset_type,
            'features_info': features_info
        }

