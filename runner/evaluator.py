"""
Evaluator class for LookBench
Handles the core evaluation logic
"""

import logging
from typing import Dict, Any, Optional
import torch
import numpy as np

from metrics import create_evaluator, get_evaluator_factory
from manager import ConfigManager, ModelManager, DataManager
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


class Evaluator:
    """Core evaluator for fashion image retrieval"""

    def __init__(
        self,
        config_manager: ConfigManager,
        model_manager: ModelManager,
        data_manager: DataManager
    ):
        """
        Initialize evaluator

        Args:
            config_manager: Configuration manager
            model_manager: Model manager
            data_manager: Data manager
        """
        self.config_manager = config_manager
        self.model_manager = model_manager
        self.data_manager = data_manager

    def load_model_and_data(
        self,
        model_name: str,
        dataset_type: str
    ) -> Dict[str, Any]:
        """
        Load model and prepare datasets

        Args:
            model_name: Name of the model to load
            dataset_type: Type of dataset to use

        Returns:
            Dictionary containing model and dataset information
        """
        log_structured(logger, logging.INFO, "Loading model and data",
                     model_name=model_name, dataset_type=dataset_type)

        # Load model
        model, model_instance = self.model_manager.load_model(model_name)
        transform = self.model_manager.get_transform(model_name)

        # Show available datasets
        available_datasets = self.data_manager.get_available_datasets()
        log_structured(logger, logging.INFO, "Available datasets retrieved",
                     available_datasets=available_datasets)

        # Load datasets
        datasets = self.data_manager.load_dataset(dataset_type, transform)

        # Log initialization info
        log_structured(logger, logging.INFO, "Model and data loading completed",
                     available_models=self.model_manager.get_available_models(),
                     dataset_type=dataset_type,
                     query_count=len(datasets['query']),
                     gallery_count=len(datasets['gallery']),
                     dataloader_config=self.data_manager.get_dataloader_config(),
                     evaluation_metric=self.config_manager.get_metric())

        return {
            'model': model,
            'model_instance': model_instance,
            'transform': transform,
            'datasets': datasets
        }

    def extract_features(
        self,
        model: torch.nn.Module,
        datasets: Dict[str, Any],
        batch_size: Optional[int] = None,
        num_workers: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Extract features from datasets

        Args:
            model: Model to extract features from
            datasets: Dictionary containing query and gallery datasets
            batch_size: Optional batch size (uses config if not provided)
            num_workers: Optional number of workers (uses config if not provided)

        Returns:
            Dictionary containing extracted features and labels
        """
        log_structured(logger, logging.INFO, "Starting feature extraction")

        # Get dataset configuration
        dataset_config = self.config_manager.get_dataset_config()
        batch_size = batch_size or dataset_config.get('batch_size', 128)
        num_workers = num_workers or dataset_config.get('num_workers', 8)

        # Extract query features
        logger.info("Extracting query features...")
        query_features, query_labels = self.data_manager.extract_features(
            model=model,
            dataset=datasets['query'],
            batch_size=batch_size,
            num_workers=num_workers
        )

        # Extract gallery features
        logger.info("Extracting gallery features...")
        gallery_features, gallery_labels = self.data_manager.extract_features(
            model=model,
            dataset=datasets['gallery'],
            batch_size=batch_size,
            num_workers=num_workers
        )

        log_structured(logger, logging.INFO, "Feature extraction completed",
                     query_features_shape=query_features.shape,
                     gallery_features_shape=gallery_features.shape,
                     batch_size=batch_size, num_workers=num_workers)

        return {
            'query_features': query_features,
            'query_labels': query_labels,
            'gallery_features': gallery_features,
            'gallery_labels': gallery_labels
        }

    def evaluate(
        self,
        features_info: Dict[str, Any],
        model_name: str,
        metric: Optional[str] = None,
        top_k: Optional[list] = None,
        l2norm: Optional[bool] = None
    ) -> Dict[str, float]:
        """
        Run evaluation on extracted features

        Args:
            features_info: Dictionary containing query and gallery features and labels
            model_name: Name of the model (for config lookup)
            metric: Optional metric name (uses config if not provided)
            top_k: Optional list of K values (uses config if not provided)
            l2norm: Optional L2 normalization flag (uses config if not provided)

        Returns:
            Dictionary containing evaluation metrics
        """
        log_structured(logger, logging.INFO, "Starting evaluation")

        # Get evaluation configuration
        eval_config = self.config_manager.get_evaluation_config()
        metric = metric or eval_config.get('metric', 'recall')
        top_k = top_k or eval_config.get('top_k', [1, 5, 10, 20])
        l2norm = l2norm if l2norm is not None else eval_config.get('l2norm', True)

        # Get model configuration for num_feat
        model_config = self.config_manager.get_model_config(model_name)
        num_feat = model_config.get('num_features', model_config.get('embedding_dim'))

        # Get evaluator type from config
        logger.info(f"Using evaluator type: {metric}")

        # Create evaluator
        try:
            evaluator_params = eval_config.get('evaluator_params', {})
            evaluator_params['rank_values'] = top_k

            evaluator = create_evaluator(metric, **evaluator_params)

            log_structured(logger, logging.INFO, "Evaluator created successfully",
                         evaluator_type=metric,
                         evaluator_class=evaluator.__class__.__name__)
        except Exception as e:
            log_structured(logger, logging.ERROR, "Failed to create evaluator",
                         evaluator_type=metric, error=str(e))
            raise

        # Run evaluation
        metrics = evaluator.evaluate(
            query_features=features_info['query_features'],
            query_labels=features_info['query_labels'],
            gallery_features=features_info['gallery_features'],
            gallery_labels=features_info['gallery_labels'],
            l2norm=l2norm,
            num_feat=num_feat
        )

        # Log results
        self._log_metrics(metrics, "Evaluation")

        return metrics

    def _log_metrics(self, metrics: Dict[str, float], prefix: str = ""):
        """Log evaluation metrics in a formatted way"""
        logger.info(f"{prefix} results:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {metric_name}: {value:.2f}%")
            else:
                logger.info(f"  {metric_name}: {value}")

