"""
Evaluator Factory for dynamic evaluator selection
"""

import logging
from typing import Dict, Any, Optional, Type
from .base import BaseEvaluator
from .rank import RankEvaluator
from .mrr import MRREvaluator
from .ndcg import NDCGEvaluator
from .map import MAPEvaluator
from utils.logging import get_logger, log_structured

logger = get_logger(__name__)


class EvaluatorFactory:
    """Factory class for creating evaluators based on configuration"""

    def __init__(self):
        """Initialize the evaluator factory with available evaluators"""
        self._evaluators: Dict[str, Type[BaseEvaluator]] = {}
        self._register_default_evaluators()

    def _register_default_evaluators(self):
        """Register default evaluators"""
        self.register_evaluator("rank", RankEvaluator)
        self.register_evaluator("recall", RankEvaluator)  # Alias
        self.register_evaluator("mrr", MRREvaluator)
        self.register_evaluator("ndcg", NDCGEvaluator)
        self.register_evaluator("map", MAPEvaluator)

        log_structured(logger, logging.INFO, "Default evaluators registered",
                       available_evaluators=list(self._evaluators.keys()))

    def register_evaluator(self, name: str, evaluator_class: Type[BaseEvaluator]):
        """
        Register a new evaluator type

        Args:
            name: Name of the evaluator
            evaluator_class: Class implementing BaseEvaluator
        """
        if not issubclass(evaluator_class, BaseEvaluator):
            raise ValueError(
                f"Evaluator class must inherit from BaseEvaluator, got {evaluator_class}")

        self._evaluators[name] = evaluator_class
        log_structured(logger, logging.INFO, "Evaluator registered",
                       evaluator_name=name, evaluator_class=evaluator_class.__name__)

    def create_evaluator(self, evaluator_type: str, **kwargs) -> BaseEvaluator:
        """
        Create an evaluator instance based on type

        Args:
            evaluator_type: Type of evaluator to create
            **kwargs: Additional arguments to pass to evaluator constructor

        Returns:
            Evaluator instance

        Raises:
            ValueError: If evaluator type is not supported
        """
        if evaluator_type not in self._evaluators:
            available = list(self._evaluators.keys())
            raise ValueError(f"Unsupported evaluator type: '{evaluator_type}'. "
                             f"Available types: {available}")

        evaluator_class = self._evaluators[evaluator_type]

        try:
            evaluator = evaluator_class(**kwargs)
            log_structured(logger, logging.INFO, "Evaluator created successfully",
                           evaluator_type=evaluator_type, evaluator_class=evaluator_class.__name__)
            return evaluator
        except Exception as e:
            log_structured(logger, logging.ERROR, "Failed to create evaluator",
                           evaluator_type=evaluator_type, error=str(e))
            raise

    def get_available_evaluators(self) -> list:
        """Get list of available evaluator types"""
        return list(self._evaluators.keys())

    def is_evaluator_supported(self, evaluator_type: str) -> bool:
        """Check if an evaluator type is supported"""
        return evaluator_type in self._evaluators


# Global factory instance
_evaluator_factory = None


def get_evaluator_factory() -> EvaluatorFactory:
    """Get the global evaluator factory instance"""
    global _evaluator_factory
    if _evaluator_factory is None:
        _evaluator_factory = EvaluatorFactory()
    return _evaluator_factory


def create_evaluator(evaluator_type: str, **kwargs) -> BaseEvaluator:
    """
    Convenience function to create an evaluator

    Args:
        evaluator_type: Type of evaluator to create
        **kwargs: Additional arguments for evaluator

    Returns:
        Evaluator instance
    """
    factory = get_evaluator_factory()
    return factory.create_evaluator(evaluator_type, **kwargs)


def register_evaluator(name: str, evaluator_class: Type[BaseEvaluator]):
    """
    Convenience function to register a new evaluator

    Args:
        name: Name of the evaluator
        evaluator_class: Class implementing BaseEvaluator
    """
    factory = get_evaluator_factory()
    factory.register_evaluator(name, evaluator_class)


__all__ = [
    'EvaluatorFactory',
    'get_evaluator_factory',
    'create_evaluator',
    'register_evaluator'
]

