"""
Rank-based evaluation metrics for LookBench
Fashion image retrieval
"""

import numpy as np
import torch
from typing import Optional, Dict

from .base import BaseEvaluator
from utils.logging import get_logger

logger = get_logger(__name__)


class RankEvaluator(BaseEvaluator):
    """Evaluator for Rank-based accuracy (Recall@K)"""

    def get_metric_name(self) -> str:
        return "Recall"

    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        """
        Evaluate recall at rank k
        
        Returns 1.0 if at least one relevant item is in top-k, 0.0 otherwise
        """
        search_limit = min(rank_val, len(sorted_indices))
        
        for gallery_idx in sorted_indices[:search_limit]:
            if gallery_label[gallery_idx] == query_label:
                return 1.0
        
        return 0.0


def evaluate_rank_accuracy(
    query_features: torch.Tensor,
    query_labels: np.ndarray,
    gallery_features: torch.Tensor,
    gallery_labels: np.ndarray,
    rank_values: list = None,
    num_feat: Optional[int] = None,
    l2norm: bool = True,
    **kwargs
) -> Dict[str, float]:
    """
    Convenience function to evaluate rank-based accuracy

    Args:
        query_features: Pre-extracted query features
        query_labels: Query labels
        gallery_features: Pre-extracted gallery features
        gallery_labels: Gallery labels
        rank_values: List of rank values to compute
        num_feat: Number of features to use
        l2norm: Whether to L2 normalize features

    Returns:
        Dictionary containing rank metrics
    """
    evaluator = RankEvaluator(rank_values=rank_values)

    return evaluator.evaluate(
        query_features=query_features,
        query_labels=query_labels,
        gallery_features=gallery_features,
        gallery_labels=gallery_labels,
        num_feat=num_feat,
        l2norm=l2norm,
        **kwargs
    )


__all__ = ['RankEvaluator', 'evaluate_rank_accuracy']

