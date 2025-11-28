"""
Normalized Discounted Cumulative Gain (NDCG) evaluation for LookBench
Fashion image retrieval
"""

import numpy as np
from .base import BaseEvaluator
from utils.logging import get_logger

logger = get_logger(__name__)


class NDCGEvaluator(BaseEvaluator):
    """Evaluator for Normalized Discounted Cumulative Gain (NDCG)"""

    def get_metric_name(self) -> str:
        return "NDCG"

    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        """
        Compute NDCG@k
        
        Returns NDCG score considering relevance and position
        """
        search_limit = min(rank_val, len(sorted_indices))
        
        # Compute DCG (Discounted Cumulative Gain)
        dcg = 0.0
        idcg = 0.0
        
        # Count relevant items for IDCG
        num_relevant = np.sum(gallery_label == query_label)
        
        for rank, gallery_idx in enumerate(sorted_indices[:search_limit], start=1):
            relevance = 1.0 if gallery_label[gallery_idx] == query_label else 0.0
            dcg += relevance / np.log2(rank + 1)
        
        # Compute IDCG (Ideal DCG)
        for rank in range(1, min(num_relevant, search_limit) + 1):
            idcg += 1.0 / np.log2(rank + 1)
        
        # Normalize
        if idcg > 0:
            return dcg / idcg
        
        return 0.0


__all__ = ['NDCGEvaluator']

