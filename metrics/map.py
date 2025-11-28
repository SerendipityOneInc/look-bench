"""
Mean Average Precision (MAP) evaluation for LookBench
Fashion image retrieval
"""

import numpy as np
from .base import BaseEvaluator
from utils.logging import get_logger

logger = get_logger(__name__)


class MAPEvaluator(BaseEvaluator):
    """Evaluator for Mean Average Precision (MAP)"""

    def get_metric_name(self) -> str:
        return "MAP"

    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        """
        Compute Average Precision (AP) for a single query
        
        Returns AP considering all relevant items in top-k
        """
        search_limit = min(rank_val, len(sorted_indices))
        
        relevant_count = 0
        precision_sum = 0.0
        
        for rank, gallery_idx in enumerate(sorted_indices[:search_limit], start=1):
            if gallery_label[gallery_idx] == query_label:
                relevant_count += 1
                precision_at_k = relevant_count / rank
                precision_sum += precision_at_k
        
        # Count total relevant items
        total_relevant = np.sum(gallery_label == query_label)
        
        if total_relevant > 0:
            return precision_sum / min(total_relevant, search_limit)
        
        return 0.0


__all__ = ['MAPEvaluator']

