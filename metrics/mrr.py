"""
Mean Reciprocal Rank (MRR) evaluation for LookBench
Fashion image retrieval
"""

import numpy as np
from .base import BaseEvaluator
from utils.logging import get_logger

logger = get_logger(__name__)


class MRREvaluator(BaseEvaluator):
    """Evaluator for Mean Reciprocal Rank (MRR)"""

    def get_metric_name(self) -> str:
        return "MRR"

    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        """
        Compute reciprocal rank of first relevant item
        
        Returns 1/rank of first match, or 0 if no match in top-k
        """
        search_limit = min(rank_val, len(sorted_indices))
        
        for rank, gallery_idx in enumerate(sorted_indices[:search_limit], start=1):
            if gallery_label[gallery_idx] == query_label:
                return 1.0 / rank
        
        return 0.0


__all__ = ['MRREvaluator']

