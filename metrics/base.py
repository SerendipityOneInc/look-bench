"""
Base Evaluator for LookBench
Fashion Image Retrieval Benchmark
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from utils.logging import get_logger

logger = get_logger(__name__)


class BaseEvaluator(ABC):
    """Base class for all evaluators"""

    def __init__(self, rank_values: Optional[List[int]] = None):
        """
        Initialize evaluator

        Args:
            rank_values: List of rank values to compute (e.g., [1, 5, 10, 20])
        """
        self.rank_values = rank_values or [1, 5, 10, 20]

    @abstractmethod
    def get_metric_name(self) -> str:
        """Get the metric name"""
        pass

    @abstractmethod
    def metric_eval(self, sorted_indices, rank_val, query_label, gallery_label):
        """
        Evaluate the metric for a single query

        Args:
            sorted_indices: Sorted gallery indices by similarity
            rank_val: Rank value to evaluate at
            query_label: Label of the query
            gallery_label: Labels of gallery items

        Returns:
            Metric value for this query
        """
        pass

    def evaluate(
        self,
        query_features: torch.Tensor,
        query_labels: np.ndarray,
        gallery_features: torch.Tensor,
        gallery_labels: np.ndarray,
        l2norm: bool = True,
        num_feat: Optional[int] = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Evaluate retrieval performance

        Args:
            query_features: Query feature embeddings [N_query, D]
            query_labels: Query labels [N_query]
            gallery_features: Gallery feature embeddings [N_gallery, D]
            gallery_labels: Gallery labels [N_gallery]
            l2norm: Whether to L2 normalize features
            num_feat: Optional number of features to use

        Returns:
            Dictionary containing metric results
        """
        # Move to GPU if available
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        query_features = query_features.to(device)
        gallery_features = gallery_features.to(device)

        # Optional feature selection
        if num_feat is not None:
            query_features = query_features[:, :num_feat]
            gallery_features = gallery_features[:, :num_feat]

        # L2 normalization
        if l2norm:
            query_features = F.normalize(query_features, p=2, dim=1)
            gallery_features = F.normalize(gallery_features, p=2, dim=1)

        # Compute similarity matrix
        logger.info("Computing similarity matrix...")
        similarity = torch.mm(query_features, gallery_features.t())

        # Evaluate metrics
        results = {}
        metric_name = self.get_metric_name()

        for rank_val in self.rank_values:
            scores = []
            
            for i in tqdm(range(len(query_labels)), desc=f"Evaluating {metric_name}@{rank_val}"):
                query_label = query_labels[i]
                sim_scores = similarity[i]
                
                # Sort gallery by similarity (descending)
                sorted_indices = torch.argsort(sim_scores, descending=True).cpu().numpy()
                
                # Compute metric
                score = self.metric_eval(sorted_indices, rank_val, query_label, gallery_labels)
                scores.append(score)

            # Average over all queries
            avg_score = np.mean(scores) * 100  # Convert to percentage
            metric_key = f"{metric_name.lower()}@{rank_val}"
            results[metric_key] = round(avg_score, 2)
            
            logger.info(f"{metric_name}@{rank_val}: {avg_score:.2f}%")

        return results

