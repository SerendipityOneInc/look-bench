"""
Pairwise Accuracy Evaluator for LookBench
Cross-encoder reranking track
"""

from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from tqdm import tqdm

from utils.logging import get_logger

logger = get_logger(__name__)


class PairwiseAccuracyEvaluator:
    """Accuracy on preference pairs, overall and per agreement tier.

    The retrieval evaluators in :mod:`metrics.base` rank a gallery and score the
    result at cut-offs. A preference pair has no gallery and no cut-off: the
    reranker sees two candidates and is correct when it puts the preferred one
    first. That makes this a sibling of :class:`~metrics.base.BaseEvaluator`
    rather than a subclass of it.

    Ties are reported, never silently resolved. A model that assigns both
    candidates the same score has not made a decision, so ``tie_credit`` says
    what such a pair is worth: ``0.0`` (default, a tie is wrong) or ``0.5`` (the
    coin-flip expectation). Both are defensible; leaving it implicit is not.
    """

    def __init__(self, tiers: Optional[Sequence[str]] = None, tie_credit: float = 0.0):
        """
        Initialize evaluator

        Args:
            tiers: Tier names to break results out by, in report order
            tie_credit: Credit given to a tied pair, between 0.0 and 1.0
        """
        if not 0.0 <= tie_credit <= 1.0:
            raise ValueError(f"tie_credit must be in [0.0, 1.0], got {tie_credit}")
        self.tiers = list(tiers) if tiers is not None else ["gold", "silver", "bronze"]
        self.tie_credit = tie_credit

    def get_metric_name(self) -> str:
        """Get the metric name"""
        return "pairwise_accuracy"

    def evaluate(
        self,
        preferred_scores: Sequence[float],
        rejected_scores: Sequence[float],
        tiers: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, float]:
        """
        Compute pairwise accuracy from paired scores.

        Args:
            preferred_scores: Score given to the labelled-preferred candidate
            rejected_scores: Score given to the other candidate
            tiers: Optional per-pair tier label, same length as the scores

        Returns:
            Dictionary of metrics: overall accuracy, tie count, per-tier
            accuracy, and the counts each is computed from
        """
        pref = np.asarray(preferred_scores, dtype=np.float64)
        rej = np.asarray(rejected_scores, dtype=np.float64)
        if pref.shape != rej.shape:
            raise ValueError(f"score arrays differ in shape: {pref.shape} vs {rej.shape}")
        if pref.size == 0:
            raise ValueError("no pairs to evaluate")

        wins = pref > rej
        ties = pref == rej
        credit = wins.astype(np.float64) + self.tie_credit * ties

        results: Dict[str, float] = OrderedDict()
        results["pairwise_accuracy"] = float(credit.mean())
        results["n_pairs"] = int(pref.size)
        results["n_ties"] = int(ties.sum())

        if tiers is not None:
            tier_arr = np.asarray(tiers)
            if tier_arr.shape[0] != pref.shape[0]:
                raise ValueError(
                    f"tiers has {tier_arr.shape[0]} entries for {pref.shape[0]} pairs"
                )
            present = {str(t) for t in np.unique(tier_arr)}
            seen = [t for t in self.tiers if t in present]
            # Tier labels outside the configured order are still reported.
            seen += sorted(present - set(self.tiers))
            for tier in seen:
                mask = tier_arr == tier
                results[f"pairwise_accuracy_{tier}"] = float(credit[mask].mean())
                results[f"n_pairs_{tier}"] = int(mask.sum())

        return results

    def evaluate_reranker(
        self,
        reranker: Any,
        records: Iterable[Dict[str, Any]],
        query_key: str = "query",
        preferred_key: str = "preferred_structured",
        rejected_key: str = "rejected_structured",
        tier_key: str = "tier",
        image_key: Optional[str] = None,
        show_progress: bool = True,
    ) -> Dict[str, float]:
        """
        Score a set of preference records with a reranker and evaluate them.

        Args:
            reranker: A :class:`~models.reranker_base.BaseReranker` instance
            records: Preference records
            query_key: Field holding the query text
            preferred_key: Field holding the labelled-preferred candidate
            rejected_key: Field holding the other candidate
            tier_key: Field holding the agreement tier, if present
            image_key: Field holding a query image, for multimodal rerankers
            show_progress: Whether to show a progress bar

        Returns:
            Dictionary of metrics, as returned by :meth:`evaluate`
        """
        pref_scores: List[float] = []
        rej_scores: List[float] = []
        tiers: List[str] = []

        records = list(records)
        for record in tqdm(records, disable=not show_progress, desc="scoring pairs"):
            query_image = record.get(image_key) if image_key else None
            pref, rej = reranker.score_pair(
                record[query_key],
                record[preferred_key],
                record[rejected_key],
                query_image=query_image,
            )
            pref_scores.append(pref)
            rej_scores.append(rej)
            if tier_key in record:
                tiers.append(record[tier_key])

        if tiers and len(tiers) != len(records):
            logger.warning(
                "tier field missing on some records; reporting overall accuracy only"
            )
            tiers = []

        return self.evaluate(pref_scores, rej_scores, tiers=tiers or None)
