"""
Base reranker classes for LookBench
Cross-encoder reranking track
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from PIL import Image

from utils.logging import get_logger

logger = get_logger(__name__)


class BaseReranker(ABC):
    """Abstract base class for reranking models.

    Retrieval models in :mod:`models.base` map one input to an embedding and are
    compared against a gallery. A reranker is a different shape: it is handed a
    query and the candidates to judge, and returns a score per candidate. There
    is no gallery, no embedding contract and no ``top_k``, which is why this does
    not subclass :class:`~models.base.BaseModel`.

    ``query_image`` is optional and always part of the signature. A text-only
    reranker ignores it; a multimodal reranker consumes it. Keeping one signature
    for both means they register, configure and evaluate through the same
    pipeline, and a track can mix them.
    """

    #: Subclasses that consume ``query_image`` set this to True.
    supports_multimodal: bool = False

    @classmethod
    @abstractmethod
    def load_model(
        cls,
        model_name: str,
        model_path: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[Any, "BaseReranker"]:
        """
        Load a reranker and return the scorer and the model instance.

        Args:
            model_name: Model name or path (e.g., HuggingFace repo id)
            model_path: Optional adapter or checkpoint path applied on top of
                ``model_name``, for models released as LoRA adapters
            **kwargs: Model-specific loading options

        Returns:
            Tuple of (scorer, model_instance)
        """
        raise NotImplementedError

    @abstractmethod
    def score(
        self,
        query: str,
        candidates: Sequence[str],
        query_image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    ) -> List[float]:
        """
        Score every candidate against the query.

        Higher is more relevant. Scores are only required to be comparable
        within one call, since the benchmark compares candidates for the same
        query and never across queries.

        Args:
            query: Query text
            candidates: Candidate product texts to score
            query_image: Optional query image; ignored by text-only rerankers

        Returns:
            One score per candidate, in the order given
        """
        raise NotImplementedError

    def score_pair(
        self,
        query: str,
        preferred: str,
        rejected: str,
        query_image: Optional[Union[str, Image.Image, torch.Tensor]] = None,
    ) -> Tuple[float, float]:
        """
        Score a preference pair in one call.

        Both candidates go through a single :meth:`score` call so that any
        batching or shared query encoding a subclass does applies to the pair,
        and so the two scores are always produced under identical conditions.

        Args:
            query: Query text
            preferred: The candidate the label prefers
            rejected: The other candidate
            query_image: Optional query image

        Returns:
            Tuple of (preferred_score, rejected_score)
        """
        scores = self.score(query, [preferred, rejected], query_image=query_image)
        if len(scores) != 2:
            raise ValueError(
                f"{type(self).__name__}.score returned {len(scores)} scores for 2 candidates"
            )
        return float(scores[0]), float(scores[1])

    def get_model_info(self, model: Any) -> Dict[str, Any]:
        """
        Get model information

        Args:
            model: The loaded scorer

        Returns:
            Dictionary containing model information
        """
        base_model = getattr(model, "model", model)
        info: Dict[str, Any] = {
            "model_type": self.get_model_type(),
            "task": "reranking",
            "supports_multimodal": self.supports_multimodal,
            "framework": "PyTorch",
        }
        try:
            params = list(base_model.parameters())
            info["total_params"] = sum(p.numel() for p in params)
            info["device"] = str(params[0].device)
        except (AttributeError, IndexError, StopIteration):
            # Remote or API-backed rerankers expose no parameters.
            info["total_params"] = None
            info["device"] = "n/a"
        return info

    @classmethod
    def get_model_type(cls) -> str:
        """Get the model type identifier"""
        return cls.__name__.lower().replace("reranker", "").strip("_") or "reranker"

    @classmethod
    def get_model_name(cls) -> str:
        """Get the human-readable model name"""
        return cls.__name__
