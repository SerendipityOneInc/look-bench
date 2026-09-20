"""
ShopRank-Bench preference pair loader for LookBench.

Loads the e-commerce preference track as (query, preferred, rejected) records in
either the structured attribute schema or the natural-language rendering.
"""

import json
from collections import Counter
from typing import Any, Dict, Iterator, List, Optional, Sequence

from utils.logging import get_logger

logger = get_logger(__name__)

FORMATS = {
    "structured": ("preferred_structured", "rejected_structured"),
    "natural": ("preferred_natural", "rejected_natural"),
}

#: Agreement tiers, by how many judge families committed a verdict.
TIERS = ("gold", "silver", "bronze")


class ShopRankPairs:
    """Preference pairs from ShopRank-Bench.

    Each record pairs a query with the candidate the judge panel preferred and
    the one it rejected, plus the agreement tier. A model is correct on a record
    when it scores ``preferred`` above ``rejected``.

    The same pair is released in two formats. ``structured`` is the canonical
    pipe-delimited attribute schema; ``natural`` is a model-rendered prose view
    of the same attributes. They carry the same label, so a per-format accuracy
    gap measures format sensitivity rather than a difference in labels.
    """

    def __init__(
        self,
        records: List[Dict[str, Any]],
        text_format: str = "structured",
    ):
        """
        Initialize the dataset

        Args:
            records: Released preference records
            text_format: ``structured`` or ``natural``
        """
        if text_format not in FORMATS:
            raise ValueError(
                f"unknown text_format {text_format!r}; expected one of {sorted(FORMATS)}"
            )
        self.records = records
        self.text_format = text_format
        self.preferred_key, self.rejected_key = FORMATS[text_format]

        missing = [r for r in records if self.preferred_key not in r or self.rejected_key not in r]
        if missing:
            raise ValueError(
                f"{len(missing)} records lack {self.preferred_key!r}/{self.rejected_key!r}; "
                f"is this track released in the {text_format!r} format?"
            )
        self._check_tiers()

    def _check_tiers(self) -> None:
        """Warn when the tier field is not the released three-level one.

        Internal pre-release files carry a stale two-level tier (gold/silver
        only). Evaluating against those silently produces a per-tier table that
        is wrong while the overall accuracy still looks right, so it is worth
        naming loudly rather than discovering later.
        """
        present = {r.get("tier") for r in self.records if r.get("tier") is not None}
        if not present:
            logger.warning("records carry no tier field; per-tier results unavailable")
            return
        unknown = present - set(TIERS)
        if unknown:
            logger.warning("unexpected tier labels present: %s", sorted(map(str, unknown)))
        if not present & {"bronze"}:
            logger.warning(
                "no 'bronze' tier present (found %s) -- this looks like the stale "
                "two-level tier field, not the released three-level one; per-tier "
                "numbers will not match published results",
                sorted(map(str, present)),
            )

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        text_format: str = "structured",
        track: Optional[str] = "preference",
    ) -> "ShopRankPairs":
        """
        Load from a local JSONL release file.

        Args:
            path: Path to the released JSONL
            text_format: ``structured`` or ``natural``
            track: Keep only this track, or None to keep everything

        Returns:
            A loaded dataset
        """
        with open(path, encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
        return cls(cls._filter(records, track), text_format=text_format)

    @classmethod
    def from_hub(
        cls,
        repo_id: str = "srpone/shoprank-bench",
        split: str = "preference",
        text_format: str = "structured",
        token: Optional[str] = None,
    ) -> "ShopRankPairs":
        """
        Load from the Hugging Face Hub.

        Args:
            repo_id: Dataset repo id
            split: Track to load
            text_format: ``structured`` or ``natural``
            token: Access token, for a gated or private release

        Returns:
            A loaded dataset
        """
        from datasets import load_dataset

        data = load_dataset(repo_id, split=split, token=token)
        return cls([dict(row) for row in data], text_format=text_format)

    @staticmethod
    def _filter(records: Sequence[Dict[str, Any]], track: Optional[str]) -> List[Dict[str, Any]]:
        if track is None:
            return list(records)
        kept = [r for r in records if r.get("track", "preference") == track]
        if not kept:
            available = sorted({str(r.get("track")) for r in records})
            raise ValueError(f"no records for track {track!r}; available: {available}")
        return kept

    def tier_counts(self) -> Dict[str, int]:
        """Return the number of pairs per tier."""
        counts = Counter(r.get("tier") for r in self.records)
        return {str(k): v for k, v in counts.items()}

    def queries(self) -> List[str]:
        """Query string per record, for query-clustered significance tests."""
        return [r["query"] for r in self.records]

    def tiers(self) -> List[Optional[str]]:
        """Tier label per record, aligned with :meth:`queries`."""
        return [r.get("tier") for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        record = self.records[index]
        return {
            "query": record["query"],
            "preferred": record[self.preferred_key],
            "rejected": record[self.rejected_key],
            "tier": record.get("tier"),
        }

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for index in range(len(self)):
            yield self[index]
