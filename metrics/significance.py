"""
Paired Significance Tests for LookBench
Cross-encoder reranking track
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from utils.logging import get_logger

logger = get_logger(__name__)


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> Dict[str, float]:
    """
    Continuity-corrected paired McNemar test over the same items.

    Args:
        correct_a: Per-item correctness for the first system
        correct_b: Per-item correctness for the second system

    Returns:
        Dictionary with chi2, p, and the two discordant counts
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"correctness arrays differ in shape: {a.shape} vs {b.shape}")

    first_only = int(np.sum(a & ~b))
    second_only = int(np.sum(b & ~a))
    discordant = first_only + second_only
    if discordant == 0:
        return {"chi2": 0.0, "p": 1.0, "first_only": 0, "second_only": 0}

    chi2 = (abs(first_only - second_only) - 1.0) ** 2 / discordant
    return {
        "chi2": float(chi2),
        "p": float(math.erfc(math.sqrt(chi2 / 2.0))),
        "first_only": first_only,
        "second_only": second_only,
    }


def clustered_bootstrap(
    groups: Sequence[str],
    correct_a: Sequence[bool],
    correct_b: Sequence[bool],
    n_boot: int = 5000,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Bootstrap a paired accuracy difference, resampling whole groups.

    Benchmark items are often not independent: several pairs share a query, its
    slate and its labels. McNemar treats each item as its own observation, which
    understates the variance of a paired difference. Resampling whole groups with
    replacement, so a group contributes all of its items or none, keeps that
    dependence in the interval.

    Args:
        groups: Group key per item (e.g., the query string)
        correct_a: Per-item correctness for the first system
        correct_b: Per-item correctness for the second system
        n_boot: Number of bootstrap resamples
        seed: Random seed

    Returns:
        Dictionary with the observed difference, its 95% CI, a two-sided
        bootstrap p, the implied design effect, and the group count
    """
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(f"correctness arrays differ in shape: {a.shape} vs {b.shape}")
    if len(groups) != a.shape[0]:
        raise ValueError(f"groups has {len(groups)} entries for {a.shape[0]} items")

    index: Dict[str, int] = {}
    for key in groups:
        index.setdefault(key, len(index))
    gidx = np.array([index[key] for key in groups])
    n_groups = len(index)

    diff = a.astype(np.int64) - b.astype(np.int64)
    group_sum = np.bincount(gidx, weights=diff, minlength=n_groups)
    group_count = np.bincount(gidx, minlength=n_groups).astype(np.float64)

    observed = float(diff.sum() / diff.size)
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot)
    for start in range(0, n_boot, 500):
        chunk = min(500, n_boot - start)
        pick = rng.integers(0, n_groups, size=(chunk, n_groups))
        stats[start:start + chunk] = group_sum[pick].sum(axis=1) / group_count[pick].sum(axis=1)

    lo, hi = np.percentile(stats, [2.5, 97.5])
    side = np.mean(stats <= 0) if observed > 0 else np.mean(stats >= 0)
    p = min(1.0, max(float(side) * 2.0, 1.0 / n_boot))

    naive_var = diff.var(ddof=1) / diff.size
    design_effect = float(stats.var(ddof=1) / naive_var) if naive_var > 0 else float("nan")

    return {
        "diff": observed,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p": p,
        "design_effect": design_effect,
        "n_items": int(diff.size),
        "n_groups": n_groups,
    }


def holm_bonferroni(pvals: Sequence[float]) -> List[float]:
    """
    Holm-Bonferroni adjusted p-values, in the order given.

    Args:
        pvals: Raw p-values for one comparison family

    Returns:
        Adjusted p-values, aligned with the input
    """
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted


def compare_systems(
    groups: Sequence[str],
    correctness: Dict[str, Sequence[bool]],
    comparisons: Sequence[Tuple[str, str]],
    n_boot: int = 5000,
    seed: int = 0,
    correct_family: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a family of paired comparisons three ways.

    Each comparison is reported as a naive McNemar test, a group-clustered
    bootstrap of the paired difference, and the implied design effect. Holm
    correction is applied across the family, since the same items back every
    comparison.

    Args:
        groups: Group key per item, shared by all systems
        correctness: Per-item correctness, keyed by system name
        comparisons: (left, right) system-name pairs to test
        n_boot: Number of bootstrap resamples
        seed: Random seed
        correct_family: Whether to Holm-correct across the comparisons

    Returns:
        One result dictionary per comparison, in the order given
    """
    rows: List[Dict[str, Any]] = []
    for left, right in comparisons:
        missing = [name for name in (left, right) if name not in correctness]
        if missing:
            logger.warning("skipping comparison, missing systems: %s", missing)
            continue

        a, b = correctness[left], correctness[right]
        naive = mcnemar(a, b)
        clustered = clustered_bootstrap(groups, a, b, n_boot=n_boot, seed=seed)
        rows.append({
            "left": left,
            "right": right,
            "acc_left": float(np.mean(np.asarray(a, dtype=bool))),
            "acc_right": float(np.mean(np.asarray(b, dtype=bool))),
            "chi2": naive["chi2"],
            "p_mcnemar": naive["p"],
            "p_clustered": clustered["p"],
            "diff": clustered["diff"],
            "ci_low": clustered["ci_low"],
            "ci_high": clustered["ci_high"],
            "design_effect": clustered["design_effect"],
            "n_items": clustered["n_items"],
            "n_groups": clustered["n_groups"],
        })

    if correct_family and rows:
        for row, adjusted in zip(rows, holm_bonferroni([r["p_clustered"] for r in rows])):
            row["p_clustered_holm"] = adjusted

    return rows
