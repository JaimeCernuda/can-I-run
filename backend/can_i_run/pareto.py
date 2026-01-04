"""
Pareto Frontier Algorithm

A model is "Pareto optimal" if no other model is:
    - Higher quality at the same or lower VRAM, AND
    - Lower VRAM at the same or higher quality

The frontier shows the best options—points OFF the line are "dominated"
(there's always a better option on the frontier).

This module computes three separate Pareto frontiers:
    1. Quality frontier: Maximize quality, minimize VRAM
    2. Performance frontier: Maximize tokens/sec, minimize VRAM
    3. Efficiency frontier: Maximize efficiency score, minimize VRAM

Visual representation:
    ┌─────────────────────────────────────────┐
    │ Quality                                 │
    │   ▲                                     │
    │   │  A ← Best quality (uses most VRAM)  │
    │   │   ╲                                 │
    │   │    B ← Balanced choice              │
    │   │     ╲                               │
    │   │      C ← Most headroom (lower qual) │
    │   └──────────────────────────────────▶  │
    │              VRAM Headroom              │
    └─────────────────────────────────────────┘
"""

from dataclasses import dataclass
from typing import TypeVar, Callable
from .models import Candidate


T = TypeVar('T')


def compute_pareto_frontier_generic(
    candidates: list[T],
    get_metric: Callable[[T], float],
    get_vram: Callable[[T], float],
    higher_is_better: bool = True
) -> list[T]:
    """
    Generic Pareto frontier computation.

    Finds candidates where no other candidate dominates them. A candidate A
    dominates B if A has better or equal metric AND lower or equal VRAM,
    with at least one strict improvement.

    Args:
        candidates: List of candidates to evaluate
        get_metric: Function to extract the metric to optimize (quality, perf, etc.)
        get_vram: Function to extract VRAM usage
        higher_is_better: Whether higher metric values are better

    Returns:
        List of Pareto-optimal candidates
    """
    if not candidates:
        return []

    # Sort by metric (best first) for efficiency
    sorted_candidates = sorted(
        candidates,
        key=lambda c: get_metric(c),
        reverse=higher_is_better
    )

    frontier = []
    min_vram_so_far = float('inf')

    for candidate in sorted_candidates:
        vram = get_vram(candidate)

        # A candidate is Pareto-optimal if it uses less VRAM than all
        # candidates with better or equal metric seen so far
        if vram < min_vram_so_far:
            frontier.append(candidate)
            min_vram_so_far = vram

    return frontier


def compute_quality_frontier(candidates: list[Candidate]) -> list[Candidate]:
    """
    Compute Pareto frontier for quality vs VRAM.

    Args:
        candidates: List of candidates

    Returns:
        Quality Pareto-optimal candidates
    """
    return compute_pareto_frontier_generic(
        candidates,
        get_metric=lambda c: c.quality_score,
        get_vram=lambda c: c.vram_required,
        higher_is_better=True
    )


def compute_performance_frontier(candidates: list[Candidate]) -> list[Candidate]:
    """
    Compute Pareto frontier for performance (tokens/sec) vs VRAM.

    Args:
        candidates: List of candidates

    Returns:
        Performance Pareto-optimal candidates
    """
    return compute_pareto_frontier_generic(
        candidates,
        get_metric=lambda c: c.tokens_per_second,
        get_vram=lambda c: c.vram_required,
        higher_is_better=True
    )


def compute_efficiency_frontier(candidates: list[Candidate]) -> list[Candidate]:
    """
    Compute Pareto frontier for efficiency vs VRAM.

    Args:
        candidates: List of candidates

    Returns:
        Efficiency Pareto-optimal candidates
    """
    return compute_pareto_frontier_generic(
        candidates,
        get_metric=lambda c: c.efficiency_score,
        get_vram=lambda c: c.vram_required,
        higher_is_better=True
    )


def compute_pareto_frontier(
    candidates: list[Candidate]
) -> list[Candidate]:
    """
    Compute all three Pareto frontiers and mark candidates.

    This modifies the candidates in-place to set the is_pareto_* flags,
    then returns only candidates that are on at least one frontier.

    Args:
        candidates: List of candidates to evaluate

    Returns:
        Candidates that are Pareto-optimal on at least one metric
    """
    # Compute each frontier
    quality_frontier = set(id(c) for c in compute_quality_frontier(candidates))
    perf_frontier = set(id(c) for c in compute_performance_frontier(candidates))
    efficiency_frontier = set(id(c) for c in compute_efficiency_frontier(candidates))

    # Mark candidates
    for candidate in candidates:
        candidate.is_pareto_quality = id(candidate) in quality_frontier
        candidate.is_pareto_performance = id(candidate) in perf_frontier
        candidate.is_pareto_efficiency = id(candidate) in efficiency_frontier

    # Return candidates on any frontier
    return [
        c for c in candidates
        if c.is_pareto_quality or c.is_pareto_performance or c.is_pareto_efficiency
    ]


def filter_dominated_candidates(
    candidates: list[Candidate],
    vram_available: float
) -> list[Candidate]:
    """
    Filter out candidates that don't fit in available VRAM.

    Args:
        candidates: List of candidates
        vram_available: Available VRAM in GB

    Returns:
        Candidates that fit in available VRAM
    """
    return [c for c in candidates if c.vram_required <= vram_available]


def get_pareto_summary(
    candidates: list[Candidate]
) -> dict:
    """
    Get summary statistics about the Pareto frontiers.

    Args:
        candidates: List of candidates (after compute_pareto_frontier)

    Returns:
        Dictionary with frontier statistics
    """
    quality_frontier = [c for c in candidates if c.is_pareto_quality]
    perf_frontier = [c for c in candidates if c.is_pareto_performance]
    efficiency_frontier = [c for c in candidates if c.is_pareto_efficiency]

    all_pareto = [
        c for c in candidates
        if c.is_pareto_quality or c.is_pareto_performance or c.is_pareto_efficiency
    ]

    # Find candidates on multiple frontiers (balanced choices)
    multi_frontier = [
        c for c in candidates
        if sum([c.is_pareto_quality, c.is_pareto_performance, c.is_pareto_efficiency]) >= 2
    ]

    return {
        "total_candidates": len(candidates),
        "quality_frontier_count": len(quality_frontier),
        "performance_frontier_count": len(perf_frontier),
        "efficiency_frontier_count": len(efficiency_frontier),
        "any_frontier_count": len(all_pareto),
        "multi_frontier_count": len(multi_frontier),
        "multi_frontier_models": [
            {
                "name": f"{c.model.name} {c.quant.name}",
                "frontiers": [
                    f for f in ["quality", "performance", "efficiency"]
                    if getattr(c, f"is_pareto_{f}")
                ]
            }
            for c in multi_frontier
        ]
    }


def rank_candidates(
    candidates: list[Candidate],
    weights: dict = None
) -> list[Candidate]:
    """
    Rank all candidates by a weighted combination of metrics.

    Default weights balance quality, performance, and efficiency equally.

    Args:
        candidates: List of candidates
        weights: Dictionary with weights for "quality", "performance", "efficiency"

    Returns:
        Candidates sorted by weighted score (best first)
    """
    if weights is None:
        weights = {"quality": 1.0, "performance": 1.0, "efficiency": 1.0}

    # Normalize weights
    total = sum(weights.values())
    norm_weights = {k: v / total for k, v in weights.items()}

    # Find max values for normalization
    max_quality = max(c.quality_score for c in candidates) if candidates else 1
    max_perf = max(c.tokens_per_second for c in candidates) if candidates else 1
    max_eff = max(c.efficiency_score for c in candidates) if candidates else 1

    def weighted_score(c: Candidate) -> float:
        return (
            norm_weights.get("quality", 0) * (c.quality_score / max_quality) +
            norm_weights.get("performance", 0) * (c.tokens_per_second / max_perf) +
            norm_weights.get("efficiency", 0) * (c.efficiency_score / max_eff)
        )

    return sorted(candidates, key=weighted_score, reverse=True)
