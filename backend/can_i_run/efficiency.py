"""
Efficiency Metric Calculation Functions

The efficiency score helps users find the "best bang for VRAM buck" by
combining quality, performance, and VRAM usage into a single metric.

Formula:
    Efficiency = (Quality × Performance) ÷ VRAM

More precisely:
    1. Normalize quality to 0-1 range (assuming max ~90 for top models)
    2. Normalize performance (assuming max ~150 tok/s for small models on fast GPUs)
    3. Efficiency = geometric_mean(normalized_quality, normalized_perf) ÷ VRAM × 100

Interpretation:
    - High efficiency = good quality AND speed for the VRAM used
    - An 8B model at Q4 might have HIGHER efficiency than a 70B at Q3,
      even though 70B has better raw quality
    - Helps users on limited VRAM find optimal tradeoffs

Use Cases:
    - "I have 12GB VRAM, what gives me the best overall experience?"
    - "I need fast responses AND good quality—what's the sweet spot?"
    - "Is it worth the extra VRAM to go from 8B to 13B?"
"""

import math
from typing import Optional


# Normalization constants
MAX_QUALITY_SCORE = 90.0    # Approximate max for best models
MAX_TOKENS_PER_SEC = 150.0  # Approximate max for small models on fast GPUs


def normalize_quality(quality_score: float) -> float:
    """
    Normalize quality score to 0-1 range.

    Args:
        quality_score: Quality score (0-100 scale)

    Returns:
        Normalized quality (0-1)
    """
    return min(quality_score / MAX_QUALITY_SCORE, 1.0)


def normalize_performance(tokens_per_second: float) -> float:
    """
    Normalize performance to 0-1 range.

    Args:
        tokens_per_second: Estimated tokens per second

    Returns:
        Normalized performance (0-1)
    """
    return min(tokens_per_second / MAX_TOKENS_PER_SEC, 1.0)


def calculate_efficiency(
    quality_score: float,
    tokens_per_second: float,
    vram_required: float
) -> float:
    """
    Calculate the efficiency score for a model configuration.

    Efficiency = geometric_mean(quality, performance) ÷ VRAM × 100

    Using geometric mean ensures that both quality and performance
    must be reasonably good for high efficiency—a model that's fast
    but low quality won't score well.

    Args:
        quality_score: Quality score (0-100 scale)
        tokens_per_second: Estimated tokens per second
        vram_required: Total VRAM required in GB

    Returns:
        Efficiency score (higher is better)
    """
    if vram_required <= 0:
        return 0.0

    # Normalize both metrics
    norm_quality = normalize_quality(quality_score)
    norm_perf = normalize_performance(tokens_per_second)

    # Geometric mean of normalized scores
    combined_score = math.sqrt(norm_quality * norm_perf)

    # Divide by VRAM and scale
    efficiency = (combined_score / vram_required) * 100

    return round(efficiency, 2)


def calculate_efficiency_percentile(
    efficiency: float,
    all_efficiencies: list[float]
) -> float:
    """
    Calculate the percentile rank of an efficiency score.

    Args:
        efficiency: The efficiency score to rank
        all_efficiencies: List of all efficiency scores for comparison

    Returns:
        Percentile rank (0-100)
    """
    if not all_efficiencies:
        return 50.0

    count_below = sum(1 for e in all_efficiencies if e < efficiency)
    percentile = (count_below / len(all_efficiencies)) * 100

    return round(percentile, 1)


def get_efficiency_breakdown(
    quality_score: float,
    tokens_per_second: float,
    vram_required: float,
    model_name: str,
    quant_name: str
) -> dict:
    """
    Get detailed efficiency calculation breakdown for "How It Works" panel.

    Args:
        quality_score: Quality score (0-100)
        tokens_per_second: Estimated tokens per second
        vram_required: Total VRAM required in GB
        model_name: Name of the model
        quant_name: Name of the quantization

    Returns:
        Dictionary with efficiency calculation details
    """
    norm_quality = normalize_quality(quality_score)
    norm_perf = normalize_performance(tokens_per_second)
    combined = math.sqrt(norm_quality * norm_perf)
    efficiency = calculate_efficiency(quality_score, tokens_per_second, vram_required)

    return {
        "model_name": model_name,
        "quantization": quant_name,
        "quality_score": round(quality_score, 2),
        "tokens_per_second": round(tokens_per_second, 1),
        "vram_required_gb": round(vram_required, 2),
        "normalized_quality": round(norm_quality, 3),
        "normalized_performance": round(norm_perf, 3),
        "geometric_mean": round(combined, 3),
        "efficiency_score": efficiency,
        "formula": (
            f"sqrt({norm_quality:.3f} × {norm_perf:.3f}) ÷ {vram_required:.1f} × 100 = {efficiency:.2f}"
        ),
        "interpretation": get_efficiency_interpretation(efficiency)
    }


def get_efficiency_interpretation(efficiency: float) -> str:
    """
    Get a human-readable interpretation of the efficiency score.

    Args:
        efficiency: Efficiency score

    Returns:
        Interpretation string
    """
    if efficiency >= 15:
        return "Excellent efficiency - great bang for VRAM"
    elif efficiency >= 10:
        return "Good efficiency - balanced choice"
    elif efficiency >= 5:
        return "Moderate efficiency - acceptable tradeoff"
    elif efficiency >= 2:
        return "Low efficiency - consider smaller models"
    else:
        return "Poor efficiency - not recommended"


def compare_efficiency(
    candidates: list[dict]
) -> list[dict]:
    """
    Compare efficiency across multiple candidates and add rankings.

    Args:
        candidates: List of candidates with efficiency scores

    Returns:
        Candidates with added rank and percentile fields
    """
    # Sort by efficiency descending
    sorted_candidates = sorted(
        candidates,
        key=lambda x: x.get("efficiency_score", 0),
        reverse=True
    )

    all_efficiencies = [c.get("efficiency_score", 0) for c in candidates]

    for i, candidate in enumerate(sorted_candidates):
        candidate["efficiency_rank"] = i + 1
        candidate["efficiency_percentile"] = calculate_efficiency_percentile(
            candidate.get("efficiency_score", 0),
            all_efficiencies
        )

    return sorted_candidates
