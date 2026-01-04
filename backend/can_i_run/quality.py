"""
Quality Score Calculation Functions

Calculates composite quality scores based on benchmark performance
and quantization degradation.

Formula:
    Quality Score = Base Model Quality × Quantization Factor × Size Adjustment

Where:
    Base Model Quality = Weighted benchmark scores (0-100 scale)
    Quantization Factor = How much quality is preserved (0.0-1.0)
    Size Adjustment = Larger models tolerate quantization better

Domain-specific weighting:
    - General: MMLU × 0.5 + HumanEval × 0.25 + GSM8K × 0.25
    - Code: HumanEval × 0.6 + MMLU × 0.2 + GSM8K × 0.2
    - Tool-calling: BFCL × 0.5 + MMLU × 0.3 + HumanEval × 0.2
    - Math: GSM8K × 0.5 + MMLU × 0.3 + HumanEval × 0.2

Note on quantization factors:
    These are derived from empirical perplexity measurements, not
    arbitrary estimates. Larger models (70B+) tolerate quantization
    better than smaller models (7-8B).

Data Sources:
    - llama.cpp quantize README
    - Intel Low-bit Leaderboard
    - Unsloth Dynamic 2.0 benchmarks
"""

from .models import Model, Quantization, ModelDomain, Benchmarks


# Domain-specific benchmark weights
DOMAIN_WEIGHTS = {
    ModelDomain.GENERAL: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
    ModelDomain.CODE: {"mmlu": 0.2, "humaneval": 0.6, "gsm8k": 0.2, "bfcl": 0.0},
    ModelDomain.TOOL_CALLING: {"mmlu": 0.3, "humaneval": 0.2, "gsm8k": 0.0, "bfcl": 0.5},
    ModelDomain.MATH: {"mmlu": 0.3, "humaneval": 0.2, "gsm8k": 0.5, "bfcl": 0.0},
    ModelDomain.VISION: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
    ModelDomain.ROLEPLAY: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
}


# Size-based quantization penalty factors
# Larger models tolerate quantization better (from Intel Low-bit Leaderboard)
SIZE_PENALTIES = {
    "small": 0.85,    # <10B: significant extra degradation
    "medium": 0.92,   # 10-30B: moderate extra degradation
    "large": 0.97,    # 30-65B: minor extra degradation
    "xlarge": 1.00,   # >65B: minimal extra degradation
}


def get_size_category(params_b: float) -> str:
    """
    Categorize model size for quantization penalty calculation.

    Args:
        params_b: Model parameters in billions

    Returns:
        Size category string
    """
    if params_b < 10:
        return "small"
    elif params_b < 30:
        return "medium"
    elif params_b < 65:
        return "large"
    else:
        return "xlarge"


def adjusted_quality_factor(
    base_quality: float,
    model_size_b: float,
    quant: Quantization
) -> float:
    """
    Apply size-adjusted quantization penalty to base quality.

    Larger models (70B+) tolerate quantization better than smaller
    models (7-8B). This adjustment is based on findings from the
    Intel Low-bit Leaderboard.

    Args:
        base_quality: Unadjusted quality score
        model_size_b: Model size in billions of parameters
        quant: Quantization specification

    Returns:
        Quality score adjusted for quantization and model size
    """
    size_category = get_size_category(model_size_b)
    size_penalty = SIZE_PENALTIES[size_category]

    # Apply both quantization factor and size penalty
    return base_quality * quant.quality_factor * size_penalty


def calculate_base_quality(
    benchmarks: Benchmarks,
    domain: ModelDomain = ModelDomain.GENERAL,
    default_score: float = 50.0
) -> float:
    """
    Calculate base quality score from benchmarks with domain weighting.

    Missing benchmarks are filled with the default score.

    Args:
        benchmarks: Model benchmark scores
        domain: Use case domain for weighting
        default_score: Score to use for missing benchmarks

    Returns:
        Weighted quality score (0-100)
    """
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[ModelDomain.GENERAL])

    # Get benchmark values, using default for missing
    mmlu = benchmarks.mmlu if benchmarks.mmlu is not None else default_score
    humaneval = benchmarks.humaneval if benchmarks.humaneval is not None else default_score
    gsm8k = benchmarks.gsm8k if benchmarks.gsm8k is not None else default_score
    bfcl = benchmarks.bfcl if benchmarks.bfcl is not None else default_score

    # Calculate weighted sum
    score = (
        mmlu * weights.get("mmlu", 0) +
        humaneval * weights.get("humaneval", 0) +
        gsm8k * weights.get("gsm8k", 0) +
        bfcl * weights.get("bfcl", 0)
    )

    return score


def calculate_quality_score(
    model: Model,
    quant: Quantization,
    domain: ModelDomain = ModelDomain.GENERAL
) -> float:
    """
    Calculate final quality score for a model+quantization combination.

    This combines:
    1. Domain-weighted benchmark scores
    2. Quantization quality factor
    3. Size-based adjustment (larger models tolerate quant better)

    Args:
        model: Model specification with benchmarks
        quant: Quantization specification with quality factor
        domain: Use case domain for benchmark weighting

    Returns:
        Final quality score (0-100)
    """
    base = calculate_base_quality(model.benchmarks, domain)
    adjusted = adjusted_quality_factor(base, model.total_params_b, quant)
    return round(adjusted, 2)


def get_quality_breakdown(
    model: Model,
    quant: Quantization,
    domain: ModelDomain = ModelDomain.GENERAL
) -> dict:
    """
    Get detailed quality score breakdown for the "How It Works" panel.

    Args:
        model: Model specification
        quant: Quantization specification
        domain: Use case domain

    Returns:
        Dictionary with quality calculation details
    """
    base = calculate_base_quality(model.benchmarks, domain)
    size_category = get_size_category(model.total_params_b)
    size_penalty = SIZE_PENALTIES[size_category]
    final = calculate_quality_score(model, quant, domain)

    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[ModelDomain.GENERAL])

    return {
        "model_name": model.name,
        "quantization": quant.name,
        "domain": domain.value,
        "weights": weights,
        "benchmarks": {
            "mmlu": model.benchmarks.mmlu,
            "humaneval": model.benchmarks.humaneval,
            "gsm8k": model.benchmarks.gsm8k,
            "bfcl": model.benchmarks.bfcl,
        },
        "base_quality": round(base, 2),
        "quant_factor": quant.quality_factor,
        "quant_tier": quant.quality_tier.value,
        "size_category": size_category,
        "size_penalty": size_penalty,
        "final_quality": final,
        "formula": f"{base:.2f} × {quant.quality_factor:.3f} × {size_penalty:.2f} = {final:.2f}"
    }
