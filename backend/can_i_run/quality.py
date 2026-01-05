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

Scale Normalization (for newer benchmarks):
    MMLU-PRO and MATH are significantly harder than MMLU and GSM8K.
    When original benchmarks are not available, we use scaled approximations:
    - MMLU ≈ min(100, MMLU-PRO × 1.8)  (MMLU-PRO scores ~40-60% of MMLU)
    - GSM8K ≈ min(100, MATH × 3.5)     (MATH Lvl5 scores ~20-40% of GSM8K)

Data Sources:
    Benchmark scores:
        - MMLU, GSM8K: Open LLM Leaderboard v1 (archived June 2024)
        - MMLU-PRO, MATH: Open LLM Leaderboard v2 (current)
        - HumanEval: EvalPlus Leaderboard (https://evalplus.github.io/leaderboard.html)
        - BFCL: Berkeley Function Calling Leaderboard (https://gorilla.cs.berkeley.edu/leaderboard.html)
    Quantization quality:
        - llama.cpp quantize tool (https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
    Size-based degradation:
        - Intel Low-bit Quantized Open LLM Leaderboard (https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)
"""

from .models import Model, Quantization, ModelDomain, Benchmarks


# Scale factors for converting newer benchmarks to original benchmark scale
# Empirically derived from Llama-3.1-8B and Llama-3.1-70B model cards:
#   8B:  MMLU(73)/MMLU-PRO(48.3)=1.51, GSM8K(84.5)/MATH(51.9)=1.63
#   70B: MMLU(86)/MMLU-PRO(66.4)=1.30, GSM8K(95.1)/MATH(68.0)=1.40
# Note: Larger models have smaller gaps due to ceiling effects on easier benchmarks
MMLU_PRO_TO_MMLU_SCALE = 1.4  # Average ratio from empirical data
GSM8K_MATH_SCALE = 1.5  # Average ratio from empirical data


# Domain-specific benchmark weights
DOMAIN_WEIGHTS = {
    ModelDomain.GENERAL: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
    ModelDomain.CODE: {"mmlu": 0.2, "humaneval": 0.6, "gsm8k": 0.2, "bfcl": 0.0},
    ModelDomain.TOOL_CALLING: {
        "mmlu": 0.3,
        "humaneval": 0.2,
        "gsm8k": 0.0,
        "bfcl": 0.5,
    },
    ModelDomain.MATH: {"mmlu": 0.3, "humaneval": 0.2, "gsm8k": 0.5, "bfcl": 0.0},
    ModelDomain.VISION: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
    ModelDomain.ROLEPLAY: {"mmlu": 0.5, "humaneval": 0.25, "gsm8k": 0.25, "bfcl": 0.0},
}


# Size-based quantization penalty factors
# Larger models tolerate quantization better (from Intel Low-bit Leaderboard)
SIZE_PENALTIES = {
    "small": 0.85,  # <10B: significant extra degradation
    "medium": 0.92,  # 10-30B: moderate extra degradation
    "large": 0.97,  # 30-65B: minor extra degradation
    "xlarge": 1.00,  # >65B: minimal extra degradation
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
    base_quality: float, model_size_b: float, quant: Quantization
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


def get_effective_mmlu(benchmarks: Benchmarks, default_score: float = 50.0) -> float:
    """
    Get effective MMLU score, using scaled MMLU-PRO as fallback.

    Args:
        benchmarks: Model benchmark scores
        default_score: Score to use if no benchmark available

    Returns:
        Effective MMLU score (0-100)
    """
    if benchmarks.mmlu is not None:
        return benchmarks.mmlu
    if benchmarks.mmlu_pro is not None:
        # Scale MMLU-PRO to approximate MMLU (cap at 100)
        return min(100.0, benchmarks.mmlu_pro * MMLU_PRO_TO_MMLU_SCALE)
    return default_score


def get_effective_gsm8k(benchmarks: Benchmarks, default_score: float = 50.0) -> float:
    """
    Get effective GSM8K score, using scaled MATH as fallback.

    Args:
        benchmarks: Model benchmark scores
        default_score: Score to use if no benchmark available

    Returns:
        Effective GSM8K score (0-100)
    """
    if benchmarks.gsm8k is not None:
        return benchmarks.gsm8k
    if benchmarks.math is not None:
        # Scale MATH Lvl5 to approximate GSM8K (cap at 100)
        return min(100.0, benchmarks.math * GSM8K_MATH_SCALE)
    return default_score


def calculate_base_quality(
    benchmarks: Benchmarks,
    domain: ModelDomain = ModelDomain.GENERAL,
    default_score: float = 50.0,
) -> float:
    """
    Calculate base quality score from benchmarks with domain weighting.

    Uses scale normalization to convert newer benchmarks (MMLU-PRO, MATH)
    to original benchmark scale (MMLU, GSM8K) when originals are unavailable.

    Args:
        benchmarks: Model benchmark scores
        domain: Use case domain for weighting
        default_score: Score to use for missing benchmarks

    Returns:
        Weighted quality score (0-100)
    """
    weights = DOMAIN_WEIGHTS.get(domain, DOMAIN_WEIGHTS[ModelDomain.GENERAL])

    # Get benchmark values with scale normalization fallbacks
    mmlu = get_effective_mmlu(benchmarks, default_score)
    humaneval = (
        benchmarks.humaneval if benchmarks.humaneval is not None else default_score
    )
    gsm8k = get_effective_gsm8k(benchmarks, default_score)
    bfcl = benchmarks.bfcl if benchmarks.bfcl is not None else default_score

    # Calculate weighted sum
    score = (
        mmlu * weights.get("mmlu", 0)
        + humaneval * weights.get("humaneval", 0)
        + gsm8k * weights.get("gsm8k", 0)
        + bfcl * weights.get("bfcl", 0)
    )

    return score


def calculate_quality_score(
    model: Model, quant: Quantization, domain: ModelDomain = ModelDomain.GENERAL
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
    model: Model, quant: Quantization, domain: ModelDomain = ModelDomain.GENERAL
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

    # Get effective values with scale normalization info
    effective_mmlu = get_effective_mmlu(model.benchmarks)
    effective_gsm8k = get_effective_gsm8k(model.benchmarks)
    mmlu_source = (
        "mmlu"
        if model.benchmarks.mmlu is not None
        else (
            "mmlu_pro (scaled ×1.8)"
            if model.benchmarks.mmlu_pro is not None
            else "default"
        )
    )
    gsm8k_source = (
        "gsm8k"
        if model.benchmarks.gsm8k is not None
        else ("math (scaled ×3.5)" if model.benchmarks.math is not None else "default")
    )

    return {
        "model_name": model.name,
        "quantization": quant.name,
        "domain": domain.value,
        "weights": weights,
        "benchmarks": {
            "mmlu": model.benchmarks.mmlu,
            "mmlu_pro": model.benchmarks.mmlu_pro,
            "humaneval": model.benchmarks.humaneval,
            "gsm8k": model.benchmarks.gsm8k,
            "math": model.benchmarks.math,
            "bfcl": model.benchmarks.bfcl,
        },
        "effective_benchmarks": {
            "mmlu": round(effective_mmlu, 2),
            "mmlu_source": mmlu_source,
            "humaneval": model.benchmarks.humaneval,
            "gsm8k": round(effective_gsm8k, 2),
            "gsm8k_source": gsm8k_source,
            "bfcl": model.benchmarks.bfcl,
        },
        "base_quality": round(base, 2),
        "quant_factor": quant.quality_factor,
        "quant_tier": quant.quality_tier.value,
        "size_category": size_category,
        "size_penalty": size_penalty,
        "final_quality": final,
        "formula": f"{base:.2f} × {quant.quality_factor:.3f} × {size_penalty:.2f} = {final:.2f}",
    }
