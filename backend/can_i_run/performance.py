"""
Performance Estimation Functions

Estimates inference speed (tokens per second) based on GPU specifications
and model size.

Theory:
    Token generation speed is primarily bounded by memory bandwidth for
    inference (compute-bound only at very small batch sizes or during prefill).

    For autoregressive generation, each token requires reading ALL model
    weights from VRAM once. This makes memory bandwidth the primary constraint.

Formula:
    Tokens/sec ≈ Memory Bandwidth (GB/s) ÷ Model Size (GB) × Efficiency

    Where efficiency is typically 60-80% of theoretical maximum due to:
    - Memory access patterns
    - Dequantization overhead
    - Software stack overhead

Example: RTX 4090 + Llama-70B Q4_K_M
    Bandwidth = 1,008 GB/s
    Model = 31.6 GB
    Theoretical = 1,008 ÷ 31.6 = 31.9 tok/s
    With ~70% efficiency = 22 tok/s

MoE models:
    Memory reads all experts (total params) but computes only active params.
    Result: similar speed to dense models of same total size.

Data Sources:
    - XiongjieDai/GPU-Benchmarks-on-LLM-Inference (GitHub)
    - llama.cpp official benchmarks
"""

from typing import Optional
from .models import Model, Quantization, GPU
from .vram import calculate_model_vram


# Efficiency factors by quantization level
# Lower bit quantization has more dequantization overhead
QUANT_EFFICIENCY = {
    "high": 0.75,    # FP16, Q8
    "medium": 0.70,  # Q4-Q6
    "low": 0.60,     # Q2-Q3
}


# Baseline tokens/sec for LLaMA-3-8B Q4_K_M at 1024 context
# Used to scale estimates for GPUs with measured data
BASELINE_MODEL_SIZE_GB = 4.58  # 8B at Q4_K_M


def get_efficiency_factor(bits_per_weight: float) -> float:
    """
    Get efficiency factor based on quantization level.

    Higher bit quantization is more efficient due to less dequantization overhead.

    Args:
        bits_per_weight: Effective bits per weight

    Returns:
        Efficiency factor (0.0-1.0)
    """
    if bits_per_weight >= 8:
        return QUANT_EFFICIENCY["high"]
    elif bits_per_weight >= 4:
        return QUANT_EFFICIENCY["medium"]
    else:
        return QUANT_EFFICIENCY["low"]


def estimate_tokens_per_second_theoretical(
    model: Model,
    quant: Quantization,
    gpu: GPU
) -> float:
    """
    Estimate tokens/sec using theoretical bandwidth calculation.

    This is a fallback when no measured performance data is available.

    Args:
        model: Model specification
        quant: Quantization specification
        gpu: GPU specification with bandwidth

    Returns:
        Estimated tokens per second
    """
    model_size_gb = calculate_model_vram(model, quant)

    # Calculate theoretical maximum from bandwidth
    theoretical_max = gpu.memory_bandwidth_gbps / model_size_gb

    # Apply efficiency factor
    efficiency = get_efficiency_factor(quant.bits_per_weight)

    return theoretical_max * efficiency


def estimate_tokens_per_second_scaled(
    model: Model,
    quant: Quantization,
    gpu: GPU
) -> Optional[float]:
    """
    Estimate tokens/sec by scaling from measured baseline performance.

    If the GPU has measured baseline performance (baseline_tps_8b_q4),
    we scale that measurement based on model size. This is more accurate
    than pure theoretical calculation.

    Memory-bound scaling: tok/s scales inversely with model size.

    Args:
        model: Model specification
        quant: Quantization specification
        gpu: GPU specification with baseline performance

    Returns:
        Estimated tokens per second, or None if no baseline available
    """
    if gpu.baseline_tps_8b_q4 is None:
        return None

    model_size_gb = calculate_model_vram(model, quant)

    # Scale inversely with model size
    scale_factor = BASELINE_MODEL_SIZE_GB / model_size_gb

    return gpu.baseline_tps_8b_q4 * scale_factor


def estimate_tokens_per_second(
    model: Model,
    quant: Quantization,
    gpu: GPU
) -> float:
    """
    Estimate tokens per second for a model+quantization+GPU combination.

    Uses measured baseline data when available, falls back to theoretical
    bandwidth calculation otherwise.

    Args:
        model: Model specification
        quant: Quantization specification
        gpu: GPU specification

    Returns:
        Estimated tokens per second
    """
    # Try scaled estimate first (more accurate if baseline data available)
    scaled = estimate_tokens_per_second_scaled(model, quant, gpu)
    if scaled is not None:
        return round(scaled, 1)

    # Fall back to theoretical calculation
    theoretical = estimate_tokens_per_second_theoretical(model, quant, gpu)
    return round(theoretical, 1)


def estimate_moe_tokens_per_second(
    model: Model,
    quant: Quantization,
    gpu: GPU
) -> float:
    """
    Estimate tokens/sec specifically for MoE models.

    MoE models have a subtle performance characteristic:
    - Memory: reads ALL parameters (all experts must be in VRAM)
    - Compute: only processes ACTIVE parameters per token

    In practice, MoE models are memory-bound like dense models,
    so performance is similar to a dense model of the same total size.

    Args:
        model: MoE model specification
        quant: Quantization specification
        gpu: GPU specification

    Returns:
        Estimated tokens per second
    """
    # For memory-bound inference, use total params (all experts read)
    # Apply slightly lower efficiency due to expert routing overhead
    model_size_gb = calculate_model_vram(model, quant)

    if gpu.baseline_tps_8b_q4 is not None:
        scale_factor = BASELINE_MODEL_SIZE_GB / model_size_gb
        return round(gpu.baseline_tps_8b_q4 * scale_factor * 0.9, 1)  # 90% MoE efficiency

    theoretical = estimate_tokens_per_second_theoretical(model, quant, gpu)
    return round(theoretical * 0.9, 1)


def get_performance_breakdown(
    model: Model,
    quant: Quantization,
    gpu: GPU
) -> dict:
    """
    Get detailed performance estimation breakdown for "How It Works" panel.

    Args:
        model: Model specification
        quant: Quantization specification
        gpu: GPU specification

    Returns:
        Dictionary with performance calculation details
    """
    model_size_gb = calculate_model_vram(model, quant)
    theoretical_max = gpu.memory_bandwidth_gbps / model_size_gb
    efficiency = get_efficiency_factor(quant.bits_per_weight)

    estimated = (
        estimate_moe_tokens_per_second(model, quant, gpu)
        if model.is_moe
        else estimate_tokens_per_second(model, quant, gpu)
    )

    breakdown = {
        "model_name": model.name,
        "quantization": quant.name,
        "gpu_name": gpu.name,
        "model_size_gb": round(model_size_gb, 2),
        "gpu_bandwidth_gbps": gpu.memory_bandwidth_gbps,
        "theoretical_max_tps": round(theoretical_max, 1),
        "efficiency_factor": efficiency,
        "is_moe": model.is_moe,
        "estimated_tps": estimated,
        "estimation_method": "scaled" if gpu.baseline_tps_8b_q4 else "theoretical",
    }

    if gpu.baseline_tps_8b_q4:
        breakdown["baseline_tps"] = gpu.baseline_tps_8b_q4
        breakdown["scale_factor"] = round(BASELINE_MODEL_SIZE_GB / model_size_gb, 3)

    if model.is_moe:
        breakdown["total_params_b"] = model.total_params_b
        breakdown["active_params_b"] = model.active_params_b
        breakdown["moe_note"] = "MoE: reads all experts, computes only active ones"

    breakdown["formula"] = (
        f"({gpu.memory_bandwidth_gbps} GB/s ÷ {model_size_gb:.1f} GB) × {efficiency} = "
        f"{round(theoretical_max * efficiency, 1)} tok/s"
    )

    return breakdown
