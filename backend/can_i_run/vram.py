"""
VRAM Calculation Functions

Calculates the VRAM required for model weights based on model size and quantization.

Formula:
    Model VRAM (GB) = Parameters × Bits per Weight ÷ 8 ÷ 1,073,741,824

For MoE models:
    Use TOTAL parameters (all experts must be in VRAM), even though only
    a subset are "active" per token.

Example:
    Llama-3.1-70B at Q4_K_M:
    = 70,600,000,000 params × 4.8 bits ÷ 8 ÷ 1,073,741,824
    = 31.6 GB
"""

from .models import Model, Quantization


# Fixed overhead for CUDA/driver in GB
CUDA_OVERHEAD_GB = 0.5


def calculate_model_vram(model: Model, quant: Quantization) -> float:
    """
    Calculate the VRAM required for model weights.

    For MoE models, ALL experts must fit in VRAM (not just active ones).
    This is because inference requires access to all experts, even if only
    a subset are computed per token.

    Args:
        model: Model specification with parameter count
        quant: Quantization specification with bits per weight

    Returns:
        VRAM required for model weights in GB
    """
    # Use total params for VRAM calculation (all experts for MoE)
    params = model.total_params_b * 1e9
    bytes_required = params * (quant.bits_per_weight / 8)
    gb_required = bytes_required / (1024 ** 3)
    return gb_required


def calculate_total_vram(
    model: Model,
    quant: Quantization,
    kv_cache_gb: float,
    overhead_gb: float = CUDA_OVERHEAD_GB
) -> float:
    """
    Calculate total VRAM required for inference.

    Total = Model Weights + KV Cache + CUDA/Driver Overhead

    Args:
        model: Model specification
        quant: Quantization specification
        kv_cache_gb: KV cache size in GB (from calculate_kv_cache)
        overhead_gb: Fixed overhead for CUDA/drivers (default 0.5 GB)

    Returns:
        Total VRAM required in GB
    """
    model_vram = calculate_model_vram(model, quant)
    return model_vram + kv_cache_gb + overhead_gb


def calculate_vram_headroom(
    total_vram_available: float,
    vram_required: float
) -> float:
    """
    Calculate remaining VRAM headroom after loading model.

    Headroom = Available VRAM - Required VRAM

    A positive value indicates the model will fit with room to spare.
    A negative value indicates the model won't fit.

    Recommended minimum headroom: 1 GB (to avoid OOM during inference)

    Args:
        total_vram_available: GPU's total VRAM in GB
        vram_required: Total required VRAM in GB

    Returns:
        VRAM headroom in GB (negative if model won't fit)
    """
    return total_vram_available - vram_required


def get_vram_breakdown(
    model: Model,
    quant: Quantization,
    kv_cache_gb: float,
    total_vram_available: float,
    overhead_gb: float = CUDA_OVERHEAD_GB
) -> dict:
    """
    Get a detailed breakdown of VRAM usage for display.

    Returns a dictionary with all components for the "How It Works" panel.

    Args:
        model: Model specification
        quant: Quantization specification
        kv_cache_gb: KV cache size in GB
        total_vram_available: GPU's total VRAM in GB
        overhead_gb: Fixed overhead for CUDA/drivers

    Returns:
        Dictionary with VRAM breakdown details
    """
    model_vram = calculate_model_vram(model, quant)
    total_required = model_vram + kv_cache_gb + overhead_gb
    headroom = total_vram_available - total_required

    return {
        "total_available_gb": total_vram_available,
        "model_weights_gb": round(model_vram, 2),
        "kv_cache_gb": round(kv_cache_gb, 2),
        "overhead_gb": overhead_gb,
        "total_required_gb": round(total_required, 2),
        "headroom_gb": round(headroom, 2),
        "headroom_percent": round((headroom / total_vram_available) * 100, 1) if total_vram_available > 0 else 0,
        "fits": headroom >= 0,
        "safe": headroom >= 1.0,  # 1 GB minimum recommended headroom
    }
