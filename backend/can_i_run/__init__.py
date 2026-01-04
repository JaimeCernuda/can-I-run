"""
GPU-to-Model Pareto Selector

A tool for calculating optimal LLM model/quantization combinations
based on available GPU VRAM, with quality vs performance tradeoffs.
"""

from .models import GPU, Model, Quantization, Candidate, ModelDomain, ModelCapability
from .vram import calculate_model_vram, calculate_total_vram
from .kv_cache import calculate_kv_cache
from .quality import calculate_quality_score, adjusted_quality_factor
from .performance import estimate_tokens_per_second
from .efficiency import calculate_efficiency
from .pareto import compute_pareto_frontier

__version__ = "0.1.0"

__all__ = [
    "GPU",
    "Model",
    "Quantization",
    "Candidate",
    "ModelDomain",
    "ModelCapability",
    "calculate_model_vram",
    "calculate_total_vram",
    "calculate_kv_cache",
    "calculate_quality_score",
    "adjusted_quality_factor",
    "estimate_tokens_per_second",
    "calculate_efficiency",
    "compute_pareto_frontier",
]
