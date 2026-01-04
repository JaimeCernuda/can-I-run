"""
Data models for GPU, Model, Quantization, and Candidate configurations.

These models define the structure of the data used throughout the calculations.
All fields are documented for transparency in the "How It Works" panel.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ModelDomain(str, Enum):
    """Use case categories for models."""
    GENERAL = "general"        # General text/chat
    CODE = "code"              # Code generation optimized
    TOOL_CALLING = "tool-calling"  # Function/tool calling optimized
    MATH = "math"              # Math/reasoning focused
    VISION = "vision"          # Multimodal with image understanding
    ROLEPLAY = "roleplay"      # Creative/roleplay focused


class ModelCapability(str, Enum):
    """Special capabilities a model may have."""
    FUNCTION_CALLING = "function_calling"  # Native tool/function support
    JSON_MODE = "json_mode"                # Structured JSON output
    VISION = "vision"                      # Image input
    LONG_CONTEXT = "long_context"          # >32K reliable context
    MULTILINGUAL = "multilingual"          # Strong non-English performance
    REASONING = "reasoning"                # Chain-of-thought / reasoning traces


class QualityTier(str, Enum):
    """Human-readable quality assessment for quantization."""
    NEAR_LOSSLESS = "near_lossless"        # <0.01 PPL increase
    VERY_LOW_LOSS = "very_low_loss"        # 0.01-0.05 PPL increase
    RECOMMENDED = "recommended"            # 0.05-0.10 PPL increase (sweet spot)
    BALANCED = "balanced"                  # 0.10-0.25 PPL increase
    NOTICEABLE_LOSS = "noticeable_loss"    # 0.25-0.50 PPL increase
    HIGH_LOSS = "high_loss"                # 0.50-1.00 PPL increase
    EXTREME_LOSS = "extreme_loss"          # >1.00 PPL increase


@dataclass
class GPU:
    """
    GPU specification for VRAM and performance calculations.

    Attributes:
        name: Display name (e.g., "RTX 4090")
        vendor: GPU vendor ("NVIDIA", "AMD", "Intel", "Apple")
        vram_gb: Total VRAM in gigabytes
        memory_bandwidth_gbps: Memory bandwidth in GB/s (critical for inference speed)
        fp16_tflops: FP16 compute performance in TFLOPS (optional)
        int8_tops: INT8 compute performance in TOPS (optional)
        generation: Architecture generation (e.g., "Ada Lovelace")
        baseline_tps_8b_q4: Measured tokens/sec for LLaMA-3-8B Q4_K_M at 1024 context
    """
    name: str
    vendor: str
    vram_gb: float
    memory_bandwidth_gbps: float
    fp16_tflops: Optional[float] = None
    int8_tops: Optional[float] = None
    generation: Optional[str] = None
    baseline_tps_8b_q4: Optional[float] = None


@dataclass
class Benchmarks:
    """
    Benchmark scores for a model.

    All scores are on a 0-100 scale where higher is better.
    None indicates the benchmark was not run for this model.
    """
    mmlu: Optional[float] = None          # Massive Multitask Language Understanding
    humaneval: Optional[float] = None     # Code generation benchmark
    gsm8k: Optional[float] = None         # Grade school math
    bfcl: Optional[float] = None          # Berkeley Function Calling Leaderboard
    tool_accuracy: Optional[float] = None # Tool/function calling accuracy


@dataclass
class Model:
    """
    LLM model specification.

    Attributes:
        name: Model identifier (e.g., "Llama-3.1-70B-Instruct")
        total_params_b: Total parameters in billions (for VRAM calculation)
        active_params_b: Active parameters per token (same as total for dense models)
        is_moe: Whether this is a Mixture of Experts model
        hidden_dim: Hidden dimension size
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        num_kv_heads: Number of key-value heads (may be < num_heads due to GQA)
        vocab_size: Vocabulary size
        max_context_length: Maximum supported context length
        effective_context_length: Tested reliable context length
        domains: List of use case categories
        capabilities: List of special capabilities
        benchmarks: Benchmark scores
        num_experts: Number of experts (for MoE models)
        num_active_experts: Number of active experts per token (for MoE models)
        notes: Additional notes (e.g., warnings about quantization sensitivity)
    """
    name: str
    total_params_b: float
    active_params_b: float
    is_moe: bool
    hidden_dim: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    vocab_size: int
    max_context_length: int
    effective_context_length: int
    domains: list[ModelDomain] = field(default_factory=list)
    capabilities: list[ModelCapability] = field(default_factory=list)
    benchmarks: Benchmarks = field(default_factory=Benchmarks)
    num_experts: Optional[int] = None
    num_active_experts: Optional[int] = None
    notes: Optional[str] = None


@dataclass
class Quantization:
    """
    Quantization specification with quality metrics.

    Quality factors are derived from empirical perplexity measurements,
    not arbitrary estimates. See data sources in documentation.

    Attributes:
        name: Quantization identifier (e.g., "Q4_K_M")
        bits_per_weight: Effective bits per weight (accounts for mixed precision)
        quality_factor: Quality preservation factor (0.0-1.0) relative to FP16
        ppl_increase: Absolute perplexity increase vs FP16
        quality_tier: Human-readable quality assessment
        source: Citation for the data source
    """
    name: str
    bits_per_weight: float
    quality_factor: float
    ppl_increase: float
    quality_tier: QualityTier
    source: str


@dataclass
class Candidate:
    """
    A model+quantization combination evaluated for a specific GPU configuration.

    This represents one point on the Pareto charts.

    Attributes:
        model: The model specification
        quant: The quantization specification
        vram_required: Total VRAM required (model + KV cache + overhead) in GB
        vram_headroom: Available VRAM minus required VRAM in GB
        quality_score: Composite quality score (0-100)
        tokens_per_second: Estimated inference speed
        efficiency_score: Combined quality*performance/VRAM metric
        is_pareto_quality: Whether this is on the quality Pareto frontier
        is_pareto_performance: Whether this is on the performance Pareto frontier
        is_pareto_efficiency: Whether this is on the efficiency Pareto frontier
    """
    model: Model
    quant: Quantization
    vram_required: float
    vram_headroom: float
    quality_score: float
    tokens_per_second: float
    efficiency_score: float
    is_pareto_quality: bool = False
    is_pareto_performance: bool = False
    is_pareto_efficiency: bool = False
