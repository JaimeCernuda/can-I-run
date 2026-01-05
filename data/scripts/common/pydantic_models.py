"""
Pydantic validation models for data update scripts.

These schemas validate the structure and values of GPU, Model, and Quantization
data before writing to JSON files.
"""

from datetime import date
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QualityTier(str, Enum):
    """Human-readable quality assessment for quantization."""

    NEAR_LOSSLESS = "near_lossless"
    VERY_LOW_LOSS = "very_low_loss"
    RECOMMENDED = "recommended"
    BALANCED = "balanced"
    NOTICEABLE_LOSS = "noticeable_loss"
    HIGH_LOSS = "high_loss"
    EXTREME_LOSS = "extreme_loss"


class GPUSchema(BaseModel):
    """Validated GPU specification."""

    name: str = Field(..., min_length=1)
    vendor: Literal["NVIDIA", "AMD", "Intel", "Apple"]
    vram_gb: float = Field(..., gt=0)
    memory_bandwidth_gbps: float = Field(..., gt=0)
    fp16_tflops: Optional[float] = Field(default=None, ge=0)
    generation: Optional[str] = None
    release_date: Optional[date] = None

    @field_validator("vram_gb", mode="before")
    @classmethod
    def validate_vram(cls, v: float) -> float:
        """Convert MB to GB if value seems too large."""
        if v > 500:  # Likely in MB, convert to GB
            return v / 1024
        return v

    @field_validator("memory_bandwidth_gbps", mode="before")
    @classmethod
    def validate_bandwidth(cls, v: float) -> float:
        """Ensure bandwidth is a reasonable value."""
        if v <= 0:
            raise ValueError("Memory bandwidth must be positive")
        return v


class BenchmarksSchema(BaseModel):
    """
    Validated benchmark scores.

    Data Sources:
        - MMLU-PRO, MATH: Open LLM Leaderboard (v2)
        - HumanEval: EvalPlus Leaderboard
        - BFCL: Berkeley Function Calling Leaderboard

    Note: MMLU-PRO is a harder version of MMLU, and MATH Lvl 5 is harder than GSM8K.
    Scores are typically lower than the original benchmarks.
    """

    mmlu: Optional[float] = Field(
        default=None, ge=0, le=100
    )  # Original MMLU (deprecated)
    mmlu_pro: Optional[float] = Field(default=None, ge=0, le=100)  # MMLU-PRO (harder)
    humaneval: Optional[float] = Field(default=None, ge=0, le=100)
    gsm8k: Optional[float] = Field(
        default=None, ge=0, le=100
    )  # Original GSM8K (deprecated)
    math: Optional[float] = Field(default=None, ge=0, le=100)  # MATH Lvl 5 (harder)
    bfcl: Optional[float] = Field(default=None, ge=0, le=100)


class ModelSchema(BaseModel):
    """Validated model specification."""

    name: str = Field(..., min_length=1)
    total_params_b: float = Field(..., gt=0)
    active_params_b: float = Field(..., gt=0)
    is_moe: bool
    hidden_dim: int = Field(..., gt=0)
    num_layers: int = Field(..., gt=0)
    num_heads: int = Field(..., gt=0)
    num_kv_heads: int = Field(..., gt=0)
    vocab_size: int = Field(..., gt=0)
    max_context_length: int = Field(..., gt=0)
    effective_context_length: int = Field(..., gt=0)
    domains: list[str] = Field(default_factory=lambda: ["general"])
    capabilities: list[str] = Field(default_factory=list)
    benchmarks: BenchmarksSchema = Field(default_factory=BenchmarksSchema)
    num_experts: Optional[int] = None
    num_active_experts: Optional[int] = None
    notes: Optional[str] = None
    hf_model_id: Optional[str] = None  # For benchmark lookup

    @field_validator("active_params_b")
    @classmethod
    def validate_active_params(cls, v: float, info) -> float:
        """Active params should not exceed total params."""
        total = info.data.get("total_params_b")
        if total and v > total:
            return total
        return v


class QuantizationSchema(BaseModel):
    """Validated quantization specification."""

    name: str = Field(..., min_length=1)
    bits_per_weight: float = Field(..., gt=0, le=16)
    quality_factor: float = Field(..., ge=0, le=1)
    ppl_increase: float = Field(..., ge=0)
    quality_tier: QualityTier
    source: str

    @field_validator("quality_tier", mode="before")
    @classmethod
    def validate_tier(cls, v):
        """Accept string or QualityTier enum."""
        if isinstance(v, str):
            return QualityTier(v)
        return v


class BenchmarkMatchResult(BaseModel):
    """Result of attempting to match a model to benchmarks."""

    model_name: str
    matched: bool
    match_method: Optional[Literal["fuzzy", "manual", "hf_lookup"]] = None
    matched_name: Optional[str] = None
    confidence: Optional[float] = None
    benchmarks: BenchmarksSchema = Field(default_factory=BenchmarksSchema)
