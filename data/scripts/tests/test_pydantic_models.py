"""Tests for Pydantic validation models."""

import pytest
from pydantic import ValidationError

from ..common.pydantic_models import (
    GPUSchema,
    BenchmarksSchema,
    QuantizationSchema,
    QualityTier,
)


class TestGPUSchema:
    """Tests for GPU validation."""

    def test_valid_gpu(self):
        gpu = GPUSchema(
            name="RTX 4090",
            vendor="NVIDIA",
            vram_gb=24.0,
            memory_bandwidth_gbps=1008.0,
        )
        assert gpu.name == "RTX 4090"
        assert gpu.vendor == "NVIDIA"

    def test_invalid_vendor(self):
        with pytest.raises(ValidationError):
            GPUSchema(
                name="Test GPU",
                vendor="Unknown",
                vram_gb=24.0,
                memory_bandwidth_gbps=1008.0,
            )

    def test_negative_vram(self):
        with pytest.raises(ValidationError):
            GPUSchema(
                name="Test GPU",
                vendor="NVIDIA",
                vram_gb=-1.0,
                memory_bandwidth_gbps=1008.0,
            )

    def test_vram_mb_conversion(self):
        """Test that large VRAM values get converted from MB to GB."""
        gpu = GPUSchema(
            name="Test GPU",
            vendor="NVIDIA",
            vram_gb=24576,  # 24576 MB
            memory_bandwidth_gbps=1008.0,
        )
        assert gpu.vram_gb == 24.0  # Should be converted to 24 GB


class TestBenchmarksSchema:
    """Tests for benchmark score validation."""

    def test_valid_benchmarks(self):
        benchmarks = BenchmarksSchema(
            humaneval=75.5,
            mmlu_pro=30.0,
            math=15.0,
        )
        assert benchmarks.humaneval == 75.5
        assert benchmarks.mmlu_pro == 30.0

    def test_all_none(self):
        benchmarks = BenchmarksSchema()
        assert benchmarks.humaneval is None
        assert benchmarks.mmlu_pro is None

    def test_score_over_100(self):
        with pytest.raises(ValidationError):
            BenchmarksSchema(humaneval=150.0)

    def test_negative_score(self):
        with pytest.raises(ValidationError):
            BenchmarksSchema(humaneval=-10.0)


class TestQuantizationSchema:
    """Tests for quantization validation."""

    def test_valid_quantization(self):
        quant = QuantizationSchema(
            name="Q4_K_M",
            bits_per_weight=4.83,
            quality_factor=0.97,
            ppl_increase=0.07,
            quality_tier=QualityTier.RECOMMENDED,
            source="llama.cpp",
        )
        assert quant.name == "Q4_K_M"
        assert quant.quality_tier == QualityTier.RECOMMENDED

    def test_string_quality_tier(self):
        """Test that string tier values are converted."""
        quant = QuantizationSchema(
            name="Q4_K_M",
            bits_per_weight=4.83,
            quality_factor=0.97,
            ppl_increase=0.07,
            quality_tier="recommended",
            source="llama.cpp",
        )
        assert quant.quality_tier == QualityTier.RECOMMENDED

    def test_invalid_bits(self):
        with pytest.raises(ValidationError):
            QuantizationSchema(
                name="Invalid",
                bits_per_weight=20.0,  # Max is 16
                quality_factor=0.97,
                ppl_increase=0.07,
                quality_tier=QualityTier.RECOMMENDED,
                source="test",
            )
