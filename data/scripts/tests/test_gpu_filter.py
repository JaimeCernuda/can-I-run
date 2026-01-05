"""Tests for GPU age filtering and vendor detection."""

from datetime import datetime

from ..common.gpu_age_filter import (
    detect_vendor,
    detect_generation,
    parse_release_date,
    infer_release_year_from_architecture,
    is_gpu_recent,
)


class TestDetectVendor:
    """Tests for vendor detection from GPU names."""

    def test_nvidia_geforce(self):
        assert detect_vendor("GeForce RTX 4090") == "NVIDIA"

    def test_nvidia_rtx(self):
        assert detect_vendor("RTX 5090") == "NVIDIA"

    def test_nvidia_quadro(self):
        assert detect_vendor("Quadro RTX 8000") == "NVIDIA"

    def test_nvidia_key_prefix(self):
        assert detect_vendor("NVIDIA_RTX 4090") == "NVIDIA"

    def test_amd_radeon(self):
        assert detect_vendor("Radeon RX 7900 XTX") == "AMD"

    def test_amd_rx_pattern(self):
        assert detect_vendor("RX 7900") == "AMD"
        assert detect_vendor("RX7900") == "AMD"

    def test_amd_key_prefix(self):
        assert detect_vendor("AMD_Radeon RX 6800") == "AMD"

    def test_amd_instinct(self):
        assert detect_vendor("AMD Instinct MI300X") == "AMD"

    def test_intel_arc(self):
        assert (
            detect_vendor("Arc A770") == "INTEL" or detect_vendor("Arc A770") == "Intel"
        )

    def test_intel_arc_battlemage(self):
        assert detect_vendor("Arc B580") == "Intel"

    def test_intel_key_prefix(self):
        assert detect_vendor("Intel_Arc A770") == "Intel"

    def test_apple_m_series(self):
        assert detect_vendor("Apple M3 Max") == "Apple"

    def test_unknown_vendor(self):
        assert detect_vendor("Unknown GPU XYZ") is None


class TestDetectGeneration:
    """Tests for GPU architecture generation detection."""

    def test_nvidia_rtx_50(self):
        assert detect_generation("RTX 5090") == "Blackwell"

    def test_nvidia_rtx_40(self):
        assert detect_generation("RTX 4090") == "Ada Lovelace"

    def test_nvidia_rtx_30(self):
        assert detect_generation("RTX 3090") == "Ampere"

    def test_nvidia_rtx_20(self):
        assert detect_generation("RTX 2080") == "Turing"

    def test_amd_rdna3(self):
        assert detect_generation("RX 7900 XTX") == "RDNA 3"

    def test_amd_rdna2(self):
        assert detect_generation("RX 6800") == "RDNA 2"

    def test_intel_battlemage(self):
        assert detect_generation("Arc B580") == "Battlemage"

    def test_intel_alchemist(self):
        assert detect_generation("Arc A770") == "Alchemist"

    def test_apple_m4(self):
        assert detect_generation("M4 Max") == "Apple M4"

    def test_unknown_generation(self):
        assert detect_generation("Unknown GPU") is None


class TestParseReleaseDate:
    """Tests for release date parsing."""

    def test_standard_format(self):
        result = parse_release_date("2024-01-15 00:00:00")
        assert result is not None
        assert result.year == 2024
        assert result.month == 1
        assert result.day == 15

    def test_date_only(self):
        result = parse_release_date("2024-01-15")
        assert result is not None
        assert result.year == 2024

    def test_none_input(self):
        assert parse_release_date(None) is None

    def test_empty_string(self):
        assert parse_release_date("") is None

    def test_invalid_format(self):
        assert parse_release_date("not a date") is None


class TestInferReleaseYear:
    """Tests for release year inference from GPU names."""

    def test_rtx_50_series(self):
        assert infer_release_year_from_architecture("RTX 5090") == 2025

    def test_rtx_40_series(self):
        assert infer_release_year_from_architecture("RTX 4090") == 2022

    def test_rx_7000_series(self):
        assert infer_release_year_from_architecture("RX 7900 XTX") == 2022

    def test_arc_b_series(self):
        assert infer_release_year_from_architecture("Arc B580") == 2024

    def test_architecture_fallback(self):
        assert infer_release_year_from_architecture("Unknown", "Ada Lovelace") == 2022

    def test_unknown(self):
        assert infer_release_year_from_architecture("Unknown GPU") is None


class TestIsGpuRecent:
    """Tests for GPU recency filtering."""

    def test_recent_by_date(self):
        # GPU from last year should be recent
        recent_date = datetime(datetime.now().year - 1, 6, 1)
        assert is_gpu_recent(recent_date, None, "", 5) is True

    def test_old_by_date(self):
        # GPU from 10 years ago should not be recent
        old_date = datetime(datetime.now().year - 10, 1, 1)
        assert is_gpu_recent(old_date, None, "", 5) is False

    def test_recent_by_architecture(self):
        # RTX 40 series (2022) should be recent with 5 year cutoff
        assert is_gpu_recent(None, "Ada Lovelace", "RTX 4090", 5) is True

    def test_recent_by_name_pattern(self):
        # RTX 50 series should be recent
        assert is_gpu_recent(None, None, "RTX 5090", 5) is True

    def test_unknown_not_recent(self):
        # Unknown GPUs should default to not recent (conservative)
        assert is_gpu_recent(None, None, "Unknown GPU", 5) is False
