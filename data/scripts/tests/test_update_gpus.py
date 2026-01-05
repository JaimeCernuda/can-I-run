"""Tests for GPU update script parsing functions."""

from ..update_gpus import parse_vram, parse_bandwidth


class TestParseVram:
    """Tests for VRAM parsing from various formats."""

    def test_gb_format(self):
        """Test '16 GB' format."""
        assert parse_vram("16 GB") == 16.0

    def test_gb_format_lowercase(self):
        """Test '24 gb' format."""
        assert parse_vram("24 gb") == 24.0

    def test_mb_value(self):
        """Test large number that's clearly in MB."""
        assert parse_vram("24576") == 24.0  # 24576 MB = 24 GB

    def test_mb_multiple_values(self):
        """Test '12288 24576' format - should take max."""
        assert parse_vram("12288 24576") == 24.0

    def test_empty_string(self):
        assert parse_vram("") == 0.0

    def test_none(self):
        assert parse_vram(None) == 0.0

    def test_numeric_value(self):
        """Test numeric input."""
        assert parse_vram(16384) == 16.0  # 16384 MB = 16 GB

    def test_small_gb_value(self):
        """Test value that's already in GB."""
        assert parse_vram("8") == 8.0


class TestParseBandwidth:
    """Tests for memory bandwidth parsing."""

    def test_simple_value(self):
        assert parse_bandwidth("1008") == 1008.0

    def test_multiple_values(self):
        """Test '448.0 576.0' format - should take max."""
        assert parse_bandwidth("448.0 576.0") == 576.0

    def test_float_value(self):
        assert parse_bandwidth("960.0") == 960.0

    def test_empty_string(self):
        assert parse_bandwidth("") == 0.0

    def test_none(self):
        assert parse_bandwidth(None) == 0.0

    def test_numeric_input(self):
        assert parse_bandwidth(1008) == 1008.0
