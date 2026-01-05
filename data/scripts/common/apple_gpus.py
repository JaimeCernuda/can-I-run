"""
Manual Apple Silicon GPU specifications.

Apple Silicon GPUs are not in the gpu-info-api source, so we maintain them manually.
Data sourced from:
- Apple official specs (developer.apple.com)
- AnandTech analysis
- Macworld benchmarks

Note: Apple Silicon uses unified memory, so VRAM = system RAM available to GPU.
Memory bandwidth is the unified memory bandwidth.
"""

from typing import Any

# Apple Silicon specifications
# Format: (name, vram_gb, memory_bandwidth_gbps, generation)
APPLE_SILICON_GPUS: list[tuple[str, float, float, str]] = [
    # M1 Family (2020)
    ("Apple M1", 8.0, 68.25, "Apple M1"),
    ("Apple M1", 16.0, 68.25, "Apple M1"),
    ("Apple M1 Pro (14-core)", 16.0, 200.0, "Apple M1"),
    ("Apple M1 Pro (16-core)", 16.0, 200.0, "Apple M1"),
    ("Apple M1 Pro (14-core)", 32.0, 200.0, "Apple M1"),
    ("Apple M1 Pro (16-core)", 32.0, 200.0, "Apple M1"),
    ("Apple M1 Max (24-core)", 32.0, 400.0, "Apple M1"),
    ("Apple M1 Max (32-core)", 32.0, 400.0, "Apple M1"),
    ("Apple M1 Max (24-core)", 64.0, 400.0, "Apple M1"),
    ("Apple M1 Max (32-core)", 64.0, 400.0, "Apple M1"),
    ("Apple M1 Ultra (48-core)", 64.0, 800.0, "Apple M1"),
    ("Apple M1 Ultra (64-core)", 64.0, 800.0, "Apple M1"),
    ("Apple M1 Ultra (48-core)", 128.0, 800.0, "Apple M1"),
    ("Apple M1 Ultra (64-core)", 128.0, 800.0, "Apple M1"),
    # M2 Family (2022)
    ("Apple M2", 8.0, 100.0, "Apple M2"),
    ("Apple M2", 16.0, 100.0, "Apple M2"),
    ("Apple M2", 24.0, 100.0, "Apple M2"),
    ("Apple M2 Pro (16-core)", 16.0, 200.0, "Apple M2"),
    ("Apple M2 Pro (19-core)", 16.0, 200.0, "Apple M2"),
    ("Apple M2 Pro (16-core)", 32.0, 200.0, "Apple M2"),
    ("Apple M2 Pro (19-core)", 32.0, 200.0, "Apple M2"),
    ("Apple M2 Max (30-core)", 32.0, 400.0, "Apple M2"),
    ("Apple M2 Max (38-core)", 32.0, 400.0, "Apple M2"),
    ("Apple M2 Max (30-core)", 64.0, 400.0, "Apple M2"),
    ("Apple M2 Max (38-core)", 64.0, 400.0, "Apple M2"),
    ("Apple M2 Max (30-core)", 96.0, 400.0, "Apple M2"),
    ("Apple M2 Max (38-core)", 96.0, 400.0, "Apple M2"),
    ("Apple M2 Ultra (60-core)", 64.0, 800.0, "Apple M2"),
    ("Apple M2 Ultra (76-core)", 64.0, 800.0, "Apple M2"),
    ("Apple M2 Ultra (60-core)", 128.0, 800.0, "Apple M2"),
    ("Apple M2 Ultra (76-core)", 128.0, 800.0, "Apple M2"),
    ("Apple M2 Ultra (60-core)", 192.0, 800.0, "Apple M2"),
    ("Apple M2 Ultra (76-core)", 192.0, 800.0, "Apple M2"),
    # M3 Family (2023)
    ("Apple M3", 8.0, 100.0, "Apple M3"),
    ("Apple M3", 16.0, 100.0, "Apple M3"),
    ("Apple M3", 24.0, 100.0, "Apple M3"),
    ("Apple M3 Pro (14-core)", 18.0, 150.0, "Apple M3"),
    ("Apple M3 Pro (18-core)", 18.0, 150.0, "Apple M3"),
    ("Apple M3 Pro (14-core)", 36.0, 150.0, "Apple M3"),
    ("Apple M3 Pro (18-core)", 36.0, 150.0, "Apple M3"),
    ("Apple M3 Max (30-core)", 36.0, 300.0, "Apple M3"),
    ("Apple M3 Max (40-core)", 48.0, 400.0, "Apple M3"),
    ("Apple M3 Max (30-core)", 64.0, 300.0, "Apple M3"),
    ("Apple M3 Max (40-core)", 64.0, 400.0, "Apple M3"),
    ("Apple M3 Max (30-core)", 96.0, 300.0, "Apple M3"),
    ("Apple M3 Max (40-core)", 128.0, 400.0, "Apple M3"),
    # M4 Family (2024)
    ("Apple M4", 16.0, 120.0, "Apple M4"),
    ("Apple M4", 24.0, 120.0, "Apple M4"),
    ("Apple M4", 32.0, 120.0, "Apple M4"),
    ("Apple M4 Pro (16-core)", 24.0, 273.0, "Apple M4"),
    ("Apple M4 Pro (20-core)", 24.0, 273.0, "Apple M4"),
    ("Apple M4 Pro (16-core)", 48.0, 273.0, "Apple M4"),
    ("Apple M4 Pro (20-core)", 48.0, 273.0, "Apple M4"),
    ("Apple M4 Max (32-core)", 36.0, 400.0, "Apple M4"),
    ("Apple M4 Max (40-core)", 48.0, 546.0, "Apple M4"),
    ("Apple M4 Max (32-core)", 64.0, 400.0, "Apple M4"),
    ("Apple M4 Max (40-core)", 64.0, 546.0, "Apple M4"),
    ("Apple M4 Max (32-core)", 128.0, 400.0, "Apple M4"),
    ("Apple M4 Max (40-core)", 128.0, 546.0, "Apple M4"),
]


def get_apple_gpus() -> list[dict[str, Any]]:
    """
    Get Apple Silicon GPU specifications.

    Returns:
        List of GPU dicts ready for GPUSchema validation
    """
    gpus = []
    seen = set()

    for name, vram_gb, bandwidth_gbps, generation in APPLE_SILICON_GPUS:
        # Create unique key to avoid duplicates
        unique_key = f"{name}_{vram_gb}"
        if unique_key in seen:
            continue
        seen.add(unique_key)

        gpus.append(
            {
                "name": name,
                "vendor": "Apple",
                "vram_gb": vram_gb,
                "memory_bandwidth_gbps": bandwidth_gbps,
                "generation": generation,
            }
        )

    return gpus
