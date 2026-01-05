"""
GPU age filtering utilities.

Filters GPUs by release date, with fallback to architecture-based inference.
"""

import re
from datetime import datetime, timedelta
from typing import Optional

# Architecture to approximate release year mapping
ARCHITECTURE_RELEASE_YEARS: dict[str, int] = {
    # NVIDIA Consumer
    "Blackwell": 2025,
    "Ada Lovelace": 2022,
    "Ampere": 2020,
    "Turing": 2018,
    "Pascal": 2016,
    "Maxwell": 2014,
    # NVIDIA Data Center
    "Hopper": 2022,
    "Volta": 2017,
    # AMD
    "RDNA 4": 2025,
    "RDNA 3.5": 2024,
    "RDNA 3": 2022,
    "RDNA 2": 2020,
    "RDNA": 2019,
    "Vega": 2017,
    # Intel
    "Battlemage": 2024,
    "Alchemist": 2022,
    "Xe": 2020,
    # Apple
    "Apple M4": 2024,
    "Apple M3": 2023,
    "Apple M2": 2022,
    "Apple M1": 2020,
}

# GPU name patterns to infer release year
GPU_NAME_PATTERNS: list[tuple[str, int]] = [
    # NVIDIA RTX series
    (r"RTX\s*50[0-9]{2}", 2025),
    (r"RTX\s*40[0-9]{2}", 2022),
    (r"RTX\s*30[0-9]{2}", 2020),
    (r"RTX\s*20[0-9]{2}", 2018),
    # NVIDIA GTX series
    (r"GTX\s*16[0-9]{2}", 2019),
    (r"GTX\s*10[0-9]{2}", 2016),
    # NVIDIA Professional
    (r"RTX\s*A[0-9]{4}", 2021),
    (r"RTX\s*[456]000\s*Ada", 2023),
    (r"L40", 2023),
    (r"H100", 2022),
    (r"H200", 2024),
    (r"A100", 2020),
    (r"A10G?", 2021),
    # AMD RX series
    (r"RX\s*9[0-9]{3}", 2025),
    (r"RX\s*7[0-9]{3}", 2022),
    (r"RX\s*6[0-9]{3}", 2020),
    (r"RX\s*5[0-9]{3}", 2019),
    # Intel Arc
    (r"Arc\s*B[0-9]{3}", 2024),
    (r"Arc\s*A[0-9]{3}", 2022),
    # Apple Silicon
    (r"M4", 2024),
    (r"M3", 2023),
    (r"M2", 2022),
    (r"M1", 2020),
]


def parse_release_date(launch_str: Optional[str]) -> Optional[datetime]:
    """
    Parse release date from gpu-info-api format.

    Args:
        launch_str: Date string like '2022-10-12 00:00:00' or '2022-10-12'

    Returns:
        Parsed datetime or None if parsing fails
    """
    if not launch_str:
        return None

    # Clean up the string
    clean_str = launch_str.strip()

    # Try various formats
    formats = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d-%m-%Y",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(clean_str, fmt)
        except ValueError:
            continue

    return None


def infer_release_year_from_architecture(
    gpu_name: str, generation: Optional[str] = None
) -> Optional[int]:
    """
    Infer approximate release year from architecture or GPU name patterns.

    Args:
        gpu_name: GPU model name (e.g., "GeForce RTX 4090")
        generation: Architecture generation if known (e.g., "Ada Lovelace")

    Returns:
        Inferred release year or None if cannot determine
    """
    # First, try architecture mapping
    if generation and generation in ARCHITECTURE_RELEASE_YEARS:
        return ARCHITECTURE_RELEASE_YEARS[generation]

    # Fall back to name pattern matching
    name_upper = gpu_name.upper()
    for pattern, year in GPU_NAME_PATTERNS:
        if re.search(pattern, name_upper, re.IGNORECASE):
            return year

    return None


def is_gpu_recent(
    launch_date: Optional[datetime] = None,
    architecture: Optional[str] = None,
    gpu_name: str = "",
    cutoff_years: int = 5,
) -> bool:
    """
    Determine if GPU was released within cutoff_years.

    Uses two strategies:
    1. Direct date comparison (if launch_date available)
    2. Architecture/name inference (fallback)

    Args:
        launch_date: Parsed release date
        architecture: GPU architecture generation
        gpu_name: GPU model name
        cutoff_years: Number of years to consider "recent"

    Returns:
        True if GPU is recent, False otherwise
    """
    current_year = datetime.now().year
    cutoff_year = current_year - cutoff_years
    cutoff_date = datetime.now() - timedelta(days=cutoff_years * 365)

    # Strategy 1: Direct date comparison
    if launch_date:
        return launch_date >= cutoff_date

    # Strategy 2: Architecture/name inference
    inferred_year = infer_release_year_from_architecture(gpu_name, architecture)
    if inferred_year:
        return inferred_year >= cutoff_year

    # Default: exclude if we cannot determine age
    # This is conservative - we'd rather miss a GPU than include ancient ones
    return False


def detect_vendor(gpu_name: str) -> Optional[str]:
    """
    Detect GPU vendor from model name or key.

    Args:
        gpu_name: GPU model name or API key (e.g., "AMD_Radeon RX 6800")

    Returns:
        Vendor string or None if unknown
    """
    name_lower = gpu_name.lower()

    # Check for key prefix format (e.g., "AMD_xxx", "Intel_xxx", "NVIDIA_xxx")
    if name_lower.startswith("amd_"):
        return "AMD"
    if name_lower.startswith("intel_"):
        return "Intel"
    if name_lower.startswith("nvidia_"):
        return "NVIDIA"
    if name_lower.startswith("apple_"):
        return "Apple"

    # Check for NVIDIA patterns
    if any(
        x in name_lower
        for x in ["geforce", "rtx", "gtx", "quadro", "tesla", "titan", "nvidia"]
    ):
        return "NVIDIA"

    # Check for AMD patterns (including RX without trailing space)
    if any(x in name_lower for x in ["radeon", "vega", "amd", "instinct"]):
        return "AMD"
    # Special check for RX followed by digit (e.g., "rx 7900", "rx7900")
    if re.search(r"\brx\s*\d", name_lower):
        return "AMD"

    # Check for Intel patterns (including Arc without trailing space)
    if any(x in name_lower for x in ["intel", "battlemage", "alchemist"]):
        return "Intel"
    # Special check for Arc followed by letter (e.g., "arc a770", "arc b580")
    if re.search(r"\barc\s*[ab]\d", name_lower):
        return "Intel"
    if "xe " in name_lower or name_lower.startswith("xe"):
        return "Intel"

    # Check for Apple patterns
    if any(x in name_lower for x in ["apple"]):
        return "Apple"
    # Special check for M-series chips
    if re.search(r"\bm[1-4]\b", name_lower):
        return "Apple"

    return None


def detect_generation(gpu_name: str) -> Optional[str]:
    """
    Detect GPU architecture generation from model name.

    Args:
        gpu_name: GPU model name

    Returns:
        Generation string or None if unknown
    """
    name_upper = gpu_name.upper()

    # NVIDIA patterns
    if re.search(r"RTX\s*50[0-9]{2}", name_upper):
        return "Blackwell"
    if re.search(r"RTX\s*40[0-9]{2}", name_upper) or "ADA" in name_upper:
        return "Ada Lovelace"
    if re.search(r"RTX\s*30[0-9]{2}", name_upper) or "A100" in name_upper:
        return "Ampere"
    if re.search(r"RTX\s*20[0-9]{2}", name_upper) or re.search(
        r"GTX\s*16[0-9]{2}", name_upper
    ):
        return "Turing"
    if re.search(r"GTX\s*10[0-9]{2}", name_upper):
        return "Pascal"
    if "H100" in name_upper or "H200" in name_upper:
        return "Hopper"

    # AMD patterns
    if re.search(r"RX\s*9[0-9]{3}", name_upper):
        return "RDNA 4"
    if re.search(r"RX\s*7[0-9]{3}", name_upper):
        return "RDNA 3"
    if re.search(r"RX\s*6[0-9]{3}", name_upper):
        return "RDNA 2"

    # Intel patterns
    if re.search(r"ARC\s*B[0-9]{3}", name_upper):
        return "Battlemage"
    if re.search(r"ARC\s*A[0-9]{3}", name_upper):
        return "Alchemist"

    # Apple patterns
    if " M4" in name_upper or name_upper.startswith("M4"):
        return "Apple M4"
    if " M3" in name_upper or name_upper.startswith("M3"):
        return "Apple M3"
    if " M2" in name_upper or name_upper.startswith("M2"):
        return "Apple M2"
    if " M1" in name_upper or name_upper.startswith("M1"):
        return "Apple M1"

    return None
