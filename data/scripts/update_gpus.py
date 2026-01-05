"""
Script to regenerate gpus.json from verified internet sources.

Source: https://github.com/voidful/gpu-info-api
    (URL: https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json)

Filtering: Includes all GPUs released in the last 5 years from all vendors.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from .common.http import fetch_json
from .common.gpu_age_filter import (
    detect_generation,
    detect_vendor,
    is_gpu_recent,
    parse_release_date,
)
from .common.pydantic_models import GPUSchema
from .common.apple_gpus import get_apple_gpus

logger = logging.getLogger(__name__)

# Setup paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
GPUS_JSON_PATH = DATA_DIR / "gpus.json"

URL_GPU_INFO_API = (
    "https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json"
)

# Minimum VRAM threshold (GB)
MIN_VRAM_GB = 6.0


def parse_vram(val: Any) -> float:
    """
    Parse Memory Size field which can be '24576', '12288 24576', '16 GB', etc.

    Args:
        val: Raw value from API (could be string, number, or None)

    Returns:
        VRAM in GB (float)
    """
    if not val:
        return 0.0

    s = str(val).strip().lower()

    # Handle "16 GB" or "24 GB" format directly
    gb_match = re.search(r"(\d+(?:\.\d+)?)\s*gb", s)
    if gb_match:
        return float(gb_match.group(1))

    # Handle MB values (e.g. "24576", "12288 24576")
    parts = s.split()
    max_mb = 0.0

    for p in parts:
        clean = re.sub(r"[^\d\.]", "", p)
        if clean:
            try:
                v = float(clean)
                if v > max_mb:
                    max_mb = v
            except ValueError:
                pass

    # Convert MB to GB (values over 100 are definitely in MB)
    if max_mb > 100:
        return max_mb / 1024.0
    return max_mb


def parse_bandwidth(val: Any) -> float:
    """
    Parse Memory Bandwidth (GB/s).

    Args:
        val: Raw value from API

    Returns:
        Bandwidth in GB/s (float)
    """
    if not val:
        return 0.0

    s = str(val).strip()

    # Handle "448.0 576.0" -> take max
    parts = s.split()
    max_bw = 0.0

    for p in parts:
        clean = re.sub(r"[^\d\.]", "", p)
        try:
            v = float(clean)
            if v > max_bw:
                max_bw = v
        except ValueError:
            pass

    return max_bw


def get_value_flexible(info: dict[str, Any], keys: list[str]) -> Optional[Any]:
    """
    Get value from dict with flexible key matching.

    Args:
        info: Dictionary to search
        keys: List of potential keys to try

    Returns:
        Value if found, None otherwise
    """
    for k in keys:
        if k in info:
            return info[k]
        # Try case-insensitive match
        for raw_k in info:
            if raw_k.strip().lower() == k.lower():
                return info[raw_k]
    return None


def process_gpu_entry(
    key: str, info: dict[str, Any], cutoff_years: int = 5
) -> Optional[GPUSchema]:
    """
    Process a single GPU entry from the API.

    Args:
        key: GPU key from API
        info: GPU info dictionary
        cutoff_years: Only include GPUs from last N years

    Returns:
        Validated GPUSchema or None if filtered out
    """
    # Try multiple fields for model name (different vendors use different fields)
    name = info.get("Model name")
    if not name or str(name).lower() == "nan":
        # Some entries use just "Model" field (Intel Data Center GPUs)
        name = info.get("Model")
    if not name or str(name).lower() == "nan":
        # AMD often uses "Model (Code name)" or "Model (Codename)"
        name = info.get("Model (Code name)") or info.get("Model (Codename)")

    if not name or str(name).lower() == "nan":
        # Intel uses "Branding and Model" fields (e.g., "Arc 5" + "B570" -> "Arc B570")
        branding = info.get("Branding and Model", "")
        model_suffix = info.get("Branding and Model Branding and Model.1", "")
        if branding and str(branding).lower() != "nan":
            if model_suffix and str(model_suffix).lower() != "nan":
                name = f"{branding} {model_suffix}"  # e.g., "Arc B570"
            else:
                name = branding

    if not name or str(name).lower() == "nan":
        # Fall back to key (which contains vendor_model format)
        name = key
        # Strip vendor prefix from key (e.g., "AMD_Radeon RX 6800" -> "Radeon RX 6800")
        if "_" in name:
            name = name.split("_", 1)[1]

    if not name or str(name).lower() == "nan":
        return None

    # Filter out console/integrated GPUs (not usable for desktop LLM inference)
    bus_interface = str(info.get("Bus interface", "")).lower()
    name_lower = name.lower()
    if "integrated" in bus_interface:
        return None
    if any(x in name_lower for x in ["playstation", "xbox", "nintendo", "switch"]):
        return None
    # Check for console codenames in fields
    console_field = info.get("Code name (console model)", "")
    if console_field and str(console_field).lower() != "nan":
        return None

    # Parse release date
    launch_str = info.get("Launch")
    launch_date = parse_release_date(launch_str)

    # Detect generation
    generation = detect_generation(name)

    # Age filter - only include recent GPUs
    if not is_gpu_recent(launch_date, generation, name, cutoff_years):
        return None

    # Detect vendor - try API field first, then infer from name/key
    vendor = info.get("Vendor")
    if not vendor or str(vendor).lower() == "nan":
        vendor = detect_vendor(name)
    if not vendor:
        # Try detecting from key prefix (e.g., "AMD_xxx", "Intel_xxx")
        vendor = detect_vendor(key)
    if not vendor:
        logger.debug(f"Could not detect vendor for: {name}")
        return None

    # Parse VRAM
    mem_keys = [
        "Memory Size (MiB)",
        "Memory Size (GiB)",
        "Memory Size",  # AMD uses this with "16 GB" format
        "Memory",
        "VRAM",
        "Memory size",
        "Memory config",
    ]
    mem_raw = get_value_flexible(info, mem_keys)

    # Check which key was used to determine if it's GiB or MiB
    used_key = None
    for k in mem_keys:
        if k in info:
            used_key = k
            break

    if used_key == "Memory Size (GiB)":
        # Direct GB value
        try:
            s = str(mem_raw).strip()
            clean = re.sub(r"[^\d\.]", "", s.split()[0])
            vram_gb = float(clean)
        except (ValueError, IndexError):
            vram_gb = 0.0
    else:
        vram_gb = parse_vram(mem_raw)

    # Skip low VRAM GPUs
    if vram_gb < MIN_VRAM_GB:
        return None

    # Parse bandwidth
    bw_keys = ["Memory Bandwidth (GB/s)", "Bandwidth"]
    bw_raw = get_value_flexible(info, bw_keys)
    bw_gbps = parse_bandwidth(bw_raw)

    if bw_gbps <= 0:
        logger.debug(f"No bandwidth data for: {name}")
        return None

    # Validate with Pydantic
    try:
        gpu = GPUSchema(
            name=name,
            vendor=vendor,
            vram_gb=vram_gb,
            memory_bandwidth_gbps=bw_gbps,
            generation=generation,
            release_date=launch_date.date() if launch_date else None,
        )
        return gpu
    except ValidationError as e:
        logger.warning(f"Validation failed for {name}: {e}")
        return None


def main(dry_run: bool = False) -> dict[str, Any]:
    """
    Regenerate gpus.json with age-filtered GPU data.

    Args:
        dry_run: If True, parse but don't write files

    Returns:
        Result dict with count and status
    """
    result = {
        "success": True,
        "count": 0,
        "errors": [],
        "warnings": [],
    }

    try:
        logger.info("Fetching GPU data from gpu-info-api...")
        raw_data = fetch_json(URL_GPU_INFO_API)
        logger.info(f"Loaded {len(raw_data)} raw GPU entries")
    except Exception as e:
        result["success"] = False
        result["errors"].append(f"Fetch failed: {e}")
        logger.error(f"Failed to fetch GPU data: {e}")
        return result

    # Process all GPUs with age filtering
    valid_gpus: list[GPUSchema] = []

    for key, info in raw_data.items():
        gpu = process_gpu_entry(key, info)
        if gpu:
            valid_gpus.append(gpu)

    logger.info(f"Filtered to {len(valid_gpus)} recent GPUs")

    # Deduplicate by name + vram
    unique_map: dict[str, GPUSchema] = {}
    for gpu in valid_gpus:
        unique_key = f"{gpu.name}_{gpu.vram_gb:.1f}"
        # Keep the one with more complete data
        if unique_key not in unique_map:
            unique_map[unique_key] = gpu
        else:
            existing = unique_map[unique_key]
            # Prefer entry with generation info
            if gpu.generation and not existing.generation:
                unique_map[unique_key] = gpu

    # Add Apple Silicon GPUs (not in gpu-info-api)
    apple_gpus = get_apple_gpus()
    apple_count = 0
    for apple_gpu in apple_gpus:
        try:
            gpu = GPUSchema(**apple_gpu)
            unique_key = f"{gpu.name}_{gpu.vram_gb:.1f}"
            if unique_key not in unique_map:
                unique_map[unique_key] = gpu
                apple_count += 1
        except ValidationError as e:
            logger.warning(f"Invalid Apple GPU {apple_gpu.get('name')}: {e}")

    logger.info(f"Added {apple_count} Apple Silicon GPUs")

    # Sort by vendor then by VRAM descending
    sorted_gpus = sorted(
        unique_map.values(),
        key=lambda x: (x.vendor, -x.vram_gb),
    )

    result["count"] = len(sorted_gpus)

    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(sorted_gpus)} GPUs")
        return result

    # Write output (excluding None values and release_date which is internal)
    output_gpus = []
    for gpu in sorted_gpus:
        gpu_dict = gpu.model_dump(exclude_none=True)
        # Remove release_date from output - it's only used for filtering
        gpu_dict.pop("release_date", None)
        output_gpus.append(gpu_dict)

    output = {"gpus": output_gpus}

    try:
        with open(GPUS_JSON_PATH, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(sorted_gpus)} GPUs to {GPUS_JSON_PATH}")
    except IOError as e:
        result["success"] = False
        result["errors"].append(f"Write failed: {e}")
        logger.error(f"Failed to write {GPUS_JSON_PATH}: {e}")

    return result


if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    result = main()
    sys.exit(0 if result["success"] else 1)
