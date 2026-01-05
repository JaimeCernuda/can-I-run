"""
Script to regenerate quantizations.json by parsing llama.cpp perplexity README.

Source: https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md

The README contains perplexity (PPL) tables for various quantization formats
measured on LLaMA-3-8B. We parse these tables to extract quality metrics.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .common.http import fetch_text
from .common.pydantic_models import QualityTier, QuantizationSchema

logger = logging.getLogger(__name__)

# Setup paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
QUANTS_JSON_PATH = DATA_DIR / "quantizations.json"

LLAMA_CPP_PPL_README = "https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/perplexity/README.md"

# Known bits per weight for each quantization format
# Source: llama.cpp quantize documentation
BITS_PER_WEIGHT: dict[str, float] = {
    "F16": 16.00,
    "BF16": 16.00,
    "Q8_0": 8.50,
    "Q6_K": 6.57,
    "Q5_K_M": 5.67,
    "Q5_K_S": 5.53,
    "Q5_1": 5.50,
    "Q5_0": 5.50,
    "Q4_K_M": 4.83,
    "Q4_K_S": 4.58,
    "Q4_1": 4.50,
    "Q4_0": 4.34,
    "Q3_K_L": 4.03,
    "Q3_K_M": 3.89,
    "Q3_K_S": 3.50,
    "Q2_K": 3.00,
    "IQ4_XS": 4.25,
    "IQ4_NL": 4.35,
    "IQ3_M": 3.44,
    "IQ3_S": 3.25,
    "IQ3_XS": 3.28,
    "IQ3_XXS": 3.06,
    "IQ2_M": 2.70,
    "IQ2_S": 2.50,
    "IQ2_XS": 2.43,
    "IQ2_XXS": 2.24,
    "IQ1_M": 2.01,
    "IQ1_S": 1.88,
}

# Baseline PPL for LLaMA-3-8B FP16 (approximate)
BASELINE_PPL = 6.23


def determine_quality_tier(ppl_increase: float) -> QualityTier:
    """
    Map PPL increase to quality tier.

    Args:
        ppl_increase: Absolute PPL increase vs FP16

    Returns:
        QualityTier enum value
    """
    if ppl_increase < 0.01:
        return QualityTier.NEAR_LOSSLESS
    elif ppl_increase < 0.1:
        return QualityTier.VERY_LOW_LOSS
    elif ppl_increase < 0.25:
        return QualityTier.RECOMMENDED
    elif ppl_increase < 0.5:
        return QualityTier.BALANCED
    elif ppl_increase < 1.0:
        return QualityTier.NOTICEABLE_LOSS
    elif ppl_increase < 2.0:
        return QualityTier.HIGH_LOSS
    else:
        return QualityTier.EXTREME_LOSS


def calculate_quality_factor(
    ppl_increase: float, baseline_ppl: float = BASELINE_PPL
) -> float:
    """
    Convert PPL increase to quality factor (0-1).

    Quality factor represents how much quality is preserved relative to FP16.
    Higher PPL increase = lower quality factor.

    Args:
        ppl_increase: Absolute PPL increase
        baseline_ppl: Baseline FP16 perplexity

    Returns:
        Quality factor between 0 and 1
    """
    if ppl_increase <= 0:
        return 1.0

    # Formula: quality = baseline_ppl / (baseline_ppl + ppl_increase)
    new_ppl = baseline_ppl + ppl_increase
    quality = baseline_ppl / new_ppl
    return round(quality, 4)


def parse_ppl_tables(readme_text: str) -> list[dict[str, Any]]:
    """
    Parse PPL tables from llama.cpp perplexity README.

    Looks for markdown tables with format:
    | Model | Size (GiB) | PPL | ΔPPL | ... |

    Args:
        readme_text: Full README content

    Returns:
        List of parsed quantization entries
    """
    quantizations: list[dict[str, Any]] = []
    seen_quants: set[str] = set()

    # Pattern to match table rows
    # Example: | q8_0  | 7.96 | 6.234 | +0.004 |
    table_pattern = (
        r"\|\s*(\S+)\s*\|\s*([\d.]+)\s*\|\s*([\d.]+)\s*\|\s*[+]?([\d.]+)\s*\|"
    )

    for match in re.finditer(table_pattern, readme_text, re.IGNORECASE):
        quant_name_raw = match.group(1).strip()
        _model_size_gib = float(match.group(2))  # Parsed but not used
        ppl = float(match.group(3))
        ppl_delta = float(match.group(4))

        # Normalize quantization name
        quant_name = quant_name_raw.upper().replace("_", "_")

        # Skip header rows and invalid entries
        if quant_name in ["MODEL", "QUANT", "NAME", "TYPE", "---"]:
            continue

        # Skip if we've already seen this quantization
        if quant_name in seen_quants:
            continue

        # Skip if not in our known formats
        if quant_name not in BITS_PER_WEIGHT:
            # Try common variations
            if quant_name.replace("-", "_") in BITS_PER_WEIGHT:
                quant_name = quant_name.replace("-", "_")
            else:
                logger.debug(f"Unknown quantization format: {quant_name}")
                continue

        bits_per_weight = BITS_PER_WEIGHT[quant_name]
        quality_factor = calculate_quality_factor(ppl_delta)
        tier = determine_quality_tier(ppl_delta)

        quantizations.append(
            {
                "name": quant_name,
                "bits_per_weight": bits_per_weight,
                "quality_factor": quality_factor,
                "ppl_increase": ppl_delta,
                "quality_tier": tier.value,
                "source": f"llama.cpp perplexity README - LLaMA-3-8B (PPL: {ppl})",
            }
        )
        seen_quants.add(quant_name)

    return quantizations


def get_default_quantizations() -> list[dict[str, Any]]:
    """
    Return default quantization specs when parsing fails.

    These are based on well-known values from llama.cpp documentation.
    """
    defaults = [
        ("F16", 16.00, 0.000, "near_lossless", "Baseline (no quantization)"),
        ("Q8_0", 8.50, 0.003, "near_lossless", "llama.cpp"),
        ("Q6_K", 6.57, 0.022, "very_low_loss", "llama.cpp"),
        ("Q5_K_M", 5.67, 0.057, "recommended", "llama.cpp"),
        ("Q5_K_S", 5.53, 0.105, "balanced", "llama.cpp"),
        ("Q4_K_M", 4.83, 0.054, "recommended", "llama.cpp - Sweet spot"),
        ("Q4_K_S", 4.58, 0.080, "balanced", "llama.cpp"),
        ("IQ4_XS", 4.25, 0.090, "balanced", "llama.cpp IQ"),
        ("Q4_0", 4.34, 0.469, "noticeable_loss", "llama.cpp (legacy)"),
        ("Q3_K_M", 3.89, 0.244, "noticeable_loss", "llama.cpp"),
        ("Q3_K_S", 3.50, 0.657, "high_loss", "llama.cpp"),
        ("IQ3_M", 3.44, 0.350, "noticeable_loss", "llama.cpp IQ"),
        ("IQ3_S", 3.25, 0.450, "noticeable_loss", "llama.cpp IQ"),
        ("Q2_K", 3.00, 0.870, "high_loss", "llama.cpp"),
        ("IQ2_M", 2.70, 1.200, "high_loss", "llama.cpp IQ"),
        ("IQ2_S", 2.50, 1.800, "extreme_loss", "llama.cpp IQ"),
        ("IQ2_XS", 2.43, 2.500, "extreme_loss", "llama.cpp IQ"),
        ("IQ2_XXS", 2.24, 3.520, "extreme_loss", "llama.cpp"),
        ("IQ1_M", 2.01, 8.000, "extreme_loss", "llama.cpp IQ"),
        ("IQ1_S", 1.88, 12.000, "extreme_loss", "llama.cpp IQ"),
    ]

    return [
        {
            "name": name,
            "bits_per_weight": bits,
            "quality_factor": calculate_quality_factor(ppl_inc),
            "ppl_increase": ppl_inc,
            "quality_tier": tier,
            "source": source,
        }
        for name, bits, ppl_inc, tier, source in defaults
    ]


def main(dry_run: bool = False) -> dict[str, Any]:
    """
    Regenerate quantizations.json from llama.cpp README.

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
        "used_fallback": False,  # Track if fallback was used
    }

    # Try to fetch and parse from llama.cpp
    try:
        logger.info("Fetching llama.cpp perplexity README...")
        readme_text = fetch_text(LLAMA_CPP_PPL_README)
        parsed_quants = parse_ppl_tables(readme_text)
        logger.info(f"Parsed {len(parsed_quants)} quantizations from README")
    except Exception as e:
        logger.error(f"FALLBACK TRIGGERED: Failed to fetch/parse llama.cpp README: {e}")
        result["warnings"].append(f"Parsing failed, using fallback defaults: {e}")
        parsed_quants = []

    # Use defaults if parsing failed or returned too few results
    if len(parsed_quants) < 10:
        result["used_fallback"] = True
        result["warnings"].append(
            f"FALLBACK USED: Only parsed {len(parsed_quants)} quantizations "
            f"(minimum 10 required). Using hardcoded defaults instead. "
            f"This may indicate the llama.cpp README format has changed."
        )
        logger.warning("=" * 60)
        logger.warning("QUANTIZATION FALLBACK TRIGGERED")
        logger.warning(f"Only parsed {len(parsed_quants)} entries from source")
        logger.warning("Using hardcoded default values instead")
        logger.warning("This may indicate the llama.cpp README format changed")
        logger.warning("Please verify the source and update the parser")
        logger.warning("=" * 60)
        parsed_quants = get_default_quantizations()

    # Ensure F16 baseline is present
    if not any(q["name"] == "F16" for q in parsed_quants):
        parsed_quants.insert(
            0,
            {
                "name": "F16",
                "bits_per_weight": 16.0,
                "quality_factor": 1.0,
                "ppl_increase": 0.0,
                "quality_tier": "near_lossless",
                "source": "Baseline (no quantization)",
            },
        )

    # Validate with Pydantic
    validated: list[dict] = []
    for q in parsed_quants:
        try:
            schema = QuantizationSchema(**q)
            validated.append(schema.model_dump())
        except ValidationError as e:
            result["warnings"].append(
                f"Invalid quantization {q.get('name', 'unknown')}: {e}"
            )
            logger.warning(f"Validation failed for {q.get('name')}: {e}")

    # Sort by bits per weight (descending)
    validated.sort(key=lambda x: -x["bits_per_weight"])

    result["count"] = len(validated)

    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(validated)} quantizations")
        return result

    # Write output
    output = {
        "_comment": "Quantization specifications with quality factors. Sources: llama.cpp quantize tool (https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md), Intel Low-bit Quantized Open LLM Leaderboard (https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)",
        "quantizations": validated,
    }

    try:
        with open(QUANTS_JSON_PATH, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(validated)} quantizations to {QUANTS_JSON_PATH}")
    except IOError as e:
        result["success"] = False
        result["errors"].append(f"Write failed: {e}")
        logger.error(f"Failed to write {QUANTS_JSON_PATH}: {e}")

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
