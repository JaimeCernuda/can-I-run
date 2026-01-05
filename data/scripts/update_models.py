"""
Script to fetch model data from Unsloth's HuggingFace repository and
match against benchmark leaderboards.

Sources:
    - Model architecture: Unsloth HuggingFace API
    - MMLU, GSM8K: Open LLM Leaderboard
    - HumanEval: EvalPlus Leaderboard
    - BFCL: Berkeley Function Calling Leaderboard

This script implements three benchmark matching strategies:
    1. Fuzzy matching (difflib)
    2. Manual mapping (manual_mappings.py)
    3. HuggingFace model_id lookup
"""

import difflib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from pydantic import ValidationError

from .common.http import create_session, fetch_json
from .common.pydantic_models import BenchmarksSchema
from .manual_mappings import lookup_manual_mapping
from .model_card_benchmarks import lookup_model_card_benchmarks

logger = logging.getLogger(__name__)

# Setup paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent
MODELS_JSON_PATH = DATA_DIR / "models.json"
MATCHING_REPORT_PATH = DATA_DIR / "benchmark_matching_report.json"

# API URLs
UNSLOTH_API_URL = "https://huggingface.co/api/models?author=unsloth&sort=downloads&direction=-1&limit=100"
EVALPLUS_RESULTS_URL = "https://evalplus.github.io/results.json"
OPEN_LLM_LEADERBOARD_URL = (
    "https://huggingface.co/datasets/open-llm-leaderboard/contents/resolve/main/"
    "data/train-00000-of-00001.parquet"
)


@dataclass
class MatchAttempt:
    """Result of a single matching strategy."""

    strategy: str
    matched: bool
    matched_name: Optional[str] = None
    confidence: float = 0.0
    benchmarks: Optional[dict[str, float]] = None


@dataclass
class MatchResult:
    """Complete matching result for a model."""

    model_name: str
    strategies: list[MatchAttempt] = field(default_factory=list)
    best_match: Optional[MatchAttempt] = None
    final_benchmarks: dict[str, Optional[float]] = field(default_factory=dict)


class BenchmarkMatcher:
    """Multi-strategy benchmark matcher with comparison output."""

    def __init__(self):
        self.open_llm_data: dict[str, dict] = {}
        self.evalplus_data: dict[str, dict] = {}
        self.bfcl_data: dict[str, dict] = {}
        self.session = create_session()

    def load_all_sources(self) -> None:
        """Fetch all benchmark data sources."""
        logger.info("Loading benchmark data sources...")

        # Load EvalPlus (HumanEval)
        try:
            self.evalplus_data = self._fetch_evalplus()
            logger.info(f"Loaded {len(self.evalplus_data)} models from EvalPlus")
        except Exception as e:
            logger.warning(f"Failed to load EvalPlus: {e}")

        # Load Open LLM Leaderboard (MMLU-PRO, MATH)
        try:
            self.open_llm_data = self._fetch_open_llm_leaderboard()
            logger.info(
                f"Loaded {len(self.open_llm_data)} models from Open LLM Leaderboard"
            )
        except Exception as e:
            logger.warning(f"Failed to load Open LLM Leaderboard: {e}")

        logger.info("Benchmark sources loaded")

    def _fetch_open_llm_leaderboard(self) -> dict[str, dict]:
        """Fetch MMLU-PRO and MATH scores from Open LLM Leaderboard."""
        import io
        import pandas as pd

        try:
            resp = self.session.get(OPEN_LLM_LEADERBOARD_URL, timeout=60)
            resp.raise_for_status()
            df = pd.read_parquet(io.BytesIO(resp.content))

            result = {}
            for _, row in df.iterrows():
                fullname = row.get("fullname", "")
                if not fullname:
                    continue

                # Store benchmarks - use MMLU-PRO and MATH Lvl 5
                mmlu_pro = row.get("MMLU-PRO", 0)
                math_lvl5 = row.get("MATH Lvl 5", 0)

                # Store under both full name and short name
                benchmarks = {}
                if mmlu_pro and mmlu_pro > 0:
                    benchmarks["mmlu_pro"] = round(mmlu_pro, 1)
                if math_lvl5 and math_lvl5 > 0:
                    benchmarks["math"] = round(math_lvl5, 1)

                if benchmarks:
                    result[fullname] = benchmarks
                    # Also store under short name (without org prefix)
                    short_name = (
                        fullname.split("/")[-1] if "/" in fullname else fullname
                    )
                    if short_name not in result:
                        result[short_name] = benchmarks

            return result
        except Exception as e:
            logger.warning(f"Error fetching Open LLM Leaderboard: {e}")
            return {}

    def _fetch_evalplus(self) -> dict[str, dict]:
        """Fetch HumanEval scores from EvalPlus."""
        try:
            data = fetch_json(EVALPLUS_RESULTS_URL, self.session)
            result = {}
            for model_name, scores in data.items():
                if isinstance(scores, dict):
                    # EvalPlus format: scores["pass@1"]["humaneval"]
                    pass_at_1 = scores.get("pass@1", {})
                    if isinstance(pass_at_1, dict):
                        humaneval = pass_at_1.get("humaneval", 0)
                        # Convert to percentage if needed
                        if humaneval <= 1:
                            humaneval = humaneval * 100
                        result[model_name] = {"humaneval": humaneval}
            return result
        except Exception as e:
            logger.warning(f"Error fetching EvalPlus: {e}")
            return {}

    def match(self, model_name: str, hf_model_id: Optional[str] = None) -> MatchResult:
        """
        Try all matching strategies and return comparison.

        Args:
            model_name: The model name from Unsloth
            hf_model_id: Optional full HuggingFace model ID

        Returns:
            MatchResult with all strategy attempts
        """
        result = MatchResult(model_name=model_name)

        # Strategy 0: Model card benchmarks (highest priority - most reliable)
        model_card_result = self._try_model_card_benchmarks(model_name)
        result.strategies.append(model_card_result)

        # Strategy 1: Fuzzy matching
        fuzzy_result = self._try_fuzzy_match(model_name)
        result.strategies.append(fuzzy_result)

        # Strategy 2: Manual mapping
        manual_result = self._try_manual_mapping(model_name)
        result.strategies.append(manual_result)

        # Strategy 3: HuggingFace model_id lookup
        hf_result = self._try_hf_lookup(hf_model_id or model_name)
        result.strategies.append(hf_result)

        # Determine best match (prefer model_card > manual > hf > fuzzy)
        for strategy in ["model_card", "manual", "hf_lookup", "fuzzy"]:
            match = next(
                (s for s in result.strategies if s.strategy == strategy and s.matched),
                None,
            )
            if match and match.benchmarks:
                result.best_match = match
                result.final_benchmarks = match.benchmarks
                break

        return result

    def _try_model_card_benchmarks(self, model_name: str) -> MatchAttempt:
        """Look up benchmarks from model card data."""
        benchmarks = lookup_model_card_benchmarks(model_name)

        if benchmarks:
            return MatchAttempt(
                strategy="model_card",
                matched=True,
                matched_name=model_name,
                confidence=1.0,
                benchmarks=benchmarks,
            )

        return MatchAttempt(strategy="model_card", matched=False)

    def _try_fuzzy_match(self, model_name: str) -> MatchAttempt:
        """Use difflib for fuzzy string matching."""
        normalized = self._normalize_name(model_name)

        # Search in EvalPlus data
        all_names = list(self.evalplus_data.keys())
        normalized_names = [self._normalize_name(n) for n in all_names]

        matches = difflib.get_close_matches(
            normalized, normalized_names, n=1, cutoff=0.6
        )

        if matches:
            # Find original name
            idx = normalized_names.index(matches[0])
            matched_name = all_names[idx]
            confidence = difflib.SequenceMatcher(None, normalized, matches[0]).ratio()

            return MatchAttempt(
                strategy="fuzzy",
                matched=True,
                matched_name=matched_name,
                confidence=confidence,
                benchmarks=self._get_benchmarks_for(matched_name),
            )

        return MatchAttempt(strategy="fuzzy", matched=False)

    def _try_manual_mapping(self, model_name: str) -> MatchAttempt:
        """Check manual mappings file."""
        leaderboard_name = lookup_manual_mapping(model_name)

        if leaderboard_name:
            benchmarks = self._get_benchmarks_for(leaderboard_name)
            return MatchAttempt(
                strategy="manual",
                matched=True,
                matched_name=leaderboard_name,
                confidence=1.0,
                benchmarks=benchmarks if benchmarks else None,
            )

        return MatchAttempt(strategy="manual", matched=False)

    def _try_hf_lookup(self, hf_model_id: str) -> MatchAttempt:
        """Search leaderboards by HuggingFace model ID."""
        # Try various format variations
        variations = [
            hf_model_id,
            hf_model_id.split("/")[-1],  # Remove org prefix
            f"unsloth/{hf_model_id}",  # Try with unsloth prefix
        ]

        for variant in variations:
            if variant in self.evalplus_data:
                return MatchAttempt(
                    strategy="hf_lookup",
                    matched=True,
                    matched_name=variant,
                    confidence=1.0,
                    benchmarks=self._get_benchmarks_for(variant),
                )

        return MatchAttempt(strategy="hf_lookup", matched=False)

    def _normalize_name(self, name: str) -> str:
        """Normalize model name for matching."""
        result = name.lower()
        # Remove common suffixes
        for suffix in [
            "-bnb-4bit",
            "-bnb-8bit",
            "-gguf",
            "-unsloth",
            "-instruct",
            "-chat",
            "-it",
        ]:
            result = result.replace(suffix, "")
        return result.strip()

    def _get_benchmarks_for(self, model_name: str) -> dict[str, float]:
        """Get combined benchmarks from all sources for a model."""
        benchmarks = {}

        # Get from EvalPlus (HumanEval)
        if model_name in self.evalplus_data:
            benchmarks.update(self.evalplus_data[model_name])

        # Get from Open LLM Leaderboard (MMLU-PRO, MATH)
        if model_name in self.open_llm_data:
            benchmarks.update(self.open_llm_data[model_name])
        else:
            # Try fuzzy match for Open LLM data
            normalized = self._normalize_name(model_name)
            for llm_name, llm_benchmarks in self.open_llm_data.items():
                if self._normalize_name(llm_name) == normalized:
                    benchmarks.update(llm_benchmarks)
                    break

        return benchmarks


def fetch_unsloth_models(session) -> list[dict]:
    """Fetch models from Unsloth HuggingFace."""
    logger.info("Fetching models from Unsloth HF...")
    all_models = []
    url = UNSLOTH_API_URL

    while url:
        try:
            resp = session.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()

            if not data:
                break

            all_models.extend(data)
            logger.info(f"Fetched {len(data)} models. Total: {len(all_models)}")

            # Check for pagination
            url = None
            if "Link" in resp.headers:
                links = resp.headers["Link"].split(",")
                for link in links:
                    parts = link.split(";")
                    if len(parts) > 1 and 'rel="next"' in parts[1]:
                        url = parts[0].strip("<> ")
                        break
        except Exception as e:
            logger.error(f"Error fetching page: {e}")
            break

    return all_models


def get_model_config(model_id: str, session) -> Optional[dict]:
    """Fetch config.json for a model."""
    url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 200:
            return resp.json()
    except Exception as e:
        logger.debug(f"Error fetching config for {model_id}: {e}")
    return None


def detect_domains_and_capabilities(
    config: dict, model_id: str
) -> tuple[list[str], list[str]]:
    """Detect model domains and capabilities from config."""
    domains = ["general"]
    capabilities = []

    archs = config.get("architectures", [])
    arch_str = str(archs).lower()
    model_type = config.get("model_type", "").lower()

    # Vision Detection
    if "vision" in model_type or "vl" in model_type or "image" in model_type:
        domains.append("vision")
        capabilities.append("vision")
    elif any(
        x in arch_str for x in ["llava", "qwen2vl", "idefics", "pixtral", "vision"]
    ):
        domains.append("vision")
        capabilities.append("vision")

    # Code Detection
    if "coder" in model_id.lower() or "code" in model_id.lower():
        domains.append("code")

    return list(set(domains)), list(set(capabilities))


def parse_config_to_model_entry(name: str, config: dict, model_id: str) -> dict:
    """Parse HuggingFace config to model entry."""
    # Flatten Vision Configs
    eff_config = config
    if "text_config" in config:
        eff_config = {**config, **config["text_config"]}

    # Parameter Extraction
    hidden_size = eff_config.get("hidden_size", eff_config.get("d_model", 0))
    num_layers = eff_config.get("num_hidden_layers", eff_config.get("n_layer", 0))
    num_heads = eff_config.get("num_attention_heads", eff_config.get("n_head", 0))
    num_kv_heads = eff_config.get("num_key_value_heads")
    if num_kv_heads is None:
        num_kv_heads = num_heads
    vocab_size = eff_config.get("vocab_size", 32000)
    max_position_embeddings = eff_config.get("max_position_embeddings", 2048)

    # MoE Logic
    num_experts = eff_config.get("num_local_experts", 0)
    active_experts = eff_config.get("num_experts_per_tok", 0)
    is_moe = num_experts > 1

    # Parameter Estimation
    total_params_b = 0.0
    size_match = re.search(r"(\d+(?:\.\d+)?)b", name.lower())
    if size_match:
        total_params_b = float(size_match.group(1))

    if total_params_b == 0.0 and hidden_size and num_layers:
        raw_params = (12 * num_layers * (hidden_size**2)) + (vocab_size * hidden_size)
        if is_moe:
            raw_params = raw_params * num_experts * 0.8
        total_params_b = round(raw_params / 1e9, 2)

    # Active Parameters
    active_params_b = total_params_b
    if is_moe and num_experts > 0 and active_experts > 0:
        active_params_b = total_params_b * (active_experts / num_experts)
        active_params_b = round(active_params_b, 2)

    # Domains & Capabilities
    domains, capabilities = detect_domains_and_capabilities(config, name)

    # Clean name
    clean_name = name.split("/")[-1]
    for suffix in ["-bnb-4bit", "-bnb-8bit", "-GGUF", "-unsloth"]:
        clean_name = clean_name.replace(suffix, "")

    return {
        "name": clean_name,
        "total_params_b": total_params_b,
        "active_params_b": active_params_b,
        "is_moe": is_moe,
        "num_experts": num_experts if is_moe else None,
        "num_active_experts": active_experts if is_moe else None,
        "hidden_dim": hidden_size,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "vocab_size": vocab_size,
        "max_context_length": max_position_embeddings,
        "effective_context_length": max_position_embeddings,
        "domains": domains,
        "capabilities": capabilities,
        "benchmarks": {
            "mmlu": None,
            "mmlu_pro": None,
            "humaneval": None,
            "gsm8k": None,
            "math": None,
            "bfcl": None,
        },
        "hf_model_id": model_id,
        "notes": "Auto-imported from Unsloth",
    }


def main(
    dry_run: bool = False,
    matching_report_path: Optional[Path] = None,
) -> dict[str, Any]:
    """
    Regenerate models.json with benchmark data.

    Args:
        dry_run: If True, parse but don't write files
        matching_report_path: Optional path for matching report

    Returns:
        Result dict with count and status
    """
    result = {
        "success": True,
        "count": 0,
        "errors": [],
        "warnings": [],
    }

    session = create_session()

    # Initialize benchmark matcher
    matcher = BenchmarkMatcher()
    matcher.load_all_sources()

    # Fetch Unsloth models
    try:
        unsloth_models = fetch_unsloth_models(session)
        logger.info(f"Found {len(unsloth_models)} Unsloth models")
    except Exception as e:
        result["success"] = False
        result["errors"].append(f"Fetch failed: {e}")
        return result

    valid_models = []
    seen_names: set[str] = set()
    matching_results: list[dict] = []

    for m in unsloth_models:
        model_id = m.get("modelId", "")
        if not model_id:
            continue

        # Get config
        config = get_model_config(model_id, session)
        if not config:
            continue

        name = model_id.split("/")[-1]
        entry = parse_config_to_model_entry(name, config, model_id)

        # Validate critical fields
        if (
            entry["hidden_dim"] == 0
            or entry["num_heads"] == 0
            or entry["num_layers"] == 0
        ):
            continue

        if entry["name"] in seen_names:
            continue

        # Match benchmarks
        match_result = matcher.match(entry["name"], model_id)
        matching_results.append(
            {
                "model_name": entry["name"],
                "hf_model_id": model_id,
                "strategies": [
                    {
                        "strategy": s.strategy,
                        "matched": s.matched,
                        "matched_name": s.matched_name,
                        "confidence": s.confidence,
                    }
                    for s in match_result.strategies
                ],
                "best_match": match_result.best_match.strategy
                if match_result.best_match
                else None,
                "benchmarks_found": bool(match_result.final_benchmarks),
            }
        )

        # Apply matched benchmarks
        if match_result.final_benchmarks:
            entry["benchmarks"].update(match_result.final_benchmarks)
            logger.debug(
                f"Matched benchmarks for {entry['name']}: {match_result.final_benchmarks}"
            )

        # Validate with Pydantic
        try:
            benchmarks = BenchmarksSchema(**entry["benchmarks"])
            entry["benchmarks"] = benchmarks.model_dump()
            valid_models.append(entry)
            seen_names.add(entry["name"])
        except ValidationError as e:
            result["warnings"].append(f"Validation failed for {entry['name']}: {e}")

    logger.info(f"Processed {len(valid_models)} valid models")
    result["count"] = len(valid_models)

    if dry_run:
        logger.info(f"[DRY RUN] Would write {len(valid_models)} models")
        return result

    # Write models.json
    output = {"models": valid_models}
    try:
        with open(MODELS_JSON_PATH, "w") as f:
            json.dump(output, f, indent=2)
        logger.info(f"Saved {len(valid_models)} models to {MODELS_JSON_PATH}")
    except IOError as e:
        result["success"] = False
        result["errors"].append(f"Write failed: {e}")

    # Write matching report
    report_path = matching_report_path or MATCHING_REPORT_PATH
    try:
        with open(report_path, "w") as f:
            json.dump(matching_results, f, indent=2)
        logger.info(f"Saved matching report to {report_path}")
    except IOError as e:
        result["warnings"].append(f"Failed to write matching report: {e}")

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
