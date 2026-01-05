"""
Data Generator Script

Generates pre-computed JSON data for the frontend by combining:
- GPU database
- Model registry
- Quantization specifications
- All computed metrics (VRAM, quality, performance, efficiency)

The generated data is a comprehensive JSON file that the React frontend
consumes without needing any runtime calculations.

Usage:
    python -m can_i_run.generate
    # or after pip install:
    generate-data
"""

import json
from pathlib import Path

from .models import (
    GPU,
    Model,
    Quantization,
    Benchmarks,
    ModelDomain,
    ModelCapability,
    QualityTier,
)
from .vram import calculate_model_vram, CUDA_OVERHEAD_GB
from .kv_cache import calculate_kv_cache, CONTEXT_POSITIONS
from .quality import calculate_quality_score
from .performance import estimate_tokens_per_second


def load_json_data(data_dir: Path) -> tuple[list[dict], list[dict], list[dict]]:
    """Load raw JSON data from files."""
    with open(data_dir / "gpus.json") as f:
        gpus_data = json.load(f)["gpus"]

    with open(data_dir / "models.json") as f:
        models_data = json.load(f)["models"]

    with open(data_dir / "quantizations.json") as f:
        quants_data = json.load(f)["quantizations"]

    return gpus_data, models_data, quants_data


def parse_gpu(data: dict) -> GPU:
    """Parse GPU from JSON data."""
    return GPU(
        name=data["name"],
        vendor=data["vendor"],
        vram_gb=data["vram_gb"],
        memory_bandwidth_gbps=data["memory_bandwidth_gbps"],
        fp16_tflops=data.get("fp16_tflops"),
        int8_tops=data.get("int8_tops"),
        generation=data.get("generation"),
    )


def parse_model(data: dict) -> Model:
    """Parse Model from JSON data."""
    benchmarks_data = data.get("benchmarks", {})
    benchmarks = Benchmarks(
        mmlu=benchmarks_data.get("mmlu"),
        mmlu_pro=benchmarks_data.get("mmlu_pro"),
        humaneval=benchmarks_data.get("humaneval"),
        gsm8k=benchmarks_data.get("gsm8k"),
        math=benchmarks_data.get("math"),
        bfcl=benchmarks_data.get("bfcl"),
        tool_accuracy=benchmarks_data.get("tool_accuracy"),
    )

    domains = [ModelDomain(d) for d in data.get("domains", [])]
    capabilities = [ModelCapability(c) for c in data.get("capabilities", [])]

    return Model(
        name=data["name"],
        total_params_b=data["total_params_b"],
        active_params_b=data["active_params_b"],
        is_moe=data["is_moe"],
        hidden_dim=data["hidden_dim"],
        num_layers=data["num_layers"],
        num_heads=data["num_heads"],
        num_kv_heads=data["num_kv_heads"],
        vocab_size=data["vocab_size"],
        max_context_length=data["max_context_length"],
        effective_context_length=data["effective_context_length"],
        domains=domains,
        capabilities=capabilities,
        benchmarks=benchmarks,
        num_experts=data.get("num_experts"),
        num_active_experts=data.get("num_active_experts"),
        notes=data.get("notes"),
    )


def parse_quantization(data: dict) -> Quantization:
    """Parse Quantization from JSON data."""
    return Quantization(
        name=data["name"],
        bits_per_weight=data["bits_per_weight"],
        quality_factor=data["quality_factor"],
        ppl_increase=data["ppl_increase"],
        quality_tier=QualityTier(data["quality_tier"]),
        source=data["source"],
    )


def generate_model_quant_data(
    model: Model, quant: Quantization, domain: ModelDomain = ModelDomain.GENERAL
) -> dict:
    """Generate computed data for a model+quantization combination."""
    model_vram = calculate_model_vram(model, quant)
    quality = calculate_quality_score(model, quant, domain)

    return {
        "model_name": model.name,
        "quant_name": quant.name,
        "model_vram_gb": round(model_vram, 2),
        "quality_score": round(quality, 2),
        "is_moe": model.is_moe,
        "total_params_b": model.total_params_b,
        "active_params_b": model.active_params_b,
        "max_context": model.max_context_length,
        "effective_context": model.effective_context_length,
        "domains": [d.value for d in model.domains],
        "capabilities": [c.value for c in model.capabilities],
        "quant_tier": quant.quality_tier.value,
        "bits_per_weight": quant.bits_per_weight,
    }


def generate_kv_cache_data(model: Model) -> dict:
    """Generate KV cache data for all context positions."""
    kv_data = {}
    for ctx in CONTEXT_POSITIONS:
        if ctx <= model.max_context_length:
            kv_cache = calculate_kv_cache(model, ctx)
            kv_data[str(ctx)] = round(kv_cache, 3)
    return kv_data


def generate_performance_data(
    model: Model, quant: Quantization, gpus: list[GPU]
) -> dict:
    """Generate performance estimates for all GPUs."""
    perf_data = {}
    for gpu in gpus:
        tps = estimate_tokens_per_second(model, quant, gpu)
        perf_data[gpu.name] = round(tps, 1)
    return perf_data


def generate_full_dataset(
    gpus: list[GPU], models: list[Model], quants: list[Quantization]
) -> dict:
    """Generate the full dataset for the frontend."""
    # GPU data (simple, just serialize)
    gpu_list = [
        {
            "name": gpu.name,
            "vendor": gpu.vendor,
            "vram_gb": gpu.vram_gb,
            "bandwidth_gbps": gpu.memory_bandwidth_gbps,
            "generation": gpu.generation,
        }
        for gpu in gpus
    ]

    # Model + Quantization combinations with precomputed data
    model_quant_combos = []
    for model in models:
        # Generate KV cache data once per model
        kv_data = generate_kv_cache_data(model)

        for quant in quants:
            combo = generate_model_quant_data(model, quant)
            combo["kv_cache"] = kv_data
            combo["performance"] = generate_performance_data(model, quant, gpus)
            model_quant_combos.append(combo)

    # Quantization reference data
    quant_list = [
        {
            "name": q.name,
            "bits": q.bits_per_weight,
            "quality_factor": q.quality_factor,
            "tier": q.quality_tier.value,
            "ppl_increase": q.ppl_increase,
        }
        for q in quants
    ]

    # Context positions for slider
    from .kv_cache import format_context_length

    context_positions = [
        {"value": pos, "label": format_context_length(pos)} for pos in CONTEXT_POSITIONS
    ]

    return {
        "gpus": gpu_list,
        "quantizations": quant_list,
        "context_positions": context_positions,
        "model_quant_combos": model_quant_combos,
        "metadata": {
            "total_gpus": len(gpus),
            "total_models": len(models),
            "total_quants": len(quants),
            "total_combos": len(model_quant_combos),
            "overhead_gb": CUDA_OVERHEAD_GB,
        },
    }


def main():
    """Main entry point for data generation."""
    # Find data directory
    script_dir = Path(__file__).parent
    # Try different possible locations
    possible_data_dirs = [
        script_dir.parent.parent / "data",  # backend/can_i_run -> data
        script_dir.parent.parent.parent / "data",  # installed package
        Path.cwd() / "data",
    ]

    data_dir = None
    for d in possible_data_dirs:
        if d.exists():
            data_dir = d
            break

    if data_dir is None:
        print("Error: Could not find data directory")
        return 1

    print(f"Loading data from {data_dir}")

    # Load raw data
    gpus_data, models_data, quants_data = load_json_data(data_dir)

    # Parse into objects
    gpus = [parse_gpu(g) for g in gpus_data]
    models = [parse_model(m) for m in models_data]
    quants = [parse_quantization(q) for q in quants_data]

    print(f"Loaded {len(gpus)} GPUs, {len(models)} models, {len(quants)} quantizations")

    # Generate full dataset
    print("Generating computed data...")
    dataset = generate_full_dataset(gpus, models, quants)

    # Write to frontend data directory
    output_dir = data_dir.parent / "frontend" / "src" / "data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "computed.json"

    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2)

    print(f"Wrote {len(dataset['model_quant_combos'])} combos to {output_file}")
    print(f"Metadata: {dataset['metadata']}")

    return 0


if __name__ == "__main__":
    exit(main())
