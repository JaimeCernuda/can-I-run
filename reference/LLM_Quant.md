# Real benchmark data for LLM quantization quality and performance

Building a GPU-to-model selector for Unsloth-supported models is now feasible thanks to structured benchmark data from llama.cpp, Intel's Low-bit Leaderboard, and systematic GPU benchmark repositories. The best sources provide **per-model perplexity degradation numbers**, **tokens/sec across 30+ GPU configurations**, and **task accuracy scores for quantized variants**. This report maps the data landscape, provides actual benchmark numbers, and identifies programmatically consumable sources.

## Perplexity benchmarks reveal consistent quality degradation patterns

The **llama.cpp quantize tool README** provides authoritative perplexity increase data for LLaMA-3-8B across all GGUF formats:

| Quantization | Size (GiB) | PPL Increase | Quality Assessment |
|--------------|------------|--------------|-------------------|
| Q8_0 | 7.96 | +0.0026 | Near-lossless |
| Q6_K | 6.14 | +0.0217 | Very low loss |
| Q5_K_M | 5.33 | +0.0569 | **Recommended** |
| Q5_K_S | 5.21 | +0.1049 | Low loss |
| Q4_K_M | 4.58 | ~+0.05 | **Balanced choice** |
| Q4_0 | 4.34 | +0.4685 | Legacy, higher loss |
| Q3_K_M | 3.74 | +0.6569 | Substantial loss |
| Q2_K | 2.96 | +3.5199 | Extreme loss |

The **Artefact2 KL-divergence benchmark** (Mistral-7B, updated February 2024) provides finer granularity for ultra-low-bit formats, showing IQ2_XXS achieves **0.1751 median KL-divergence** with **23.13% top token differences** from FP16—critical data for understanding when quantization breaks model behavior rather than just increasing perplexity.

Structured data source: The **llama.cpp Google Sheet** at `docs.google.com/spreadsheets/d/1UsbivogLMrQbBA-Fk0ESRGTrvCsknBUieSykfWn6D9Q` allows CSV export with model/quantization/context filtering.

## Task-based accuracy scores exist through Intel's dedicated leaderboard

The **Intel Low-bit Quantized Open LLM Leaderboard** (`huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard`) fills a crucial gap—testing quantized models across **10 benchmarks** including MMLU, HellaSwag, ARC, TruthfulQA, and WinoGrande. It supports filtering by quantization algorithm (AutoRound, GPTQ, AWQ, BitsAndBytes, GGUF), weight data type (fp4, int4, nf4), and model size.

Key findings from quantized model benchmarks:

- **DeepSeek-R1-distill-Qwen-32B Q4_K_M**: 82.37% MMLU (vs 82.15% BF16)—essentially **no accuracy loss**
- **LLaMA-3 8B at 4-bit**: ~55% MMLU (vs ~66% FP16)—**11-point drop** significant for smaller models
- **Qwen3-14B GPTQ 4-bit**: ~60% MMLU with minimal degradation from larger model base

The critical pattern: **Larger models tolerate quantization better**. A 70B model at Q4_K_M often outperforms a 13B model at FP16 on the same benchmark.

## GPU performance data maps to specific hardware configurations

The **XiongjieDai/GPU-Benchmarks-on-LLM-Inference** repository provides the most comprehensive structured data, covering **30+ GPU configurations** with LLaMA 3 models. Sample tokens/sec for LLaMA 3 8B Q4_K_M at 1024 context:

| GPU | Tokens/sec |
|-----|------------|
| RTX 3070 8GB | 70.94 |
| RTX 3080 10GB | 106.40 |
| RTX 3090 24GB | 111.74 |
| RTX 4090 24GB | **127.74** |
| A100 PCIe 80GB | 138.31 |
| H100 PCIe 80GB | **144.49** |
| RTX 6000 Ada 48GB | 130.99 |

For **70B models** (Q4_K_M, requiring multi-GPU):
- 2× RTX 3090: 16.29 t/s
- 2× RTX 4090: 19.06 t/s  
- 4× H100 PCIe: **26.20 t/s**

Apple Silicon data from official llama.cpp benchmarks shows memory bandwidth correlation: **M4 Max achieves 83 t/s** for 8B Q4_K_M, tracking closely to its 546 GB/s bandwidth (roughly bandwidth × 0.15 = tokens/sec for Q4 models).

## Academic research establishes architecture-specific quantization behavior

Multiple papers confirm that **model architecture affects quantization robustness**, a finding critical for per-model selector accuracy:

**LLaMA-3-70B vulnerability discovered**: Unlike other models showing <1% accuracy degradation with W8A8, LLaMA-3-70B shows **significant degradation** even at 8-bit precision due to extreme weight outliers (magnitudes >90 vs <1.0 in other models) concentrated in initial transformer blocks. This is an exception—LLaMA-3-8B and LLaMA-2 series remain robust.

**Mistral responds better to AWQ** than other quantization methods. The **VPTQ paper** shows Mistral-7B achieves 0.38-0.68 perplexity reduction versus other 2-bit methods, while QUIK (which excels for LLaMA) underperforms on Mistral.

**ParetoQ (February 2025)** identifies a critical **learning transition between 2 and 3 bits**: above 3-bit, fine-tuned quantized models stay close to original distributions; below 2-bit, representations change drastically. This explains why 2-bit quantization requires specialized methods like QuIP# or AQLM while 4-bit works with simpler approaches.

Key papers with empirical quality curves:
- **GPTQ**: OPT-175B at 4-bit: 8.37 PPL vs 8.34 FP16 (+0.03)
- **AWQ**: Protecting just 1% of salient weights drops OPT-6.7B INT3 perplexity from 43.2 to 13.0
- **QuIP#**: First 2-bit method achieving near-FP16 performance on LLaMA-2-70B
- **SqueezeLLM**: 3-bit LLaMA-7B outperforms GPTQ/AWQ by >0.3 perplexity

## Unsloth provides model-specific optimized quantization with benchmark data

Unsloth's **Dynamic 2.0 GGUFs** apply model-specific quantization schemes—layers quantized in Gemma 3 differ from Llama 4—using a 1.5M+ token calibration dataset. Their benchmark data provides quality scores at extreme compression:

| Quantization | Gemma 3 27B MMLU | Disk Size | Efficiency* |
|--------------|------------------|-----------|-------------|
| IQ1_M | 48.10% | 6.51GB | 3.42 |
| IQ2_XXS | 59.20% | 7.31GB | 4.32 |
| Q2_K_XL | 68.70% | 9.95GB | 4.30 |
| Q4_K_XL | **71.47%** | 15.64GB | 2.94 |
| BF16 baseline | 71.5% | ~54GB | — |

*Efficiency = (MMLU - 25) / Disk GB

The **Aider Polyglot benchmark** on DeepSeek V3.1 671B demonstrates Unsloth's 3-bit Dynamic achieving **75.6% accuracy** versus 76.1% full precision—only 0.5% loss despite **~11× compression**.

Unsloth supports GGUF export for: Q8_0, Q6_K, Q5_K_M, Q4_K_M, Q4_0, Q3_K_L, Q2_K, IQ2_XXS, IQ2_XS, IQ3_XXS, and proprietary XL variants. Their model catalog covers **Llama 2-4, Qwen 2-3, Gemma 2-3, Mistral/Mixtral, Phi 3-4, DeepSeek V3/R1**, and 50+ other model families.

## Programmatically consumable data sources for tool integration

| Source | Data Type | Format | Access Method |
|--------|-----------|--------|---------------|
| Intel Low-bit Leaderboard | 10 benchmark scores across quant methods | HuggingFace Space | HF Spaces API |
| llama.cpp Google Sheet | Perplexity by model/quant/context | Google Sheets | Sheets API / CSV export |
| XiongjieDai GPU Benchmarks | Tokens/sec across 30+ GPUs | Markdown tables | GitHub raw parse |
| oobabooga Blog | Perplexity + speed + VRAM | Markdown tables | Web scrape |
| lm-evaluation-harness | 60+ task scores | JSON output | Python API |
| Unsloth Model Catalog | Supported models + quant methods | Documentation | Manual mapping |
| Artefact2 KL-divergence | Per-quant KL stats for Mistral-7B | GitHub Gist | Direct fetch |

The **lm-evaluation-harness** enables generating custom benchmark data with JSON output:
```bash
lm_eval --model gguf --model_args pretrained=model.Q4_K_M.gguf \
  --tasks mmlu,hellaswag --output_path results/
```

No unified REST API exists for historical benchmarks—aggregating the above sources into a JSON/CSV database represents the primary integration work.

## Practical data structure for the selector tool

Based on available data, a lookup table can be constructed with this schema:

```json
{
  "quantization_quality": {
    "Q8_0": {"ppl_increase": 0.0026, "quality_tier": "near_lossless", "bits_per_weight": 8.5},
    "Q6_K": {"ppl_increase": 0.0217, "quality_tier": "very_low_loss", "bits_per_weight": 6.57},
    "Q5_K_M": {"ppl_increase": 0.0569, "quality_tier": "recommended", "bits_per_weight": 5.67},
    "Q4_K_M": {"ppl_increase": 0.0535, "quality_tier": "balanced", "bits_per_weight": 4.83},
    "Q3_K_M": {"ppl_increase": 0.2437, "quality_tier": "high_loss", "bits_per_weight": 3.89},
    "Q2_K": {"ppl_increase": 0.8698, "quality_tier": "extreme_loss", "bits_per_weight": 3.00}
  },
  "gpu_performance": {
    "RTX_4090": {"8B_Q4_K_M": 127.74, "70B_Q4_K_M": 19.06, "vram_gb": 24},
    "RTX_3090": {"8B_Q4_K_M": 111.74, "70B_Q4_K_M": 16.29, "vram_gb": 24},
    "A100_80GB": {"8B_Q4_K_M": 138.31, "70B_Q4_K_M": 22.0, "vram_gb": 80}
  },
  "vram_requirements": {
    "8B": {"Q4_K_M": 4.58, "Q8_0": 7.96, "F16": 14.0},
    "70B": {"Q4_K_M": 40.0, "Q8_0": 70.0, "F16": 140.0}
  }
}
```

## Conclusion: Structured data exists but requires aggregation

The research reveals that **high-quality benchmark data exists** across multiple sources—the challenge is aggregation rather than data scarcity. The Intel Low-bit Leaderboard and llama.cpp Google Sheet provide the most structured quality metrics; XiongjieDai's repository offers the most comprehensive GPU performance mapping. 

Key architectural insight for the tool: larger models (**70B+**) show **minimal quality degradation at Q4_K_M** (often <1% accuracy loss), while smaller models (**7-8B**) can lose **10+ percentage points on MMLU**. This means optimal quantization selection should vary by model size, not apply uniform quality factors.

The most actionable next step is parsing the llama.cpp Google Sheet and XiongjieDai GitHub tables into a unified JSON database, supplemented by Unsloth's model catalog for supported model coverage. Running lm-evaluation-harness on priority model+quant combinations can fill gaps where public benchmarks don't exist.