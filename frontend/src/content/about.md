# How It Works

This document explains the mathematics, algorithms, and data sources used by the GPU-to-Model Pareto Selector. It's intended for technical users who want to validate the correctness of the system.

---

## 1. Overview

This tool solves a multi-objective optimization problem: given a GPU's VRAM and bandwidth constraints, find the Pareto-optimal combinations of (model, quantization) that maximize quality, performance, or efficiency.

**Three optimization axes:**
- **Quality**: Benchmark score adjusted for quantization degradation
- **Performance**: Tokens per second (inference speed)
- **Efficiency**: Quality × Speed / VRAM (best "bang for buck")

---

## 2. VRAM Calculation

VRAM usage has three components: model weights, KV cache, and overhead.

### 2.1 Model Weights

The static memory footprint is determined by parameter count and quantization bit-depth:

$$V_{weights} = \frac{P \times B}{8 \times 10^9} \text{ GB}$$

Where:
- $P$ = Total parameters (in absolute count, e.g., 70.6 billion for Llama-3.1-70B)
- $B$ = Average bits per weight (e.g., 4.83 for Q4_K_M)

**MoE models**: Use TOTAL parameters (all experts), not just active params. All experts must reside in VRAM even though only a subset are computed per token.

| Model | Total Params | Active Params | VRAM Calculation Uses |
|-------|-------------|---------------|----------------------|
| Llama-3.1-70B | 70.6B | 70.6B | 70.6B (dense) |
| DeepSeek V3 | 671B | 37B | 671B (all experts) |
| Qwen3-30B-A3B | 30.5B | 3B | 30.5B (all experts) |

### 2.2 KV Cache

The KV cache stores attention keys and values for all tokens in the context window. It scales linearly with context length:

$$V_{kv} = \frac{2 \times L \times H_{kv} \times D_{head} \times C \times 2}{10^9} \text{ GB}$$

Where:
- $2$ = Key AND Value caches
- $L$ = Number of transformer layers (e.g., 80 for Llama-70B)
- $H_{kv}$ = Number of KV heads (often < attention heads due to GQA)
- $D_{head}$ = Head dimension = $\frac{H_{dim}}{H_{attn}}$ (typically 128)
- $C$ = Context length in tokens
- $2$ = Bytes per value (FP16)

**Example: Llama-3.1-70B at 8K context**
```
= 2 × 80 × 8 × 128 × 8192 × 2 / 10^9
= 2.68 GB
```

**Grouped Query Attention (GQA)**: Modern models like Llama-3 use GQA where $H_{kv} < H_{attn}$. This significantly reduces KV cache size compared to full multi-head attention.

### 2.3 Total VRAM

$$V_{total} = V_{weights} + V_{kv} + 0.5 \text{ GB}$$

The 0.5 GB overhead accounts for CUDA kernels, driver allocations, and temporary tensors.

The UI shows a "safe" indicator when headroom (GPU VRAM - V_total) exceeds 1 GB, providing buffer for runtime memory spikes.

---

## 3. Quantization

Quantization reduces the precision of neural network weights to save memory. This section explains the different methods and their quality tradeoffs.

### 3.1 What is Quantization?

Neural network weights are typically stored in 16-bit floating point (FP16). Quantization reduces this to fewer bits (8, 4, 3, or even 2 bits), trading memory for minimal quality loss.

**Why it works**: Neural networks are inherently tolerant of precision loss. The weights learned during training contain redundancy, and the model's behavior is robust to small perturbations.

### 3.2 Quantization Methods

**Naive Quantization (Q4_0, Q8_0)**
- Uniform bit allocation across all weights
- Simple but suboptimal—treats all weights equally

**K-Quants (Q4_K_M, Q5_K_S, etc.)**
- Uses k-means clustering to group similar weights
- Allocates more precision to important weight distributions
- The "K" indicates k-means-based quantization
- Suffixes: S=Small, M=Medium, L=Large (precision levels)

**Importance Quantization (IQ4_XS, IQ2_XXS, etc.)**
- Uses an importance matrix to identify critical weights
- Allocates more bits to weights with higher impact on output
- Achieves better quality at extreme compression (2-3 bits)
- The "I" prefix indicates importance-weighted allocation

### 3.3 Quality Measurement: Perplexity

Perplexity (PPL) measures how "confused" a language model is when predicting the next token. Lower is better.

$$\text{PPL} = e^{-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i | x_{<i})}$$

**PPL Increase** = PPL(quantized) - PPL(FP16). This measures the quality loss from quantization.

| PPL Increase | Interpretation |
|-------------|----------------|
| < 0.01 | Imperceptible loss |
| 0.01 - 0.1 | Very low loss, recommended |
| 0.1 - 0.3 | Noticeable in edge cases |
| 0.3 - 1.0 | Visible degradation |
| > 1.0 | Significant quality loss |

### 3.4 Model Size Effect on Quantization

**Critical insight**: Larger models tolerate quantization better than smaller models.

From Intel Low-bit Leaderboard research:

| Model Size | Size Penalty Factor | Degradation |
|-----------|---------------------|-------------|
| < 10B (small) | 0.85 | 15% extra degradation |
| 10-30B (medium) | 0.92 | 8% extra degradation |
| 30-65B (large) | 0.97 | 3% extra degradation |
| > 65B (xlarge) | 1.00 | Minimal extra degradation |

**Implication**: A 70B model at Q4_K_M often outperforms a 13B model at FP16 on benchmarks, despite using similar VRAM.

### 3.5 Full Quantization Reference

All quantization formats supported by this tool, with data from llama.cpp benchmarks on LLaMA-3-8B:

| Name | Bits/Weight | PPL Increase | Quality Factor | Tier | Source |
|------|-------------|--------------|----------------|------|--------|
| F16 | 16.00 | 0.000 | 1.000 | Near Lossless | Baseline |
| Q8_0 | 8.50 | 0.003 | 0.999 | Near Lossless | llama.cpp |
| Q6_K | 6.57 | 0.022 | 0.996 | Very Low Loss | llama.cpp |
| Q5_K_M | 5.67 | 0.057 | 0.990 | Recommended | llama.cpp |
| Q5_K_S | 5.53 | 0.105 | 0.983 | Balanced | llama.cpp |
| **Q4_K_M** | **4.83** | **0.054** | **0.991** | **Recommended** | llama.cpp |
| Q4_K_S | 4.58 | 0.080 | 0.987 | Balanced | llama.cpp |
| IQ4_XS | 4.25 | 0.090 | 0.985 | Balanced | llama.cpp IQ |
| Q4_0 | 4.34 | 0.469 | 0.925 | Noticeable Loss | llama.cpp (legacy) |
| Q3_K_M | 3.89 | 0.244 | 0.960 | Noticeable Loss | llama.cpp |
| Q3_K_S | 3.50 | 0.657 | 0.890 | High Loss | llama.cpp |
| IQ3_M | 3.44 | 0.350 | 0.940 | Noticeable Loss | llama.cpp IQ |
| IQ3_S | 3.25 | 0.450 | 0.920 | Noticeable Loss | llama.cpp IQ |
| Q2_K | 3.00 | 0.870 | 0.870 | High Loss | llama.cpp |
| IQ2_M | 2.70 | 1.200 | 0.820 | High Loss | llama.cpp IQ |
| IQ2_S | 2.50 | 1.800 | 0.780 | Extreme Loss | llama.cpp IQ |
| IQ2_XS | 2.30 | 2.500 | 0.750 | Extreme Loss | llama.cpp IQ |
| IQ2_XXS | 2.10 | 3.520 | 0.700 | Extreme Loss | llama.cpp |
| IQ1_M | 1.75 | 8.000 | 0.500 | Extreme Loss | llama.cpp IQ |
| IQ1_S | 1.50 | 12.000 | 0.400 | Extreme Loss | llama.cpp IQ |

> **Recommendation**: Q4_K_M is the "golden standard" for most use cases—it offers 99.1% quality retention with ~70% VRAM savings compared to FP16.

---

## 4. Quality Score Calculation

The quality score combines benchmark performance with quantization degradation and model size effects.

### 4.1 Base Quality (Benchmark Composite)

We compute a weighted average of benchmark scores based on the selected domain:

| Domain | MMLU | HumanEval | GSM8K | BFCL |
|--------|------|-----------|-------|------|
| General | 0.50 | 0.25 | 0.25 | 0.00 |
| Code | 0.20 | 0.60 | 0.20 | 0.00 |
| Tool-Calling | 0.30 | 0.20 | 0.00 | 0.50 |
| Math | 0.30 | 0.20 | 0.50 | 0.00 |
| Vision | 0.50 | 0.25 | 0.25 | 0.00 |

$$Q_{base} = \sum_{b \in \text{benchmarks}} w_b \times \text{score}_b$$

Missing benchmarks are filled with a default score of 50.

### 4.2 Final Quality Score

The final score applies quantization and size adjustments:

$$Q_{final} = Q_{base} \times \text{quality\_factor} \times \text{size\_penalty}$$

**Example**: Llama-3.1-8B with Q4_K_M
```
Base = 68.1 × 0.5 + 62.5 × 0.25 + 75.0 × 0.25 = 68.425
Quality Factor = 0.991
Size Penalty = 0.85 (small model)
Final = 68.425 × 0.991 × 0.85 = 57.63
```

---

## 5. Performance Estimation

Token generation speed is estimated from GPU memory bandwidth and model size.

### 5.1 Why Memory-Bandwidth Bound?

LLM inference has two phases:
1. **Prefill** (processing prompt): Compute-bound, benefits from TFLOPS
2. **Decode** (generating tokens): Memory-bound, limited by bandwidth

Since most time is spent in decode, **memory bandwidth is the primary constraint**. Each generated token requires reading ALL model weights from VRAM once.

### 5.2 Theoretical Estimation

When no measured benchmarks are available:

$$T/s = \frac{BW}{V_{weights}} \times \eta$$

Where:
- $BW$ = GPU memory bandwidth (GB/s)
- $V_{weights}$ = Model size in GB
- $\eta$ = Efficiency factor

**Efficiency factors by quantization**:

| Quantization Level | Bits | Efficiency ($\eta$) |
|-------------------|------|---------------------|
| High (FP16, Q8) | ≥ 8 | 0.75 |
| Medium (Q4-Q6) | 4-6 | 0.70 |
| Low (Q2-Q3) | < 4 | 0.60 |

Lower bit quantization has more dequantization overhead, reducing efficiency.

### 5.3 Scaled Estimation

When the GPU has measured baseline performance (LLaMA-3-8B at Q4_K_M), we scale from that measurement:

$$T/s = T_{baseline} \times \frac{V_{baseline}}{V_{model}}$$

This is more accurate than pure theoretical calculation because it incorporates real-world software stack efficiency.

### 5.4 MoE Model Performance

MoE models have a subtle performance characteristic:
- **Memory**: Reads ALL parameters (all experts in VRAM)
- **Compute**: Only processes ACTIVE parameters per token

In practice, MoE inference is memory-bound like dense models, with an additional 10% overhead for expert routing:

$$T/s_{MoE} = T/s_{dense} \times 0.90$$

---

## 6. Efficiency Metric

The efficiency score helps find the "best bang for VRAM buck":

$$E = \frac{\sqrt{Q_{norm} \times P_{norm}}}{VRAM} \times 100$$

Where:
- $Q_{norm} = \min(Q_{final} / 90, 1.0)$ — Normalized quality (max ~90 for top models)
- $P_{norm} = \min(T/s / 150, 1.0)$ — Normalized performance (max ~150 tok/s)

**Why geometric mean?** The geometric mean ensures that BOTH quality AND performance must be good for high efficiency. A model with 90 quality but 10 tok/s will score lower than one with 70 quality and 70 tok/s.

**Use cases**:
- "I have 12GB VRAM, what gives me the best overall experience?"
- "Is it worth the extra VRAM to go from 8B to 13B?"

---

## 7. Pareto Optimization

### 7.1 Definition

A model is **Pareto-optimal** if no other model dominates it. Model A dominates Model B if:
- A has a **higher metric** at the same or lower VRAM, OR
- A has the **same metric** at lower VRAM

Models that are dominated are suboptimal—there's always a better choice on the Pareto frontier.

### 7.2 Three Independent Frontiers

We compute three separate Pareto frontiers:

1. **Quality Frontier**: Maximize benchmark score for given VRAM
2. **Performance Frontier**: Maximize tokens/sec for given VRAM
3. **Efficiency Frontier**: Maximize quality×speed/VRAM

**Models on multiple frontiers** are particularly well-balanced choices.

### 7.3 Algorithm

```python
def compute_pareto_frontier(candidates):
    # Sort by metric descending (higher is better)
    sorted_candidates = sorted(candidates, key=lambda c: c.metric, reverse=True)

    frontier = []
    min_vram = infinity

    for candidate in sorted_candidates:
        # Pareto optimal if: highest metric seen so far for its VRAM
        if candidate.vram < min_vram:
            frontier.append(candidate)
            min_vram = candidate.vram

    return frontier
```

---

## 8. Data Sources

### GPU Specifications

| Source | Data | How We Use It |
|--------|------|---------------|
| dbgpu (Python library) | VRAM, bandwidth, TFLOPS, architecture | Aggregates data from TechPowerUp's GPU database. We use this library to extract specs for 50+ GPUs. |
| XiongjieDai/GPU-Benchmarks-on-LLM-Inference | Measured tok/s for 30+ GPUs | Real-world LLaMA-3-8B Q4_K_M benchmarks. We use baseline_tps_8b_q4 to scale performance estimates. |

**GPU Benchmarks Repository**: [github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference)

### Quantization Quality

| Source | Data | How We Use It |
|--------|------|---------------|
| llama.cpp quantize README | Perplexity benchmarks on LLaMA-3-8B | Primary source for PPL increase values. Located in the llama.cpp repository's examples/quantize/README.md |
| Intel Low-bit Leaderboard | Model size vs quantization tolerance | Source for size penalty factors (small models degrade more than large models at same quant level) |

**llama.cpp Quantization Docs**: [github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md](https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md)

**Intel Low-bit Leaderboard**: [huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard](https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard)

### Model Benchmarks

| Source | Data | How We Use It |
|--------|------|---------------|
| HuggingFace Model Cards | Architecture (layers, heads, hidden_dim) | Extracted from individual model repositories for KV cache calculation |
| Unsloth | Curated model catalog | Primary list of inference-optimized models we support |
| Open LLM Leaderboard | MMLU, HumanEval, GSM8K scores | Benchmark scores for quality calculation |
| Berkeley Function Calling Leaderboard | BFCL scores | Tool-calling benchmark scores for the tool-calling domain filter |

**Open LLM Leaderboard**: [huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard](https://huggingface.co/spaces/open-llm-leaderboard/open_llm_leaderboard)

**BFCL Leaderboard**: [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html)
