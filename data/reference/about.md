# How It Works & Data Intelligence

## The Math of Memory (VRAM)

Large Language Model (LLM) inference memory usage is primarily determined by two factors: the model's static weights and the dynamic Key-Value (KV) cache.

### 1. Model Weights
The static memory footprint is a function of the parameter count ($P$) and the quantization bit-depth ($B$).

$$ V_{weights} = \frac{P \times B}{8 \times 10^9} \text{ GB} $$

*   **$P$**: Total parameter count (e.g., 8 Billion for LLaMA-3-8B).
*   **$B$**: Average bits per weight. For FP16, $B=16$. For Q4_K_M, $B \approx 4.83$.

### 2. KV Cache (Context Memory)
The KV cache stores the attention history for every token in the context window. As context ($C$) grows, this scales linearly.

$$ V_{kv} = \frac{2 \times L \times H_{kv} \times D_{head} \times C \times 2_{bytes}}{10^{9}} \text{ GB} $$

*   **$L$**: Number of layers in the model.
*   **$H_{kv}$**: Number of Key-Value heads (often smaller than total heads in GQA models like LLaMA-3).
*   **$D_{head}$**: Dimension of each head (typically 128).
*   **$C$**: Current context length (tokens).
*   **$2_{bytes}$**: KV cache is typically stored in FP16 (2 bytes) unless K-cache quantization is used.

### 3. Total VRAM Estimate
We add a buffer for activation overhead (CUDA kernels, temporary tensors), typically estimated as a small percentage of weights or a fixed overhead.

$$ V_{total} = V_{weights} + V_{kv} + V_{activation} $$

---

## Pareto Frontier Optimization

Our "Smart Select" logic uses **Pareto Optimization** to surface the best models. A model is considered **Pareto Optimal** if:
1.  No other model offers **higher quality** for the same or less VRAM.
2.  No other model requires **less VRAM** for the same or higher quality.

Models that do not meet these criteria are "dominated" by better options and are filtered out of the primary recommendations.

---

## Quantization: The Bits & Bytes

Quantization reduces the precision of model weights to save memory. Modern methods like **GGUF/k-quants** and **Importance Matrix (IQ)** quantization use smart clustering (k-means) to preserve accuracy even at low bit-depths.

### Quantization Tiers & Perplexity

| Quantization | Bits/Weight | Perplexity Increase | Quality Tier | Efficacy |
| :--- | :--- | :--- | :--- | :--- |
| **F16** | 16.00 | +0.00 | Near Lossless | Baseline |
| **Q8_0** | 8.50 | +0.0026 | Near Lossless | Perfect Accuracy |
| **Q6_K** | 6.57 | +0.0217 | Very Low Loss | Excellent |
| **Q5_K_M** | 5.67 | +0.0569 | Recommended | High Fidelity |
| **Q4_K_M** | 4.83 | +0.0535 | Recommended | **Golden Standard (Daily Driver)** |
| **Q4_0** | 4.34 | +0.4685 | Noticeable Loss | Legacy Format |
| **Q3_K_M** | 3.89 | +0.2437 | Noticeable Loss | VRAM constrained |
| **IQ3_M** | 3.44 | +0.3500 | Noticeable Loss | Modern SOTA for 3-bit |
| **Q2_K** | 2.56 | +0.8698 | High Loss | Severe Logic Degradation |
| **IQ2_XXS** | 2.10 | +3.5200 | Extreme Loss | Coherence Breakdown |

> **Note**: "Perplexity" measures how "confused" a model is. Lower increase is better.

---

## Data Sources

This application uses verified data from the following open-source repositories:

1.  **[Quantization Specifications](https://github.com/ggerganov/llama.cpp/tree/master/examples/quantize)**: `data/quantizations.json` derives bit-depths and perplexity scores directly from `llama.cpp` benchmarks.
2.  **[Model Architectures](https://huggingface.co/unsloth)**: Parameter counts and layer configs are sourced from `Unsloth` and `HuggingFace` model cards.
3.  **[Pareto Logic](https://en.wikipedia.org/wiki/Pareto_efficiency)**: The specific sorting algorithm used here matches standard multi-objective optimization definitions (`backend/can_i_run/pareto.py`).
