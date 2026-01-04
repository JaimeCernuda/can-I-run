# GPU-to-Model Pareto Selector

A React-based tool that recommends optimal LLM model/quantization combinations based on available GPU VRAM, visualized as a Pareto frontier.

## Goal

Build a web tool (GitHub Pages) where users can:
1. Select their GPU from a dropdown (or enter VRAM manually)
2. Set their desired context length via a slider
3. See a Pareto frontier of model+quantization options that fit their available VRAM
4. Understand the quality vs VRAM headroom tradeoffs

The core insight: **No one has built a unified tool that maps GPU VRAM → Pareto frontier of (model size × quantization) with quality tradeoffs.** The pieces exist separately but not assembled.

## User Flow

```
GPU Selection → VRAM (from database or manual entry)
     ↓
Context Slider → KV Cache estimate  
     ↓
Free VRAM for Model = Total VRAM - KV Cache - Overhead (~500MB)
     ↓
Model Selection: Pareto-optimal (model, quant) combinations that fit
```

## UI Design

### Unified Three-Chart View

Instead of a toggle, show **all three metrics simultaneously**. When hovering on a point in any chart, the same model is highlighted in ALL charts for holistic comparison.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FILTERS                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│ GPU: [RTX 4090 (24GB) ▼]    Context: [●────────] 8K tokens (max: 128K)      │
│                                                                              │
│ Domain: [All ▼]  ○ General  ○ Code  ● Tool-Calling  ○ Math  ○ Vision        │
│                                                                              │
│ Available for model: 21.2 GB  (KV cache: 2.3GB, overhead: 0.5GB)            │
├─────────────────────────────────────────────────────────────────────────────┤
│ [? How It Works]                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐    │
│  │ QUALITY (Benchmark) │ │ PERFORMANCE (tok/s) │ │ EFFICIENCY (Q×P/V)  │    │
│  │                     │ │                     │ │                     │    │
│  │  ▲                  │ │  ▲                  │ │  ▲                  │    │
│  │  │ ★A               │ │  │           ★E     │ │  │     [★C]         │    │
│  │  │   ╲              │ │  │         ╱        │ │  │    ╱   ╲         │    │
│  │  │    ★B            │ │  │       ★D         │ │  │  ★B     ★D       │    │
│  │  │      ╲           │ │  │     ╱            │ │  │          ╲       │    │
│  │  │      [★C]        │ │  │  [★C]            │ │  │           ★A     │    │
│  │  │         ╲        │ │  │ ╱                │ │  │                  │    │
│  │  │          ★D      │ │  │★B                │ │  │             ★E   │    │
│  │  │            ╲     │ │  │                  │ │  │                  │    │
│  │  │             ★E   │ │  │★A                │ │  │                  │    │
│  │  └──────────────▶   │ │  └──────────────▶   │ │  └──────────────▶   │    │
│  │       VRAM →        │ │       VRAM →        │ │       VRAM →        │    │
│  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘    │
│                                                                              │
│  [★C] = Currently hovered model (highlighted in ALL charts)                  │
│  ★ = Dense model    ○ = MoE model    ◆ = Tool-calling optimized             │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│ PARETO FRONTIERS: Quality ━━  Performance ━━  Efficiency ━━                 │
│ (Each chart shows its own Pareto frontier line)                              │
└─────────────────────────────────────────────────────────────────────────────┘

Legend:
  A = Llama-3.1-70B Q3_K_M (highest quality, slowest)
  B = Qwen3-32B Q4_K_M
  C = Llama-3.1-8B Q8 (balanced - good on all three metrics)
  D = Gemma-3-12B-Tool Q4 (tool-calling optimized)
  E = Llama-3.2-3B FP16 (fastest, lowest quality)
```

### Three Metrics Explained

1. **Quality (Y-axis: Benchmark Score)**
   - Composite of MMLU, HumanEval, GSM8K (weighted)
   - For tool-calling domain: weight BFCL score higher
   - Adjusted for quantization degradation

2. **Performance (Y-axis: Tokens/Second)**
   - Estimated from GPU bandwidth and model size
   - Shows real-world inference speed
   - Higher = faster response times

3. **Efficiency (Y-axis: Quality × Performance / VRAM)**
   - Composite score balancing all factors
   - Helps identify "best bang for buck" models
   - Formula: `(quality_score × tokens_per_sec) / vram_used`

### Hover Interaction

When user hovers on any point:
- Same model highlights in ALL THREE charts (synchronized)
- Tooltip shows full model details
- Helps users see tradeoffs at a glance

```
┌──────────────────────────────────────┐
│ Llama-3.1-8B Q8_0                    │
├──────────────────────────────────────┤
│ VRAM: 8.5 GB (3.7 GB headroom)       │
│ Quality: 67.2 (MMLU: 68.1)           │
│ Speed: ~45 tok/s on your GPU         │
│ Efficiency: 356                       │
├──────────────────────────────────────┤
│ Max Context: 128K ✓                  │
│ Domains: general, code               │
│ Tool-calling: ✗ (not optimized)      │
├──────────────────────────────────────┤
│ [Copy Ollama Command] [View Details] │
└──────────────────────────────────────┘
```

### UI Components

1. **GPU Dropdown**: Searchable select with common GPUs grouped by generation/vendor
   - Include "Custom" option for manual VRAM entry
   - Show VRAM prominently: "RTX 4090 (24 GB)"

2. **Context Length Slider**: Snapped to common values
   - Positions: 2K, 4K, 8K, 16K, 32K, 64K, 128K, 256K, 512K, 1M
   - Shows estimated KV cache size updating in real-time
   - Warning indicator when KV cache exceeds 50% of total VRAM

3. **VRAM Budget Display**: Clear breakdown
   - Total VRAM
   - Minus KV Cache estimate
   - Minus overhead (~500MB)
   - Equals: Available for model weights

4. **Pareto Chart**: Interactive scatter plot
   - X-axis: VRAM headroom (how much buffer you have)
   - Y-axis: Quality score (composite of benchmark scores)
   - Points are clickable with tooltips showing model details
   - Highlight the Pareto-optimal frontier line
   - Different markers for Dense (★) vs MoE (○) models

5. **Model Cards**: Below charts, expandable cards for each Pareto-optimal option
   - Model name and quantization
   - VRAM usage breakdown
   - Key benchmarks (MMLU, HumanEval, BFCL for tool-calling, etc.)
   - Ollama/llama.cpp command to run it
   - Context limit warning if selected context exceeds model's max

6. **Domain Selector**: Dropdown to filter models by use case
   - Options: All, General/Chat, Code, Tool-Calling, Math/Reasoning, Vision
   - Multi-select allowed (e.g., "Code + Tool-Calling")
   - Shows count of available models per domain
   - Tool-calling includes: Gemma 3 Tools, Mistral function-calling, Qwen 2.5-Coder, etc.

7. **Linked Multi-Chart View**: Three synchronized charts instead of a toggle
   
   ```
   ┌─────────────────────────────────────────────────────────────────────────────┐
   │  Filter: [All Domains ▼]  [Tool-Calling ▼]           Context: [8K ●───] 
   │  GPU: [RTX 4090 (24GB) ▼]     Available VRAM: 21.2 GB                       │
   ├─────────────────────────────────────────────────────────────────────────────┤
   │                                                                             │
   │  ┌─────────────────────┐ ┌─────────────────────┐ ┌─────────────────────┐   │
   │  │ QUALITY             │ │ PERFORMANCE         │ │ EFFICIENCY          │   │
   │  │                     │ │                     │ │                     │   │
   │  │ ▲                   │ │ tok/s ▲             │ │ Q×P/VRAM ▲          │   │
   │  │ │  ★ 70B-Q3        │ │ │        ★ 8B-Q4    │ │ │     ★ 8B-Q4       │   │
   │  │ │    ╲              │ │ │      ╱            │ │ │   ╱               │   │
   │  │ │  [★ 32B-Q4]◄─────┼─┼─┼────[★]◄───────────┼─┼─┼─[★]              │   │
   │  │ │      ╲            │ │ │    ╱              │ │ │ ╱                 │   │
   │  │ │        ★ 8B-Q8   │ │ │  ★ 70B-Q3        │ │ │★ 70B-Q3           │   │
   │  │ └────────────────▶  │ │ └────────────────▶  │ │ └────────────────▶  │   │
   │  │         VRAM        │ │         VRAM        │ │         VRAM        │   │
   │  └─────────────────────┘ └─────────────────────┘ └─────────────────────┘   │
   │                                                                             │
   │  [★] = Currently hovered model (highlighted across ALL charts)              │
   │  ○ = MoE model    ★ = Dense model    ◆ = Tool-calling optimized            │
   └─────────────────────────────────────────────────────────────────────────────┘
   ```
   
   **Chart Definitions:**
   - **Quality Chart**: Y = benchmark score (MMLU-weighted), X = VRAM usage
   - **Performance Chart**: Y = tokens/sec estimate, X = VRAM usage  
   - **Efficiency Chart**: Y = (Quality × Performance) / VRAM, X = VRAM usage
   
   **Linked Behavior:**
   - Hover on ANY point highlights the same model across all three charts
   - Click to select and pin a model (stays highlighted)
   - Multiple selections allowed for comparison
   - Pareto frontier line shown on each chart independently

8. **"How It Works" Panel**: Collapsible explanation of all calculations
   - VRAM breakdown formula
   - KV cache math
   - Quality scoring methodology (with data sources cited)
   - Performance estimation approach
   - Efficiency metric explanation
   - Links to source papers/benchmarks
   - Model context length limitations

## Efficiency Metric

The efficiency score helps users find the "best bang for VRAM buck":

```typescript
interface EfficiencyScore {
  // Raw efficiency: how much quality+speed per GB of VRAM
  efficiency: number;
  
  // Normalized to 0-100 scale for display
  efficiency_percentile: number;
}

function calculateEfficiency(
  qualityScore: number,      // 0-100 benchmark composite
  tokensPerSecond: number,   // Estimated tok/s
  vramRequired: number       // GB
): number {
  // Normalize quality to 0-1 range (assuming max ~90 for top models)
  const normalizedQuality = qualityScore / 90;
  
  // Normalize performance (assuming max ~150 tok/s for small models on fast GPUs)
  const normalizedPerf = Math.min(tokensPerSecond / 150, 1);
  
  // Efficiency = geometric mean of quality and perf, divided by VRAM
  const qpScore = Math.sqrt(normalizedQuality * normalizedPerf);
  
  return (qpScore / vramRequired) * 100;
}
```

**Interpretation:**
- High efficiency = good quality AND speed for the VRAM used
- A 8B model at Q4 might have HIGHER efficiency than a 70B at Q3, even though 70B has better raw quality
- Helps users on limited VRAM find optimal tradeoffs

**Use Cases:**
- "I have 12GB VRAM, what gives me the best overall experience?"
- "I need fast responses AND good quality—what's the sweet spot?"
- "Is it worth the extra VRAM to go from 8B to 13B?"

## Performance Estimation (Tokens/Second)

### The Math

Token generation speed is primarily bounded by **memory bandwidth** for inference (compute-bound only at very small batch sizes or during prefill).

```typescript
interface GPUSpecs {
  vram_gb: number;
  memory_bandwidth_gbps: number;  // e.g., RTX 4090 = 1008 GB/s
  fp16_tflops: number;            // e.g., RTX 4090 = 82.6 TFLOPS
  int8_tops?: number;             // INT8 tensor ops if available
}

function estimateTokensPerSecond(
  model: Model,
  quant: Quantization,
  gpu: GPUSpecs,
  batchSize: number = 1
): number {
  // Bytes that must be read from VRAM per token
  // For autoregressive generation: read all weights once per token
  const modelSizeBytes = model.total_params_b * 1e9 * (quant.bits_per_weight / 8);
  
  // Memory bandwidth limit (theoretical max)
  const memoryBoundTPS = gpu.memory_bandwidth_gbps * 1e9 / modelSizeBytes;
  
  // Compute bound (rarely the bottleneck for single-batch inference)
  // FLOPs per token ≈ 2 * params (multiply-add for each weight)
  const flopsPerToken = 2 * model.active_params_b * 1e9;
  const effectiveTflops = quant.bits_per_weight <= 8 
    ? (gpu.int8_tops || gpu.fp16_tflops * 2)  // INT8 doubles throughput
    : gpu.fp16_tflops;
  const computeBoundTPS = (effectiveTflops * 1e12) / flopsPerToken;
  
  // Actual speed is minimum of the two bounds
  // Apply efficiency factor (~60-80% of theoretical)
  const efficiency = 0.7;
  return Math.min(memoryBoundTPS, computeBoundTPS) * efficiency;
}
```

### Why Memory Bandwidth Matters

For single-user inference:
- **Prefill phase** (processing prompt): Compute-bound, benefits from TFLOPS
- **Decode phase** (generating tokens): Memory-bound, limited by bandwidth

Since most time is spent in decode, **memory bandwidth is the primary constraint**.

Example calculation for RTX 4090 + Llama-3.1-70B Q4_K_M:
```
Model size = 70B × 4.8 bits / 8 = 42 GB
Bandwidth = 1008 GB/s
Theoretical max = 1008 / 42 = 24 tokens/sec
With 70% efficiency = ~17 tokens/sec
```

### GPU Bandwidth Reference

| GPU | VRAM | Bandwidth | Notes |
|-----|------|-----------|-------|
| RTX 4090 | 24 GB | 1008 GB/s | Consumer king |
| RTX 4080 | 16 GB | 717 GB/s | |
| RTX 4070 Ti | 12 GB | 504 GB/s | |
| RTX 3090 | 24 GB | 936 GB/s | Good VRAM, slightly slower |
| RTX 3080 | 10 GB | 760 GB/s | |
| RTX 3070 | 8 GB | 448 GB/s | |
| A100 40GB | 40 GB | 1555 GB/s | Datacenter |
| A100 80GB | 80 GB | 2039 GB/s | Datacenter |
| H100 SXM | 80 GB | 3350 GB/s | Current fastest |
| H100 PCIe | 80 GB | 2000 GB/s | |
| RTX 6000 Ada | 48 GB | 960 GB/s | Workstation |
| M1 Max | 32-64 GB | 400 GB/s | Unified memory |
| M2 Ultra | 192 GB | 800 GB/s | Huge VRAM, moderate bandwidth |
| M4 Max | 128 GB | 546 GB/s | |

### Real GPU Performance Benchmarks

**Source**: XiongjieDai/GPU-Benchmarks-on-LLM-Inference (GitHub)

Actual measured tokens/sec for LLaMA 3 8B Q4_K_M at 1024 context:

| GPU | Tokens/sec | VRAM | Price Tier |
|-----|------------|------|------------|
| RTX 3070 8GB | 70.94 | 8 GB | Budget |
| RTX 3080 10GB | 106.40 | 10 GB | Mid |
| RTX 3090 24GB | 111.74 | 24 GB | Mid-High |
| RTX 4070 Ti 12GB | ~95 | 12 GB | Mid |
| **RTX 4090 24GB** | **127.74** | 24 GB | High |
| RTX 6000 Ada 48GB | 130.99 | 48 GB | Workstation |
| A100 PCIe 40GB | 125.50 | 40 GB | Datacenter |
| A100 PCIe 80GB | 138.31 | 80 GB | Datacenter |
| **H100 PCIe 80GB** | **144.49** | 80 GB | Datacenter |
| M4 Max | ~83 | 128 GB | Apple Silicon |

**70B Models** (Q4_K_M, multi-GPU required):

| Configuration | Tokens/sec | Notes |
|---------------|------------|-------|
| 2× RTX 3090 | 16.29 | NVLink recommended |
| 2× RTX 4090 | 19.06 | |
| 2× A100 80GB | 22.0 | |
| 4× H100 PCIe | 26.20 | |

**Performance Estimation Formula (when real data unavailable):**

```typescript
// Approximation based on memory bandwidth correlation
// Derived from benchmark data fitting
function estimateTokensPerSecond(
  modelSizeGB: number,
  gpuBandwidthGBps: number,
  quantBits: number
): number {
  // Base formula: bandwidth / model_size * efficiency
  const theoreticalMax = gpuBandwidthGBps / modelSizeGB;
  
  // Efficiency varies by quant (lower bits = less efficient due to dequant overhead)
  const quantEfficiency = quantBits >= 8 ? 0.75 : 
                          quantBits >= 4 ? 0.70 : 
                          0.60;
  
  // Apply efficiency factor
  return theoreticalMax * quantEfficiency;
}

// Example: RTX 4090 + Llama-70B Q4_K_M
// Model size = 70B × 4.83 bits / 8 = 42.3 GB
// Theoretical = 1008 / 42.3 = 23.8 tok/s
// With 70% efficiency = ~16.7 tok/s (close to measured 2× RTX 4090 = 19 tok/s)
```

**Data Sources for Performance:**
- Primary: https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference
- Secondary: llama.cpp official benchmarks
- Apple Silicon: llama.cpp M-series benchmarks

### MoE Performance Advantage

MoE models have a performance benefit: only **active parameters** are computed per token, but bandwidth reads all experts. However, with proper implementation (expert offloading, speculative expert loading), MoE can be faster than dense models of equivalent quality.

```typescript
function estimateMoETokensPerSecond(
  model: Model,  // is_moe = true
  quant: Quantization,
  gpu: GPUSpecs
): number {
  // Memory: still read total params (all experts in VRAM)
  const modelSizeBytes = model.total_params_b * 1e9 * (quant.bits_per_weight / 8);
  const memoryBoundTPS = gpu.memory_bandwidth_gbps * 1e9 / modelSizeBytes;
  
  // Compute: only active params matter
  const flopsPerToken = 2 * model.active_params_b * 1e9;
  const computeBoundTPS = (gpu.fp16_tflops * 1e12) / flopsPerToken;
  
  // MoE is more likely to be memory-bound
  return Math.min(memoryBoundTPS, computeBoundTPS) * 0.65;
}

## "How It Works" Panel Content

This collapsible panel should explain all calculations transparently to users.

### VRAM Budget Breakdown

```
┌─────────────────────────────────────────────────────────────────┐
│ YOUR GPU: RTX 4090                                              │
│ Total VRAM: 24.0 GB                                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│ ┌─────────────────────────────────────┐                         │
│ │ Model Weights        │ 18.5 GB      │ ← Depends on model +    │
│ │                      │              │   quantization          │
│ ├──────────────────────┼──────────────┤                         │
│ │ KV Cache (8K ctx)    │  2.3 GB      │ ← Scales with context   │
│ ├──────────────────────┼──────────────┤                         │
│ │ CUDA/Driver Overhead │  0.5 GB      │ ← Fixed overhead        │
│ ├──────────────────────┼──────────────┤                         │
│ │ Safety Buffer        │  2.7 GB      │ ← Headroom (11%)        │
│ └──────────────────────┴──────────────┘                         │
│                                                                 │
│ Formula:                                                        │
│   Available = Total - KV Cache - Overhead                       │
│   Headroom = Available - Model Size                             │
└─────────────────────────────────────────────────────────────────┘
```

### Model Weights Calculation

```
Model VRAM (GB) = Parameters × Bits per Weight ÷ 8 ÷ 1,073,741,824

Example: Llama-3.1-70B at Q4_K_M
  = 70,600,000,000 params × 4.8 bits ÷ 8 ÷ 1,073,741,824
  = 31.6 GB

For MoE models: Use TOTAL parameters (all experts must be in VRAM)
  DeepSeek V3: 671B total params → need full 671B in memory
  Even though only 37B are "active" per token
```

### KV Cache Calculation

```
KV Cache (GB) = 2 × Layers × KV_Heads × Head_Dim × Context × Batch × 2 bytes
               ─────────────────────────────────────────────────────────────
                                    1,073,741,824

Where:
  2           = Key AND Value caches
  Layers      = Number of transformer layers (e.g., 80 for Llama-70B)
  KV_Heads    = Number of key-value heads (may be < attention heads due to GQA)
  Head_Dim    = Hidden dimension ÷ Number of attention heads
  Context     = Sequence length (e.g., 8192 tokens)
  Batch       = Batch size (typically 1 for interactive use)
  2 bytes     = FP16 storage per value

Example: Llama-3.1-70B at 8K context
  = 2 × 80 layers × 8 KV heads × 128 head_dim × 8192 tokens × 1 × 2
  = 2.68 GB
```

### Quality Score Explanation

```
Quality Score = Base Model Quality × Quantization Factor

Where:
  Base Model Quality = Weighted benchmark scores (0-100 scale)
    = MMLU × 0.4 + HumanEval × 0.3 + GSM8K × 0.3
  
  Quantization Factor = How much quality is preserved (0.0-1.0)
    = Derived from perplexity benchmarks
    = baseline_perplexity ÷ quantized_perplexity

⚠️ Note: Quantization factors are derived from empirical 
   perplexity measurements, not arbitrary estimates.
   See data sources in documentation.
```

### Token Speed Estimation

```
Tokens/sec ≈ Memory Bandwidth ÷ Model Size × Efficiency

For decode (token generation), LLMs are memory-bandwidth bound:
  - Each token requires reading ALL model weights from VRAM
  - GPU compute is rarely the bottleneck for single-user inference

Example: RTX 4090 + Llama-70B Q4_K_M
  Bandwidth = 1,008 GB/s
  Model = 31.6 GB
  Theoretical = 1,008 ÷ 31.6 = 31.9 tok/s
  With ~70% efficiency = 22 tok/s

MoE models: Same memory read (all experts), but less compute
  → Often memory-bound, similar speed to dense models of same size
```

### Pareto Frontier Explained

```
A model is "Pareto optimal" if no other model is:
  - Higher quality at the same or lower VRAM, AND
  - Lower VRAM at the same or higher quality

The frontier shows your best options:
  ┌─────────────────────────────────────────┐
  │ Quality                                 │
  │   ▲                                     │
  │   │  A ← Best quality (uses most VRAM)  │
  │   │   ╲                                 │
  │   │    B ← Balanced choice              │
  │   │     ╲                               │
  │   │      C ← Most headroom (lower qual) │
  │   └──────────────────────────────────▶  │
  │              VRAM Headroom              │
  └─────────────────────────────────────────┘

Points OFF the line are "dominated" - there's always 
a better option on the frontier.
```

## Data Sources

### GPU Database

Use **one of these** for GPU specs:

1. **`dbgpu`** — Python library with 2000+ GPUs from TechPowerUp
   - Install: `pip install dbgpu`
   - Has: memory, bandwidth, CUDA cores, TDP, architecture, compute capability
   - Formats: JSON, CSV, PKL

2. **`gpu-info-api`** — Wikipedia-sourced, updated weekly via GitHub Actions
   - Endpoint: `https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json`
   - Covers NVIDIA, AMD, Intel
   - Good for static bundling

3. **TechPowerUp GPU Database** — For reference/validation
   - URL: https://www.techpowerup.com/gpu-specs/
   - Most comprehensive human-readable source

**Recommended approach**: Download `gpu-info-api` JSON at build time, bundle common gaming/workstation GPUs (filter to ~100 most relevant), allow manual VRAM override. **Must include memory bandwidth** for performance estimation.

```typescript
interface GPU {
  name: string;              // "RTX 4090"
  vendor: string;            // "NVIDIA"
  vram_gb: number;           // 24
  memory_bandwidth_gbps: number;  // 1008
  fp16_tflops?: number;      // 82.6
  generation?: string;       // "Ada Lovelace"
}
```

### Model Registry

Build a registry of popular inference models with:

```typescript
interface Model {
  name: string;              // "Llama-3.1-70B"
  total_params_b: number;    // 70.6
  active_params_b: number;   // 70.6 (same for dense, different for MoE)
  is_moe: boolean;
  num_experts?: number;      // for MoE
  num_active_experts?: number;
  hidden_dim: number;
  num_layers: number;
  num_heads: number;
  num_kv_heads: number;
  vocab_size: number;
  
  // Context limits - NOT all models support 1M!
  max_context_length: number;  // e.g., 8192, 32768, 131072, 1048576
  effective_context_length?: number;  // Tested performance (may be lower than max)
  
  // Domain/capability tags for filtering
  domains: ModelDomain[];
  capabilities: ModelCapability[];
  
  benchmarks: {
    mmlu?: number;
    humaneval?: number;
    gsm8k?: number;
    // Tool-calling specific
    bfcl?: number;           // Berkeley Function Calling Leaderboard
    tool_accuracy?: number;
    // etc
  };
}

type ModelDomain = 
  | "general"      // General text/chat
  | "code"         // Code generation optimized
  | "tool-calling" // Function/tool calling optimized
  | "math"         // Math/reasoning focused
  | "vision"       // Multimodal with image understanding
  | "roleplay";    // Creative/roleplay focused

type ModelCapability =
  | "function_calling"  // Native tool/function support
  | "json_mode"         // Structured JSON output
  | "vision"            // Image input
  | "long_context"      // >32K reliable context
  | "multilingual"      // Strong non-English performance
  | "reasoning";        // Chain-of-thought / reasoning traces
```

### Model Context Length Reference

**Critical**: The context slider must be capped at the selected model's `max_context_length`. Not all models support long context!

| Model Family | Max Context | Effective Context | Notes |
|--------------|-------------|-------------------|-------|
| Llama 3.1 | 128K | ~128K | Works well at full length |
| Llama 3.2 | 128K | ~128K | |
| Llama 3.3 | 128K | ~128K | |
| Qwen 2.5 | 128K | ~64K | Some variants claim 1M |
| Qwen 3 | 128K | ~128K | |
| Gemma 2 | 8K | 8K | Limited context |
| Gemma 3 | 128K | ~32K | Tool-calling variant |
| Mistral v0.1 | 8K | 8K | Original |
| Mistral v0.2+ | 32K | 32K | Extended context |
| Mistral Large | 128K | ~128K | |
| Mixtral 8x7B | 32K | 32K | MoE |
| DeepSeek V3 | 128K | ~64K | MoE, 671B total |
| Phi-3 Mini | 128K | ~32K | Small but long context |
| Phi-4 | 16K | 16K | |
| Command R+ | 128K | ~128K | |
| Jamba | 256K | ~256K | Mamba-based |

**UI Behavior**: When user selects a model, cap the context slider at that model's max. Show warning if selected context exceeds `effective_context_length`.

**Sources for model data:**
- Unsloth's HuggingFace models
- Open LLM Leaderboard
- Model cards on HuggingFace

### Quantization Specs

```typescript
interface Quantization {
  name: string;                    // "Q4_K_M"
  bits_per_weight: number;         // 4.83 (effective, accounts for mixed precision)
  quality_factor: number;          // 0.0-1.0, relative to FP16
  perplexity_increase: number;     // Absolute PPL increase vs FP16
  quality_tier: QualityTier;       // Human-readable assessment
  source: string;                  // Citation for the data
}

type QualityTier = 
  | "near_lossless"    // <0.01 PPL increase
  | "very_low_loss"    // 0.01-0.05 PPL increase
  | "recommended"      // 0.05-0.10 PPL increase (sweet spot)
  | "balanced"         // 0.10-0.25 PPL increase
  | "noticeable_loss"  // 0.25-0.50 PPL increase
  | "high_loss"        // 0.50-1.00 PPL increase
  | "extreme_loss";    // >1.00 PPL increase

// REAL DATA from llama.cpp benchmarks (LLaMA-3-8B)
// Source: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md
const QUANT_SPECS: Record<string, Quantization> = {
  "F16": { 
    bits: 16.00, 
    quality_factor: 1.000, 
    ppl_increase: 0.0000,
    quality_tier: "near_lossless",
    source: "baseline"
  },
  "Q8_0": { 
    bits: 8.50, 
    quality_factor: 0.999, 
    ppl_increase: 0.0026,
    quality_tier: "near_lossless",
    source: "llama.cpp quantize README"
  },
  "Q6_K": { 
    bits: 6.57, 
    quality_factor: 0.996, 
    ppl_increase: 0.0217,
    quality_tier: "very_low_loss",
    source: "llama.cpp quantize README"
  },
  "Q5_K_M": { 
    bits: 5.67, 
    quality_factor: 0.990, 
    ppl_increase: 0.0569,
    quality_tier: "recommended",
    source: "llama.cpp quantize README"
  },
  "Q5_K_S": { 
    bits: 5.53, 
    quality_factor: 0.983, 
    ppl_increase: 0.1049,
    quality_tier: "balanced",
    source: "llama.cpp quantize README"
  },
  "Q4_K_M": { 
    bits: 4.83, 
    quality_factor: 0.991, 
    ppl_increase: 0.0535,
    quality_tier: "recommended",  // Sweet spot for most users
    source: "llama.cpp quantize README"
  },
  "Q4_K_S": { 
    bits: 4.58, 
    quality_factor: 0.987, 
    ppl_increase: 0.0796,
    quality_tier: "balanced",
    source: "llama.cpp quantize README"
  },
  "Q4_0": { 
    bits: 4.34, 
    quality_factor: 0.925, 
    ppl_increase: 0.4685,
    quality_tier: "noticeable_loss",  // Legacy format, avoid
    source: "llama.cpp quantize README"
  },
  "Q3_K_M": { 
    bits: 3.89, 
    quality_factor: 0.960, 
    ppl_increase: 0.2437,
    quality_tier: "noticeable_loss",
    source: "llama.cpp quantize README"
  },
  "Q3_K_S": { 
    bits: 3.50, 
    quality_factor: 0.890, 
    ppl_increase: 0.6569,
    quality_tier: "high_loss",
    source: "llama.cpp quantize README"
  },
  "Q2_K": { 
    bits: 3.00, 
    quality_factor: 0.870, 
    ppl_increase: 0.8698,
    quality_tier: "high_loss",
    source: "llama.cpp quantize README"
  },
  "IQ2_XXS": { 
    bits: 2.10, 
    quality_factor: 0.700, 
    ppl_increase: 3.5199,
    quality_tier: "extreme_loss",
    source: "llama.cpp quantize README"
  },
  "IQ2_XS": { 
    bits: 2.30, 
    quality_factor: 0.750, 
    ppl_increase: 2.5000,  // Estimated
    quality_tier: "extreme_loss",
    source: "llama.cpp benchmarks"
  },
};

// Unsloth Dynamic 2.0 variants (model-specific optimized quantization)
const UNSLOTH_DYNAMIC_QUANTS = {
  "UD_Q2_K_XL": {
    bits: 2.80,
    quality_factor: 0.960,  // 68.7% MMLU on Gemma 27B vs 71.5% BF16
    quality_tier: "balanced",
    source: "Unsloth Dynamic 2.0 benchmarks"
  },
  "UD_Q4_K_XL": {
    bits: 4.50,
    quality_factor: 0.999,  // 71.47% MMLU vs 71.5% BF16
    quality_tier: "near_lossless",
    source: "Unsloth Dynamic 2.0 benchmarks"
  },
  "UD_IQ2_XXS": {
    bits: 2.10,
    quality_factor: 0.830,  // 59.2% MMLU on Gemma 27B
    quality_tier: "noticeable_loss",
    source: "Unsloth Dynamic 2.0 benchmarks"
  },
};
```

**Critical Finding: Model Size Affects Quantization Tolerance**

From Intel Low-bit Leaderboard and academic research:

| Model Size | Q4_K_M Impact | Notes |
|------------|---------------|-------|
| 7-8B | -10 to -15% MMLU | Significant degradation |
| 13-14B | -5 to -8% MMLU | Moderate degradation |
| 32-34B | -2 to -4% MMLU | Minor degradation |
| 70B+ | -0.5 to -2% MMLU | Near-lossless |

**Implication for the tool**: Apply size-adjusted quality factors:

```typescript
function adjustedQualityFactor(
  baseQuality: number,
  modelSizeB: number,
  quant: Quantization
): number {
  // Larger models tolerate quantization better
  // Based on Intel Low-bit Leaderboard findings
  const sizePenalty = {
    small: 0.85,   // <10B: significant extra degradation
    medium: 0.92,  // 10-30B: moderate extra degradation
    large: 0.97,   // 30-65B: minor extra degradation
    xlarge: 1.00,  // >65B: minimal extra degradation
  };
  
  const sizeCategory = 
    modelSizeB < 10 ? 'small' :
    modelSizeB < 30 ? 'medium' :
    modelSizeB < 65 ? 'large' : 'xlarge';
  
  return baseQuality * quant.quality_factor * sizePenalty[sizeCategory];
}
```

**LLaMA-3-70B Exception**: Research shows LLaMA-3-70B has extreme weight outliers (magnitudes >90 vs <1.0 in other models) that cause unusual degradation even at W8A8. Flag this model specifically in the tool.

**Data Sources for Quality Factors:**
- llama.cpp quantize README: https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md
- Intel Low-bit Leaderboard: https://huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard
- Unsloth Dynamic 2.0 benchmarks: https://docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs
- Artefact2 KL-divergence study (Mistral-7B): GitHub Gist

## VRAM Calculation Formulas

### Model Weights VRAM

```typescript
function calculateModelVRAM(model: Model, quant: Quantization): number {
  // For MoE: ALL experts must fit in VRAM (not just active ones)
  const params = model.total_params_b * 1e9;
  const bytes = params * (quant.bits_per_weight / 8);
  return bytes / (1024 ** 3); // Convert to GB
}
```

### KV Cache VRAM

```typescript
function calculateKVCache(
  model: Model, 
  contextLength: number,
  batchSize: number = 1
): number {
  // KV cache stores key and value vectors for each layer
  const bytesPerToken = 
    2 *                          // K and V
    model.num_layers * 
    model.num_kv_heads *        // Use KV heads (GQA optimization)
    (model.hidden_dim / model.num_heads) *  // Head dimension
    2;                          // FP16 = 2 bytes
  
  return (bytesPerToken * contextLength * batchSize) / (1024 ** 3);
}
```

### Total VRAM Required

```typescript
function totalVRAM(model: Model, quant: Quantization, context: number): number {
  const modelVRAM = calculateModelVRAM(model, quant);
  const kvCache = calculateKVCache(model, context);
  const overhead = 0.5; // CUDA/driver overhead in GB
  
  return modelVRAM + kvCache + overhead;
}
```

## Pareto Frontier Algorithm

```typescript
interface Candidate {
  model: Model;
  quant: Quantization;
  vram_required: number;
  quality_score: number;  // Composite of benchmarks * quant.quality_factor
  vram_headroom: number;  // available_vram - vram_required
}

function computeParetoFrontier(
  candidates: Candidate[]
): Candidate[] {
  // Sort by quality descending
  const sorted = [...candidates].sort((a, b) => b.quality_score - a.quality_score);
  
  const frontier: Candidate[] = [];
  let minVRAM = Infinity;
  
  for (const c of sorted) {
    // Pareto optimal if: highest quality seen so far for its VRAM budget
    // (no other point has better quality with less or equal VRAM)
    if (c.vram_required < minVRAM) {
      frontier.push(c);
      minVRAM = c.vram_required;
    }
  }
  
  return frontier;
}
```

## MoE Model Handling

MoE (Mixture of Experts) models are tricky:

| Model | Total Params | Active Params | VRAM Behavior |
|-------|-------------|---------------|---------------|
| DeepSeek V3 | 671B | 37B | Need ALL 671B in VRAM |
| Qwen3-30B-A3B | 30B | 3B | Need 30B, runs like 3B |
| Mixtral 8x7B | 47B | 13B | Need 47B in VRAM |

**Display strategy:**
- Show MoE models with different marker (○ vs ★)
- Tooltip: "30B total (3B active per token)"
- Quality based on active params, VRAM based on total params

## Tech Stack

- **Framework**: React 18+ with TypeScript
- **Build**: Vite
- **Styling**: Tailwind CSS
- **Charts**: Recharts or Plotly.js
- **Deployment**: GitHub Pages (static, no backend needed)
- **Data**: Bundled JSON files (no runtime API calls needed)

## File Structure

```
/
├── src/
│   ├── components/
│   │   ├── GPUSelector.tsx
│   │   ├── ContextSlider.tsx
│   │   ├── DomainFilter.tsx        # Domain/capability multi-select filter
│   │   ├── VRAMBreakdown.tsx
│   │   ├── LinkedCharts.tsx        # Container for synchronized 3-chart view
│   │   ├── ParetoChart.tsx         # Individual chart (Quality/Perf/Efficiency)
│   │   ├── HowItWorks.tsx          # Collapsible explanation panel
│   │   └── ModelCard.tsx
│   ├── data/
│   │   ├── gpus.json               # GPU database (VRAM + bandwidth + tok/s)
│   │   ├── models.json             # Model registry with domains/context limits
│   │   ├── tool-calling-models.json # Tool-calling specific models + BFCL scores
│   │   ├── quantizations.json      # Quant specs with quality factors
│   │   ├── quant-benchmarks.json   # Raw perplexity data (source of truth)
│   │   └── gpu-performance.json    # Measured tok/s across GPU × Model matrix
│   ├── lib/
│   │   ├── vram.ts                 # VRAM calculation functions
│   │   ├── pareto.ts               # Pareto frontier algorithm
│   │   ├── quality.ts              # Quality scoring with size adjustments
│   │   ├── performance.ts          # Tokens/sec estimation + lookup
│   │   └── efficiency.ts           # Efficiency metric calculation
│   ├── hooks/
│   │   └── useLinkedHighlight.ts   # Cross-chart hover synchronization
│   ├── App.tsx
│   └── main.tsx
├── public/
├── docs/                           # GitHub Pages serves from here
│   └── (built files)
├── research/                       # Data collection scripts
│   ├── scrape-perplexity.py
│   ├── scrape-gpu-benchmarks.py
│   └── collect-bfcl-scores.py
├── index.html
├── package.json
├── vite.config.ts
└── CLAUDE.md                       # This file
```

## Tool-Calling Models Registry

For users selecting "Tool-Calling" domain, include these Unsloth-supported models:

```typescript
const TOOL_CALLING_MODELS: Model[] = [
  {
    name: "Gemma-3-27B-IT",
    total_params_b: 27,
    domains: ["tool-calling", "general"],
    capabilities: ["function_calling", "json_mode"],
    max_context_length: 131072,
    benchmarks: {
      mmlu: 75.6,
      bfcl: 82.3,  // Berkeley Function Calling Leaderboard
    },
    notes: "Google's tool-calling optimized model"
  },
  {
    name: "Qwen2.5-Coder-32B-Instruct",
    total_params_b: 32,
    domains: ["tool-calling", "code"],
    capabilities: ["function_calling", "json_mode"],
    max_context_length: 131072,
    benchmarks: {
      humaneval: 92.7,
      bfcl: 78.5,
    },
    notes: "Strong code + tool calling"
  },
  {
    name: "Mistral-Small-24B-Instruct-2501",
    total_params_b: 24,
    domains: ["tool-calling", "general"],
    capabilities: ["function_calling", "json_mode"],
    max_context_length: 32768,
    benchmarks: {
      mmlu: 71.2,
      bfcl: 76.8,
    },
    notes: "Mistral's function calling model"
  },
  {
    name: "Llama-3.3-70B-Instruct",
    total_params_b: 70,
    domains: ["general", "tool-calling", "code"],
    capabilities: ["function_calling", "json_mode", "long_context"],
    max_context_length: 131072,
    benchmarks: {
      mmlu: 86.0,
      bfcl: 84.1,
    },
    notes: "Best overall with tool support"
  },
  {
    name: "Hermes-3-Llama-3.1-8B",
    total_params_b: 8,
    domains: ["tool-calling", "general"],
    capabilities: ["function_calling", "json_mode"],
    max_context_length: 131072,
    benchmarks: {
      bfcl: 71.2,
    },
    notes: "Small but capable tool-calling"
  },
  {
    name: "DeepSeek-V3",
    total_params_b: 671,
    active_params_b: 37,
    is_moe: true,
    domains: ["general", "code", "tool-calling", "math"],
    capabilities: ["function_calling", "json_mode", "reasoning"],
    max_context_length: 131072,
    benchmarks: {
      mmlu: 87.1,
      humaneval: 82.6,
      bfcl: 79.3,
    },
    notes: "MoE - 671B total but only 37B active"
  },
];
```

**Tool-Calling Benchmark Sources:**
- Berkeley Function Calling Leaderboard: https://gorilla.cs.berkeley.edu/leaderboard.html
- Nexus Function Calling Benchmark
- API-Bank evaluation

## Development Commands

```bash
# Install dependencies
npm install

# Start dev server
npm run dev

# Build for production
npm run build

# Preview production build
npm run preview

# Deploy to GitHub Pages
npm run deploy
```

## Implementation Phases

### Phase 1: Core Logic (No UI)
- [ ] Set up Vite + React + TypeScript project
- [ ] Implement VRAM calculation functions with tests
- [ ] Build model registry JSON with domains and context limits
- [ ] Build GPU database JSON with bandwidth and tok/s data
- [ ] Implement Pareto frontier algorithm
- [ ] Implement efficiency metric calculation

### Phase 2: Basic UI
- [ ] GPU dropdown with search
- [ ] Domain filter multi-select
- [ ] Context length slider with KV cache display (capped by model max)
- [ ] VRAM breakdown component

### Phase 3: Linked Charts
- [ ] Three synchronized Pareto charts (Quality, Performance, Efficiency)
- [ ] Cross-chart hover highlighting (useLinkedHighlight hook)
- [ ] Click-to-select and pin models
- [ ] Different markers for Dense/MoE/Tool-calling models

### Phase 4: Polish
- [ ] Model cards with domain-specific benchmarks (BFCL for tool-calling, etc.)
- [ ] Copy ollama/llama.cpp commands
- [ ] "How It Works" collapsible panel
- [ ] Mobile responsive layout
- [ ] Dark mode
- [ ] URL state for sharing configurations

### Phase 5: Deployment
- [ ] GitHub Actions for CI
- [ ] Deploy to GitHub Pages
- [ ] Add contribution guide for model/GPU/benchmark additions

## Quality Score Formula

Composite quality score with domain-aware weighting:

```typescript
function qualityScore(
  model: Model, 
  quant: Quantization,
  domain: ModelDomain
): number {
  // Domain-specific benchmark weighting
  const weights = {
    general: { mmlu: 0.5, humaneval: 0.25, gsm8k: 0.25 },
    code: { humaneval: 0.6, mmlu: 0.2, gsm8k: 0.2 },
    "tool-calling": { bfcl: 0.5, mmlu: 0.3, humaneval: 0.2 },
    math: { gsm8k: 0.5, mmlu: 0.3, humaneval: 0.2 },
  };
  
  const w = weights[domain] || weights.general;
  
  const baseQuality = (
    (model.benchmarks.mmlu || 50) * (w.mmlu || 0) +
    (model.benchmarks.humaneval || 50) * (w.humaneval || 0) +
    (model.benchmarks.gsm8k || 50) * (w.gsm8k || 0) +
    (model.benchmarks.bfcl || 50) * (w.bfcl || 0)
  );
  
  // Apply quantization degradation + size adjustment
  const sizeAdjustedQuality = adjustedQualityFactor(
    baseQuality, 
    model.total_params_b, 
    quant
  );
  
  return sizeAdjustedQuality;
}
```

## Data Sources Reference (Completed Research)

### Quality Metrics
| Source | Data Type | URL | Format |
|--------|-----------|-----|--------|
| llama.cpp quantize README | Perplexity by quant | github.com/ggerganov/llama.cpp | Markdown table |
| Intel Low-bit Leaderboard | 10 benchmark scores | huggingface.co/spaces/Intel/low_bit_open_llm_leaderboard | HF Space (API) |
| Unsloth Dynamic 2.0 | MMLU by quant | docs.unsloth.ai/basics/unsloth-dynamic-2.0-ggufs | Docs |
| Artefact2 KL-divergence | KL stats for Mistral-7B | GitHub Gist | JSON |

### Performance Metrics
| Source | Data Type | URL | Format |
|--------|-----------|-----|--------|
| XiongjieDai GPU Benchmarks | tok/s across 30+ GPUs | github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference | Markdown tables |
| llama.cpp official | Apple Silicon benchmarks | github.com/ggerganov/llama.cpp | Wiki |
| llama.cpp Google Sheet | Community benchmarks | docs.google.com/spreadsheets (see repo) | CSV export |

### Tool-Calling Benchmarks
| Source | Data Type | URL | Format |
|--------|-----------|-----|--------|
| Berkeley Function Calling | BFCL scores | gorilla.cs.berkeley.edu/leaderboard.html | Web table |
| Nexus Function Calling | Tool accuracy | nexusflow.ai | Leaderboard |

### Academic Papers
- **GPTQ** (2022): OPT-175B at 4-bit: 8.37 PPL vs 8.34 FP16
- **AWQ** (2023): 1% salient weight protection drops INT3 PPL from 43.2 to 13.0
- **ParetoQ** (2025): Learning transition between 2-3 bits
- **QuIP#** (2024): First 2-bit method achieving near-FP16 on LLaMA-2-70B
- **SqueezeLLM** (2023): 3-bit outperforms GPTQ/AWQ by >0.3 PPL

## Notes

- Context length slider capped by model's `max_context_length`
- Show warning when selected context exceeds `effective_context_length`
- Show warning when VRAM headroom < 1GB (risky, may OOM)
- For MoE models, inference speed scales with active params but VRAM with total
- LLaMA-3-70B has unusual quantization sensitivity—flag in tool
- Tool-calling models should show BFCL score prominently when domain filter is active
- Efficiency metric helps users on limited VRAM find optimal tradeoffs

---

## Sample Model Registry (Unsloth-Supported)

This is a starter registry focused on Unsloth-supported models. Each entry includes domains, context limits, and benchmark data.

```typescript
const UNSLOTH_MODEL_REGISTRY: Model[] = [
  // === GENERAL/CHAT MODELS ===
  {
    name: "Llama-3.1-8B-Instruct",
    total_params_b: 8.03,
    active_params_b: 8.03,
    is_moe: false,
    hidden_dim: 4096,
    num_layers: 32,
    num_heads: 32,
    num_kv_heads: 8,
    vocab_size: 128256,
    max_context_length: 131072,
    effective_context_length: 131072,
    domains: ["general", "code"],
    capabilities: ["function_calling", "json_mode", "long_context"],
    benchmarks: { mmlu: 68.1, humaneval: 62.5, gsm8k: 75.0 }
  },
  {
    name: "Llama-3.1-70B-Instruct",
    total_params_b: 70.6,
    active_params_b: 70.6,
    is_moe: false,
    hidden_dim: 8192,
    num_layers: 80,
    num_heads: 64,
    num_kv_heads: 8,
    vocab_size: 128256,
    max_context_length: 131072,
    effective_context_length: 131072,
    domains: ["general", "code", "math"],
    capabilities: ["function_calling", "json_mode", "long_context", "reasoning"],
    benchmarks: { mmlu: 82.0, humaneval: 80.5, gsm8k: 93.0 },
    notes: "⚠️ Unusual quantization sensitivity due to weight outliers"
  },
  {
    name: "Qwen2.5-32B-Instruct",
    total_params_b: 32.5,
    active_params_b: 32.5,
    is_moe: false,
    hidden_dim: 5120,
    num_layers: 64,
    num_heads: 40,
    num_kv_heads: 8,
    vocab_size: 152064,
    max_context_length: 131072,
    effective_context_length: 65536,
    domains: ["general", "code", "math"],
    capabilities: ["function_calling", "json_mode", "long_context", "multilingual"],
    benchmarks: { mmlu: 79.0, humaneval: 75.0, gsm8k: 88.0 }
  },
  
  // === TOOL-CALLING OPTIMIZED MODELS ===
  {
    name: "Gemma-3-12B-Tool",
    total_params_b: 12.0,
    active_params_b: 12.0,
    is_moe: false,
    hidden_dim: 3840,
    num_layers: 40,
    num_heads: 16,
    num_kv_heads: 8,
    vocab_size: 262144,
    max_context_length: 131072,
    effective_context_length: 32768,
    domains: ["tool-calling", "general"],
    capabilities: ["function_calling", "json_mode"],
    benchmarks: { 
      mmlu: 64.0, 
      humaneval: 55.0, 
      bfcl: 78.5,  // Berkeley Function Calling Leaderboard
      tool_accuracy: 82.0 
    }
  },
  {
    name: "Qwen2.5-Coder-32B-Instruct",
    total_params_b: 32.5,
    active_params_b: 32.5,
    is_moe: false,
    hidden_dim: 5120,
    num_layers: 64,
    num_heads: 40,
    num_kv_heads: 8,
    vocab_size: 152064,
    max_context_length: 131072,
    effective_context_length: 65536,
    domains: ["code", "tool-calling"],
    capabilities: ["function_calling", "json_mode", "long_context"],
    benchmarks: { 
      mmlu: 75.0, 
      humaneval: 88.0,  // Strong coding
      bfcl: 72.0,
      gsm8k: 80.0 
    }
  },
  {
    name: "Mistral-Small-24B-Instruct",
    total_params_b: 24.0,
    active_params_b: 24.0,
    is_moe: false,
    hidden_dim: 5120,
    num_layers: 56,
    num_heads: 32,
    num_kv_heads: 8,
    vocab_size: 32768,
    max_context_length: 32768,
    effective_context_length: 32768,
    domains: ["general", "tool-calling"],
    capabilities: ["function_calling", "json_mode"],
    benchmarks: { mmlu: 72.0, humaneval: 68.0, bfcl: 75.0 }
  },
  
  // === MOE MODELS ===
  {
    name: "Qwen3-30B-A3B",
    total_params_b: 30.5,
    active_params_b: 3.0,
    is_moe: true,
    num_experts: 128,
    num_active_experts: 8,
    hidden_dim: 2048,
    num_layers: 48,
    num_heads: 16,
    num_kv_heads: 4,
    vocab_size: 151936,
    max_context_length: 131072,
    effective_context_length: 32768,
    domains: ["general"],
    capabilities: ["long_context"],
    benchmarks: { mmlu: 65.0, humaneval: 58.0, gsm8k: 70.0 }
  },
  {
    name: "DeepSeek-V3",
    total_params_b: 671.0,
    active_params_b: 37.0,
    is_moe: true,
    num_experts: 256,
    num_active_experts: 8,
    hidden_dim: 7168,
    num_layers: 61,
    num_heads: 128,
    num_kv_heads: 128,
    vocab_size: 129280,
    max_context_length: 131072,
    effective_context_length: 65536,
    domains: ["general", "code", "math", "reasoning"],
    capabilities: ["function_calling", "json_mode", "long_context", "reasoning"],
    benchmarks: { mmlu: 87.0, humaneval: 85.0, gsm8k: 95.0 }
  },
  
  // === SMALL/FAST MODELS ===
  {
    name: "Llama-3.2-3B-Instruct",
    total_params_b: 3.21,
    active_params_b: 3.21,
    is_moe: false,
    hidden_dim: 3072,
    num_layers: 28,
    num_heads: 24,
    num_kv_heads: 8,
    vocab_size: 128256,
    max_context_length: 131072,
    effective_context_length: 8192,
    domains: ["general"],
    capabilities: [],
    benchmarks: { mmlu: 55.0, humaneval: 40.0, gsm8k: 48.0 }
  },
  {
    name: "Phi-4",
    total_params_b: 14.0,
    active_params_b: 14.0,
    is_moe: false,
    hidden_dim: 4096,
    num_layers: 40,
    num_heads: 32,
    num_kv_heads: 8,
    vocab_size: 100352,
    max_context_length: 16384,
    effective_context_length: 16384,
    domains: ["general", "code", "math"],
    capabilities: ["reasoning"],
    benchmarks: { mmlu: 78.0, humaneval: 70.0, gsm8k: 85.0 }
  },
  
  // === LIMITED CONTEXT MODELS (important for slider capping) ===
  {
    name: "Gemma-2-9B-Instruct",
    total_params_b: 9.24,
    active_params_b: 9.24,
    is_moe: false,
    hidden_dim: 3584,
    num_layers: 42,
    num_heads: 16,
    num_kv_heads: 8,
    vocab_size: 256128,
    max_context_length: 8192,  // LIMITED!
    effective_context_length: 8192,
    domains: ["general"],
    capabilities: [],
    benchmarks: { mmlu: 70.0, humaneval: 54.0, gsm8k: 68.0 }
  },
  {
    name: "Mistral-7B-Instruct-v0.1",
    total_params_b: 7.24,
    active_params_b: 7.24,
    is_moe: false,
    hidden_dim: 4096,
    num_layers: 32,
    num_heads: 32,
    num_kv_heads: 8,
    vocab_size: 32000,
    max_context_length: 8192,  // LIMITED!
    effective_context_length: 8192,
    domains: ["general"],
    capabilities: [],
    benchmarks: { mmlu: 60.0, humaneval: 45.0, gsm8k: 52.0 }
  },
];
```

### Context Length Handling in UI

```typescript
function getAvailableContextPositions(
  selectedModel: Model | null,
  allPositions: number[] = [2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
): { position: number; available: boolean; warning: boolean }[] {
  return allPositions.map(pos => {
    if (!selectedModel) {
      return { position: pos, available: true, warning: false };
    }
    
    const available = pos <= selectedModel.max_context_length;
    const warning = pos > selectedModel.effective_context_length && available;
    
    return { position: pos, available, warning };
  });
}

// UI should:
// 1. Gray out positions beyond max_context_length
// 2. Show warning icon on positions between effective and max
// 3. Show tooltip: "This model's performance may degrade above {effective}K context"
```

### Domain Filtering

```typescript
function filterByDomain(
  models: Model[],
  selectedDomains: ModelDomain[]
): Model[] {
  if (selectedDomains.length === 0 || selectedDomains.includes("all" as any)) {
    return models;
  }
  
  return models.filter(model => 
    model.domains.some(d => selectedDomains.includes(d))
  );
}

// When "tool-calling" domain is selected:
// 1. Show BFCL score in quality metric (weighted higher)
// 2. Add ◆ marker to tool-optimized models
// 3. Sort by tool_accuracy in model cards
```

### GPU Performance Lookup Table

Based on XiongjieDai GPU Benchmarks repository:

```typescript
const GPU_PERFORMANCE_DATA: Record<string, {
  vram_gb: number;
  bandwidth_gbps: number;
  // Measured tok/s for LLaMA-3-8B Q4_K_M at 1024 context
  baseline_tps_8b_q4: number;
}> = {
  "RTX 3070": { vram_gb: 8, bandwidth_gbps: 448, baseline_tps_8b_q4: 70.94 },
  "RTX 3080": { vram_gb: 10, bandwidth_gbps: 760, baseline_tps_8b_q4: 106.40 },
  "RTX 3090": { vram_gb: 24, bandwidth_gbps: 936, baseline_tps_8b_q4: 111.74 },
  "RTX 4070 Ti": { vram_gb: 12, bandwidth_gbps: 504, baseline_tps_8b_q4: 85.0 },
  "RTX 4080": { vram_gb: 16, bandwidth_gbps: 717, baseline_tps_8b_q4: 110.0 },
  "RTX 4090": { vram_gb: 24, bandwidth_gbps: 1008, baseline_tps_8b_q4: 127.74 },
  "RTX 6000 Ada": { vram_gb: 48, bandwidth_gbps: 960, baseline_tps_8b_q4: 130.99 },
  "A100 40GB": { vram_gb: 40, bandwidth_gbps: 1555, baseline_tps_8b_q4: 135.0 },
  "A100 80GB": { vram_gb: 80, bandwidth_gbps: 2039, baseline_tps_8b_q4: 138.31 },
  "H100 PCIe": { vram_gb: 80, bandwidth_gbps: 2039, baseline_tps_8b_q4: 144.49 },
  "H100 SXM": { vram_gb: 80, bandwidth_gbps: 3350, baseline_tps_8b_q4: 180.0 },
  // Apple Silicon (unified memory, different characteristics)
  "M1 Max": { vram_gb: 32, bandwidth_gbps: 400, baseline_tps_8b_q4: 45.0 },
  "M2 Ultra": { vram_gb: 192, bandwidth_gbps: 800, baseline_tps_8b_q4: 65.0 },
  "M4 Max": { vram_gb: 128, bandwidth_gbps: 546, baseline_tps_8b_q4: 83.0 },
};

// Scale from baseline to estimate other model sizes:
function estimateTokensPerSecond(
  gpu: string,
  modelSizeGb: number,  // After quantization
  baselineModelSizeGb: number = 4.58  // 8B Q4_K_M
): number {
  const gpuData = GPU_PERFORMANCE_DATA[gpu];
  if (!gpuData) return 0;
  
  // Memory-bound: tok/s scales inversely with model size
  const scaleFactor = baselineModelSizeGb / modelSizeGb;
  return gpuData.baseline_tps_8b_q4 * scaleFactor;
}
```
