# GPU-to-Model Pareto Selector

A web tool that recommends optimal LLM model/quantization combinations based on available GPU VRAM, visualized as Pareto frontiers.

**Live Demo:** [https://JaimeCernuda.github.io/can-I-run/](https://JaimeCernuda.github.io/can-I-run/)

## Features

- **GPU Selection**: Choose from 50+ GPUs (NVIDIA, AMD, Apple Silicon, Intel) or enter custom VRAM
- **Context Length Slider**: Adjust from 2K to 1M tokens with real-time KV cache estimation
- **Domain Filtering**: Filter models by use case (General, Code, Tool-Calling, Math, Vision)
- **Three Synchronized Charts**:
  - Quality vs VRAM (benchmark scores)
  - Performance vs VRAM (tokens/second)
  - Efficiency vs VRAM (quality × speed / VRAM)
- **Pareto Frontiers**: Visualize optimal tradeoffs for each metric
- **Model Cards**: Detailed info with Ollama commands for quick deployment

## How It Works

### VRAM Budget Breakdown

```
Available = Total VRAM - KV Cache - Overhead (0.5GB)
```

- **Model Weights**: `Parameters × Bits per Weight ÷ 8`
- **KV Cache**: `2 × Layers × KV_Heads × Head_Dim × Context × 2 bytes`
- **Overhead**: ~0.5GB for CUDA/driver allocations

For MoE models, ALL experts must fit in VRAM, even though only a subset are used per token.

### Quality Score

```
Quality = Base Score × Quantization Factor × Size Adjustment
```

- **Base Score**: Weighted average of MMLU, HumanEval, GSM8K (weights vary by domain)
- **Quantization Factor**: Quality retention from perplexity measurements
- **Size Adjustment**: Larger models (70B+) tolerate quantization better than small models

### Performance Estimation

```
Tokens/sec ≈ Memory Bandwidth (GB/s) ÷ Model Size (GB) × Efficiency
```

Token generation is primarily memory-bandwidth bound during decode. When available, we use measured benchmarks instead of theoretical estimates.

### Efficiency Metric

```
Efficiency = √(Quality × Performance) ÷ VRAM × 100
```

Uses geometric mean so both quality AND speed must be good for high efficiency.

### Pareto Frontier

A model is Pareto-optimal if no other model dominates it:
- Higher metric at same or lower VRAM, OR
- Same metric at lower VRAM

## Data Sources

| Source | Data Type | URL |
|--------|-----------|-----|
| llama.cpp | Quantization quality | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| XiongjieDai | GPU benchmarks | [github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference](https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference) |
| TechPowerUp | GPU specs | [techpowerup.com/gpu-specs](https://www.techpowerup.com/gpu-specs/) |
| HuggingFace | Model specs | [huggingface.co](https://huggingface.co) |
| BFCL | Tool-calling scores | [gorilla.cs.berkeley.edu/leaderboard.html](https://gorilla.cs.berkeley.edu/leaderboard.html) |

## Quantization Quality Reference

| Quant | Bits | Quality Factor | PPL Increase | Tier |
|-------|------|----------------|--------------|------|
| F16 | 16.00 | 1.000 | 0.00 | Near Lossless |
| Q8_0 | 8.50 | 0.999 | 0.003 | Near Lossless |
| Q6_K | 6.57 | 0.996 | 0.022 | Very Low Loss |
| Q5_K_M | 5.67 | 0.990 | 0.057 | Recommended |
| Q4_K_M | 4.83 | 0.991 | 0.054 | **Recommended** |
| Q4_K_S | 4.58 | 0.987 | 0.080 | Balanced |
| Q3_K_M | 3.89 | 0.960 | 0.244 | Noticeable Loss |
| Q2_K | 3.00 | 0.870 | 0.870 | High Loss |

## Project Structure

```
/
├── backend/                 # Python calculations
│   ├── can_i_run/
│   │   ├── vram.py         # VRAM calculations
│   │   ├── kv_cache.py     # KV cache formulas
│   │   ├── quality.py      # Quality scoring
│   │   ├── performance.py  # Speed estimation
│   │   ├── efficiency.py   # Efficiency metric
│   │   ├── pareto.py       # Pareto algorithm
│   │   └── generate.py     # Data generator
│   └── pyproject.toml
├── frontend/               # React application
│   ├── src/
│   │   ├── components/     # UI components
│   │   ├── lib/           # Calculations
│   │   ├── hooks/         # React hooks
│   │   └── data/          # Generated JSON
│   └── package.json
├── data/                   # Raw data files
│   ├── gpus.json
│   ├── models.json
│   └── quantizations.json
└── .github/workflows/      # CI/CD
```

## Development

### Prerequisites

- Python 3.9+
- Node.js 18+
- npm

### Setup

```bash
# Install Python backend
cd backend
pip install -e .

# Generate data
python -m can_i_run.generate

# Install frontend
cd ../frontend
npm install

# Start dev server
npm run dev
```

### Building

```bash
cd frontend
npm run build
```

### Updating Data

To add new models, GPUs, or quantizations:

1. Edit the JSON files in `/data/`
2. Run `python -m can_i_run.generate` from `/backend/`
3. The frontend will use the updated data

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your changes
4. Run tests (if applicable)
5. Submit a pull request

### Adding a New GPU

Edit `data/gpus.json`:

```json
{
  "name": "RTX 5090",
  "vendor": "NVIDIA",
  "vram_gb": 32,
  "memory_bandwidth_gbps": 1792,
  "fp16_tflops": 104.8,
  "generation": "Blackwell",
  "baseline_tps_8b_q4": 180.0
}
```

### Adding a New Model

Edit `data/models.json`:

```json
{
  "name": "New-Model-70B-Instruct",
  "total_params_b": 70.0,
  "active_params_b": 70.0,
  "is_moe": false,
  "hidden_dim": 8192,
  "num_layers": 80,
  "num_heads": 64,
  "num_kv_heads": 8,
  "vocab_size": 128256,
  "max_context_length": 131072,
  "effective_context_length": 131072,
  "domains": ["general", "code"],
  "capabilities": ["function_calling", "long_context"],
  "benchmarks": {
    "mmlu": 85.0,
    "humaneval": 80.0,
    "gsm8k": 90.0
  }
}
```

## Deployment

This project uses **GitHub Actions** (not branch-based Pages). Here's how to deploy:

### First-Time Setup

1. Go to your repo: `https://github.com/JaimeCernuda/can-I-run`
2. Click **Settings** → **Pages** (in left sidebar)
3. Under "Build and deployment":
   - **Source**: Select **GitHub Actions** (NOT "Deploy from a branch")
4. Merge this branch to `main`
5. The GitHub Action will automatically build and deploy

### How It Works

The `.github/workflows/deploy.yml` workflow:
1. Installs Python and generates computed data
2. Builds the React frontend
3. Deploys to GitHub Pages

Every push to `main` triggers a new deployment. You can also trigger manually via the Actions tab.

### Manual Deployment

If you want to deploy manually without pushing to main:
1. Go to **Actions** tab
2. Select "Deploy to GitHub Pages" workflow
3. Click "Run workflow"

## License

MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) for quantization research
- [Unsloth](https://unsloth.ai) for model optimization work
- [XiongjieDai](https://github.com/XiongjieDai) for GPU benchmark data
- The open-source LLM community
