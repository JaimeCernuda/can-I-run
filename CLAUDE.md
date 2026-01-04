# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GPU-to-Model Pareto Selector: A web tool that recommends optimal LLM model/quantization combinations based on available GPU VRAM, visualized as Pareto frontiers. Users select a GPU, context length, and domain filters to see which models fit and which are Pareto-optimal for quality, performance, or efficiency.

## Build Commands

```bash
# Backend: Generate computed data (must run before frontend build)
cd backend
pip install -e .
python -m can_i_run.generate

# Frontend: Install and run dev server
cd frontend
npm install
npm run dev      # Start dev server
npm run build    # Build for production (runs tsc -b && vite build)
npm run lint     # ESLint
```

## Architecture

The project has a **build-time data generation** pattern:

1. **Raw data** (`/data/`) - JSON files for GPUs, models, and quantizations
2. **Python backend** (`/backend/can_i_run/`) - Generates pre-computed metrics at build time
3. **Generated JSON** (`/frontend/src/data/computed.json`) - All model×quant combinations with VRAM, quality, performance data
4. **React frontend** (`/frontend/`) - Consumes pre-computed data, does real-time filtering and Pareto calculations

### Data Flow

```
data/*.json → backend/can_i_run/generate.py → frontend/src/data/computed.json → React app
```

### Backend Modules

- `vram.py` - VRAM calculation: model weights + KV cache + overhead
- `kv_cache.py` - KV cache formula: `2 × layers × kv_heads × head_dim × context × 2`
- `quality.py` - Quality scoring from benchmarks × quantization factor
- `performance.py` - Token/sec estimation from memory bandwidth
- `efficiency.py` - Efficiency metric: `sqrt(quality × perf) / VRAM`
- `pareto.py` - Pareto frontier algorithm
- `generate.py` - Main script that combines all calculations

### Frontend Structure

- `lib/calculations.ts` - Mirrors Python calculations for real-time updates
- `lib/types.ts` - TypeScript types for all data structures
- `components/LinkedCharts.tsx` - Three synchronized Recharts (quality, perf, efficiency vs VRAM)
- `components/GPUSelector.tsx` - GPU dropdown with 50+ options or custom VRAM
- `components/ContextSlider.tsx` - Context length from 2K to 1M tokens
- `hooks/useLinkedHighlight.ts` - Shared highlight state across charts

### Key Calculations

VRAM required = `model_weights + kv_cache + 0.5GB overhead`

A model is Pareto-optimal if no other model dominates it (better metric at ≤ VRAM).

## Adding Data

To add new GPUs/models/quantizations:
1. Edit JSON files in `/data/`
2. Run `python -m can_i_run.generate` from `/backend/`
3. Rebuild frontend

## Git Conventions

Write conventional commit messages (feat:, fix:, refactor:, etc.) without co-author attribution.
