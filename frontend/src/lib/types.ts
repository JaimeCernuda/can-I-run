/**
 * Type definitions for the GPU-to-Model Pareto Selector
 */

export interface GPU {
  name: string;
  vendor: string;
  vram_gb: number;
  bandwidth_gbps: number;
  generation?: string;
}

export interface Quantization {
  name: string;
  bits: number;
  quality_factor: number;
  tier: QualityTier;
  ppl_increase: number;
}

export type QualityTier =
  | "near_lossless"
  | "very_low_loss"
  | "recommended"
  | "balanced"
  | "noticeable_loss"
  | "high_loss"
  | "extreme_loss";

export type ModelDomain =
  | "general"
  | "code"
  | "tool-calling"
  | "math"
  | "math"
  | "vision";

export type ModelCapability =
  | "function_calling"
  | "json_mode"
  | "vision"
  | "long_context"
  | "multilingual"
  | "reasoning";

export interface ModelQuantCombo {
  model_name: string;
  quant_name: string;
  model_vram_gb: number;
  quality_score: number;
  is_moe: boolean;
  total_params_b: number;
  active_params_b: number;
  max_context: number;
  effective_context: number;
  domains: ModelDomain[];
  capabilities: ModelCapability[];
  quant_tier: QualityTier;
  bits_per_weight: number;
  kv_cache: Record<string, number>; // context length -> kv cache GB
  performance: Record<string, number>; // gpu name -> tokens/sec
}

export interface ContextPosition {
  value: number;
  label: string;
}

export interface ComputedData {
  gpus: GPU[];
  quantizations: Quantization[];
  context_positions: ContextPosition[];
  model_quant_combos: ModelQuantCombo[];
  metadata: {
    total_gpus: number;
    total_models: number;
    total_quants: number;
    total_combos: number;
    overhead_gb: number;
  };
}

export interface ChartPoint {
  id: string;
  model_name: string;
  quant_name: string;
  vram_required: number;
  vram_headroom: number;
  quality_score: number;
  tokens_per_second: number;
  efficiency_score: number;
  is_moe: boolean;
  is_tool_calling: boolean;
  is_pareto_quality: boolean;
  is_pareto_performance: boolean;
  is_pareto_efficiency: boolean;
  domains: ModelDomain[];
  max_context: number;
  bits_per_weight: number;
  total_params_b: number;
}

export interface VRAMBreakdown {
  total_available_gb: number;
  model_weights_gb: number;
  kv_cache_gb: number;
  overhead_gb: number;
  total_required_gb: number;
  headroom_gb: number;
  headroom_percent: number;
  fits: boolean;
  safe: boolean;
}

export type ChartType = "quality" | "performance" | "efficiency";
