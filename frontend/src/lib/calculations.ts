/**
 * Client-side calculations for VRAM, quality, performance, efficiency, and Pareto frontiers.
 *
 * These mirror the Python backend calculations but run in the browser for real-time updates.
 */

import type { ChartPoint, ModelQuantCombo, VRAMBreakdown } from "./types";

const CUDA_OVERHEAD_GB = 0.5;
const MAX_QUALITY_SCORE = 90.0;
const MAX_TOKENS_PER_SEC = 150.0;

/**
 * Calculate total VRAM required for a model configuration.
 */
export function calculateTotalVRAM(
  modelVramGb: number,
  kvCacheGb: number,
  overheadGb: number = CUDA_OVERHEAD_GB
): number {
  return modelVramGb + kvCacheGb + overheadGb;
}

/**
 * Get VRAM breakdown for display.
 */
export function getVRAMBreakdown(
  totalVramAvailable: number,
  modelVramGb: number,
  kvCacheGb: number,
  overheadGb: number = CUDA_OVERHEAD_GB
): VRAMBreakdown {
  const totalRequired = calculateTotalVRAM(modelVramGb, kvCacheGb, overheadGb);
  const headroom = totalVramAvailable - totalRequired;
  const headroomPercent = totalVramAvailable > 0
    ? (headroom / totalVramAvailable) * 100
    : 0;

  return {
    total_available_gb: totalVramAvailable,
    model_weights_gb: Math.round(modelVramGb * 100) / 100,
    kv_cache_gb: Math.round(kvCacheGb * 100) / 100,
    overhead_gb: overheadGb,
    total_required_gb: Math.round(totalRequired * 100) / 100,
    headroom_gb: Math.round(headroom * 100) / 100,
    headroom_percent: Math.round(headroomPercent * 10) / 10,
    fits: headroom >= 0,
    safe: headroom >= 1.0,
  };
}

/**
 * Calculate efficiency score.
 * Efficiency = sqrt(normalized_quality * normalized_performance) / VRAM * 100
 */
export function calculateEfficiency(
  qualityScore: number,
  tokensPerSecond: number,
  vramRequired: number
): number {
  if (vramRequired <= 0) return 0;

  const normQuality = Math.min(qualityScore / MAX_QUALITY_SCORE, 1.0);
  const normPerf = Math.min(tokensPerSecond / MAX_TOKENS_PER_SEC, 1.0);
  const combined = Math.sqrt(normQuality * normPerf);

  return Math.round((combined / vramRequired) * 100 * 100) / 100;
}

/**
 * Compute Pareto frontier for a single metric.
 * A point is Pareto-optimal if no other point dominates it
 * (better metric with same or less VRAM, or same metric with less VRAM).
 */
function computeParetoFrontier(
  points: ChartPoint[],
  getMetric: (p: ChartPoint) => number
): Set<string> {
  if (points.length === 0) return new Set();

  // Sort by metric descending (higher is better)
  const sorted = [...points].sort((a, b) => getMetric(b) - getMetric(a));

  const frontier = new Set<string>();
  let minVram = Infinity;

  for (const point of sorted) {
    if (point.vram_required < minVram) {
      frontier.add(point.id);
      minVram = point.vram_required;
    }
  }

  return frontier;
}

/**
 * Convert model+quant combos to chart points with computed metrics.
 */
export function createChartPoints(
  combos: ModelQuantCombo[],
  gpuName: string,
  contextLength: number,
  totalVramGb: number
): ChartPoint[] {
  const contextKey = contextLength.toString();

  return combos
    .filter((combo) => {
      // Filter out combos that don't have data for this context length
      return combo.kv_cache[contextKey] !== undefined;
    })
    .map((combo) => {
      const kvCacheGb = combo.kv_cache[contextKey] || 0;
      const vramRequired = calculateTotalVRAM(combo.model_vram_gb, kvCacheGb);
      const vramHeadroom = totalVramGb - vramRequired;
      const tokensPerSecond = combo.performance[gpuName] || 0;
      const efficiencyScore = calculateEfficiency(
        combo.quality_score,
        tokensPerSecond,
        vramRequired
      );

      const id = `${combo.model_name}-${combo.quant_name}`;

      return {
        id,
        model_name: combo.model_name,
        quant_name: combo.quant_name,
        vram_required: Math.round(vramRequired * 100) / 100,
        vram_headroom: Math.round(vramHeadroom * 100) / 100,
        quality_score: combo.quality_score,
        tokens_per_second: tokensPerSecond,
        efficiency_score: efficiencyScore,
        is_moe: combo.is_moe,
        is_tool_calling: combo.domains.includes("tool-calling"),
        is_pareto_quality: false,
        is_pareto_performance: false,
        is_pareto_efficiency: false,
        domains: combo.domains,
        max_context: combo.max_context,
        bits_per_weight: combo.bits_per_weight,
        total_params_b: combo.total_params_b,
      };
    });
}

/**
 * Mark Pareto-optimal points and filter to those that fit in VRAM.
 */
export function computeParetoPoints(
  points: ChartPoint[],
  maxVram: number
): ChartPoint[] {
  // Filter to points that fit
  const fittingPoints = points.filter((p) => p.vram_required <= maxVram);

  // Compute frontiers
  const qualityFrontier = computeParetoFrontier(
    fittingPoints,
    (p) => p.quality_score
  );
  const performanceFrontier = computeParetoFrontier(
    fittingPoints,
    (p) => p.tokens_per_second
  );
  const efficiencyFrontier = computeParetoFrontier(
    fittingPoints,
    (p) => p.efficiency_score
  );

  // Mark points
  return fittingPoints.map((point) => ({
    ...point,
    is_pareto_quality: qualityFrontier.has(point.id),
    is_pareto_performance: performanceFrontier.has(point.id),
    is_pareto_efficiency: efficiencyFrontier.has(point.id),
  }));
}

/**
 * Filter points by domain.
 */
export function filterByDomain(
  points: ChartPoint[],
  domains: string[]
): ChartPoint[] {
  if (domains.length === 0) return points;

  return points.filter((point) =>
    point.domains.some((d) => domains.includes(d))
  );
}

/**
 * Get Pareto frontier line points (sorted by VRAM for connecting lines).
 */
export function getParetoFrontierLine(
  points: ChartPoint[],
  frontierType: "quality" | "performance" | "efficiency"
): ChartPoint[] {
  const frontierPoints = points.filter((p) => {
    switch (frontierType) {
      case "quality":
        return p.is_pareto_quality;
      case "performance":
        return p.is_pareto_performance;
      case "efficiency":
        return p.is_pareto_efficiency;
    }
  });

  // Sort by VRAM ascending for line drawing
  return frontierPoints.sort((a, b) => a.vram_required - b.vram_required);
}

/**
 * Format VRAM for display.
 */
export function formatVRAM(gb: number): string {
  if (gb >= 100) {
    return `${Math.round(gb)} GB`;
  }
  return `${gb.toFixed(1)} GB`;
}

/**
 * Format tokens per second for display.
 */
export function formatTPS(tps: number): string {
  if (tps >= 100) {
    return `${Math.round(tps)} tok/s`;
  }
  return `${tps.toFixed(1)} tok/s`;
}

/**
 * Get quality tier label.
 */
export function getQualityTierLabel(tier: string): string {
  const labels: Record<string, string> = {
    near_lossless: "Near Lossless",
    very_low_loss: "Very Low Loss",
    recommended: "Recommended",
    balanced: "Balanced",
    noticeable_loss: "Noticeable Loss",
    high_loss: "High Loss",
    extreme_loss: "Extreme Loss",
  };
  return labels[tier] || tier;
}

/**
 * Get color for quality tier.
 */
export function getQualityTierColor(tier: string): string {
  const colors: Record<string, string> = {
    near_lossless: "#22c55e",
    very_low_loss: "#84cc16",
    recommended: "#3b82f6",
    balanced: "#f59e0b",
    noticeable_loss: "#f97316",
    high_loss: "#ef4444",
    extreme_loss: "#dc2626",
  };
  return colors[tier] || "#6b7280";
}
