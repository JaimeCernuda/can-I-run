/**
 * GPU-to-Model Pareto Selector
 *
 * Main application component that orchestrates the UI and state management.
 */

import { useState, useMemo } from "react";
import { GPUSelector } from "./components/GPUSelector";
import { ContextSlider } from "./components/ContextSlider";
import { DomainFilter } from "./components/DomainFilter";
import { VRAMBreakdown } from "./components/VRAMBreakdown";
import { LinkedCharts } from "./components/LinkedCharts";
import { ModelCard } from "./components/ModelCard";
import { HowItWorks } from "./components/HowItWorks";
import {
  createChartPoints,
  computeParetoPoints,
  filterByDomain,
  getVRAMBreakdown,
} from "./lib/calculations";
import type { GPU, ModelDomain } from "./lib/types";
import computedData from "./data/computed.json";

// Type assertion for imported JSON
const data = computedData as {
  gpus: GPU[];
  quantizations: { name: string; bits: number; quality_factor: number; tier: string; ppl_increase: number }[];
  context_positions: { value: number; label: string }[];
  model_quant_combos: any[];
  metadata: { total_gpus: number; total_models: number; total_quants: number; total_combos: number; overhead_gb: number };
};

export default function App() {
  // GPU selection state
  const [selectedGpu, setSelectedGpu] = useState<GPU | null>(null);
  const [customVram, setCustomVram] = useState<number | null>(null);

  // Context length state
  const [contextIndex, setContextIndex] = useState(2); // Default to 8K (index 2)

  // Domain filter state
  const [selectedDomains, setSelectedDomains] = useState<ModelDomain[]>([]);

  // Selected models for comparison
  const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(new Set());

  // Derived values
  const totalVram = selectedGpu?.vram_gb ?? customVram ?? 0;
  const gpuName = selectedGpu?.name ?? "Custom GPU";
  const contextLength = data.context_positions[contextIndex]?.value ?? 8192;

  // Create chart points with all calculations
  const chartPoints = useMemo(() => {
    if (totalVram === 0) return [];

    const points = createChartPoints(
      data.model_quant_combos,
      selectedGpu?.name ?? "RTX 4090", // Use RTX 4090 as fallback for performance estimates
      contextLength,
      totalVram
    );

    // Filter by domain
    const filteredPoints = filterByDomain(points, selectedDomains);

    // Compute Pareto frontiers and filter to fitting models
    return computeParetoPoints(filteredPoints, totalVram);
  }, [totalVram, selectedGpu, contextLength, selectedDomains]);

  // Get VRAM breakdown for selected model (first Pareto-optimal model if any)
  const selectedPoint = chartPoints.find(p => p.is_pareto_quality) ?? chartPoints[0];
  const vramBreakdown = useMemo(() => {
    if (!totalVram || !selectedPoint) return null;

    // Find the combo data for KV cache
    const combo = data.model_quant_combos.find(
      c => `${c.model_name}-${c.quant_name}` === selectedPoint.id
    );
    const kvCache = combo?.kv_cache[contextLength.toString()] ?? 0;

    return getVRAMBreakdown(
      totalVram,
      selectedPoint.vram_required - kvCache - 0.5, // Extract model weights
      kvCache,
      0.5
    );
  }, [totalVram, selectedPoint, contextLength]);

  // Get Pareto-optimal models sorted by quality
  const paretoModels = useMemo(() => {
    return chartPoints
      .filter(p => p.is_pareto_quality || p.is_pareto_performance || p.is_pareto_efficiency)
      .sort((a, b) => b.quality_score - a.quality_score);
  }, [chartPoints]);

  // Count models per domain
  const domainCounts = useMemo(() => {
    const counts: Record<ModelDomain, number> = {
      general: 0,
      code: 0,
      "tool-calling": 0,
      math: 0,
      vision: 0,
      roleplay: 0,
    };

    for (const point of chartPoints) {
      for (const domain of point.domains) {
        counts[domain]++;
      }
    }

    return counts;
  }, [chartPoints]);

  // Toggle model selection
  const toggleModelSelection = (id: string) => {
    setSelectedModelIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  return (
    <div className="min-h-screen bg-[linear-gradient(135deg,#0a0a0f_0%,#1a1a2e_50%,#0f0f1a_100%)] text-gray-200 font-mono">
      {/* Header */}
      {/* Header */}
      <header className="border-b border-gray-800 bg-[#0a0a0f]/50 backdrop-blur-md sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <h1 className="text-2xl font-bold flex items-center gap-3">
            <span className="text-transparent bg-clip-text bg-gradient-to-br from-amber-400 to-red-500">◆</span>
            <span className="text-gray-100">Model ↔ Quant Selector</span>
            <span className="text-[0.7rem] px-2 py-1 bg-amber-500/10 border border-amber-500/30 rounded text-amber-500 tracking-wider">
              UNSLOTH
            </span>
          </h1>
          <p className="text-sm text-gray-400 mt-1 ml-7">
            Find the optimal model size × quantization for your GPU and context needs
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
        {/* Filters Section */}
        <section className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <GPUSelector
              gpus={data.gpus}
              selectedGpu={selectedGpu}
              customVram={customVram}
              onGpuChange={setSelectedGpu}
              onCustomVramChange={setCustomVram}
            />
            <ContextSlider
              positions={data.context_positions}
              selectedIndex={contextIndex}
              kvCacheGb={
                selectedPoint
                  ? data.model_quant_combos.find(
                    c => `${c.model_name}-${c.quant_name}` === selectedPoint.id
                  )?.kv_cache[contextLength.toString()] ?? 0
                  : 0
              }
              maxContext={selectedPoint?.max_context ?? null}
              effectiveContext={selectedPoint?.max_context ?? null}
              totalVramGb={totalVram}
              onIndexChange={setContextIndex}
            />
          </div>

          <DomainFilter
            selectedDomains={selectedDomains}
            onDomainsChange={setSelectedDomains}
            modelCounts={domainCounts}
          />

          {totalVram > 0 && (
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Available for model:{" "}
              <span className="font-semibold text-gray-900 dark:text-gray-100">
                {(totalVram - 0.5).toFixed(1)} GB
              </span>
              <span className="mx-2">|</span>
              Total VRAM: {totalVram} GB
              <span className="mx-2">|</span>
              Overhead: 0.5 GB
            </div>
          )}
        </section>

        {/* How It Works */}
        <HowItWorks />

        {/* Charts Section */}
        {totalVram > 0 ? (
          chartPoints.length > 0 ? (
            <section className="space-y-4">
              <LinkedCharts points={chartPoints} maxVram={totalVram} />
            </section>
          ) : (
            <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
              <p className="text-gray-600 dark:text-gray-400">
                No models fit in {totalVram} GB VRAM with {contextLength / 1000}K
                context.
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-500 mt-2">
                Try reducing context length or selecting a GPU with more VRAM.
              </p>
            </div>
          )
        ) : (
          <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-8 text-center">
            <p className="text-gray-600 dark:text-gray-400">
              Select a GPU or enter custom VRAM to see recommendations.
            </p>
          </div>
        )}

        {/* VRAM Breakdown */}
        {totalVram > 0 && selectedPoint && (
          <VRAMBreakdown breakdown={vramBreakdown} gpuName={gpuName} />
        )}

        {/* Pareto-Optimal Model Cards */}
        {paretoModels.length > 0 && (
          <section className="space-y-4">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Recommended Models ({paretoModels.length})
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {paretoModels.slice(0, 12).map((point, idx) => (
                <ModelCard
                  key={point.id}
                  point={point}
                  rank={idx + 1}
                  isSelected={selectedModelIds.has(point.id)}
                  onSelect={() => toggleModelSelection(point.id)}
                />
              ))}
            </div>
            {paretoModels.length > 12 && (
              <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                Showing top 12 of {paretoModels.length} Pareto-optimal models
              </p>
            )}
          </section>
        )}

        {/* Footer */}
        <footer className="text-center text-sm text-gray-500 dark:text-gray-400 py-4 border-t border-gray-200 dark:border-gray-700">
          <p>
            Data sources: llama.cpp, TechPowerUp, HuggingFace, Open LLM
            Leaderboard
          </p>
          <p className="mt-1">
            Built with React, Tailwind CSS, and Recharts |{" "}
            <a
              href="https://github.com"
              className="text-blue-500 hover:underline"
            >
              View on GitHub
            </a>
          </p>
        </footer>
      </main>
    </div>
  );
}
