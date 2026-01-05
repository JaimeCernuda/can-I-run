/**
 * GPU-to-Model Pareto Selector
 *
 * Main application component that orchestrates the UI and state management.
 */

import { useState, useMemo, useDeferredValue, useEffect } from "react";
import { GPUSelector } from "./components/GPUSelector";
import { ContextSlider } from "./components/ContextSlider";
import { DualRangeSlider } from "./components/DualRangeSlider";
import { DomainFilter } from "./components/DomainFilter";
import { VRAMBreakdown } from "./components/VRAMBreakdown";
import { LinkedCharts } from "./components/LinkedCharts";
import { ModelCard } from "./components/ModelCard";
import About from "./components/About";
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

const MODEL_SIZE_POINTS = [0, 0.5, 1, 3, 7, 14, 30, 70, 150, 300, 405, 700];

export default function App() {
    // Simple URL routing
    const [view, setView] = useState<'home' | 'about'>(() => {
        const params = new URLSearchParams(window.location.search);
        return params.get('view') === 'about' ? 'about' : 'home';
    });

    // Update URL when view changes
    useEffect(() => {
        const params = new URLSearchParams(window.location.search);
        const currentView = params.get('view') === 'about' ? 'about' : 'home';

        // Only push state if actual view mismatch to avoid loops
        if (currentView !== view) {
            if (view === 'about') {
                params.set('view', 'about');
            } else {
                params.delete('view');
            }
            const newUrl = `${window.location.pathname}${params.toString() ? '?' + params.toString() : ''}`;
            window.history.pushState({}, '', newUrl);
        }
    }, [view]);

    // Handle Back Button
    useEffect(() => {
        const handlePopState = () => {
            const params = new URLSearchParams(window.location.search);
            const newView = params.get('view') === 'about' ? 'about' : 'home';
            setView(newView);
        };

        window.addEventListener('popstate', handlePopState);
        return () => window.removeEventListener('popstate', handlePopState);
    }, []);

    // GPU selection state
    const [selectedGpu, setSelectedGpu] = useState<GPU | null>(null);
    const [customVram, setCustomVram] = useState<number | null>(null);

    // Search and Sort state
    const [searchQuery, setSearchQuery] = useState("");
    const [sortBy, setSortBy] = useState<'quality' | 'performance' | 'efficiency'>('quality');
    const [visibleCount, setVisibleCount] = useState(12);

    // Context length state
    const [contextIndex, setContextIndex] = useState(2); // Default to 8K (index 2)
    const deferredContextIndex = useDeferredValue(contextIndex);

    // Domain filter state
    const [selectedDomains, setSelectedDomains] = useState<ModelDomain[]>([]);

    // Model Parameter filter state (Indices of MODEL_SIZE_POINTS)
    const [paramRangeIndices, setParamRangeIndices] = useState<{ min: number; max: number }>({ min: 0, max: MODEL_SIZE_POINTS.length - 1 });

    // Selected models for comparison
    const [selectedModelIds, setSelectedModelIds] = useState<Set<string>>(new Set());

    // Derived values
    const totalVram = selectedGpu?.vram_gb ?? customVram ?? 0;
    const gpuName = selectedGpu?.name ?? "Custom GPU";
    const contextLength = data.context_positions[deferredContextIndex]?.value ?? 8192;

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
        let filteredPoints = filterByDomain(points, selectedDomains);

        // Filter by params using stepped points
        const minParams = MODEL_SIZE_POINTS[paramRangeIndices.min];
        const maxParams = MODEL_SIZE_POINTS[paramRangeIndices.max];

        filteredPoints = filteredPoints.filter(p =>
            p.total_params_b >= minParams &&
            (paramRangeIndices.max === MODEL_SIZE_POINTS.length - 1 ? true : p.total_params_b <= maxParams)
        );

        // Compute Pareto frontiers and filter to fitting models
        return computeParetoPoints(filteredPoints, totalVram);
    }, [totalVram, selectedGpu, contextLength, selectedDomains, paramRangeIndices]);

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

    // Get Filtered and Sorted Models for the list
    const filteredModels = useMemo(() => {
        let models = chartPoints.filter(p => p.is_pareto_quality || p.is_pareto_performance || p.is_pareto_efficiency);

        // Apply text search
        if (searchQuery) {
            const query = searchQuery.toLowerCase();
            models = models.filter(p => p.model_name.toLowerCase().includes(query));
        }

        // Apply sorting
        return models.sort((a, b) => {
            if (sortBy === 'quality') return b.quality_score - a.quality_score;
            if (sortBy === 'performance') return b.tokens_per_second - a.tokens_per_second;
            if (sortBy === 'efficiency') {
                // Approximate efficiency as Quality * Speed / VRAM
                const effA = (a.quality_score * a.tokens_per_second) / a.vram_required;
                const effB = (b.quality_score * b.tokens_per_second) / b.vram_required;
                return effB - effA;
            }
            return 0;
        });
    }, [chartPoints, searchQuery, sortBy]);

    // Count models per domain
    const domainCounts = useMemo(() => {
        const counts: Record<ModelDomain, number> = {
            general: 0,
            code: 0,
            "tool-calling": 0,
            math: 0,
            vision: 0,
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

    if (view === 'about') {
        return (
            <div className="min-h-screen bg-[linear-gradient(135deg,#0a0a0f_0%,#1a1a2e_50%,#0f0f1a_100%)] text-gray-200 font-mono">
                <About onBack={() => setView('home')} />
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-[linear-gradient(135deg,#0a0a0f_0%,#1a1a2e_50%,#0f0f1a_100%)] text-gray-200 font-mono">
            {/* Header */}
            <header className="border-b border-gray-800 bg-[#0a0a0f]/50 backdrop-blur-md sticky top-0 z-10">
                <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div>
                        <h1 className="text-2xl font-bold flex items-center gap-3">
                            <span className="text-transparent bg-clip-text bg-gradient-to-br from-amber-400 to-red-500">◆</span>
                            <span className="text-gray-100">Model ↔ Quant Selector</span>
                        </h1>
                        <p className="text-sm text-gray-400 mt-1 ml-7">
                            Find the optimal model size × quantization for your GPU and context needs
                        </p>
                    </div>
                    <button
                        onClick={() => setView('about')}
                        className="text-sm font-medium text-blue-400 hover:text-blue-300 transition-colors border border-blue-900 px-3 py-1.5 rounded-lg hover:bg-blue-900/10"
                    >
                        How It Works
                    </button>
                </div>
            </header>

            <main className="max-w-7xl mx-auto px-4 py-6 space-y-6">
                {/* Filters Section */}
                <section className="bg-gray-900/50 backdrop-blur-sm rounded-lg border border-gray-700 p-6 space-y-8">
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

                    {/* Model Params Filter */}
                    <div>
                        <div className="flex justify-between text-sm font-medium text-gray-300 mb-2">
                            <span>Model Size</span>
                            <span className="text-blue-400">
                                {MODEL_SIZE_POINTS[paramRangeIndices.min]}B - {
                                    paramRangeIndices.max === MODEL_SIZE_POINTS.length - 1
                                        ? "Any"
                                        : `${MODEL_SIZE_POINTS[paramRangeIndices.max]}B`
                                }
                            </span>
                        </div>
                        <DualRangeSlider
                            min={0}
                            max={MODEL_SIZE_POINTS.length - 1}
                            step={1}
                            minVal={paramRangeIndices.min}
                            maxVal={paramRangeIndices.max}
                            onChange={(min, max) => setParamRangeIndices({ min, max })}
                            formatLabel={(val) => {
                                const size = MODEL_SIZE_POINTS[val];
                                return size >= 700 ? "Max" : `${size}B`;
                            }}
                            steps={MODEL_SIZE_POINTS}
                        />
                    </div>

                    {totalVram > 0 && (
                        <div className="text-sm text-gray-400">
                            Available for model:{" "}
                            <span className="font-semibold text-gray-100">
                                {(totalVram - 0.5).toFixed(1)} GB
                            </span>
                            <span className="mx-2">|</span>
                            Total VRAM: {totalVram} GB
                            <span className="mx-2">|</span>
                            Overhead: 0.5 GB
                        </div>
                    )}
                </section>

                {/* Charts Section */}
                {totalVram > 0 ? (
                    chartPoints.length > 0 ? (
                        <section className="space-y-4">
                            <LinkedCharts points={chartPoints} maxVram={totalVram} />
                        </section>
                    ) : (
                        <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
                            <p className="text-gray-400">
                                No models fit in {totalVram} GB VRAM with {contextLength / 1000}K
                                context.
                            </p>
                            <p className="text-sm text-gray-500 mt-2">
                                Try reducing context length or selecting a GPU with more VRAM.
                            </p>
                        </div>
                    )
                ) : (
                    <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
                        <p className="text-gray-400">
                            Select a GPU or enter custom VRAM to see recommendations.
                        </p>
                    </div>
                )}

                {/* VRAM Breakdown */}
                {totalVram > 0 && selectedPoint && (
                    <VRAMBreakdown breakdown={vramBreakdown} gpuName={gpuName} />
                )}

                {/* Pareto-Optimal Model Cards */}
                {filteredModels.length > 0 && (
                    <section className="space-y-4">
                        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
                            <h2 className="text-lg font-semibold text-gray-100">
                                Recommended Models ({filteredModels.length})
                            </h2>
                            <div className="flex flex-wrap gap-2">
                                <input
                                    type="text"
                                    placeholder="Search models..."
                                    value={searchQuery}
                                    onChange={(e) => setSearchQuery(e.target.value)}
                                    className="px-3 py-1.5 text-sm bg-gray-800 border border-gray-600 rounded-lg text-gray-100 focus:outline-none focus:border-blue-500"
                                />
                                <div className="flex bg-gray-800 border border-gray-600 rounded-lg overflow-hidden">
                                    <button
                                        onClick={() => setSortBy('quality')}
                                        className={`px-3 py-1.5 text-xs font-medium transition-colors ${sortBy === 'quality' ? 'bg-blue-900 text-blue-300' : 'text-gray-400 hover:bg-gray-700'}`}
                                    >
                                        Quality
                                    </button>
                                    <button
                                        onClick={() => setSortBy('performance')}
                                        className={`px-3 py-1.5 text-xs font-medium border-l border-gray-600 transition-colors ${sortBy === 'performance' ? 'bg-green-900 text-green-300' : 'text-gray-400 hover:bg-gray-700'}`}
                                    >
                                        Speed
                                    </button>
                                    <button
                                        onClick={() => setSortBy('efficiency')}
                                        className={`px-3 py-1.5 text-xs font-medium border-l border-gray-600 transition-colors ${sortBy === 'efficiency' ? 'bg-amber-900 text-amber-300' : 'text-gray-400 hover:bg-gray-700'}`}
                                    >
                                        Efficiency
                                    </button>
                                </div>
                            </div>
                        </div>

                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                            {filteredModels.slice(0, visibleCount).map((point, idx) => (
                                <ModelCard
                                    key={point.id}
                                    point={point}
                                    rank={idx + 1}
                                    isSelected={selectedModelIds.has(point.id)}
                                    onSelect={() => toggleModelSelection(point.id)}
                                />
                            ))}
                        </div>

                        {filteredModels.length > visibleCount && (
                            <div className="flex justify-center pt-4">
                                <button
                                    onClick={() => setVisibleCount(c => c + 12)}
                                    className="px-6 py-2 text-sm font-medium text-blue-300 bg-blue-900/30 border border-blue-900/50 rounded-lg hover:bg-blue-900/50 transition-colors"
                                >
                                    Load More Models ({filteredModels.length - visibleCount} remaining)
                                </button>
                            </div>
                        )}
                    </section>
                )}

                {/* Footer */}
                <footer className="text-center text-sm text-gray-400 py-4 border-t border-gray-800">

                </footer>
            </main>
        </div >
    );
}
