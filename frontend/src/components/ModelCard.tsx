/**
 * Model Card Component
 *
 * Expandable card showing details for a Pareto-optimal model configuration.
 * Includes benchmarks, VRAM breakdown, and copy commands.
 */

import { useState } from "react";
import type { ChartPoint } from "../lib/types";
import {
  formatVRAM,
  formatTPS,
  getQualityTierColor,
} from "../lib/calculations";

interface ModelCardProps {
  point: ChartPoint;
  rank: number;
  onSelect: () => void;
  isSelected: boolean;
}

export function ModelCard({ point, rank, onSelect, isSelected }: ModelCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [copied, setCopied] = useState(false);

  // Generate Ollama command
  const ollamaModel = point.model_name.toLowerCase().replace(/-instruct$/i, "");
  const ollamaQuant = point.quant_name.toLowerCase();
  const ollamaCommand = `ollama run ${ollamaModel}:${ollamaQuant}`;

  const copyCommand = () => {
    navigator.clipboard.writeText(ollamaCommand);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  // Determine badge colors
  const paretoTypes = [];
  if (point.is_pareto_quality) paretoTypes.push("Quality");
  if (point.is_pareto_performance) paretoTypes.push("Performance");
  if (point.is_pareto_efficiency) paretoTypes.push("Efficiency");

  return (
    <div
      className={`bg-gray-800 rounded-lg border-2 transition-colors ${isSelected
        ? "border-blue-400"
        : "border-gray-700 hover:border-gray-600"
        }`}
    >
      {/* Header - always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-4 text-left"
      >
        <div className="flex items-start justify-between">
          <div className="flex items-start gap-3">
            {/* Rank badge */}
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-900/30 flex items-center justify-center">
              <span className="text-sm font-bold text-blue-300">
                #{rank}
              </span>
            </div>

            <div>
              {/* Model name */}
              <h3 className="font-medium text-gray-100">
                {point.model_name}
              </h3>

              {/* Quantization */}
              <div className="flex items-center gap-2 mt-0.5">
                <span className="text-sm text-gray-400">
                  {point.quant_name}
                </span>
                <span
                  className="px-1.5 py-0.5 text-xs rounded"
                  style={{
                    backgroundColor: `${getQualityTierColor(point.quant_name.includes("Q8") ? "near_lossless" : point.quant_name.includes("Q4") ? "recommended" : "balanced")}20`,
                    color: getQualityTierColor(point.quant_name.includes("Q8") ? "near_lossless" : point.quant_name.includes("Q4") ? "recommended" : "balanced"),
                  }}
                >
                  {point.bits_per_weight.toFixed(1)} bits
                </span>
              </div>

              <div className="flex flex-wrap gap-1 mt-1">
                {point.is_moe && (
                  <span className="px-1.5 py-0.5 bg-purple-900/30 text-purple-300 text-xs rounded">
                    MoE
                  </span>
                )}
                {point.is_tool_calling && (
                  <span className="px-1.5 py-0.5 bg-green-900/30 text-green-300 text-xs rounded">
                    Tool-Calling
                  </span>
                )}
                {paretoTypes.map((type) => (
                  <span
                    key={type}
                    className="px-1.5 py-0.5 bg-blue-900/30 text-blue-300 text-xs rounded"
                  >
                    {type} Optimal
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Stats summary */}
          <div className="text-right flex-shrink-0">
            <div className="text-lg font-semibold text-gray-100">
              {formatVRAM(point.vram_required)}
            </div>
            <div className="text-sm text-gray-400">
              {formatTPS(point.tokens_per_second)}
            </div>
            <div className="text-xs text-gray-500">
              Efficiency: {point.efficiency_score.toFixed(1)}
            </div>
          </div>
        </div>

        {/* Expand indicator */}
        <div className="mt-2 text-center">
          <svg
            className={`w-5 h-5 mx-auto text-gray-400 transition-transform ${isExpanded ? "rotate-180" : ""
              }`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M19 9l-7 7-7-7"
            />
          </svg>
        </div>
      </button>

      {/* Expanded details */}
      {isExpanded && (
        <div className="px-4 pb-4 border-t border-gray-200 dark:border-gray-700">
          <div className="pt-4 grid grid-cols-2 gap-4">
            {/* Metrics */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-300">
                Metrics
              </h4>
              <div className="space-y-1 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-400">
                    Quality Score:
                  </span>
                  <span className="font-medium text-gray-100">
                    {point.quality_score.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">
                    Speed:
                  </span>
                  <span className="font-medium text-gray-100">
                    {formatTPS(point.tokens_per_second)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">
                    VRAM Headroom:
                  </span>
                  <span
                    className={`font-medium ${point.vram_headroom >= 2
                      ? "text-green-600 dark:text-green-400"
                      : point.vram_headroom >= 0
                        ? "text-amber-600 dark:text-amber-400"
                        : "text-red-600 dark:text-red-400"
                      }`}
                  >
                    {point.vram_headroom.toFixed(1)} GB
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">
                    Max Context:
                  </span>
                  <span className="font-medium text-gray-100">
                    {point.max_context >= 1_000_000
                      ? `${point.max_context / 1_000_000}M`
                      : `${point.max_context / 1_000}K`}
                  </span>
                </div>
              </div>
            </div>

            {/* Domains */}
            <div className="space-y-2">
              <h4 className="text-sm font-medium text-gray-300">
                Domains
              </h4>
              <div className="flex flex-wrap gap-1">
                {point.domains.map((domain) => (
                  <span
                    key={domain}
                    className="px-2 py-0.5 bg-gray-700 text-gray-300 text-xs rounded"
                  >
                    {domain}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Command */}
          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Ollama Command
            </h4>
            <div className="flex items-center gap-2">
              <code className="flex-1 px-3 py-2 bg-gray-900 dark:bg-black text-green-400 text-sm rounded font-mono overflow-x-auto">
                {ollamaCommand}
              </code>
              <button
                onClick={copyCommand}
                className="flex-shrink-0 px-3 py-2 bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 text-sm rounded hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
              >
                {copied ? "Copied!" : "Copy"}
              </button>
            </div>
          </div>

          {/* Select for comparison */}
          <div className="mt-4">
            <button
              onClick={onSelect}
              className={`w-full py-2 px-4 rounded-lg text-sm font-medium transition-colors ${isSelected
                ? "bg-blue-500 text-white hover:bg-blue-600"
                : "bg-gray-700 text-gray-300 hover:bg-gray-600"
                }`}
            >
              {isSelected ? "Selected for Comparison" : "Select for Comparison"}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
