/**
 * VRAM Breakdown Component
 *
 * Visual breakdown of VRAM usage showing model weights, KV cache,
 * overhead, and remaining headroom.
 */

import type { VRAMBreakdown as VRAMBreakdownType } from "../lib/types";

interface VRAMBreakdownProps {
  breakdown: VRAMBreakdownType | null;
  gpuName: string | null;
}

export function VRAMBreakdown({ breakdown, gpuName }: VRAMBreakdownProps) {
  if (!breakdown) {
    return (
      <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Select a GPU and model to see VRAM breakdown
        </p>
      </div>
    );
  }

  const {
    total_available_gb,
    model_weights_gb,
    kv_cache_gb,
    overhead_gb,
    total_required_gb,
    headroom_gb,
    headroom_percent,
    fits,
    safe,
  } = breakdown;

  // Calculate percentages for the bar
  const modelPercent = (model_weights_gb / total_available_gb) * 100;
  const kvPercent = (kv_cache_gb / total_available_gb) * 100;
  const overheadPercent = (overhead_gb / total_available_gb) * 100;
  const headroomBarPercent = Math.max(0, headroom_percent);

  return (
    <div className="p-4 bg-gray-800 rounded-lg space-y-3">
      {/* Header */}
      <div className="flex justify-between items-center">
        <h3 className="text-sm font-medium text-gray-300">
          VRAM Budget
        </h3>
        {gpuName && (
          <span className="text-sm text-gray-400">
            {gpuName}
          </span>
        )}
      </div>

      {/* Visual bar */}
      <div className="h-6 bg-gray-700 rounded-full overflow-hidden flex">
        {/* Model weights */}
        <div
          className="bg-blue-500 h-full transition-all duration-300"
          style={{ width: `${Math.min(modelPercent, 100)}%` }}
          title={`Model: ${model_weights_gb} GB`}
        />
        {/* KV Cache */}
        <div
          className="bg-purple-500 h-full transition-all duration-300"
          style={{ width: `${Math.min(kvPercent, 100 - modelPercent)}%` }}
          title={`KV Cache: ${kv_cache_gb} GB`}
        />
        {/* Overhead */}
        <div
          className="bg-gray-500 h-full transition-all duration-300"
          style={{
            width: `${Math.min(overheadPercent, 100 - modelPercent - kvPercent)}%`,
          }}
          title={`Overhead: ${overhead_gb} GB`}
        />
        {/* Headroom */}
        {headroom_gb > 0 && (
          <div
            className={`h-full transition-all duration-300 ${safe
              ? "bg-green-400 dark:bg-green-600"
              : "bg-amber-400 dark:bg-amber-600"
              }`}
            style={{ width: `${headroomBarPercent}%` }}
            title={`Headroom: ${headroom_gb} GB`}
          />
        )}
      </div>

      {/* Legend */}
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-blue-500 rounded-sm" />
          <span className="text-gray-400">
            Model: {model_weights_gb} GB
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-purple-500 rounded-sm" />
          <span className="text-gray-400">
            KV Cache: {kv_cache_gb} GB
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-3 h-3 bg-gray-500 rounded-sm" />
          <span className="text-gray-400">
            Overhead: {overhead_gb} GB
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className={`w-3 h-3 rounded-sm ${headroom_gb > 0
              ? safe
                ? "bg-green-600"
                : "bg-amber-600"
              : "bg-red-600"
              }`}
          />
          <span className="text-gray-400">
            Headroom: {headroom_gb} GB ({headroom_percent}%)
          </span>
        </div>
      </div>

      {/* Status message */}
      <div
        className={`text-sm font-medium ${!fits
          ? "text-red-400"
          : safe
            ? "text-green-400"
            : "text-amber-400"
          }`}
      >
        {!fits ? (
          <>
            <span className="mr-1">⚠️</span>
            Model requires {total_required_gb} GB but only {total_available_gb}{" "}
            GB available
          </>
        ) : safe ? (
          <>
            <span className="mr-1">✅</span>
            {headroom_gb} GB headroom - safe to run
          </>
        ) : (
          <>
            <span className="mr-1">⚠️</span>
            Only {headroom_gb} GB headroom - may OOM under load
          </>
        )}
      </div>
    </div>
  );
}
