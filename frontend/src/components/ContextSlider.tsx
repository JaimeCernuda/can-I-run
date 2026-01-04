/**
 * Context Length Slider Component
 *
 * A slider for selecting context length with real-time KV cache estimation.
 * Shows warnings when KV cache is too large or exceeds model limits.
 */

import { useMemo } from "react";
import type { ContextPosition } from "../lib/types";

interface ContextSliderProps {
  positions: ContextPosition[];
  selectedIndex: number;
  kvCacheGb: number;
  maxContext: number | null; // Model's max context, or null if no model selected
  effectiveContext: number | null; // Model's effective context
  totalVramGb: number;
  onIndexChange: (index: number) => void;
}

export function ContextSlider({
  positions,
  selectedIndex,
  kvCacheGb,
  maxContext,
  effectiveContext,
  totalVramGb,
  onIndexChange,
}: ContextSliderProps) {
  const selectedPosition = positions[selectedIndex];
  const selectedValue = selectedPosition?.value ?? 8192;

  // Check context warnings
  const warnings = useMemo(() => {
    const result: string[] = [];

    // Check if KV cache is too large
    const kvCachePercent = totalVramGb > 0 ? (kvCacheGb / totalVramGb) * 100 : 0;
    if (kvCachePercent > 50) {
      result.push(
        `KV cache uses ${kvCachePercent.toFixed(0)}% of VRAM (>50% warning)`
      );
    }

    // Check if context exceeds model max
    if (maxContext && selectedValue > maxContext) {
      result.push(
        `Context exceeds model's max (${formatContext(maxContext)})`
      );
    }

    // Check if context exceeds effective context
    if (
      effectiveContext &&
      selectedValue > effectiveContext &&
      (!maxContext || selectedValue <= maxContext)
    ) {
      result.push(
        `Performance may degrade above ${formatContext(effectiveContext)}`
      );
    }

    return result;
  }, [kvCacheGb, totalVramGb, selectedValue, maxContext, effectiveContext]);

  // Determine which positions are available
  const positionStates = useMemo(() => {
    return positions.map((pos) => {
      if (maxContext && pos.value > maxContext) {
        return "unavailable";
      }
      if (effectiveContext && pos.value > effectiveContext) {
        return "warning";
      }
      return "available";
    });
  }, [positions, maxContext, effectiveContext]);

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
        Context Length
      </label>

      <div className="space-y-2">
        {/* Slider */}
        <input
          type="range"
          min={0}
          max={positions.length - 1}
          value={selectedIndex}
          onChange={(e) => onIndexChange(parseInt(e.target.value))}
          className="w-full"
        />

        {/* Position markers */}
        <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400">
          {positions.map((pos, idx) => (
            <span
              key={pos.value}
              className={`${
                positionStates[idx] === "unavailable"
                  ? "text-gray-300 dark:text-gray-600"
                  : positionStates[idx] === "warning"
                    ? "text-amber-500"
                    : ""
              } ${idx === selectedIndex ? "font-bold text-blue-600 dark:text-blue-400" : ""}`}
            >
              {pos.label}
            </span>
          ))}
        </div>

        {/* Current value and KV cache */}
        <div className="flex justify-between items-center text-sm">
          <div>
            <span className="text-gray-600 dark:text-gray-400">Selected: </span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {selectedPosition?.label ?? "8K"} tokens
            </span>
          </div>
          <div>
            <span className="text-gray-600 dark:text-gray-400">KV Cache: </span>
            <span
              className={`font-medium ${
                warnings.length > 0
                  ? "text-amber-600 dark:text-amber-400"
                  : "text-gray-900 dark:text-gray-100"
              }`}
            >
              {kvCacheGb.toFixed(2)} GB
            </span>
          </div>
        </div>

        {/* Warnings */}
        {warnings.length > 0 && (
          <div className="space-y-1">
            {warnings.map((warning, idx) => (
              <div
                key={idx}
                className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400"
              >
                <svg
                  className="w-4 h-4"
                  fill="currentColor"
                  viewBox="0 0 20 20"
                >
                  <path
                    fillRule="evenodd"
                    d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z"
                    clipRule="evenodd"
                  />
                </svg>
                <span>{warning}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    return `${tokens / 1_000_000}M`;
  }
  if (tokens >= 1_000) {
    return `${tokens / 1_000}K`;
  }
  return String(tokens);
}
