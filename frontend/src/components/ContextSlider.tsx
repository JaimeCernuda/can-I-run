/**
 * Context Length Slider Component
 *
 * A slider for selecting context length with real-time KV cache estimation.
 * Shows warnings when KV cache is too large or exceeds model limits.
 */

import { useMemo, useState, useRef, useEffect } from "react";
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
  // Internal state for dragging
  const [localIndex, setLocalIndex] = useState(selectedIndex);
  const [isDragging, setIsDragging] = useState(false);

  // Sync local index with prop when not dragging
  if (selectedIndex !== localIndex && !isDragging) {
    setLocalIndex(selectedIndex);
  }

  const handlePointerUp = () => {
    setIsDragging(false);
    if (localIndex !== selectedIndex) {
      onIndexChange(localIndex);
    }
  };

  const selectedPosition = positions[localIndex];
  const selectedValue = selectedPosition?.value ?? 8192;

  // Check context warnings
  const warnings = useMemo(() => {
    const result: string[] = [];

    // Check if KV cache is too large
    // Note: kvCacheGb is from the parent based on COMMITTED index. 
    // We could try to estimate it here but it's complex. 
    // User requested "on release" updates so stale KV cache during drag is acceptable.
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

  // Helper to get percent
  const getPercent = (index: number) => {
    return (index / (positions.length - 1)) * 100;
  };

  // Handle interactive drag
  const trackRef = useRef<HTMLDivElement>(null);

  const handleDrag = (clientX: number) => {
    if (!trackRef.current) return;
    const rect = trackRef.current.getBoundingClientRect();
    const percent = Math.min(Math.max(0, (clientX - rect.left) / rect.width), 1);

    const steps = positions.length - 1;
    const newIndex = Math.round(percent * steps);

    if (newIndex !== localIndex) {
      setLocalIndex(newIndex);
      setIsDragging(true);
    }
  };

  useEffect(() => {
    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (isDragging) {
        const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;
        handleDrag(clientX);
      }
    };

    const handleUp = () => {
      if (isDragging) {
        setIsDragging(false);
        onIndexChange(localIndex);
      }
    };

    if (isDragging) {
      document.addEventListener('mousemove', handleMove);
      document.addEventListener('mouseup', handleUp);
      document.addEventListener('touchmove', handleMove);
      document.addEventListener('touchend', handleUp);
    }

    return () => {
      document.removeEventListener('mousemove', handleMove);
      document.removeEventListener('mouseup', handleUp);
      document.removeEventListener('touchmove', handleMove);
      document.removeEventListener('touchend', handleUp);
    };
  }, [isDragging, localIndex, onIndexChange, positions.length]);

  const handleTrackClick = (e: React.MouseEvent | React.TouchEvent) => {
    const clientX = 'touches' in e ? e.touches[0].clientX : (e as React.MouseEvent).clientX;
    handleDrag(clientX);
    // Immediate update on click
    const rect = trackRef.current!.getBoundingClientRect();
    const percent = Math.min(Math.max(0, (clientX - rect.left) / rect.width), 1);
    const newIndex = Math.round(percent * (positions.length - 1));
    onIndexChange(newIndex);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-4">
        Context Length
      </label>

      <div className="relative w-full h-12 flex items-center select-none">
        {/* Track */}
        <div
          ref={trackRef}
          className="relative w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full cursor-pointer"
          onMouseDown={(e) => {
            handleTrackClick(e);
            setIsDragging(true);
          }}
          onTouchStart={(e) => {
            handleTrackClick(e);
            setIsDragging(true);
          }}
        >
          {/* Tick Marks (Base) */}
          {positions.map((_, idx) => (
            <div
              key={idx}
              className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-gray-400 dark:bg-gray-600 rounded-full transform -translate-x-1/2 pointer-events-none"
              style={{ left: `${getPercent(idx)}%` }}
            />
          ))}

          {/* Active Track (Fill) - Optional for single value, usually left filled */}
          <div
            className="absolute h-full bg-blue-500 rounded-full pointer-events-none"
            style={{ width: `${getPercent(localIndex)}%` }}
          />

          {/* Thumb */}
          <div
            className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-blue-500 rounded-full shadow hover:scale-110 transition-transform z-10 ${isDragging ? "scale-110 ring-4 ring-blue-500/20" : ""
              }`}
            style={{ left: `${getPercent(localIndex)}%` }}
          />
        </div>

        {/* Labels and Status Ticks (Below) */}
        <div className="absolute top-9 w-full h-8 pointer-events-none">
          {positions.map((pos, idx) => (
            <div key={pos.value}>
              {/* Visual Tick below track */}
              <div
                className="absolute top-0 w-1 h-1 bg-gray-400 dark:bg-gray-500 rounded-full transform -translate-x-1/2"
                style={{ left: `${getPercent(idx)}%`, top: '-2px' }}
              />

              <span
                className={`absolute top-4 -translate-x-1/2 text-[10px] transition-colors whitespace-nowrap ${positionStates[idx] === "unavailable"
                  ? "text-gray-300 dark:text-gray-600"
                  : positionStates[idx] === "warning"
                    ? "text-amber-500"
                    : "text-gray-500 dark:text-gray-400"
                  } ${idx === localIndex ? "font-bold text-blue-600 dark:text-blue-400" : ""}`}
                style={{ left: `${getPercent(idx)}%` }}
              >
                {pos.label}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* Current value and KV cache */}
      <div className="flex justify-between items-center text-sm mt-6">
        <div>
          <span className="text-gray-600 dark:text-gray-400">Selected: </span>
          <span className="font-medium text-gray-900 dark:text-gray-100">
            {selectedPosition?.label ?? "8K"} tokens
          </span>
        </div>
        <div>
          <span className="text-gray-600 dark:text-gray-400">KV Cache: </span>
          <span
            className={`font-medium ${warnings.length > 0
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
        <div className="space-y-1 mt-2">
          {warnings.map((warning, idx) => (
            <div
              key={idx}
              className="flex items-center gap-1 text-xs text-amber-600 dark:text-amber-400"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span>{warning}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function formatContext(tokens: number): string {
  if (tokens >= 1_000_000) {
    const val = tokens / 1_000_000;
    return Number.isInteger(val) ? `${val}M` : `${val.toFixed(1)}M`;
  }
  if (tokens >= 1_000) {
    return `${Math.floor(tokens / 1_000)}K`;
  }
  return String(tokens);
}
