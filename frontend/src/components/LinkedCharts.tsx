/**
 * Linked Charts Component
 *
 * Container for the three synchronized Pareto charts.
 * Manages shared hover/selection state across all charts.
 */

import { ParetoChart } from "./ParetoChart";
import type { ChartPoint } from "../lib/types";
import { useLinkedHighlight } from "../hooks/useLinkedHighlight";

interface LinkedChartsProps {
  points: ChartPoint[];
  maxVram: number;
}

export function LinkedCharts({ points, maxVram }: LinkedChartsProps) {
  const {
    highlightedId,
    selectedIds,
    onPointHover,
    onPointLeave,
    onPointClick,
    clearSelections,
  } = useLinkedHighlight();

  // Count Pareto-optimal points on each frontier
  const qualityCount = points.filter((p) => p.is_pareto_quality).length;
  const perfCount = points.filter((p) => p.is_pareto_performance).length;
  const effCount = points.filter((p) => p.is_pareto_efficiency).length;

  return (
    <div className="space-y-4">
      {/* Header with stats */}
      <div className="flex justify-between items-center">
        <div className="text-sm text-gray-600 dark:text-gray-400">
          <span className="font-medium text-gray-900 dark:text-gray-100">
            {points.length}
          </span>{" "}
          model configurations fit your GPU
        </div>

        {selectedIds.size > 0 && (
          <button
            onClick={clearSelections}
            className="text-sm text-blue-600 dark:text-blue-400 hover:underline"
          >
            Clear selections ({selectedIds.size})
          </button>
        )}
      </div>

      {/* Pareto frontier stats */}
      <div className="flex gap-4 text-xs">
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-blue-500" />
          <span className="text-gray-600 dark:text-gray-400">
            Quality frontier: {qualityCount}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-green-500" />
          <span className="text-gray-600 dark:text-gray-400">
            Performance frontier: {perfCount}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="w-3 h-0.5 bg-amber-500" />
          <span className="text-gray-600 dark:text-gray-400">
            Efficiency frontier: {effCount}
          </span>
        </div>
      </div>

      {/* Three charts */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ParetoChart
          points={points}
          chartType="quality"
          maxVram={maxVram}
          highlightedId={highlightedId}
          selectedIds={selectedIds}
          onPointHover={onPointHover}
          onPointLeave={onPointLeave}
          onPointClick={onPointClick}
        />
        <ParetoChart
          points={points}
          chartType="performance"
          maxVram={maxVram}
          highlightedId={highlightedId}
          selectedIds={selectedIds}
          onPointHover={onPointHover}
          onPointLeave={onPointLeave}
          onPointClick={onPointClick}
        />
        <ParetoChart
          points={points}
          chartType="efficiency"
          maxVram={maxVram}
          highlightedId={highlightedId}
          selectedIds={selectedIds}
          onPointHover={onPointHover}
          onPointLeave={onPointLeave}
          onPointClick={onPointClick}
        />
      </div>

      {/* Highlighted/selected point details */}
      {highlightedId && (
        <div className="text-sm text-gray-600 dark:text-gray-400 text-center">
          Hover:{" "}
          <span className="font-medium text-gray-900 dark:text-gray-100">
            {highlightedId}
          </span>
          <span className="ml-2 text-xs">(click to pin for comparison)</span>
        </div>
      )}
    </div>
  );
}
