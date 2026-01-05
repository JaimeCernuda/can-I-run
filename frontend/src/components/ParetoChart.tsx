/**
 * Pareto Chart Component
 *
 * A scatter plot showing model/quant combinations with a Pareto frontier line.
 * Used for Quality, Performance, and Efficiency charts.
 */

import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Line,
  ReferenceLine,
} from "recharts";
import type { ChartPoint, ChartType } from "../lib/types";
import { getParetoFrontierLine, formatVRAM, formatTPS } from "../lib/calculations";

interface ParetoChartProps {
  points: ChartPoint[];
  chartType: ChartType;
  maxVram: number;
  highlightedId: string | null;
  selectedIds: Set<string>;
  onPointHover: (id: string) => void;
  onPointLeave: () => void;
  onPointClick: (id: string) => void;
}

const CHART_CONFIG: Record<
  ChartType,
  {
    title: string;
    yLabel: string;
    color: string;
    getY: (p: ChartPoint) => number;
    formatY: (v: number) => string;
  }
> = {
  quality: {
    title: "Quality (Benchmark)",
    yLabel: "Quality Score",
    color: "#3b82f6",
    getY: (p) => p.quality_score,
    formatY: (v) => v.toFixed(1),
  },
  performance: {
    title: "Performance (tok/s)",
    yLabel: "Tokens/sec",
    color: "#22c55e",
    getY: (p) => p.tokens_per_second,
    formatY: (v) => `${v.toFixed(1)} tok/s`,
  },
  efficiency: {
    title: "Efficiency (Q×P/V)",
    yLabel: "Efficiency Score",
    color: "#f59e0b",
    getY: (p) => p.efficiency_score,
    formatY: (v) => v.toFixed(2),
  },
};

export function ParetoChart({
  points,
  chartType,
  maxVram,
  highlightedId,
  selectedIds,
  onPointHover,
  onPointLeave,
  onPointClick,
}: ParetoChartProps) {
  const config = CHART_CONFIG[chartType];
  const frontierPoints = getParetoFrontierLine(points, chartType);

  // Prepare data for scatter
  const scatterData = points.map((p) => ({
    ...p,
    x: p.vram_required,
    y: config.getY(p),
    isHighlighted: p.id === highlightedId || selectedIds.has(p.id),
    isPareto:
      chartType === "quality"
        ? p.is_pareto_quality
        : chartType === "performance"
          ? p.is_pareto_performance
          : p.is_pareto_efficiency,
  }));

  // Prepare frontier line data
  const lineData = frontierPoints.map((p) => ({
    x: p.vram_required,
    y: config.getY(p),
  }));

  // Custom dot renderer
  const renderDot = (props: any) => {
    const { cx, cy, payload } = props;
    if (!cx || !cy) return <g />;

    const isHighlighted = payload.isHighlighted;
    const isMoe = payload.is_moe;
    const isToolCalling = payload.is_tool_calling;
    const isPareto = payload.isPareto;

    // Determine color based on type
    let fill = "#6b7280"; // default gray
    if (isToolCalling) fill = "#22c55e"; // green for tool-calling
    else if (isMoe) fill = "#8b5cf6"; // purple for MoE
    else fill = "#3b82f6"; // blue for dense

    // Size based on highlight
    const r = isHighlighted ? 8 : isPareto ? 6 : 5;

    // Opacity for non-pareto points
    const opacity = isPareto ? 1 : 0.6;

    return (
      <g>
        {/* Highlight ring */}
        {isHighlighted && (
          <circle
            cx={cx}
            cy={cy}
            r={r + 3}
            fill="none"
            stroke={fill}
            strokeWidth={2}
            opacity={0.5}
          />
        )}
        {/* Main dot */}
        <circle
          cx={cx}
          cy={cy}
          r={r}
          fill={fill}
          opacity={opacity}
          stroke={isHighlighted ? "#000" : "none"}
          strokeWidth={isHighlighted ? 2 : 0}
          style={{ cursor: "pointer" }}
          onMouseEnter={() => onPointHover(payload.id)}
          onMouseLeave={onPointLeave}
          onClick={() => onPointClick(payload.id)}
        />
        {/* MoE indicator (hollow center) */}
        {isMoe && (
          <circle
            cx={cx}
            cy={cy}
            r={r * 0.4}
            fill="white"
            style={{ pointerEvents: "none" }}
          />
        )}
      </g>
    );
  };

  // Custom tooltip
  const CustomTooltip = ({ active, payload }: any) => {
    if (!active || !payload || !payload[0]) return null;

    const data = payload[0].payload as ChartPoint & { isPareto: boolean };

    return (
      <div className="bg-white dark:bg-gray-800 p-3 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 max-w-xs">
        <div className="font-medium text-gray-900 dark:text-gray-100">
          {data.model_name}
        </div>
        <div className="text-sm text-gray-500 dark:text-gray-400">
          {data.quant_name} ({data.bits_per_weight.toFixed(1)} bits)
        </div>

        <div className="mt-2 space-y-1 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">VRAM:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {formatVRAM(data.vram_required)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Quality:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {data.quality_score.toFixed(1)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Speed:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {formatTPS(data.tokens_per_second)}
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-600 dark:text-gray-400">Efficiency:</span>
            <span className="font-medium text-gray-900 dark:text-gray-100">
              {data.efficiency_score.toFixed(2)}
            </span>
          </div>
        </div>

        <div className="mt-2 flex flex-wrap gap-1">
          {data.is_moe && (
            <span className="px-1.5 py-0.5 bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 text-xs rounded">
              MoE
            </span>
          )}
          {data.is_tool_calling && (
            <span className="px-1.5 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 text-xs rounded">
              Tool-Calling
            </span>
          )}
          {data.isPareto && (
            <span className="px-1.5 py-0.5 bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300 text-xs rounded">
              Pareto Optimal
            </span>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700">
      <h3 className="text-sm font-medium text-gray-300 mb-2">
        {config.title}
      </h3>

      <ResponsiveContainer width="100%" height={300}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 20, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" opacity={0.3} />
          <XAxis
            type="number"
            dataKey="x"
            name="VRAM"
            domain={[0, maxVram]}
            tickFormatter={(v) => `${v}GB`}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            label={{
              value: "VRAM (GB)",
              position: "bottom",
              offset: 0,
              style: { fontSize: 12, fill: "#9ca3af" },
            }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name={config.yLabel}
            tickFormatter={(v) => config.formatY(v)}
            tick={{ fill: "#9ca3af", fontSize: 11 }}
            label={{
              value: config.yLabel,
              angle: -90,
              position: "left",
              offset: 10,
              style: { fontSize: 12, fill: "#9ca3af" },
            }}
          />
          <Tooltip content={<CustomTooltip />} />

          {/* VRAM limit reference line */}
          <ReferenceLine
            x={maxVram}
            stroke="#ef4444"
            strokeDasharray="5 5"
            label={{
              value: "Max VRAM",
              position: "top",
              fill: "#ef4444",
              fontSize: 10,
            }}
          />

          {/* Frontier line */}
          {lineData.length > 1 && (
            <Line
              type="monotone"
              data={lineData}
              dataKey="y"
              stroke={config.color}
              strokeWidth={2}
              dot={false}
              isAnimationActive={false}
            />
          )}

          {/* Scatter points */}
          <Scatter
            data={scatterData}
            shape={renderDot}
            isAnimationActive={false}
          />
        </ScatterChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="mt-2 flex flex-wrap gap-3 text-xs text-gray-400">
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 bg-blue-500 rounded-full" />
          Dense
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 bg-purple-500 rounded-full border-2 border-white" />
          MoE
        </div>
        <div className="flex items-center gap-1">
          <span className="w-2.5 h-2.5 bg-green-500 rounded-full" />
          Tool-Calling
        </div>
        <div className="flex items-center gap-1">
          <span
            className="w-4 h-0.5"
            style={{ backgroundColor: config.color }}
          />
          Pareto Frontier
        </div>
      </div>
    </div>
  );
}
