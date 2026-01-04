import React, { useState, useMemo } from 'react';

// GPU Database (subset from TechPowerUp/dbgpu)
const GPU_DATABASE = [
  { name: "RTX 4090", vram: 24, bandwidth: 1008 },
  { name: "RTX 4080", vram: 16, bandwidth: 717 },
  { name: "RTX 4070 Ti", vram: 12, bandwidth: 504 },
  { name: "RTX 4060 Ti 16GB", vram: 16, bandwidth: 288 },
  { name: "RTX 4060", vram: 8, bandwidth: 272 },
  { name: "RTX 3090", vram: 24, bandwidth: 936 },
  { name: "RTX 3080 10GB", vram: 10, bandwidth: 760 },
  { name: "RTX 3070", vram: 8, bandwidth: 448 },
  { name: "A100 40GB", vram: 40, bandwidth: 1555 },
  { name: "A100 80GB", vram: 80, bandwidth: 2039 },
  { name: "H100 80GB", vram: 80, bandwidth: 3350 },
  { name: "L4", vram: 24, bandwidth: 300 },
  { name: "T4", vram: 16, bandwidth: 300 },
  { name: "A10", vram: 24, bandwidth: 600 },
  { name: "Arc B580", vram: 12, bandwidth: 456 },
  { name: "RTX 5090", vram: 32, bandwidth: 1792 },
];

// Unsloth Model Database
// For MoE: total_params = what loads into VRAM, active_params = what fires per token
const MODEL_DATABASE = [
  // Dense models
  { name: "Llama-3.2-1B", family: "llama", total_params: 1, active_params: 1, arch: "dense", hidden_dim: 2048, layers: 16, kv_heads: 8, base_quality: 45 },
  { name: "Llama-3.2-3B", family: "llama", total_params: 3, active_params: 3, arch: "dense", hidden_dim: 3072, layers: 28, kv_heads: 8, base_quality: 58 },
  { name: "Llama-3.1-8B", family: "llama", total_params: 8, active_params: 8, arch: "dense", hidden_dim: 4096, layers: 32, kv_heads: 8, base_quality: 69 },
  { name: "Llama-3.3-70B", family: "llama", total_params: 70, active_params: 70, arch: "dense", hidden_dim: 8192, layers: 80, kv_heads: 8, base_quality: 86 },
  { name: "Gemma-3-4B", family: "gemma", total_params: 4, active_params: 4, arch: "dense", hidden_dim: 2560, layers: 34, kv_heads: 4, base_quality: 62 },
  { name: "Gemma-3-12B", family: "gemma", total_params: 12, active_params: 12, arch: "dense", hidden_dim: 3840, layers: 48, kv_heads: 4, base_quality: 74 },
  { name: "Gemma-3-27B", family: "gemma", total_params: 27, active_params: 27, arch: "dense", hidden_dim: 4608, layers: 62, kv_heads: 8, base_quality: 81 },
  { name: "Qwen3-8B", family: "qwen", total_params: 8, active_params: 8, arch: "dense", hidden_dim: 4096, layers: 36, kv_heads: 8, base_quality: 71 },
  { name: "Qwen3-14B", family: "qwen", total_params: 14, active_params: 14, arch: "dense", hidden_dim: 5120, layers: 40, kv_heads: 8, base_quality: 76 },
  { name: "Qwen3-32B", family: "qwen", total_params: 32, active_params: 32, arch: "dense", hidden_dim: 5120, layers: 64, kv_heads: 8, base_quality: 83 },
  { name: "Phi-4-14B", family: "phi", total_params: 14, active_params: 14, arch: "dense", hidden_dim: 5120, layers: 40, kv_heads: 8, base_quality: 77 },
  { name: "Mistral-7B-v0.3", family: "mistral", total_params: 7, active_params: 7, arch: "dense", hidden_dim: 4096, layers: 32, kv_heads: 8, base_quality: 67 },
  
  // MoE models - total_params is what loads, active_params is per-token
  { name: "Qwen3-30B-A3B", family: "qwen", total_params: 30, active_params: 3, arch: "moe", experts: 128, active_experts: 8, hidden_dim: 2048, layers: 48, kv_heads: 4, base_quality: 70 },
  { name: "Mixtral-8x7B", family: "mistral", total_params: 46.7, active_params: 12.9, arch: "moe", experts: 8, active_experts: 2, hidden_dim: 4096, layers: 32, kv_heads: 8, base_quality: 75 },
  { name: "Mixtral-8x22B", family: "mistral", total_params: 141, active_params: 39, arch: "moe", experts: 8, active_experts: 2, hidden_dim: 6144, layers: 56, kv_heads: 8, base_quality: 84 },
  { name: "DeepSeek-V3", family: "deepseek", total_params: 671, active_params: 37, arch: "moe", experts: 256, active_experts: 8, hidden_dim: 7168, layers: 61, kv_heads: 8, base_quality: 90 },
  { name: "gpt-oss-20B", family: "openai", total_params: 21, active_params: 3.6, arch: "moe", experts: 16, active_experts: 4, hidden_dim: 3072, layers: 32, kv_heads: 8, base_quality: 72 },
  { name: "gpt-oss-120B", family: "openai", total_params: 117, active_params: 5.1, arch: "moe", experts: 16, active_experts: 4, hidden_dim: 4096, layers: 48, kv_heads: 8, base_quality: 82 },
];

// Quantization configurations
const QUANT_CONFIGS = [
  { name: "FP16", bits: 16, bytes: 2, quality_mult: 1.0, color: "#10b981" },
  { name: "Q8", bits: 8, bytes: 1, quality_mult: 0.98, color: "#3b82f6" },
  { name: "Q6_K", bits: 6, bytes: 0.78, quality_mult: 0.95, color: "#8b5cf6" },
  { name: "Q5_K_M", bits: 5, bytes: 0.68, quality_mult: 0.92, color: "#a855f7" },
  { name: "Q4_K_M", bits: 4, bytes: 0.57, quality_mult: 0.88, color: "#f59e0b" },
  { name: "Q3_K_M", bits: 3, bytes: 0.43, quality_mult: 0.80, color: "#ef4444" },
  { name: "Q2_K", bits: 2, bytes: 0.31, quality_mult: 0.65, color: "#dc2626" },
];

const CONTEXT_SNAPS = [2048, 4096, 8192, 16384, 32768, 65536, 131072];

// Calculate model VRAM for given quantization
function calcModelVram(model, quant) {
  return model.total_params * quant.bytes * 1.1; // 10% overhead
}

// Calculate KV cache VRAM for given context length
function calcKvCache(model, contextLen, kvQuantBits = 16) {
  const headDim = model.hidden_dim / (model.kv_heads * 4); // approximate
  const bytesPerKv = kvQuantBits / 8;
  const cacheSize = 2 * model.layers * model.kv_heads * headDim * contextLen * bytesPerKv;
  return cacheSize / 1e9;
}

// Calculate effective quality score
// MoE models get a bonus but don't match equivalent dense params
function calcQuality(model, quant) {
  let score = model.base_quality * quant.quality_mult;
  return Math.round(score);
}

// Find Pareto frontier
function paretoFrontier(points) {
  const dominated = new Set();
  
  for (let i = 0; i < points.length; i++) {
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      // Point j dominates i if j has higher quality AND lower vram
      if (points[j].quality >= points[i].quality && points[j].totalVram <= points[i].totalVram) {
        if (points[j].quality > points[i].quality || points[j].totalVram < points[i].totalVram) {
          dominated.add(i);
        }
      }
    }
  }
  
  return points.map((p, i) => ({ ...p, isPareto: !dominated.has(i) }));
}

export default function ModelSelector() {
  const [selectedGpu, setSelectedGpu] = useState(GPU_DATABASE[0]);
  const [contextLen, setContextLen] = useState(8192);
  const [showAll, setShowAll] = useState(false);
  const [hoveredPoint, setHoveredPoint] = useState(null);
  
  const systemOverhead = 0.5; // GB for CUDA, etc
  
  // Calculate all feasible model+quant combinations
  const modelConfigs = useMemo(() => {
    const configs = [];
    
    for (const model of MODEL_DATABASE) {
      for (const quant of QUANT_CONFIGS) {
        const modelVram = calcModelVram(model, quant);
        const kvVram = calcKvCache(model, contextLen);
        const totalVram = modelVram + kvVram + systemOverhead;
        const quality = calcQuality(model, quant);
        const fits = totalVram <= selectedGpu.vram;
        
        // Calculate headroom
        const headroom = selectedGpu.vram - totalVram;
        
        configs.push({
          model,
          quant,
          modelVram,
          kvVram,
          totalVram,
          quality,
          fits,
          headroom,
          id: `${model.name}-${quant.name}`,
        });
      }
    }
    
    // Add Pareto info
    const feasible = configs.filter(c => c.fits);
    const withPareto = paretoFrontier(feasible);
    
    return configs.map(c => {
      const paretoMatch = withPareto.find(p => p.id === c.id);
      return { ...c, isPareto: paretoMatch?.isPareto || false };
    });
  }, [selectedGpu, contextLen]);
  
  const feasibleConfigs = modelConfigs.filter(c => c.fits);
  const paretoConfigs = feasibleConfigs.filter(c => c.isPareto);
  
  // Chart dimensions
  const chartWidth = 700;
  const chartHeight = 400;
  const padding = { top: 40, right: 40, bottom: 60, left: 60 };
  const innerWidth = chartWidth - padding.left - padding.right;
  const innerHeight = chartHeight - padding.top - padding.bottom;
  
  // Scales
  const maxVram = selectedGpu.vram;
  const maxQuality = 100;
  
  const xScale = (vram) => padding.left + (vram / maxVram) * innerWidth;
  const yScale = (quality) => padding.top + innerHeight - (quality / maxQuality) * innerHeight;
  
  // Format context length for display
  const formatContext = (len) => len >= 1024 ? `${len/1024}K` : len;
  
  // Get closest context snap
  const contextSnapIndex = CONTEXT_SNAPS.findIndex(s => s >= contextLen) || 0;
  
  return (
    <div style={{
      minHeight: '100vh',
      background: 'linear-gradient(135deg, #0a0a0f 0%, #1a1a2e 50%, #0f0f1a 100%)',
      color: '#e2e8f0',
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: '2rem',
    }}>
      {/* Header */}
      <div style={{ marginBottom: '2rem' }}>
        <h1 style={{
          fontSize: '1.5rem',
          fontWeight: 600,
          color: '#f8fafc',
          marginBottom: '0.5rem',
          display: 'flex',
          alignItems: 'center',
          gap: '0.75rem',
        }}>
          <span style={{ 
            background: 'linear-gradient(135deg, #f59e0b 0%, #ef4444 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>◆</span>
          Model ↔ Quant Selector
          <span style={{
            fontSize: '0.7rem',
            padding: '0.25rem 0.5rem',
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            borderRadius: '4px',
            color: '#f59e0b',
          }}>UNSLOTH</span>
        </h1>
        <p style={{ fontSize: '0.85rem', color: '#64748b' }}>
          Find the optimal model size × quantization for your GPU and context needs
        </p>
      </div>
      
      {/* Controls */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
        gap: '1.5rem',
        marginBottom: '2rem',
      }}>
        {/* GPU Selection */}
        <div style={{
          background: 'rgba(30, 30, 45, 0.6)',
          border: '1px solid rgba(100, 100, 140, 0.2)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <label style={{ 
            fontSize: '0.7rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.1em',
            color: '#64748b',
            display: 'block',
            marginBottom: '0.75rem',
          }}>
            Target GPU
          </label>
          <select
            value={selectedGpu.name}
            onChange={(e) => setSelectedGpu(GPU_DATABASE.find(g => g.name === e.target.value))}
            style={{
              width: '100%',
              padding: '0.75rem 1rem',
              background: 'rgba(15, 15, 25, 0.8)',
              border: '1px solid rgba(100, 100, 140, 0.3)',
              borderRadius: '8px',
              color: '#f8fafc',
              fontSize: '0.95rem',
              fontFamily: 'inherit',
              cursor: 'pointer',
            }}
          >
            {GPU_DATABASE.map(gpu => (
              <option key={gpu.name} value={gpu.name}>
                {gpu.name} — {gpu.vram}GB
              </option>
            ))}
          </select>
          <div style={{
            display: 'flex',
            gap: '1.5rem',
            marginTop: '1rem',
            fontSize: '0.8rem',
          }}>
            <div>
              <span style={{ color: '#64748b' }}>VRAM: </span>
              <span style={{ color: '#10b981', fontWeight: 600 }}>{selectedGpu.vram} GB</span>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>BW: </span>
              <span style={{ color: '#3b82f6' }}>{selectedGpu.bandwidth} GB/s</span>
            </div>
          </div>
        </div>
        
        {/* Context Length */}
        <div style={{
          background: 'rgba(30, 30, 45, 0.6)',
          border: '1px solid rgba(100, 100, 140, 0.2)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <label style={{ 
            fontSize: '0.7rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.1em',
            color: '#64748b',
            display: 'block',
            marginBottom: '0.75rem',
          }}>
            Target Context Length
          </label>
          <input
            type="range"
            min={0}
            max={CONTEXT_SNAPS.length - 1}
            value={CONTEXT_SNAPS.indexOf(contextLen)}
            onChange={(e) => setContextLen(CONTEXT_SNAPS[parseInt(e.target.value)])}
            style={{
              width: '100%',
              height: '8px',
              borderRadius: '4px',
              background: `linear-gradient(to right, #f59e0b 0%, #f59e0b ${(CONTEXT_SNAPS.indexOf(contextLen) / (CONTEXT_SNAPS.length - 1)) * 100}%, rgba(100,100,140,0.3) ${(CONTEXT_SNAPS.indexOf(contextLen) / (CONTEXT_SNAPS.length - 1)) * 100}%, rgba(100,100,140,0.3) 100%)`,
              cursor: 'pointer',
              appearance: 'none',
            }}
          />
          <div style={{
            display: 'flex',
            justifyContent: 'space-between',
            marginTop: '0.5rem',
            fontSize: '0.65rem',
            color: '#475569',
          }}>
            {CONTEXT_SNAPS.map((snap, i) => (
              <span key={snap} style={{ 
                color: snap === contextLen ? '#f59e0b' : '#475569',
                fontWeight: snap === contextLen ? 600 : 400,
              }}>
                {formatContext(snap)}
              </span>
            ))}
          </div>
          <div style={{
            marginTop: '1rem',
            fontSize: '0.8rem',
          }}>
            <span style={{ color: '#64748b' }}>Selected: </span>
            <span style={{ color: '#f59e0b', fontWeight: 600 }}>{formatContext(contextLen)} tokens</span>
          </div>
        </div>
        
        {/* Summary Stats */}
        <div style={{
          background: 'rgba(30, 30, 45, 0.6)',
          border: '1px solid rgba(100, 100, 140, 0.2)',
          borderRadius: '12px',
          padding: '1.25rem',
        }}>
          <label style={{ 
            fontSize: '0.7rem', 
            textTransform: 'uppercase', 
            letterSpacing: '0.1em',
            color: '#64748b',
            display: 'block',
            marginBottom: '0.75rem',
          }}>
            Feasibility Summary
          </label>
          <div style={{ display: 'flex', gap: '2rem' }}>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 700, color: '#10b981' }}>
                {feasibleConfigs.length}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>configs fit</div>
            </div>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 700, color: '#f59e0b' }}>
                {paretoConfigs.length}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>on Pareto frontier</div>
            </div>
            <div>
              <div style={{ fontSize: '2rem', fontWeight: 700, color: '#8b5cf6' }}>
                {modelConfigs.filter(c => !c.fits).length}
              </div>
              <div style={{ fontSize: '0.7rem', color: '#64748b' }}>don't fit</div>
            </div>
          </div>
        </div>
      </div>
      
      {/* Main Chart */}
      <div style={{
        background: 'rgba(20, 20, 35, 0.8)',
        border: '1px solid rgba(100, 100, 140, 0.2)',
        borderRadius: '12px',
        padding: '1.5rem',
        marginBottom: '2rem',
      }}>
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: '1rem',
        }}>
          <h2 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc' }}>
            Quality vs VRAM — Pareto Frontier
          </h2>
          <label style={{ 
            display: 'flex', 
            alignItems: 'center', 
            gap: '0.5rem',
            fontSize: '0.8rem',
            color: '#64748b',
            cursor: 'pointer',
          }}>
            <input
              type="checkbox"
              checked={showAll}
              onChange={(e) => setShowAll(e.target.checked)}
              style={{ accentColor: '#f59e0b' }}
            />
            Show all configs (including dominated)
          </label>
        </div>
        
        <svg width={chartWidth} height={chartHeight} style={{ display: 'block', margin: '0 auto' }}>
          {/* Grid lines */}
          {[0, 25, 50, 75, 100].map(q => (
            <g key={`grid-y-${q}`}>
              <line
                x1={padding.left}
                y1={yScale(q)}
                x2={chartWidth - padding.right}
                y2={yScale(q)}
                stroke="rgba(100, 100, 140, 0.15)"
                strokeDasharray="4,4"
              />
              <text
                x={padding.left - 10}
                y={yScale(q)}
                textAnchor="end"
                dominantBaseline="middle"
                fill="#64748b"
                fontSize="11"
                fontFamily="inherit"
              >
                {q}
              </text>
            </g>
          ))}
          
          {[0, selectedGpu.vram * 0.25, selectedGpu.vram * 0.5, selectedGpu.vram * 0.75, selectedGpu.vram].map(v => (
            <g key={`grid-x-${v}`}>
              <line
                x1={xScale(v)}
                y1={padding.top}
                x2={xScale(v)}
                y2={chartHeight - padding.bottom}
                stroke="rgba(100, 100, 140, 0.15)"
                strokeDasharray="4,4"
              />
              <text
                x={xScale(v)}
                y={chartHeight - padding.bottom + 20}
                textAnchor="middle"
                fill="#64748b"
                fontSize="11"
                fontFamily="inherit"
              >
                {v.toFixed(0)}GB
              </text>
            </g>
          ))}
          
          {/* VRAM limit line */}
          <line
            x1={xScale(selectedGpu.vram)}
            y1={padding.top}
            x2={xScale(selectedGpu.vram)}
            y2={chartHeight - padding.bottom}
            stroke="#ef4444"
            strokeWidth="2"
            strokeDasharray="6,3"
          />
          <text
            x={xScale(selectedGpu.vram) - 5}
            y={padding.top + 15}
            textAnchor="end"
            fill="#ef4444"
            fontSize="10"
            fontFamily="inherit"
          >
            VRAM LIMIT
          </text>
          
          {/* Pareto frontier line */}
          {paretoConfigs.length > 1 && (
            <path
              d={paretoConfigs
                .sort((a, b) => a.totalVram - b.totalVram)
                .map((c, i) => `${i === 0 ? 'M' : 'L'} ${xScale(c.totalVram)} ${yScale(c.quality)}`)
                .join(' ')}
              fill="none"
              stroke="rgba(245, 158, 11, 0.4)"
              strokeWidth="2"
            />
          )}
          
          {/* Data points */}
          {(showAll ? modelConfigs : feasibleConfigs).map((config) => {
            const x = xScale(config.totalVram);
            const y = yScale(config.quality);
            const isHovered = hoveredPoint?.id === config.id;
            const opacity = config.fits ? (config.isPareto ? 1 : 0.4) : 0.15;
            
            return (
              <g key={config.id}>
                <circle
                  cx={x}
                  cy={y}
                  r={isHovered ? 10 : (config.isPareto ? 7 : 5)}
                  fill={config.quant.color}
                  opacity={opacity}
                  stroke={config.isPareto ? '#fff' : 'none'}
                  strokeWidth={config.isPareto ? 2 : 0}
                  style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
                  onMouseEnter={() => setHoveredPoint(config)}
                  onMouseLeave={() => setHoveredPoint(null)}
                />
                {config.model.arch === 'moe' && (
                  <circle
                    cx={x}
                    cy={y}
                    r={isHovered ? 14 : (config.isPareto ? 11 : 8)}
                    fill="none"
                    stroke={config.quant.color}
                    strokeWidth="1"
                    strokeDasharray="2,2"
                    opacity={opacity * 0.6}
                    style={{ pointerEvents: 'none' }}
                  />
                )}
              </g>
            );
          })}
          
          {/* Axis labels */}
          <text
            x={chartWidth / 2}
            y={chartHeight - 10}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            fontFamily="inherit"
          >
            Total VRAM Usage (Model + KV Cache)
          </text>
          <text
            x={15}
            y={chartHeight / 2}
            textAnchor="middle"
            fill="#94a3b8"
            fontSize="12"
            fontFamily="inherit"
            transform={`rotate(-90, 15, ${chartHeight / 2})`}
          >
            Quality Score
          </text>
        </svg>
        
        {/* Legend */}
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '1.5rem',
          marginTop: '1rem',
          flexWrap: 'wrap',
        }}>
          {QUANT_CONFIGS.map(q => (
            <div key={q.name} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <div style={{
                width: 12,
                height: 12,
                borderRadius: '50%',
                background: q.color,
              }} />
              <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>{q.name}</span>
            </div>
          ))}
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <div style={{
              width: 12,
              height: 12,
              borderRadius: '50%',
              border: '2px dashed #94a3b8',
            }} />
            <span style={{ fontSize: '0.75rem', color: '#94a3b8' }}>MoE Model</span>
          </div>
        </div>
      </div>
      
      {/* Hover tooltip */}
      {hoveredPoint && (
        <div style={{
          position: 'fixed',
          bottom: '2rem',
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(15, 15, 25, 0.95)',
          border: '1px solid rgba(100, 100, 140, 0.4)',
          borderRadius: '12px',
          padding: '1rem 1.5rem',
          boxShadow: '0 20px 40px rgba(0,0,0,0.5)',
          zIndex: 100,
          minWidth: '400px',
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'flex-start',
            marginBottom: '0.75rem',
          }}>
            <div>
              <div style={{ fontSize: '1.1rem', fontWeight: 600, color: '#f8fafc' }}>
                {hoveredPoint.model.name}
                <span style={{ 
                  marginLeft: '0.5rem',
                  padding: '0.15rem 0.4rem',
                  background: hoveredPoint.quant.color,
                  borderRadius: '4px',
                  fontSize: '0.7rem',
                  color: '#000',
                  fontWeight: 700,
                }}>
                  {hoveredPoint.quant.name}
                </span>
              </div>
              <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '0.25rem' }}>
                {hoveredPoint.model.arch === 'moe' 
                  ? `MoE: ${hoveredPoint.model.total_params}B total / ${hoveredPoint.model.active_params}B active`
                  : `Dense: ${hoveredPoint.model.total_params}B params`
                }
              </div>
            </div>
            <div style={{
              padding: '0.25rem 0.5rem',
              background: hoveredPoint.isPareto ? 'rgba(245, 158, 11, 0.2)' : 'rgba(100, 100, 140, 0.2)',
              border: `1px solid ${hoveredPoint.isPareto ? 'rgba(245, 158, 11, 0.4)' : 'rgba(100, 100, 140, 0.3)'}`,
              borderRadius: '4px',
              fontSize: '0.65rem',
              color: hoveredPoint.isPareto ? '#f59e0b' : '#64748b',
              textTransform: 'uppercase',
              letterSpacing: '0.05em',
            }}>
              {hoveredPoint.isPareto ? '★ Pareto Optimal' : 'Dominated'}
            </div>
          </div>
          
          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: '1rem',
            fontSize: '0.8rem',
          }}>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.65rem', marginBottom: '0.2rem' }}>MODEL</div>
              <div style={{ color: '#10b981', fontWeight: 600 }}>{hoveredPoint.modelVram.toFixed(1)} GB</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.65rem', marginBottom: '0.2rem' }}>KV CACHE</div>
              <div style={{ color: '#3b82f6', fontWeight: 600 }}>{hoveredPoint.kvVram.toFixed(1)} GB</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.65rem', marginBottom: '0.2rem' }}>TOTAL</div>
              <div style={{ color: '#f8fafc', fontWeight: 600 }}>{hoveredPoint.totalVram.toFixed(1)} GB</div>
            </div>
            <div>
              <div style={{ color: '#64748b', fontSize: '0.65rem', marginBottom: '0.2rem' }}>HEADROOM</div>
              <div style={{ 
                color: hoveredPoint.headroom > 2 ? '#10b981' : hoveredPoint.headroom > 0 ? '#f59e0b' : '#ef4444', 
                fontWeight: 600 
              }}>
                {hoveredPoint.headroom > 0 ? '+' : ''}{hoveredPoint.headroom.toFixed(1)} GB
              </div>
            </div>
          </div>
          
          {/* VRAM breakdown bar */}
          <div style={{ marginTop: '0.75rem' }}>
            <div style={{
              height: '8px',
              background: 'rgba(100, 100, 140, 0.2)',
              borderRadius: '4px',
              overflow: 'hidden',
              display: 'flex',
            }}>
              <div style={{
                width: `${(hoveredPoint.modelVram / selectedGpu.vram) * 100}%`,
                background: '#10b981',
              }} />
              <div style={{
                width: `${(hoveredPoint.kvVram / selectedGpu.vram) * 100}%`,
                background: '#3b82f6',
              }} />
              <div style={{
                width: `${(systemOverhead / selectedGpu.vram) * 100}%`,
                background: '#64748b',
              }} />
            </div>
            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              fontSize: '0.6rem',
              color: '#475569',
              marginTop: '0.25rem',
            }}>
              <span>0 GB</span>
              <span>{selectedGpu.vram} GB</span>
            </div>
          </div>
        </div>
      )}
      
      {/* Recommendations Table */}
      <div style={{
        background: 'rgba(20, 20, 35, 0.8)',
        border: '1px solid rgba(100, 100, 140, 0.2)',
        borderRadius: '12px',
        padding: '1.5rem',
      }}>
        <h2 style={{ fontSize: '1rem', fontWeight: 600, color: '#f8fafc', marginBottom: '1rem' }}>
          Pareto-Optimal Configurations
        </h2>
        
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.85rem' }}>
            <thead>
              <tr style={{ borderBottom: '1px solid rgba(100, 100, 140, 0.3)' }}>
                <th style={{ textAlign: 'left', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Model</th>
                <th style={{ textAlign: 'left', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Quant</th>
                <th style={{ textAlign: 'left', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Arch</th>
                <th style={{ textAlign: 'right', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Params</th>
                <th style={{ textAlign: 'right', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>VRAM</th>
                <th style={{ textAlign: 'right', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Quality</th>
                <th style={{ textAlign: 'right', padding: '0.75rem', color: '#64748b', fontWeight: 500 }}>Headroom</th>
              </tr>
            </thead>
            <tbody>
              {paretoConfigs
                .sort((a, b) => b.quality - a.quality)
                .slice(0, 10)
                .map((config, i) => (
                <tr 
                  key={config.id}
                  style={{ 
                    borderBottom: '1px solid rgba(100, 100, 140, 0.15)',
                    background: i === 0 ? 'rgba(245, 158, 11, 0.05)' : 'transparent',
                  }}
                  onMouseEnter={() => setHoveredPoint(config)}
                  onMouseLeave={() => setHoveredPoint(null)}
                >
                  <td style={{ padding: '0.75rem', fontWeight: 500 }}>
                    {i === 0 && <span style={{ color: '#f59e0b', marginRight: '0.5rem' }}>★</span>}
                    {config.model.name}
                  </td>
                  <td style={{ padding: '0.75rem' }}>
                    <span style={{
                      padding: '0.15rem 0.4rem',
                      background: config.quant.color,
                      borderRadius: '4px',
                      fontSize: '0.7rem',
                      color: '#000',
                      fontWeight: 600,
                    }}>
                      {config.quant.name}
                    </span>
                  </td>
                  <td style={{ padding: '0.75rem', color: '#94a3b8' }}>
                    {config.model.arch === 'moe' ? 'MoE' : 'Dense'}
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: '#94a3b8' }}>
                    {config.model.arch === 'moe' 
                      ? `${config.model.active_params}B/${config.model.total_params}B`
                      : `${config.model.total_params}B`
                    }
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right', color: '#f8fafc', fontWeight: 500 }}>
                    {config.totalVram.toFixed(1)} GB
                  </td>
                  <td style={{ padding: '0.75rem', textAlign: 'right' }}>
                    <span style={{
                      color: config.quality >= 80 ? '#10b981' : config.quality >= 65 ? '#f59e0b' : '#94a3b8',
                      fontWeight: 600,
                    }}>
                      {config.quality}
                    </span>
                  </td>
                  <td style={{ 
                    padding: '0.75rem', 
                    textAlign: 'right',
                    color: config.headroom > 2 ? '#10b981' : '#f59e0b',
                  }}>
                    +{config.headroom.toFixed(1)} GB
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        
        {paretoConfigs.length === 0 && (
          <div style={{
            textAlign: 'center',
            padding: '3rem',
            color: '#64748b',
          }}>
            No configurations fit in {selectedGpu.vram}GB with {formatContext(contextLen)} context.
            <br />
            Try reducing context length or selecting a GPU with more VRAM.
          </div>
        )}
      </div>
      
      {/* Footer note */}
      <div style={{
        marginTop: '1.5rem',
        padding: '1rem',
        background: 'rgba(30, 30, 45, 0.4)',
        borderRadius: '8px',
        fontSize: '0.75rem',
        color: '#64748b',
      }}>
        <strong style={{ color: '#94a3b8' }}>Note:</strong> MoE models (dashed circles) load all parameters into VRAM 
        but only activate a subset per token. Quality scores are approximate and based on benchmark aggregates. 
        Actual VRAM usage may vary by ~10-15% depending on framework and settings.
      </div>
    </div>
  );
}
