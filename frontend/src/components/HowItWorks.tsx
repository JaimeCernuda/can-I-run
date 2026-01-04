/**
 * How It Works Component
 *
 * Collapsible panel explaining all calculations and methodology.
 * Provides transparency for users to understand the recommendations.
 */

import { useState } from "react";

export function HowItWorks() {
  const [isOpen, setIsOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<string | null>(null);

  const sections = [
    {
      id: "vram",
      title: "VRAM Budget Breakdown",
      content: (
        <div className="space-y-3">
          <p>Total VRAM is divided into three components:</p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            Available = Total VRAM - KV Cache - Overhead (0.5GB)
          </div>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>
              <strong>Model Weights:</strong> Parameters × Bits per Weight ÷ 8
            </li>
            <li>
              <strong>KV Cache:</strong> Grows with context length (see below)
            </li>
            <li>
              <strong>Overhead:</strong> ~0.5GB for CUDA/driver allocations
            </li>
          </ul>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            For MoE models, ALL experts must fit in VRAM, even though only a
            subset are used per token.
          </p>
        </div>
      ),
    },
    {
      id: "kvcache",
      title: "KV Cache Calculation",
      content: (
        <div className="space-y-3">
          <p>The KV cache stores attention keys and values for all tokens:</p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm overflow-x-auto">
            KV Cache = 2 × Layers × KV_Heads × Head_Dim × Context × 2 bytes
          </div>
          <p className="text-sm">
            Example: Llama-3.1-70B at 8K context:
          </p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            = 2 × 80 × 8 × 128 × 8192 × 2 = 2.68 GB
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Modern models use Grouped Query Attention (GQA) where KV_heads &lt;
            attention_heads, significantly reducing cache size.
          </p>
        </div>
      ),
    },
    {
      id: "quality",
      title: "Quality Score",
      content: (
        <div className="space-y-3">
          <p>Quality combines benchmark scores with quantization effects:</p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            Quality = Base Score × Quant Factor × Size Adjustment
          </div>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>
              <strong>Base Score:</strong> Weighted average of MMLU, HumanEval,
              GSM8K (weights vary by domain)
            </li>
            <li>
              <strong>Quant Factor:</strong> Quality retention from perplexity
              measurements (Q4_K_M ≈ 0.99)
            </li>
            <li>
              <strong>Size Adjustment:</strong> Larger models (70B+) tolerate
              quantization better than small models
            </li>
          </ul>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Sources: llama.cpp perplexity benchmarks, Intel Low-bit Leaderboard
          </p>
        </div>
      ),
    },
    {
      id: "performance",
      title: "Token Speed Estimation",
      content: (
        <div className="space-y-3">
          <p>
            Token generation is primarily memory-bandwidth bound during decode:
          </p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            Tokens/sec ≈ Bandwidth (GB/s) ÷ Model Size (GB) × Efficiency
          </div>
          <p className="text-sm">
            Example: RTX 4090 (1008 GB/s) + Llama-70B Q4 (31.6 GB):
          </p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            = 1008 ÷ 31.6 × 0.7 ≈ 22 tok/s
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            When available, we use measured benchmarks from XiongjieDai's
            GPU-Benchmarks repository instead of theoretical estimates.
          </p>
        </div>
      ),
    },
    {
      id: "efficiency",
      title: "Efficiency Metric",
      content: (
        <div className="space-y-3">
          <p>
            Efficiency balances quality and speed relative to VRAM cost:
          </p>
          <div className="bg-gray-100 dark:bg-gray-700 p-3 rounded font-mono text-sm">
            Efficiency = √(Quality × Performance) ÷ VRAM × 100
          </div>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>
              Uses geometric mean so both quality AND speed must be good
            </li>
            <li>
              An 8B Q4 can beat 70B Q3 in efficiency despite lower raw quality
            </li>
            <li>Helps find the "sweet spot" for limited VRAM</li>
          </ul>
        </div>
      ),
    },
    {
      id: "pareto",
      title: "Pareto Frontier",
      content: (
        <div className="space-y-3">
          <p>A model is Pareto-optimal if no other model dominates it:</p>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>Higher metric at same or lower VRAM, OR</li>
            <li>Same metric at lower VRAM</li>
          </ul>
          <p className="text-sm">
            We compute three separate frontiers:
          </p>
          <ul className="list-disc pl-5 space-y-1 text-sm">
            <li>
              <span className="text-blue-500 font-medium">Quality:</span>{" "}
              Maximize benchmark score
            </li>
            <li>
              <span className="text-green-500 font-medium">Performance:</span>{" "}
              Maximize tokens/sec
            </li>
            <li>
              <span className="text-amber-500 font-medium">Efficiency:</span>{" "}
              Maximize quality×speed/VRAM
            </li>
          </ul>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Models on multiple frontiers are particularly well-balanced choices.
          </p>
        </div>
      ),
    },
    {
      id: "sources",
      title: "Data Sources",
      content: (
        <div className="space-y-3">
          <ul className="list-disc pl-5 space-y-2 text-sm">
            <li>
              <strong>GPU Specs:</strong> TechPowerUp GPU Database, gpu-info-api
            </li>
            <li>
              <strong>Quantization Quality:</strong>{" "}
              <a
                href="https://github.com/ggerganov/llama.cpp"
                className="text-blue-500 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                llama.cpp quantize README
              </a>
            </li>
            <li>
              <strong>GPU Benchmarks:</strong>{" "}
              <a
                href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference"
                className="text-blue-500 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                XiongjieDai/GPU-Benchmarks-on-LLM-Inference
              </a>
            </li>
            <li>
              <strong>Model Benchmarks:</strong> Open LLM Leaderboard, model
              cards on HuggingFace
            </li>
            <li>
              <strong>Tool-Calling:</strong>{" "}
              <a
                href="https://gorilla.cs.berkeley.edu/leaderboard.html"
                className="text-blue-500 hover:underline"
                target="_blank"
                rel="noopener noreferrer"
              >
                Berkeley Function Calling Leaderboard
              </a>
            </li>
          </ul>
        </div>
      ),
    },
  ];

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-3 flex items-center justify-between text-left"
      >
        <span className="font-medium text-gray-900 dark:text-gray-100 flex items-center gap-2">
          <span className="text-lg">?</span>
          How It Works
        </span>
        <svg
          className={`w-5 h-5 text-gray-500 transition-transform ${
            isOpen ? "rotate-180" : ""
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
      </button>

      {isOpen && (
        <div className="border-t border-gray-200 dark:border-gray-700">
          <div className="p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              This tool calculates optimal LLM model/quantization combinations
              based on your GPU's VRAM and bandwidth. Click each section below
              to learn about the methodology.
            </p>

            <div className="space-y-2">
              {sections.map((section) => (
                <div
                  key={section.id}
                  className="border border-gray-200 dark:border-gray-700 rounded-lg"
                >
                  <button
                    onClick={() =>
                      setActiveSection(
                        activeSection === section.id ? null : section.id
                      )
                    }
                    className="w-full px-4 py-2 flex items-center justify-between text-left hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg"
                  >
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                      {section.title}
                    </span>
                    <svg
                      className={`w-4 h-4 text-gray-400 transition-transform ${
                        activeSection === section.id ? "rotate-180" : ""
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
                  </button>

                  {activeSection === section.id && (
                    <div className="px-4 pb-4 text-gray-700 dark:text-gray-300">
                      {section.content}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
