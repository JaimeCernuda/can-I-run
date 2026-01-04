
import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

const Latex: React.FC<{ children: string; block?: boolean }> = ({ children, block }) => {
    const containerRef = useRef<HTMLSpanElement>(null);

    useEffect(() => {
        if (containerRef.current) {
            try {
                katex.render(children, containerRef.current, {
                    throwOnError: false, // This usually handles math errors, but not library errors
                    displayMode: block,
                    output: 'mathml',
                });
            } catch (error) {
                console.error("Katex render error:", error);
                // Fallback to text
                containerRef.current.innerText = children;
            }
        }
    }, [children, block]);

    return <span ref={containerRef} />;
};

const About: React.FC<{ onBack: () => void }> = ({ onBack }) => {
    return (
        <div className="min-h-screen bg-[#0b0f19] text-gray-200 p-8 font-sans">
            <div className="max-w-4xl mx-auto">
                <button
                    onClick={onBack}
                    className="mb-8 px-4 py-2 bg-gray-800 hover:bg-gray-700 rounded-lg text-sm font-medium transition-colors border border-gray-700 flex items-center gap-2"
                >
                    <span>←</span> Return to Calculator
                </button>

                <header className="mb-12 border-b border-gray-800 pb-8">
                    <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                        How It Works
                    </h1>
                    <p className="text-gray-400 text-lg leading-relaxed">
                        This tool models the hardware requirements for Large Language Models (LLMs) using first-principles mathematics and empirical benchmarks.
                    </p>
                </header>

                <div className="space-y-16">
                    {/* Math Section (VRAM) */}
                    <section>
                        <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-2">
                            <span className="text-blue-400">#</span> The Math of Memory (VRAM)
                        </h2>
                        <div className="prose prose-invert max-w-none text-gray-300 space-y-6">
                            <p>
                                LLM inference memory usage is strictly determined by the model's static weights and the dynamic Key-Value (KV) cache.
                            </p>

                            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                                <h3 className="font-medium text-white mb-2">1. Model Weights</h3>
                                <div className="mb-3">
                                    <Latex block>{`V_{weights} = \\frac{P \\times B}{8 \\times 10^9} \\text{ GB}`}</Latex>
                                </div>
                                <ul className="list-disc pl-5 space-y-1 text-sm text-gray-400">
                                    <li>{"$P$: Total parameter count (e.g., 8 Billion)"}</li>
                                    <li>{"$B$: Average bits per weight (e.g., 4.83 for Q4_K_M)"}</li>
                                </ul>
                            </div>

                            <div className="bg-gray-900 rounded-xl p-6 border border-gray-800">
                                <h3 className="font-medium text-white mb-2">2. KV Cache (Context Memory)</h3>
                                <div className="mb-3">
                                    <Latex block>{`V_{kv} = \\frac{2 \\times L \\times H_{kv} \\times D_{head} \\times C \\times 2_{bytes}}{10^{9}} \\text{ GB}`}</Latex>
                                </div>
                                <ul className="list-disc pl-5 space-y-1 text-sm text-gray-400">
                                    <li>{"$L$: Layers (Transformer blocks)"}</li>
                                    <li>{"$H_{kv}$: Key-Value Heads (often < Total Heads)"}</li>
                                    <li>{"$D_{head}$: Head Dimension (typically 128)"}</li>
                                    <li>{"$C$: Context Length (tokens)"}</li>
                                </ul>
                            </div>
                        </div>
                    </section>

                    {/* Efficiency Metric Section */}
                    <section>
                        <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-2">
                            <span className="text-green-400">#</span> Efficiency Metric Calculation
                        </h2>
                        <div className="bg-gray-900/50 p-8 rounded-xl border border-gray-700 backdrop-blur-sm">
                            <p className="text-gray-300 mb-6 italic">
                                The efficiency score helps users find the "best bang for VRAM buck" by combining quality, performance, and VRAM usage into a single metric.
                            </p>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                                <div>
                                    <h4 className="font-bold text-white mb-2">Formula</h4>
                                    <div className="bg-gray-950 p-4 rounded border border-gray-800 mb-4 font-mono text-sm text-green-300">
                                        Efficiency = (Quality × Performance) ÷ VRAM
                                    </div>
                                    <p className="text-sm text-gray-400">
                                        More precisely:
                                        <br />1. Normalize quality to 0-1 range
                                        <br />2. Normalize performance (tok/s)
                                        <br />3. Geometric mean of quality & performance
                                        <br />4. Divide by VRAM usage
                                    </p>
                                </div>
                                <div>
                                    <h4 className="font-bold text-white mb-2">Interpretation</h4>
                                    <ul className="text-sm text-gray-400 space-y-2">
                                        <li className="flex gap-2">
                                            <span className="text-green-500">↑</span>
                                            High efficiency = Good quality AND speed for the VRAM used.
                                        </li>
                                        <li className="flex gap-2">
                                            <span className="text-blue-500">•</span>
                                            An 8B model at Q4 might have HIGHER efficiency than a 70B at Q3, even if the 70B has better raw quality.
                                        </li>
                                        <li className="flex gap-2">
                                            <span className="text-purple-500">?</span>
                                            "Is it worth the extra VRAM to go from 8B to 13B?"
                                        </li>
                                    </ul>
                                </div>
                            </div>
                        </div>
                    </section>

                    {/* Pareto Logic */}
                    <section>
                        <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-2">
                            <span className="text-purple-400">#</span> Pareto Frontier Optimization
                        </h2>
                        <div className="bg-gradient-to-br from-gray-900 to-gray-800 p-8 rounded-xl border border-gray-700">
                            <p className="text-gray-300 mb-4">
                                Our "Smart Select" logic uses <strong>Pareto Optimization</strong> to surface the best models. A model is strictly <strong>Pareto Optimal</strong> if:
                            </p>
                            <ul className="space-y-3 mb-6">
                                <li className="flex items-start gap-3">
                                    <span className="bg-green-500/20 text-green-400 p-1 rounded">✓</span>
                                    <span>No other model offers <strong>higher quality</strong> for the same or less VRAM.</span>
                                </li>
                                <li className="flex items-start gap-3">
                                    <span className="bg-green-500/20 text-green-400 p-1 rounded">✓</span>
                                    <span>No other model requires <strong>less VRAM</strong> for the same or higher quality.</span>
                                </li>
                            </ul>
                            <p className="text-sm text-gray-400">
                                Models that do not meet these criteria are considered "dominated" and are filtered out of recommendations.
                            </p>
                        </div>
                    </section>

                    {/* Quantization Table */}
                    <section>
                        <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-2">
                            <span className="text-amber-400">#</span> Quantization Tiers
                        </h2>

                        <div className="mb-6 bg-gray-800/40 p-6 rounded-lg border border-gray-700">
                            <h3 className="text-lg font-bold text-white mb-2">Terminology Guide</h3>
                            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-gray-300">
                                <div>
                                    <span className="font-bold text-blue-300">Q4_K_M vs Q4_0:</span> "K-quants" (K_M/K_S) use a newer, smarter quantization method that preserves key weights better than the legacy "0" method. Q4_K_M is the modern standard.
                                </div>
                                <div>
                                    <span className="font-bold text-amber-300">PPL (Perplexity):</span> How "confused" the model is. Lower is better. +0.05 increase is negligible; +0.5 is noticeable.
                                </div>
                            </div>
                        </div>

                        <div className="overflow-x-auto rounded-xl border border-gray-700 bg-gray-900">
                            <table className="w-full text-left text-sm">
                                <thead className="bg-gray-950 text-gray-400 border-b border-gray-800">
                                    <tr>
                                        <th className="p-4 font-medium">Quantization</th>
                                        <th className="p-4 font-medium">Size (GiB)<br /><span className="text-xs font-normal opacity-50 text-gray-500">(for 8B)</span></th>
                                        <th className="p-4 font-medium">PPL Increase</th>
                                        <th className="p-4 font-medium">Quality Assessment</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-800 text-gray-300">
                                    <tr>
                                        <td className="p-4 font-mono text-blue-300">Q8_0</td>
                                        <td className="p-4">7.96</td>
                                        <td className="p-4 text-green-400">+0.0026</td>
                                        <td className="p-4">Near-lossless</td>
                                    </tr>
                                    <tr>
                                        <td className="p-4 font-mono text-cyan-300">Q6_K</td>
                                        <td className="p-4">6.14</td>
                                        <td className="p-4 text-green-400">+0.0217</td>
                                        <td className="p-4">Very low loss</td>
                                    </tr>
                                    <tr className="bg-blue-500/10">
                                        <td className="p-4 font-mono text-white font-bold">Q5_K_M</td>
                                        <td className="p-4">5.33</td>
                                        <td className="p-4 text-green-300">+0.0569</td>
                                        <td className="p-4 font-bold text-blue-300">Recommended <span className="text-xs font-normal text-gray-400 ml-1">(High Fidelity)</span></td>
                                    </tr>
                                    <tr>
                                        <td className="p-4 font-mono text-gray-400">Q5_K_S</td>
                                        <td className="p-4">5.21</td>
                                        <td className="p-4 text-green-300">+0.1049</td>
                                        <td className="p-4">Low loss</td>
                                    </tr>
                                    <tr className="bg-green-500/10">
                                        <td className="p-4 font-mono text-white font-bold">Q4_K_M</td>
                                        <td className="p-4">4.58</td>
                                        <td className="p-4 text-green-300">~+0.05</td>
                                        <td className="p-4 font-bold text-green-300">Balanced Choice <span className="text-xs font-normal text-gray-400 ml-1">(Daily Driver)</span></td>
                                    </tr>
                                    <tr>
                                        <td className="p-4 font-mono text-gray-500">Q4_0</td>
                                        <td className="p-4">4.34</td>
                                        <td className="p-4 text-yellow-500">+0.4685</td>
                                        <td className="p-4 text-gray-400">Legacy, higher loss</td>
                                    </tr>
                                    <tr>
                                        <td className="p-4 font-mono text-yellow-300">Q3_K_M</td>
                                        <td className="p-4">3.74</td>
                                        <td className="p-4 text-yellow-500">+0.6569</td>
                                        <td className="p-4 text-yellow-200">Substantial loss</td>
                                    </tr>
                                    <tr>
                                        <td className="p-4 font-mono text-red-400">Q2_K</td>
                                        <td className="p-4">2.96</td>
                                        <td className="p-4 text-red-500">+3.5199</td>
                                        <td className="p-4 text-red-300">Extreme loss</td>
                                    </tr>
                                </tbody>
                            </table>
                        </div>
                    </section>

                    {/* Data Sources */}
                    <section className="mb-8">
                        <h2 className="text-2xl font-semibold mb-6 text-white">Data Sources</h2>
                        <ul className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            <li className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                                <a href="https://github.com/ggerganov/llama.cpp/blob/master/examples/quantize/README.md" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-medium block mb-1">
                                    llama.cpp / Quantization README
                                </a>
                                <p className="text-sm text-gray-400">
                                    Primary source for degradation benchmarks (Perplexity).
                                </p>
                            </li>
                            <li className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                                <a href="https://raw.githubusercontent.com/voidful/gpu-info-api/gpu-data/gpu.json" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-medium block mb-1">
                                    gpu-info-api / GPU Spec Database
                                </a>
                                <p className="text-sm text-gray-400">
                                    Comprehensive database of GPU specs (VRAM, Bandwidth, Cores).
                                </p>
                            </li>
                            <li className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                                <a href="https://raw.githubusercontent.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference/main/README.md" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-medium block mb-1">
                                    GPU-Benchmarks-on-LLM-Inference
                                </a>
                                <p className="text-sm text-gray-400">
                                    Real-world inference speed tests across different hardware.
                                </p>
                            </li>
                            <li className="p-4 bg-gray-800 rounded-lg border border-gray-700">
                                <a href="https://huggingface.co/collections/unsloth/llama-3-6618d368420215033c5e8845" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline font-medium block mb-1">
                                    Unsloth / Model Collections
                                </a>
                                <p className="text-sm text-gray-400">
                                    Source for model architecture parameters (Layers, Heads).
                                </p>
                            </li>
                        </ul>
                    </section>
                </div>
            </div>
        </div>
    );
};

export default About;

