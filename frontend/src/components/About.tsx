
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
        <div className="container mx-auto px-4 py-8 max-w-5xl text-slate-200">
            <button
                onClick={onBack}
                className="mb-8 px-5 py-2.5 bg-slate-700/50 hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors border border-slate-600/50"
            >
                ← Back to Calculator
            </button>

            <header className="mb-12 text-center">
                <h1 className="text-4xl font-bold mb-4 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
                    How it Works & Data Intelligence
                </h1>
                <p className="text-slate-400 max-w-2xl mx-auto">
                    A deep dive into the mathematics of LLM inference, quantization mechanisms, and the real-world benchmarks powering this tool.
                </p>
            </header>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 mb-16">
                {/* VRAM MATH SECTION */}
                <section className="bg-slate-900/50 p-8 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
                    <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-3">
                        <span className="p-2 bg-blue-500/10 rounded-lg text-blue-400">∑</span>
                        The Math of VRAM
                    </h2>
                    <div className="space-y-6">
                        <div>
                            <p className="text-slate-300 mb-3">
                                Total VRAM usage is the sum of model weights, KV cache, and activation overhead.
                            </p>
                            <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800 overflow-x-auto">
                                <Latex block>{`V_{total} = V_{weights} + V_{kv} + V_{activation}`}</Latex>
                            </div>
                        </div>

                        <div>
                            <h3 className="font-medium text-blue-300 mb-2">1. Model Weights</h3>
                            <p className="text-sm text-slate-400 mb-2">
                                Derived from parameter count ($P$) and bits per weight ($B$).
                            </p>
                            <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800 text-sm">
                                <Latex block>{`V_{weights} = \\frac{P \\times B}{8 \\times 10^9} \\text{ GB}`}</Latex>
                            </div>
                        </div>

                        <div>
                            <h3 className="font-medium text-green-300 mb-2">2. KV Cache (per token)</h3>
                            <p className="text-sm text-slate-400 mb-2">
                                The memory needed to store attention keys/values for context ($C$).
                            </p>
                            <div className="bg-slate-950/50 p-3 rounded-lg border border-slate-800 text-sm">
                                <Latex block>{`V_{kv} = \\frac{2 \\times L \\times H_{kv} \\times D_{head} \\times C \\times 2_{bytes}}{10^{9}}`}</Latex>
                            </div>
                            <ul className="text-xs text-slate-500 mt-2 grid grid-cols-2 gap-1">
                                <li>$L$: Layers</li>
                                <li>$H_{kv}$: KV Heads</li>
                                <li>$D_{head}$: Head Dimension</li>
                                <li>$C$: Context Length</li>
                            </ul>
                        </div>
                    </div>
                </section>

                {/* PERFORMANCE MATH SECTION */}
                <section className="bg-slate-900/50 p-8 rounded-2xl border border-slate-700/50 backdrop-blur-sm">
                    <h2 className="text-2xl font-semibold mb-6 text-white flex items-center gap-3">
                        <span className="p-2 bg-purple-500/10 rounded-lg text-purple-400">⚡</span>
                        Inference Speed & Efficiency
                    </h2>
                    <div className="space-y-6">
                        <div>
                            <h3 className="font-medium text-purple-300 mb-2">Token Generation Speed</h3>
                            <p className="text-slate-300 mb-3 text-sm">
                                LLM inference is typically memory-bandwidth bound. We estimate speed ($T/s$) using GPU Bandwidth ($BW$) and Model Size.
                            </p>
                            <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
                                <Latex block>{`T/s \\approx \\frac{BW \\text{ (GB/s)}}{V_{weights} \\text{ (GB)}} \\times \\eta`}</Latex>
                            </div>
                            <p className="text-xs text-slate-500 mt-2">
                                Where $\eta$ is an efficiency factor (typically 0.6-0.8 for optimized kernels like Unsloth/FlashAttn).
                            </p>
                        </div>

                        <div>
                            <h3 className="font-medium text-amber-300 mb-2">Efficiency Score</h3>
                            <p className="text-slate-300 mb-3 text-sm">
                                A proprietary metric to find the "sweet spot" models.
                            </p>
                            <div className="bg-slate-950/50 p-4 rounded-lg border border-slate-800">
                                <Latex block>{`E = \\frac{\\sqrt{Q_{norm} \\times P_{norm}}}{VRAM} \\times 100`}</Latex>
                            </div>
                        </div>
                    </div>
                </section>
            </div>

            {/* QUANTIZATION DEEP DIVE */}
            <section className="mb-16">
                <h2 className="text-3xl font-bold mb-8 text-white border-b border-slate-800 pb-4">
                    Quantization: The Bits & Bytes
                </h2>

                <div className="bg-slate-800/40 rounded-xl p-8 border border-slate-700 mb-8">
                    <h3 className="text-xl font-semibold text-white mb-4">Why Q4_K_M is the Golden Standard</h3>
                    <p className="text-slate-300 mb-6 leading-relaxed">
                        Modern quantization (like GGUF/k-quants) does not just truncate bits. It uses smart clustering (k-means) to optimize which weights get more precision.
                        <br /><br />
                        For <strong>LLaMA-3 and Mistral</strong> class models, benchmarks show that 4-bit quantization (Q4_K_M) retains <strong>99% of FP16 performance</strong> while reducing VRAM usage by ~50%.
                        The drop in perplexity (confusion) is often negligible compared to the gain in being able to run a larger parameter model.
                    </p>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                        <div className="bg-green-900/20 border border-green-500/30 p-4 rounded-lg">
                            <div className="text-green-400 font-bold mb-1">Recommended: Q4_K_M</div>
                            <div className="text-2xl font-mono text-white mb-2">4.83 bits</div>
                            <div className="text-sm text-green-200/70">
                                Optimal balance. "Goldilocks" zone for almost all use cases.
                            </div>
                        </div>
                        <div className="bg-yellow-900/20 border border-yellow-500/30 p-4 rounded-lg">
                            <div className="text-yellow-400 font-bold mb-1">The Cliff: Q3_K_M</div>
                            <div className="text-2xl font-mono text-white mb-2">3.9 bits</div>
                            <div className="text-sm text-yellow-200/70">
                                Noticeable logic degradation. Only use if strictly VRAM constrained.
                            </div>
                        </div>
                        <div className="bg-red-900/20 border border-red-500/30 p-4 rounded-lg">
                            <div className="text-red-400 font-bold mb-1">Brain Damage: Q2_K</div>
                            <div className="text-2xl font-mono text-white mb-2">2.56 bits</div>
                            <div className="text-sm text-red-200/70">
                                High perplexity. Model becomes incoherent/repetitive. Avoid.
                            </div>
                        </div>
                    </div>
                </div>

                <div className="overflow-x-auto rounded-xl border border-slate-700">
                    <table className="w-full text-left bg-slate-900/50">
                        <thead className="bg-slate-950 text-slate-400 border-b border-slate-700">
                            <tr>
                                <th className="p-4">Quantization</th>
                                <th className="p-4">Bits/Weight</th>
                                <th className="p-4">Perplexity (+PPL)</th>
                                <th className="p-4">VRAM Saving</th>
                                <th className="p-4">Use Case</th>
                            </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800 text-slate-300">
                            <tr className="hover:bg-slate-800/50">
                                <td className="p-4 font-mono text-blue-300">F16 / BF16</td>
                                <td className="p-4">16.0</td>
                                <td className="p-4 text-green-400">Baseline</td>
                                <td className="p-4">0%</td>
                                <td className="p-4 text-sm">Research / Training</td>
                            </tr>
                            <tr className="hover:bg-slate-800/50">
                                <td className="p-4 font-mono text-cyan-300">Q8_0</td>
                                <td className="p-4">8.0</td>
                                <td className="p-4 text-green-400">+0.002 (Null)</td>
                                <td className="p-4">50%</td>
                                <td className="p-4 text-sm">Perfect Accuracy</td>
                            </tr>
                            <tr className="bg-blue-500/5 hover:bg-blue-500/10 border-l-4 border-l-blue-500">
                                <td className="p-4 font-mono text-white font-bold">Q4_K_M</td>
                                <td className="p-4">4.83</td>
                                <td className="p-4 text-green-300">+0.05 (Negligible)</td>
                                <td className="p-4">~70%</td>
                                <td className="p-4 text-sm font-bold text-blue-300">Daily Driver (Best Value)</td>
                            </tr>
                            <tr className="hover:bg-slate-800/50">
                                <td className="p-4 font-mono text-yellow-500">Q3_K_M</td>
                                <td className="p-4">3.91</td>
                                <td className="p-4 text-yellow-400">+0.24 (Visible)</td>
                                <td className="p-4">~75%</td>
                                <td className="p-4 text-sm">Squeezing big models</td>
                            </tr>
                            <tr className="hover:bg-slate-800/50 opacity-60">
                                <td className="p-4 font-mono text-red-500">Q2_K</td>
                                <td className="p-4">2.56</td>
                                <td className="p-4 text-red-400">+3.52 (High)</td>
                                <td className="p-4">~84%</td>
                                <td className="p-4 text-sm">Not Recommended</td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </section>

            {/* DATA SOURCES SECTION */}
            <section className="mb-12">
                <h2 className="text-2xl font-semibold mb-6 text-white">Verified Data Sources</h2>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <a href="https://huggingface.co/unsloth" target="_blank" rel="noopener noreferrer"
                        className="group p-5 bg-slate-800/40 hover:bg-slate-800 rounded-xl border border-slate-700 hover:border-blue-500/50 transition-all">
                        <h3 className="font-bold text-blue-400 group-hover:text-blue-300 mb-2 flex items-center gap-2">
                            Unsloth AI
                            <span className="text-xs bg-blue-500/10 px-2 py-0.5 rounded text-blue-300">Models</span>
                        </h3>
                        <p className="text-sm text-slate-400">Direct parameter counts and architecture configs from the Unsloth library.</p>
                    </a>

                    <a href="https://github.com/ggerganov/llama.cpp" target="_blank" rel="noopener noreferrer"
                        className="group p-5 bg-slate-800/40 hover:bg-slate-800 rounded-xl border border-slate-700 hover:border-orange-500/50 transition-all">
                        <h3 className="font-bold text-orange-400 group-hover:text-orange-300 mb-2 flex items-center gap-2">
                            llama.cpp
                            <span className="text-xs bg-orange-500/10 px-2 py-0.5 rounded text-orange-300">Benchmarks</span>
                        </h3>
                        <p className="text-sm text-slate-400">Perplexity (PPL) degradation stats and memory overhead logic.</p>
                    </a>

                    <a href="https://github.com/XiongjieDai/GPU-Benchmarks-on-LLM-Inference" target="_blank" rel="noopener noreferrer"
                        className="group p-5 bg-slate-800/40 hover:bg-slate-800 rounded-xl border border-slate-700 hover:border-purple-500/50 transition-all">
                        <h3 className="font-bold text-purple-400 group-hover:text-purple-300 mb-2 flex items-center gap-2">
                            GPU Benchmarks
                            <span className="text-xs bg-purple-500/10 px-2 py-0.5 rounded text-purple-300">Throughput</span>
                        </h3>
                        <p className="text-sm text-slate-400">Real-world tokens/sec measurements across RTX 3090/4090 and H100 hardware.</p>
                    </a>
                </div>
            </section>
        </div>
    );
};

export default About;

