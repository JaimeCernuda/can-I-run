/**
 * About Page Component
 *
 * Renders the comprehensive technical documentation from about.md
 * using react-markdown with custom styled components and KaTeX for math.
 */

import ReactMarkdown from 'react-markdown';
import remarkMath from 'remark-math';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import rehypeRaw from 'rehype-raw';
import 'katex/dist/katex.min.css';

import { markdownComponents } from './MarkdownComponents';
import aboutContent from '../content/about.md?raw';

interface AboutProps {
    onBack: () => void;
}

const About: React.FC<AboutProps> = ({ onBack }) => {
    return (
        <div className="container mx-auto px-4 py-8 max-w-5xl text-slate-200">
            <button
                onClick={onBack}
                className="mb-8 px-5 py-2.5 bg-slate-700/50 hover:bg-slate-600 rounded-lg text-sm font-medium transition-colors border border-slate-600/50"
            >
                ← Back to Calculator
            </button>

            <header className="mb-12 text-center">
                <p className="text-slate-400 max-w-2xl mx-auto">
                    A deep dive into the mathematics of LLM inference, quantization mechanisms, and the real-world benchmarks powering this tool.
                </p>
            </header>

            <article className="prose-invert max-w-none">
                <ReactMarkdown
                    remarkPlugins={[remarkMath, remarkGfm]}
                    rehypePlugins={[rehypeKatex, rehypeRaw]}
                    components={markdownComponents}
                >
                    {aboutContent}
                </ReactMarkdown>
            </article>
        </div>
    );
};

export default About;
