/**
 * Custom Markdown Components
 *
 * Provides styled renderers for react-markdown to match the About.tsx visual design.
 * These components maintain the dark theme with slate colors and accent highlights.
 */

import type { Components } from 'react-markdown';
import type { ReactNode } from 'react';

// Section icon configuration
const SECTION_ICONS: Record<string, { icon: string; color: string }> = {
  overview: { icon: '◆', color: 'text-amber-400 bg-amber-500/10' },
  vram: { icon: '∑', color: 'text-blue-400 bg-blue-500/10' },
  quantization: { icon: '⚡', color: 'text-purple-400 bg-purple-500/10' },
  quality: { icon: '★', color: 'text-green-400 bg-green-500/10' },
  performance: { icon: '⚡', color: 'text-orange-400 bg-orange-500/10' },
  efficiency: { icon: '◎', color: 'text-cyan-400 bg-cyan-500/10' },
  pareto: { icon: '◇', color: 'text-pink-400 bg-pink-500/10' },
  data: { icon: '⬡', color: 'text-indigo-400 bg-indigo-500/10' },
};

// Determine section type from heading text
function getSectionType(text: string): string {
  const lower = text.toLowerCase();
  if (lower.includes('overview')) return 'overview';
  if (lower.includes('vram')) return 'vram';
  if (lower.includes('quantization')) return 'quantization';
  if (lower.includes('quality')) return 'quality';
  if (lower.includes('performance')) return 'performance';
  if (lower.includes('efficiency')) return 'efficiency';
  if (lower.includes('pareto')) return 'pareto';
  if (lower.includes('data')) return 'data';
  return 'overview';
}

// Get icon config for a section type
function getSectionIcon(type: string): { icon: string; color: string } {
  return SECTION_ICONS[type] || SECTION_ICONS.overview;
}

// Extract text from children
function getTextContent(children: ReactNode): string {
  if (typeof children === 'string') return children;
  if (Array.isArray(children)) {
    return children.map(getTextContent).join('');
  }
  if (children && typeof children === 'object' && 'props' in children) {
    return getTextContent((children as { props: { children: ReactNode } }).props.children);
  }
  return '';
}

export const markdownComponents: Components = {
  // Main title
  h1: ({ children }) => (
    <h1 className="text-4xl font-bold mb-6 bg-gradient-to-r from-blue-400 via-purple-400 to-pink-400 bg-clip-text text-transparent">
      {children}
    </h1>
  ),

  // Section headers with icons
  h2: ({ children }) => {
    const text = getTextContent(children);
    const sectionType = getSectionType(text);
    const iconConfig = getSectionIcon(sectionType);
    return (
      <h2 className="text-2xl font-semibold mb-6 mt-12 text-white flex items-center border-b border-slate-700 pb-4">
        <span className={`p-2 rounded-lg ${iconConfig.color} mr-3`}>
          {iconConfig.icon}
        </span>
        {children}
      </h2>
    );
  },

  // Subsection headers
  h3: ({ children }) => (
    <h3 className="text-xl font-medium mb-4 mt-8 text-slate-200">
      {children}
    </h3>
  ),

  // Sub-subsection headers
  h4: ({ children }) => (
    <h4 className="text-lg font-medium mb-3 mt-6 text-slate-300">
      {children}
    </h4>
  ),

  // Paragraphs
  p: ({ children }) => (
    <p className="text-slate-300 mb-4 leading-relaxed">
      {children}
    </p>
  ),

  // Strong/bold text
  strong: ({ children }) => (
    <strong className="text-white font-semibold">{children}</strong>
  ),

  // Emphasis/italic
  em: ({ children }) => (
    <em className="text-slate-200 italic">{children}</em>
  ),

  // Inline code
  code: ({ children, className }) => {
    // Check if this is a code block (has language class) vs inline code
    const isBlock = className?.includes('language-');
    if (isBlock) {
      return (
        <code className={`${className} block`}>
          {children}
        </code>
      );
    }
    return (
      <code className="bg-slate-800 text-blue-300 px-1.5 py-0.5 rounded text-sm font-mono">
        {children}
      </code>
    );
  },

  // Code blocks
  pre: ({ children }) => (
    <pre className="bg-slate-950/50 border border-slate-800 rounded-lg p-4 overflow-x-auto mb-6 text-sm font-mono text-slate-300">
      {children}
    </pre>
  ),

  // Block quotes - styled as info cards
  blockquote: ({ children }) => (
    <div className="bg-blue-500/5 border-l-4 border-blue-500 rounded-r-lg p-4 mb-6">
      <div className="text-slate-300">{children}</div>
    </div>
  ),

  // Unordered lists
  ul: ({ children }) => (
    <ul className="list-disc list-inside space-y-2 mb-6 text-slate-300 pl-2">
      {children}
    </ul>
  ),

  // Ordered lists
  ol: ({ children }) => (
    <ol className="list-decimal list-inside space-y-2 mb-6 text-slate-300 pl-2">
      {children}
    </ol>
  ),

  // List items
  li: ({ children }) => (
    <li className="text-slate-300 leading-relaxed">
      {children}
    </li>
  ),

  // Tables
  table: ({ children }) => (
    <div className="overflow-x-auto rounded-xl border border-slate-700 mb-6">
      <table className="w-full text-left bg-slate-900/50">
        {children}
      </table>
    </div>
  ),

  // Table head
  thead: ({ children }) => (
    <thead className="bg-slate-950 text-slate-400 border-b border-slate-700">
      {children}
    </thead>
  ),

  // Table body
  tbody: ({ children }) => (
    <tbody className="divide-y divide-slate-800 text-slate-300">
      {children}
    </tbody>
  ),

  // Table rows
  tr: ({ children }) => (
    <tr className="hover:bg-slate-800/50 transition-colors">
      {children}
    </tr>
  ),

  // Table header cells
  th: ({ children }) => (
    <th className="p-4 font-medium text-slate-300">
      {children}
    </th>
  ),

  // Table data cells
  td: ({ children }) => {
    const text = getTextContent(children);
    // Highlight recommended quantization
    const isRecommended = text.includes('Q4_K_M') || text.includes('Recommended');
    const isHighLoss = text.includes('High Loss') || text.includes('Extreme Loss');

    let cellClass = 'p-4';
    if (isRecommended) {
      cellClass += ' text-blue-300 font-medium';
    } else if (isHighLoss) {
      cellClass += ' text-red-400';
    }

    return <td className={cellClass}>{children}</td>;
  },

  // Horizontal rules
  hr: () => (
    <hr className="border-slate-700 my-8" />
  ),

  // Links
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="text-blue-400 hover:text-blue-300 hover:underline transition-colors"
    >
      {children}
    </a>
  ),
};

export default markdownComponents;
