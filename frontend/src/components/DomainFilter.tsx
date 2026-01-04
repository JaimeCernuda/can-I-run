/**
 * Domain Filter Component
 *
 * Multi-select filter for model domains (general, code, tool-calling, etc.).
 */

import type { ModelDomain } from "../lib/types";

const DOMAIN_OPTIONS: { value: ModelDomain; label: string; icon: string }[] = [
  { value: "general", label: "General", icon: "💬" },
  { value: "code", label: "Code", icon: "💻" },
  { value: "tool-calling", label: "Tool-Calling", icon: "🔧" },
  { value: "math", label: "Math", icon: "🔢" },
  { value: "vision", label: "Vision", icon: "👁️" },
  { value: "roleplay", label: "Roleplay", icon: "🎭" },
];

interface DomainFilterProps {
  selectedDomains: ModelDomain[];
  onDomainsChange: (domains: ModelDomain[]) => void;
  modelCounts?: Record<ModelDomain, number>;
}

export function DomainFilter({
  selectedDomains,
  onDomainsChange,
  modelCounts,
}: DomainFilterProps) {
  const toggleDomain = (domain: ModelDomain) => {
    if (selectedDomains.includes(domain)) {
      onDomainsChange(selectedDomains.filter((d) => d !== domain));
    } else {
      onDomainsChange([...selectedDomains, domain]);
    }
  };

  const selectAll = () => {
    onDomainsChange([]);
  };

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
        Domains
      </label>

      <div className="flex flex-wrap gap-2">
        {/* All button */}
        <button
          onClick={selectAll}
          className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
            selectedDomains.length === 0
              ? "bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
              : "border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:border-gray-400 dark:hover:border-gray-500"
          }`}
        >
          All
        </button>

        {/* Domain buttons */}
        {DOMAIN_OPTIONS.map((option) => {
          const isSelected = selectedDomains.includes(option.value);
          const count = modelCounts?.[option.value];

          return (
            <button
              key={option.value}
              onClick={() => toggleDomain(option.value)}
              className={`px-3 py-1.5 text-sm rounded-full border transition-colors ${
                isSelected
                  ? "bg-blue-100 dark:bg-blue-900/30 border-blue-300 dark:border-blue-700 text-blue-700 dark:text-blue-300"
                  : "border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:border-gray-400 dark:hover:border-gray-500"
              }`}
            >
              <span className="mr-1">{option.icon}</span>
              {option.label}
              {count !== undefined && (
                <span className="ml-1 text-xs opacity-60">({count})</span>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
