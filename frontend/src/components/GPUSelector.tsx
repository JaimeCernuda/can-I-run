/**
 * GPU Selector Component
 *
 * A searchable dropdown for selecting a GPU from the database,
 * with an option for custom VRAM entry.
 */

import { useState, useMemo } from "react";
import type { GPU } from "../lib/types";

interface GPUSelectorProps {
  gpus: GPU[];
  selectedGpu: GPU | null;
  customVram: number | null;
  onGpuChange: (gpu: GPU | null) => void;
  onCustomVramChange: (vram: number | null) => void;
}

export function GPUSelector({
  gpus,
  selectedGpu,
  customVram,
  onGpuChange,
  onCustomVramChange,
}: GPUSelectorProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [isOpen, setIsOpen] = useState(false);
  const [isCustom, setIsCustom] = useState(false);
  const [featureFilter, setFeatureFilter] = useState<"All" | "NVIDIA" | "AMD" | "Apple" | "Intel">("All");

  // Group GPUs by vendor
  const groupedGpus = useMemo(() => {
    const groups: Record<string, GPU[]> = {};
    for (const gpu of gpus) {
      if (!groups[gpu.vendor]) {
        groups[gpu.vendor] = [];
      }
      groups[gpu.vendor].push(gpu);
    }
    // Sort vendors: NVIDIA first, then Apple, AMD, Intel
    const vendorOrder = ["NVIDIA", "Apple", "AMD", "Intel"];
    return vendorOrder
      .filter((v) => groups[v])
      .map((vendor) => ({
        vendor,
        gpus: groups[vendor].sort((a, b) => b.vram_gb - a.vram_gb),
      }));
  }, [gpus]);

  // Filter GPUs by search query and feature filter
  const filteredGroups = useMemo(() => {
    let groups = groupedGpus;

    if (featureFilter !== "All") {
      groups = groupedGpus.filter((g) => g.vendor === featureFilter);
    }

    if (!searchQuery) return groups;

    const query = searchQuery.toLowerCase();
    return groups
      .map((group) => ({
        vendor: group.vendor,
        gpus: group.gpus.filter(
          (gpu) =>
            gpu.name.toLowerCase().includes(query) ||
            gpu.vendor.toLowerCase().includes(query)
        ),
      }))
      .filter((group) => group.gpus.length > 0);
  }, [groupedGpus, featureFilter, searchQuery]);

  const handleSelect = (gpu: GPU) => {
    onGpuChange(gpu);
    onCustomVramChange(null);
    setIsCustom(false);
    setIsOpen(false);
    setSearchQuery("");
  };

  const handleCustomToggle = () => {
    setIsCustom(true);
    onGpuChange(null);
    setIsOpen(false);
  };

  const handleCustomVramInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const value = parseFloat(e.target.value);
    if (!isNaN(value) && value > 0) {
      onCustomVramChange(value);
    } else if (e.target.value === "") {
      onCustomVramChange(null);
    }
  };

  const displayValue = isCustom
    ? customVram
      ? `Custom (${customVram} GB)`
      : "Enter VRAM..."
    : selectedGpu
      ? `${selectedGpu.name} (${selectedGpu.vram_gb} GB)`
      : "Select GPU...";

  return (
    <div className="relative">
      <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
        GPU
      </label>

      {isCustom ? (
        <div className="flex gap-2">
          <input
            type="number"
            min="1"
            max="512"
            step="0.1"
            placeholder="VRAM in GB"
            value={customVram ?? ""}
            onChange={handleCustomVramInput}
            className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={() => {
              setIsCustom(false);
              onCustomVramChange(null);
            }}
            className="px-3 py-2 text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
          >
            Cancel
          </button>
        </div>
      ) : (
        <div className="relative">
          <button
            onClick={() => setIsOpen(!isOpen)}
            className="w-full px-3 py-2 text-left border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 hover:border-gray-400 dark:hover:border-gray-500 focus:ring-2 focus:ring-blue-500"
          >
            <span className="block truncate">{displayValue}</span>
            <span className="absolute inset-y-0 right-0 flex items-center pr-2 pointer-events-none">
              <svg
                className="w-5 h-5 text-gray-400"
                viewBox="0 0 20 20"
                fill="currentColor"
              >
                <path
                  fillRule="evenodd"
                  d="M5.293 7.293a1 1 0 011.414 0L10 10.586l3.293-3.293a1 1 0 111.414 1.414l-4 4a1 1 0 01-1.414 0l-4-4a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              </svg>
            </span>
          </button>

          {isOpen && (
            <div className="absolute z-50 w-full mt-1 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg shadow-lg max-h-96 overflow-hidden">
              {/* Search input */}
              <div className="p-2 border-b border-gray-200 dark:border-gray-700">
                <input
                  type="text"
                  placeholder="Search GPUs..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                  autoFocus
                />
              </div>

              {/* Vendor Filters */}
              <div className="flex gap-1 p-2 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
                {["All", "NVIDIA", "AMD", "Apple", "Intel"].map((vendor) => (
                  <button
                    key={vendor}
                    onClick={() => setFeatureFilter(vendor as any)}
                    className={`px-2 py-1 text-xs rounded-full whitespace-nowrap transition-colors ${featureFilter === vendor
                      ? "bg-blue-100 dark:bg-blue-900 text-blue-700 dark:text-blue-300 font-medium"
                      : "bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600"
                      }`}
                  >
                    {vendor}
                  </button>
                ))}
              </div>

              {/* GPU list */}
              <div className="overflow-y-auto max-h-72">
                {/* Custom VRAM option */}
                <button
                  onClick={handleCustomToggle}
                  className="w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 text-blue-600 dark:text-blue-400 font-medium border-b border-gray-200 dark:border-gray-700"
                >
                  Enter custom VRAM...
                </button>

                {filteredGroups.map((group) => (
                  <div key={group.vendor}>
                    <div className="px-4 py-1 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900 sticky top-0">
                      {group.vendor}
                    </div>
                    {group.gpus.map((gpu) => (
                      <button
                        key={gpu.name}
                        onClick={() => handleSelect(gpu)}
                        className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 ${selectedGpu?.name === gpu.name
                          ? "bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300"
                          : "text-gray-900 dark:text-gray-100"
                          }`}
                      >
                        <span className="font-medium">{gpu.name}</span>
                        <span className="ml-2 text-gray-500 dark:text-gray-400">
                          {gpu.vram_gb} GB
                        </span>
                        {gpu.generation && (
                          <span className="ml-2 text-xs text-gray-400 dark:text-gray-500">
                            {gpu.generation}
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* VRAM display */}
      {(selectedGpu || customVram) && (
        <div className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Total VRAM:{" "}
          <span className="font-medium text-gray-700 dark:text-gray-300">
            {selectedGpu?.vram_gb ?? customVram} GB
          </span>
          {selectedGpu?.bandwidth_gbps && (
            <span className="ml-2">
              Bandwidth:{" "}
              <span className="font-medium text-gray-700 dark:text-gray-300">
                {selectedGpu.bandwidth_gbps} GB/s
              </span>
            </span>
          )}
        </div>
      )}
    </div>
  );
}
