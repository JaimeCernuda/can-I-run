/**
 * Hook for synchronized highlighting across multiple charts.
 *
 * When a user hovers on a point in one chart, the same model
 * is highlighted in all three charts simultaneously.
 */

import { useState, useCallback, useMemo } from "react";

interface UseLinkedHighlightReturn {
  /** Currently highlighted point ID (or null if none) */
  highlightedId: string | null;

  /** Set of pinned/selected point IDs for comparison */
  selectedIds: Set<string>;

  /** Handler for mouse enter on a point */
  onPointHover: (id: string) => void;

  /** Handler for mouse leave */
  onPointLeave: () => void;

  /** Handler for clicking/selecting a point */
  onPointClick: (id: string) => void;

  /** Clear all selections */
  clearSelections: () => void;

  /** Check if a point should be highlighted */
  isHighlighted: (id: string) => boolean;

  /** Check if a point is selected/pinned */
  isSelected: (id: string) => boolean;
}

export function useLinkedHighlight(): UseLinkedHighlightReturn {
  const [highlightedId, setHighlightedId] = useState<string | null>(null);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  const onPointHover = useCallback((id: string) => {
    setHighlightedId(id);
  }, []);

  const onPointLeave = useCallback(() => {
    setHighlightedId(null);
  }, []);

  const onPointClick = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }, []);

  const clearSelections = useCallback(() => {
    setSelectedIds(new Set());
  }, []);

  const isHighlighted = useCallback(
    (id: string) => {
      return id === highlightedId || selectedIds.has(id);
    },
    [highlightedId, selectedIds]
  );

  const isSelected = useCallback(
    (id: string) => {
      return selectedIds.has(id);
    },
    [selectedIds]
  );

  return useMemo(
    () => ({
      highlightedId,
      selectedIds,
      onPointHover,
      onPointLeave,
      onPointClick,
      clearSelections,
      isHighlighted,
      isSelected,
    }),
    [
      highlightedId,
      selectedIds,
      onPointHover,
      onPointLeave,
      onPointClick,
      clearSelections,
      isHighlighted,
      isSelected,
    ]
  );
}
