import { useState, useRef, useEffect, useCallback } from "react";

interface DualRangeSliderProps {
    min: number;
    max: number;
    step?: number;
    minVal: number;
    maxVal: number;
    onChange: (min: number, max: number) => void;
    formatLabel?: (val: number) => string;
    steps?: number[]; // Optional discrete steps
}

export function DualRangeSlider({
    min,
    max,
    step = 1,
    minVal,
    maxVal,
    onChange,
    formatLabel = (val) => val.toString(),
    steps,
}: DualRangeSliderProps) {
    const [activeThumb, setActiveThumb] = useState<"min" | "max" | null>(null);
    const trackRef = useRef<HTMLDivElement>(null);

    // (getPercent moved to render scope or helper below to access steps)

    // Handle drag
    // If steps provided, map value to index; else standard linear
    const handleDrag = useCallback(
        (e: MouseEvent | TouchEvent, thumb: "min" | "max") => {
            if (!trackRef.current) return;

            const trackRect = trackRef.current.getBoundingClientRect();
            const clientX = 'touches' in e ? e.touches[0].clientX : e.clientX;

            const percent = Math.min(
                Math.max(0, (clientX - trackRect.left) / trackRect.width),
                1
            );

            let newValue: number;

            if (steps) {
                // Map percent to index range 0 to steps.length-1
                const maxIndex = steps.length - 1;
                newValue = Math.round(percent * maxIndex);
            } else {
                const rawValue = min + percent * (max - min);
                newValue = Math.round(rawValue / step) * step;
            }

            if (thumb === "min") {
                const val = Math.min(newValue, maxVal - (steps ? 1 : step));
                onChange(val, maxVal);
            } else {
                const val = Math.max(newValue, minVal + (steps ? 1 : step));
                onChange(minVal, val);
            }
        },
        [min, max, step, minVal, maxVal, onChange, steps]
    );

    // Helper to get percent for UI
    const getPercent = (value: number) => {
        if (steps) {
            return (value / (steps.length - 1)) * 100;
        }
        return Math.round(((value - min) / (max - min)) * 100);
    };

    useEffect(() => {
        if (activeThumb) {
            const handleMove = (e: MouseEvent | TouchEvent) => handleDrag(e, activeThumb);
            const handleUp = () => setActiveThumb(null);

            document.addEventListener("mousemove", handleMove);
            document.addEventListener("mouseup", handleUp);
            document.addEventListener("touchmove", handleMove);
            document.addEventListener("touchend", handleUp);

            return () => {
                document.removeEventListener("mousemove", handleMove);
                document.removeEventListener("mouseup", handleUp);
                document.removeEventListener("touchmove", handleMove);
                document.removeEventListener("touchend", handleUp);
            };
        }
    }, [activeThumb, handleDrag]);

    return (
        <div className="relative w-full h-12 flex items-center select-none">
            <div ref={trackRef} className="relative w-full h-1.5 bg-gray-200 dark:bg-gray-700 rounded-full">
                {/* Step Ticks */}
                {steps && steps.map((_, index) => (
                    <div
                        key={index}
                        className="absolute top-1/2 -translate-y-1/2 w-1.5 h-1.5 bg-gray-400 dark:bg-gray-600 rounded-full z-0 transform -translate-x-1/2"
                        style={{ left: `${(index / (steps.length - 1)) * 100}%` }}
                    />
                ))}

                {/* Active Range Bar */}
                <div
                    className="absolute h-full bg-blue-500 rounded-full"
                    style={{
                        left: `${getPercent(minVal)}%`,
                        width: `${getPercent(maxVal) - getPercent(minVal)}%`,
                    }}
                />

                {/* Min Thumb */}
                <div
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-blue-500 rounded-full cursor-pointer shadow hover:scale-110 transition-transform z-10 ${activeThumb === "min" ? "scale-110 ring-4 ring-blue-500/20" : ""
                        }`}
                    style={{ left: `${getPercent(minVal)}%` }}
                    onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveThumb("min");
                    }}
                    onTouchStart={(e) => {
                        e.stopPropagation();
                        setActiveThumb("min");
                    }}
                />

                {/* Max Thumb */}
                <div
                    className={`absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-4 h-4 bg-white border-2 border-blue-500 rounded-full cursor-pointer shadow hover:scale-110 transition-transform z-10 ${activeThumb === "max" ? "scale-110 ring-4 ring-blue-500/20" : ""
                        }`}
                    style={{ left: `${getPercent(maxVal)}%` }}
                    onMouseDown={(e) => {
                        e.stopPropagation();
                        setActiveThumb("max");
                    }}
                    onTouchStart={(e) => {
                        e.stopPropagation();
                        setActiveThumb("max");
                    }}
                />
            </div>

            {/* Permanent Labels */}
            {steps && (
                <div className="absolute w-full top-9 text-xs text-gray-400 font-medium select-none pointer-events-none">
                    {steps.map((_, index) => {
                        // Skip some labels if too crowded? User asked for "numbers to also show all the time"
                        // 12 items might be crowded. Let's try showing all but small font.
                        // Or show alternate if crowded? Let's show all first.
                        return (
                            <div
                                key={index}
                                className="absolute transform -translate-x-1/2 text-center"
                                style={{ left: `${(index / (steps.length - 1)) * 100}%` }}
                            >
                                {formatLabel(index)}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
