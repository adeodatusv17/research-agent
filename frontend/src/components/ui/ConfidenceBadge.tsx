"use client";

import clsx from "clsx";

interface ConfidenceBadgeProps {
  score?: number;
  source?: "extracted" | "inferred";
  className?: string;
}

export function ConfidenceBadge({
  score,
  source = "extracted",
  className,
}: ConfidenceBadgeProps) {
  const safeScore = typeof score === "number" && Number.isFinite(score) ? score : 0;
  const safeSource = source ?? "extracted";
  const percentage = Math.round(safeScore * 100);

  const tone =
    safeScore >= 0.75
      ? "bg-status-success/10 text-status-success border-status-success/20"
      : safeScore >= 0.45
        ? "bg-status-warning/10 text-status-warning border-status-warning/20"
        : "bg-status-error/10 text-status-error border-status-error/20";

  const label = safeSource === "inferred" ? "Inferred" : "Extracted";

  return (
    <span
      className={clsx(
        "inline-flex items-center rounded-full border px-2.5 py-1 text-xs font-medium",
        tone,
        className
      )}
    >
      {label} • {percentage}%
    </span>
  );
}
