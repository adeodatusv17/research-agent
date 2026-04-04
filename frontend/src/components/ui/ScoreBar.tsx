import clsx from "clsx";

interface ScoreBarProps {
  value: number; // 0-1
  label: string;
  showValue?: boolean;
  color?: "indigo" | "cyan" | "emerald" | "warning" | "error";
}

const colorMap = {
  indigo: "bg-accent-indigo",
  cyan: "bg-accent-cyan",
  emerald: "bg-status-success",
  warning: "bg-status-warning",
  error: "bg-status-error",
};

function scoreToColor(value: number): "emerald" | "warning" | "error" | "indigo" {
  if (value >= 0.7) return "emerald";
  if (value >= 0.4) return "warning";
  if (value < 0.4) return "error";
  return "indigo";
}

export function ScoreBar({ value, label, showValue = true, color }: ScoreBarProps) {
  const resolvedColor = color ?? scoreToColor(value);
  const pct = Math.round(Math.max(0, Math.min(1, value)) * 100);

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <span className="text-xs text-text-secondary">{label}</span>
        {showValue && (
          <span className="text-xs font-mono font-medium text-text-primary">{pct}%</span>
        )}
      </div>
      <div className="score-bar-track">
        <div
          className={clsx("score-bar-fill", colorMap[resolvedColor])}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function OverallScoreDial({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = pct >= 70 ? "text-status-success" : pct >= 40 ? "text-status-warning" : "text-status-error";

  return (
    <div className="flex items-center gap-4">
      <div className="relative w-16 h-16 flex-shrink-0">
        <svg viewBox="0 0 36 36" className="w-16 h-16 -rotate-90">
          <circle cx="18" cy="18" r="15.9" fill="none" stroke="#1E2D45" strokeWidth="2.5" />
          <circle
            cx="18" cy="18" r="15.9" fill="none"
            stroke={pct >= 70 ? "#10B981" : pct >= 40 ? "#F59E0B" : "#EF4444"}
            strokeWidth="2.5"
            strokeDasharray={`${pct} ${100 - pct}`}
            strokeLinecap="round"
          />
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <span className={`text-sm font-bold font-mono ${color}`}>{pct}</span>
        </div>
      </div>
      <div>
        <div className="text-xs text-text-muted">Overall Score</div>
        <div className={`text-lg font-bold ${color}`}>{pct}%</div>
      </div>
    </div>
  );
}
