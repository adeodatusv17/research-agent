import clsx from "clsx";

type BadgeVariant = "ready" | "analyzing" | "pending" | "failed" | "default";

interface BadgeProps {
  variant?: BadgeVariant;
  children: React.ReactNode;
  className?: string;
}

const variantStyles: Record<BadgeVariant, string> = {
  ready: "bg-status-success/15 text-status-success border-status-success/20",
  analyzing: "bg-accent-indigo/15 text-accent-indigo border-accent-indigo/20",
  pending: "bg-status-warning/15 text-status-warning border-status-warning/20",
  failed: "bg-status-error/15 text-status-error border-status-error/20",
  default: "bg-bg-hover text-text-secondary border-bg-border",
};

const dotStyles: Record<BadgeVariant, string> = {
  ready: "bg-status-success",
  analyzing: "bg-accent-indigo animate-pulse",
  pending: "bg-status-warning animate-pulse-slow",
  failed: "bg-status-error",
  default: "bg-text-muted",
};

export function Badge({ variant = "default", children, className }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1.5 px-2 py-0.5 text-xs font-medium rounded-full border",
        variantStyles[variant],
        className
      )}
    >
      <span className={clsx("status-dot w-1.5 h-1.5", dotStyles[variant])} />
      {children}
    </span>
  );
}
