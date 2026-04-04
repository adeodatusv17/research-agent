"use client";

import { useState } from "react";
import { ChevronDown } from "lucide-react";
import { ConfidenceBadge } from "@/components/ui/ConfidenceBadge";

interface ChunkCardProps {
  text: string;
  summary?: string;
  confidence?: number;
  source?: "extracted" | "inferred";
}

export default function ChunkCard({
  text,
  summary,
  confidence,
  source = "extracted",
}: ChunkCardProps) {
  const [expanded, setExpanded] = useState(false);
  const fullText = text.trim();
  const previewText = (summary ?? fullText).trim();
  const canExpand = fullText.length > 0 && previewText.length > 0 && fullText !== previewText;
  const displayText = expanded || !canExpand ? fullText : previewText;

  return (
    <article className="rounded-xl border border-white/10 bg-bg-card p-4 shadow-lg shadow-black/20 transition-transform duration-150 hover:-translate-y-[2px] hover:border-white/20">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-sm leading-relaxed text-gray-300">{displayText}</p>
          <div className="mt-3 flex items-center gap-3 text-xs text-gray-500">
            <span>{source === "inferred" ? "Inferred" : "Extracted"}</span>
            {canExpand && (
              <button
                type="button"
                onClick={() => setExpanded((current) => !current)}
                className="inline-flex items-center gap-1 text-xs text-gray-400 transition-colors hover:text-white"
                aria-expanded={expanded}
              >
                <ChevronDown
                  className={`h-3.5 w-3.5 transition-transform ${expanded ? "rotate-180" : ""}`}
                />
                <span>{expanded ? "Collapse" : "Expand"}</span>
              </button>
            )}
          </div>
        </div>
        <ConfidenceBadge score={confidence} source={source} className="shrink-0" />
      </div>
    </article>
  );
}
