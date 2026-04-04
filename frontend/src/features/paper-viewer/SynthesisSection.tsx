"use client";

import { useMemo, useState } from "react";
import { ChevronDown, Sigma } from "lucide-react";
import katex from "katex";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import ChunkCard from "@/features/paper-viewer/ChunkCard";
import type { AnalysisChunk, EquationCollection, SectionSynthesisResult } from "@/lib/types";

interface SynthesisSectionProps {
  title: string;
  sectionResult?: SectionSynthesisResult | null;
  chunks?: AnalysisChunk[];
  equations?: EquationCollection | null;
}

export default function SynthesisSection({
  title,
  sectionResult,
  chunks = [],
  equations,
}: SynthesisSectionProps) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [showEquations, setShowEquations] = useState(false);
  const normalizedContent = (sectionResult?.synthesis ?? "").trim();
  const confidence = sectionResult?.confidence ?? "high";
  const warning = (sectionResult?.warning ?? "").trim();
  const fabricationFlagged = Boolean(sectionResult?.fabrication_flagged);
  const retrievalRounds = sectionResult?.retrieval_rounds ?? 0;
  const rewriteRounds = sectionResult?.rewrite_rounds ?? 0;
  const normalizedChunks = chunks.filter(
    (chunk) => (chunk.text ?? chunk.summary ?? "").trim().length > 0
  );
  const equationItems = equations?.items?.filter((item) => (item.latex ?? "").trim().length > 0) ?? [];
  const equationSource = equations?.source ?? null;
  const canShowEquations = title === "Methods & Approach" && equationItems.length > 0;

  if (!normalizedContent && normalizedChunks.length === 0 && !canShowEquations) {
    return null;
  }

  return (
    <section className="mt-8">
      <h2 className="text-base font-semibold text-white">{title}</h2>

      {(normalizedContent || canShowEquations || warning) && (
        <div className="mt-4 rounded-xl border border-white/10 bg-bg-card p-5 shadow-lg shadow-black/20">
          {fabricationFlagged && (
            <div className="mb-4 rounded-lg border border-red-500/25 bg-red-500/10 p-3 text-sm text-red-200">
              This section may contain unsupported claims. The system could not verify
              all statements against the source.
            </div>
          )}
          {warning && (
            <div className="mb-4 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-100">
              {warning}
            </div>
          )}
          {confidence === "medium" && !fabricationFlagged && (
            <div className="mb-4 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-gray-400">
              This section was revised during quality review.
            </div>
          )}
          {confidence === "low" && !fabricationFlagged && (
            <div className="mb-4 rounded-lg border border-white/10 bg-white/[0.03] px-3 py-2 text-sm text-gray-300">
              Limited source evidence - verify against the original paper.
            </div>
          )}
          {retrievalRounds > 0 && (
            <div className="mb-4 rounded-lg border border-white/10 bg-white/[0.02] px-3 py-2 text-sm text-gray-500">
              Additional evidence was retrieved to improve this section.
            </div>
          )}
          {normalizedContent && (
            <div className="prose prose-invert max-w-none text-sm text-gray-300 prose-headings:text-white prose-p:text-gray-300 prose-strong:text-white prose-li:text-gray-300 prose-th:text-white prose-td:text-gray-300">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  table: ({ children }) => (
                    <div className="overflow-x-auto">
                      <table className="min-w-full border-collapse text-sm">{children}</table>
                    </div>
                  ),
                  thead: ({ children }) => <thead className="border-b border-white/10">{children}</thead>,
                  tbody: ({ children }) => <tbody>{children}</tbody>,
                  tr: ({ children }) => <tr className="border-b border-white/5">{children}</tr>,
                  th: ({ children }) => (
                    <th className="px-3 py-2 text-left font-semibold text-white">{children}</th>
                  ),
                  td: ({ children }) => <td className="px-3 py-2 align-top text-gray-300">{children}</td>,
                  ul: ({ children }) => <ul className="space-y-2 pl-5">{children}</ul>,
                  p: ({ children }) => <p className="leading-relaxed text-gray-300">{children}</p>,
                }}
              >
                {normalizedContent}
              </ReactMarkdown>
            </div>
          )}

          {canShowEquations && (
            <div className={`${normalizedContent ? "mt-5 border-t border-white/10 pt-4" : ""}`}>
              <button
                type="button"
                onClick={() => setShowEquations((current) => !current)}
                className="inline-flex items-center gap-2 text-sm text-gray-300 transition-colors hover:text-white"
                aria-expanded={showEquations}
              >
                <Sigma className="h-4 w-4" />
                <span>View Equations</span>
                <ChevronDown
                  className={`h-4 w-4 transition-transform ${showEquations ? "rotate-180" : ""}`}
                />
              </button>

              {showEquations && (
                <div className="mt-4 rounded-xl border border-white/10 bg-bg-hover/40 p-4">
                  {equationSource === "llm_generated" && (
                    <div className="mb-4 rounded-lg border border-amber-500/20 bg-amber-500/10 p-3 text-sm text-amber-200">
                      ⚠ These equations were inferred by an AI model because none were
                      extracted directly from the paper. They may contain errors —
                      verify against the original paper before use.
                    </div>
                  )}
                  <div className="space-y-4">
                    {equationItems.map((item, index) => (
                      <EquationBlock
                        key={item.id ?? `${title}-equation-${index}`}
                        latex={(item.latex ?? "").trim()}
                        description={(item.description ?? "").trim()}
                      />
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {normalizedChunks.length > 0 && (
        <div className="mt-4">
          <button
            type="button"
            onClick={() => setShowEvidence((current) => !current)}
            className="inline-flex items-center gap-2 text-sm text-gray-500 transition-colors hover:text-white"
            aria-expanded={showEvidence}
          >
            <span>Evidence</span>
            <ChevronDown className={`h-4 w-4 transition-transform ${showEvidence ? "rotate-180" : ""}`} />
          </button>
          {rewriteRounds > 0 && (
            <p className="mt-2 text-xs text-gray-500">
              Reviewed against source evidence after one rewrite attempt.
            </p>
          )}
          {showEvidence && (
            <div className="mt-3 grid grid-cols-1 gap-3">
              {normalizedChunks.map((chunk, index) => (
                <ChunkCard
                  key={chunk.id ?? `${title}-${index}`}
                  text={(chunk.text ?? chunk.summary ?? "").trim()}
                  summary={(chunk.summary ?? chunk.text ?? "").trim()}
                  confidence={chunk.confidence}
                  source={chunk.source ?? "extracted"}
                />
              ))}
            </div>
          )}
        </div>
      )}
    </section>
  );
}

function EquationBlock({
  latex,
  description,
}: {
  latex: string;
  description: string;
}) {
  const equationSteps = useMemo(() => splitEquationSteps(latex), [latex]);
  const renderedEquations = useMemo(
    () => equationSteps.map((step) => renderEquationLatex(step)),
    [equationSteps]
  );
  const sourceExcerpt = useMemo(
    () => description.replace(/\.\.\.$/, "").trim(),
    [description]
  );

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-4">
      <div className="rounded-lg border border-white/10 bg-black/30 p-4">
        <div className="space-y-3">
          {renderedEquations.map((equation, index) => (
            <div key={`${equationSteps[index]}-${index}`} className="overflow-x-auto text-white">
              {renderedEquations.length > 1 && (
                <p className="mb-2 text-[11px] font-medium uppercase tracking-[0.14em] text-gray-500">
                  Step {index + 1}
                </p>
              )}
              <div dangerouslySetInnerHTML={{ __html: equation }} />
            </div>
          ))}
        </div>
      </div>
      {sourceExcerpt && (
        <div className="mt-3 rounded-lg border border-white/5 bg-white/[0.02] px-3 py-2">
          <p className="text-[11px] font-medium uppercase tracking-[0.14em] text-gray-500">
            Source Context
          </p>
          <p className="mt-1 text-sm leading-relaxed text-gray-400">{sourceExcerpt}</p>
        </div>
      )}
    </div>
  );
}

function splitEquationSteps(latex: string): string[] {
  const cleaned = latex.replace(/\s+/g, " ").trim();
  if (!cleaned) {
    return [];
  }

  const pieces = cleaned
    .split(/(?=(?:[A-Za-zxy~˜][A-Za-z0-9˜′'`\s]{0,14})\s*=)/g)
    .map((part) => part.replace(/^\(\d+\)\s*/, "").trim())
    .filter(Boolean);

  if (pieces.length <= 1) {
    return [cleaned];
  }

  return pieces.filter((part) => /=/.test(part));
}

function renderEquationLatex(latex: string): string {
  try {
    return katex.renderToString(latex, {
      throwOnError: false,
      displayMode: true,
    });
  } catch {
    return katex.renderToString(`\\text{${latex.replace(/[{}]/g, "")}}`, {
      throwOnError: false,
      displayMode: true,
    });
  }
}
