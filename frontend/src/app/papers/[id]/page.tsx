"use client";

import { useEffect, useMemo, useState, use } from "react";
import Link from "next/link";
import clsx from "clsx";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import katex from "katex";
import {
  BookOpen,
  Bot,
  CheckCircle2,
  ChevronDown,
  Code2,
  Download,
  ExternalLink,
  FlaskConical,
  Loader2,
  Rocket,
  Sigma,
} from "lucide-react";
import toast from "react-hot-toast";
import ArchivistShell from "@/components/archivist/ArchivistShell";
import RepositoryLinksPanel from "@/components/repositories/RepositoryLinksPanel";
import QAChat from "@/features/qa/QAChat";
import { analyzePaper, getAnalysis, getPaper } from "@/lib/api-client";
import { formatDomainLabel, normalizeDomain } from "@/lib/domain-utils";
import type {
  AnalysisChunk,
  EquationCollection,
  MethodsStructure,
  Paper,
  PaperAnalysis,
  SectionSynthesisResult,
  SynthesisOutput,
} from "@/lib/types";

type SectionKey = "key_ideas" | "methods" | "results" | "discussion";

const SECTION_META: Record<SectionKey, string> = {
  key_ideas: "01. Overview",
  methods: "02. Methods",
  results: "03. Results",
  discussion: "04. Discussion",
};

interface PageProps {
  params: Promise<{ id: string }>;
}

type PaperTab = "overview" | "qa" | "experiment";

function normalizeSectionSynthesis(
  value: SynthesisOutput[keyof SynthesisOutput]
): SectionSynthesisResult | null {
  if (!value) return null;
  if (typeof value === "string") {
    return {
      synthesis: value,
      confidence: "high",
      warning: null,
      fabrication_flagged: false,
      retrieval_rounds: 0,
      rewrite_rounds: 0,
      review_score: 0,
      review_issues: [],
      evidence_chunk_count: 0,
    };
  }
  return value;
}

function getSectionChunks(
  structure: PaperAnalysis["inferred_structure"],
  sectionKey: SectionKey
): AnalysisChunk[] {
  const sectionValue = structure?.[sectionKey];
  if (sectionKey === "methods" && sectionValue && !Array.isArray(sectionValue)) {
    return (sectionValue as MethodsStructure).chunks ?? [];
  }
  return Array.isArray(sectionValue) ? sectionValue : [];
}

function getMethodsEquations(
  structure: PaperAnalysis["inferred_structure"]
): EquationCollection | null {
  const methods = structure?.methods;
  if (!methods || Array.isArray(methods)) return null;
  return (methods as MethodsStructure).equations ?? null;
}

function formatConfidence(confidence?: number | null) {
  return `${Math.round((confidence ?? 0) * 100)}%`;
}

function formatReproScore(analysis?: PaperAnalysis | null) {
  const raw = analysis?.reproducibility?.overall_score;
  if (typeof raw !== "number") return null;
  return raw <= 1 ? raw * 10 : raw;
}

function splitEquationSteps(latex: string): string[] {
  const cleaned = latex.replace(/\s+/g, " ").trim();
  if (!cleaned) return [];

  const pieces = cleaned
    .split(/(?=(?:[A-Za-zxy~][A-Za-z0-9~'`\s]{0,14})\s*=)/g)
    .map((part) => part.replace(/^\(\d+\)\s*/, "").trim())
    .filter(Boolean);

  return pieces.length > 1 ? pieces.filter((part) => /=/.test(part)) : [cleaned];
}

function renderEquationLatex(latex: string) {
  try {
    return katex.renderToString(latex, { throwOnError: false, displayMode: true });
  } catch {
    return katex.renderToString(`\\text{${latex.replace(/[{}]/g, "")}}`, {
      throwOnError: false,
      displayMode: true,
    });
  }
}

function EvidenceCard({ chunk }: { chunk: AnalysisChunk }) {
  const [expanded, setExpanded] = useState(false);
  const text = (chunk.text ?? chunk.summary ?? "").trim();
  const summary = (chunk.summary ?? text).trim();

  return (
    <div className="rounded-sm border border-[#3d4949]/20 bg-[#282a2d]/50 p-3">
      <button
        onClick={() => setExpanded((current) => !current)}
        className="flex w-full cursor-pointer items-center justify-between gap-3 text-left"
      >
        <div>
          <div className="mb-1 flex flex-wrap items-center gap-2 font-[family-name:var(--font-label)] text-[9px] uppercase tracking-[0.14em] text-[#879392]">
            <span>{chunk.section_name ?? "Unknown Section"}</span>
            {chunk.chunk_index != null && <span>Chunk #{chunk.chunk_index}</span>}
            <span className="text-[#66dd8b]">{Math.round((chunk.confidence ?? 0) * 100)}% conf</span>
          </div>
          <p className="text-xs leading-relaxed text-[#bcc9c8]">{expanded ? text : summary}</p>
        </div>
        <ChevronDown className={clsx("h-4 w-4 shrink-0 text-[#879392] transition-transform", expanded && "rotate-180")} />
      </button>
    </div>
  );
}

function SectionBlock({
  title,
  result,
  chunks,
  equations,
}: {
  title: string;
  result: SectionSynthesisResult | null;
  chunks: AnalysisChunk[];
  equations?: EquationCollection | null;
}) {
  const [showEvidence, setShowEvidence] = useState(false);
  const [showEquations, setShowEquations] = useState(false);
  const equationItems = equations?.items?.filter((item) => (item.latex ?? "").trim().length > 0) ?? [];

  if (!result?.synthesis && chunks.length === 0) {
    return null;
  }

  return (
    <section className="group">
      <div className="mb-4 flex items-center justify-between border-b border-[#3d4949]/10 pb-2">
        <h2 className="font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.2em] text-[#afcbd8]">
          {title}
        </h2>
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="h-1 w-16 overflow-hidden bg-[#333538]">
              <div
                className={clsx(
                  "h-full",
                  result?.confidence === "low"
                    ? "bg-[#ffb4ab]"
                    : result?.confidence === "medium"
                      ? "bg-[#afcbd8]"
                      : "bg-[#66dd8b]"
                )}
                style={{ width: `${Math.max(20, (result?.review_score ?? 0) * 10)}%` }}
              />
            </div>
            <span className="font-[family-name:var(--font-label)] text-[10px] text-[#bcc9c8]">
              {result?.confidence?.toUpperCase() ?? "LOW"}
            </span>
          </div>
        </div>
      </div>

      <div className="rounded-sm border-l-2 border-[#5dd9d8]/20 bg-[#1a1c1f] p-6">
        {result?.fabrication_flagged && (
          <div className="mb-4 rounded-sm border border-[#93000a] bg-[#93000a]/20 p-3 text-sm text-[#ffdad6]">
            This section may contain unsupported claims. Verify it against the source evidence.
          </div>
        )}
        {result?.warning && (
          <div className="mb-4 rounded-sm border border-[#5dd9d8]/10 bg-[#333538]/40 p-3 text-sm text-[#bcc9c8]">
            {result.warning}
          </div>
        )}

        {result?.synthesis && (
          <div className="prose prose-invert max-w-none text-sm prose-p:text-[#bcc9c8] prose-strong:text-[#e2e2e6] prose-li:text-[#bcc9c8] prose-th:text-[#e2e2e6] prose-td:text-[#bcc9c8]">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                table: ({ children }) => (
                  <div className="overflow-x-auto">
                    <table className="min-w-full border-collapse text-sm">{children}</table>
                  </div>
                ),
                thead: ({ children }) => <thead className="border-b border-[#3d4949]/20">{children}</thead>,
                tr: ({ children }) => <tr className="border-b border-[#3d4949]/10">{children}</tr>,
                th: ({ children }) => <th className="px-3 py-2 text-left">{children}</th>,
                td: ({ children }) => <td className="px-3 py-2 align-top">{children}</td>,
              }}
            >
              {result.synthesis}
            </ReactMarkdown>
          </div>
        )}

        <div className="mt-6 flex flex-wrap gap-4">
          <div className="rounded-sm bg-[#333538] px-3 py-1.5">
            <span className="mb-0.5 block font-[family-name:var(--font-label)] text-[10px] uppercase text-[#afcbd8]">
              Retrieved
            </span>
            <span className="text-xs font-bold text-[#e2e2e6]">{result?.retrieval_rounds ?? 0} rounds</span>
          </div>
          <div className="rounded-sm bg-[#333538] px-3 py-1.5">
            <span className="mb-0.5 block font-[family-name:var(--font-label)] text-[10px] uppercase text-[#afcbd8]">
              Synthesized
            </span>
            <span className="text-xs font-bold text-[#e2e2e6]">{result?.rewrite_rounds ?? 0} rewrites</span>
          </div>
          <div className="rounded-sm bg-[#333538] px-3 py-1.5">
            <span className="mb-0.5 block font-[family-name:var(--font-label)] text-[10px] uppercase text-[#afcbd8]">
              Evidence
            </span>
            <span className="text-xs font-bold text-[#e2e2e6]">{result?.evidence_chunk_count ?? chunks.length} chunks</span>
          </div>
        </div>

        {equationItems.length > 0 && (
          <div className="mt-6">
            <button
              onClick={() => setShowEquations((current) => !current)}
              className="inline-flex cursor-pointer items-center gap-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#afcbd8] hover:text-[#e2e2e6]"
            >
              <Sigma className="h-3.5 w-3.5" />
              View Equations
              <ChevronDown className={clsx("h-4 w-4 transition-transform", showEquations && "rotate-180")} />
            </button>

            {showEquations && (
              <div className="mt-4 space-y-4 rounded-sm border border-[#3d4949]/20 bg-[#282a2d]/40 p-4">
                {equations?.source === "llm_generated" && (
                  <div className="rounded-sm border border-[#ffb4ab]/20 bg-[#93000a]/10 p-3 text-sm text-[#ffdad6]">
                    These equations were inferred by AI because no direct equations were extracted.
                    Verify them against the original paper before use.
                  </div>
                )}
                {equationItems.map((item, index) => {
                  const steps = splitEquationSteps((item.latex ?? "").trim());
                  return (
                    <div key={item.id ?? `equation-${index}`} className="rounded-sm border border-[#3d4949]/20 bg-[#111316] p-4">
                      <div className="space-y-3">
                        {steps.map((step, stepIndex) => (
                          <div key={`${stepIndex}-${step}`} className="overflow-x-auto text-[#5dd9d8]">
                            <div dangerouslySetInnerHTML={{ __html: renderEquationLatex(step) }} />
                          </div>
                        ))}
                      </div>
                      {item.description && (
                        <p className="mt-3 text-sm leading-relaxed text-[#bcc9c8]">{item.description}</p>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

        {chunks.length > 0 && (
          <details className="group/evidence mt-6">
            <summary
              onClick={(event) => {
                event.preventDefault();
                setShowEvidence((current) => !current);
              }}
              className="list-none cursor-pointer font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#5dd9d8]"
            >
              <span className="inline-flex items-center gap-2">
                <ChevronDown className={clsx("h-4 w-4 transition-transform", showEvidence && "rotate-180")} />
                Expand Raw Evidence Base
              </span>
            </summary>
            {showEvidence && (
              <div className="mt-4 space-y-2">
                {chunks.map((chunk, index) => (
                  <EvidenceCard key={chunk.id ?? `${title}-chunk-${index}`} chunk={chunk} />
                ))}
              </div>
            )}
          </details>
        )}
      </div>
    </section>
  );
}

function PaperSignalsRail({
  paper,
  analysis,
  loadState,
  onRunAnalysis,
  running,
}: {
  paper: Paper | null;
  analysis: PaperAnalysis | null;
  loadState: "loading" | "idle" | "ready" | "error" | "running";
  onRunAnalysis: () => void;
  running: boolean;
}) {
  const repro = formatReproScore(analysis);
  const repo = analysis?.repository_info?.primary_repo;
  const overview = normalizeSectionSynthesis(analysis?.synthesis_output?.key_ideas);
  const evidenceChunks = getSectionChunks(analysis?.inferred_structure, "key_ideas").slice(0, 2);
  const queryPreview =
    paper?.title != null
      ? `What should we know first about "${paper.title}"?`
      : "Run analysis to inspect this paper.";
  const apaCitation = paper
    ? `${paper.title}. (${new Date(paper.created_at).getFullYear()}). Digital Archivist paper record ${paper.id}.`
    : null;

  async function handleCopyCitation() {
    if (!apaCitation) return;
    try {
      await navigator.clipboard.writeText(apaCitation);
      toast.success("APA citation copied");
    } catch {
      toast.error("Unable to copy citation");
    }
  }

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[#3d4949]/20 p-4">
        <div className="flex items-center gap-2">
          <Bot className="h-5 w-5 text-[#5dd9d8]" />
          <h2 className="font-[family-name:var(--font-label)] text-sm font-bold uppercase text-[#e2e2e6]">
            QA Diagnostics
          </h2>
        </div>
        <div className="mt-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
          Grounded Retrieval
        </div>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto p-4">
        <div className="space-y-1 border-b border-[#3d4949]/10 pb-4">
          <div className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
            Active Model
          </div>
          <div className="text-[11px] font-medium text-[#5dd9d8]">Grounded Retrieval Engine v2.4</div>
          <div className="text-[10px] text-[#bcc9c8]">
            {analysis?.analysis_status?.status === "success" ? "Verified extraction" : "Awaiting refreshed pass"}
          </div>
        </div>

        <div className="space-y-3">
          <div className="rounded-sm border border-[#3d4949]/20 bg-[#1e2023] p-3">
            <div className="mb-1 font-[family-name:var(--font-label)] text-[10px] uppercase text-[#afcbd8]">
              User Query
            </div>
            <p className="text-xs leading-relaxed text-[#e2e2e6]">{queryPreview}</p>
          </div>
          <div className="border-l border-[#5dd9d8]/30 pl-4">
            <div className="mb-1 font-[family-name:var(--font-label)] text-[10px] uppercase text-[#5dd9d8]">
              System Response
            </div>
            <p className="text-xs leading-relaxed text-[#bcc9c8]">
              {overview?.synthesis?.slice(0, 180) ?? "Structured synthesis will appear here once analysis completes."}
              {overview?.synthesis && overview.synthesis.length > 180 ? "..." : ""}
            </p>
          </div>
        </div>

        <div className="rounded-sm border border-[#3d4949]/20 bg-[#1e2023] p-4">
          <div className="mb-3 text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
            Retrieval Metadata
          </div>
          <div className="mb-3 flex items-center justify-between text-[9px] font-[family-name:var(--font-label)] uppercase text-[#afcbd8]">
            <span>Retrieved Chunks</span>
            <span className="text-[#5dd9d8]">
              Top Similarity {formatConfidence(analysis?.domain_confidence ?? paper?.domain_confidence)}
            </span>
          </div>
          <div className="space-y-2">
            {evidenceChunks.length > 0 ? (
              evidenceChunks.map((chunk, index) => (
                <div key={chunk.id ?? `evidence-${index}`} className="rounded-sm bg-[#333538]/40 p-2">
                  <div className="mb-1 flex items-center justify-between text-[10px]">
                    <span className="text-[#bcc9c8]">#{chunk.chunk_index ?? index + 1}</span>
                    <span className="text-[#66dd8b]">
                      Rerank {(chunk.confidence ?? analysis?.domain_confidence ?? 0.9).toFixed(2)}
                    </span>
                  </div>
                  <p className="line-clamp-3 text-[10px] leading-relaxed text-[#879392]">
                    {chunk.summary ?? chunk.text ?? "Evidence available after extraction."}
                  </p>
                </div>
              ))
            ) : (
              <div className="rounded-sm bg-[#333538]/40 p-3 text-[10px] text-[#879392]">
                Evidence chunks will appear here after analysis.
              </div>
            )}
          </div>
        </div>

        <div className="space-y-4 border-t border-[#3d4949]/10 pt-6">
          <h3 className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#afcbd8]">
            External Context
          </h3>
          <div className="rounded-sm border border-[#3d4949]/20 bg-[#282a2d] p-4 transition-colors hover:border-[#5dd9d8]/30">
            <div className="mb-3 flex items-center justify-between">
              <Code2 className="h-4 w-4 text-[#5dd9d8]" />
              <span className="rounded-sm bg-[#25a55a]/10 px-1.5 py-0.5 text-[10px] font-bold text-[#66dd8b]">
                {repo ? "MATCH" : "PENDING"}
              </span>
            </div>
            <div className="truncate text-xs font-bold text-[#e2e2e6]">
              {repo ? repo.replace("https://", "") : "No repository discovered"}
            </div>
            <div className="mt-1 text-[10px] text-[#afcbd8]">
              {repo ? "Discovered repository" : "Repository discovery pending"}
            </div>
          </div>

          <div className="rounded-sm border border-[#3d4949]/20 bg-[#282a2d] p-4">
            <div className="mb-2 flex items-center justify-between">
              <span className="text-[10px] font-[family-name:var(--font-label)] uppercase text-[#afcbd8]">
                Reproducibility
              </span>
              <span className="text-sm font-bold text-[#5dd9d8]">
                {repro == null ? "--" : `${repro.toFixed(1)}/10`}
              </span>
            </div>
            <div className="h-1 w-full overflow-hidden rounded-full bg-[#333538]">
              <div
                className="h-full bg-[#5dd9d8]"
                style={{ width: `${repro == null ? 0 : Math.min(100, repro * 10)}%` }}
              />
            </div>
            <div className="mt-3 flex items-center gap-2 text-[10px] text-[#bcc9c8]">
              <CheckCircle2 className="h-3.5 w-3.5 text-[#66dd8b]" />
              <span>
                {analysis?.analysis_status?.status === "success"
                  ? "Experiment scaffold ready"
                  : "Run analysis to unlock scaffold generation"}
              </span>
            </div>
          </div>

          <button
            onClick={onRunAnalysis}
            disabled={running}
            className="inline-flex cursor-pointer items-center gap-2 rounded-sm border border-[#5dd9d8]/30 px-3 py-2 text-[10px] font-[family-name:var(--font-label)] uppercase tracking-[0.18em] text-[#5dd9d8] disabled:opacity-50 hover:bg-[#5dd9d8]/5"
          >
            {loadState === "running" ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Rocket className="h-3.5 w-3.5" />}
            {analysis ? "Re-run Analysis" : "Run Analysis"}
          </button>
        </div>
      </div>

      <div className="mt-auto border-t border-[#3d4949]/20 bg-[#0c0e11] p-4">
        <div className="flex gap-2">
          {paper?.source_url ? (
            <a
              href={paper.source_url}
              target="_blank"
              rel="noreferrer"
              className="flex flex-1 items-center justify-center gap-2 border border-[#3d4949]/20 bg-[#333538] py-2 text-[10px] font-[family-name:var(--font-label)] uppercase font-bold text-[#e2e2e6]"
            >
              <Download className="h-3.5 w-3.5" />
              Download PDF
            </a>
          ) : (
            <button
              type="button"
              disabled
              className="flex flex-1 items-center justify-center gap-2 border border-[#3d4949]/20 bg-[#333538] py-2 text-[10px] font-[family-name:var(--font-label)] uppercase font-bold text-[#879392]"
            >
              <Download className="h-3.5 w-3.5" />
              Download PDF
            </button>
          )}
          <button
            onClick={handleCopyCitation}
            className="flex flex-1 cursor-pointer items-center justify-center gap-2 border border-[#3d4949]/20 bg-[#333538] py-2 text-[10px] font-[family-name:var(--font-label)] uppercase font-bold text-[#e2e2e6]"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Cite APA
          </button>
          <Link
            href={paper ? `/experiments?paperId=${paper.id}` : "/experiments"}
            className="flex flex-1 items-center justify-center gap-2 border border-[#3d4949]/20 bg-[#333538] py-2 text-[10px] font-[family-name:var(--font-label)] uppercase font-bold text-[#e2e2e6]"
          >
            <ExternalLink className="h-3.5 w-3.5" />
            Open Lab
          </Link>
        </div>
      </div>
    </div>
  );
}

export default function PaperPage({ params }: PageProps) {
  const { id } = use(params);
  const [paper, setPaper] = useState<Paper | null>(null);
  const [analysis, setAnalysis] = useState<PaperAnalysis | null>(null);
  const [loadState, setLoadState] = useState<"loading" | "idle" | "ready" | "error" | "running">("loading");
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<PaperTab>("overview");

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoadState("loading");
      setError(null);
      try {
        const [paperData, analysisData] = await Promise.all([
          getPaper(id),
          getAnalysis(id).catch(() => null),
        ]);

        if (cancelled) return;

        setPaper(paperData);
        if (analysisData) {
          setAnalysis(analysisData);
          setLoadState("ready");
        } else {
          setLoadState("idle");
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load paper");
          setLoadState("error");
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, [id]);

  async function handleRunAnalysis() {
    setLoadState("running");
    try {
      const data = await analyzePaper(id);
      setAnalysis(data);
      setLoadState("ready");
      toast.success("Analysis complete");
    } catch (runError) {
      const message = runError instanceof Error ? runError.message : "Analysis failed";
      setError(message);
      setLoadState("error");
      toast.error(message);
    }
  }

  const normalizedDomain = normalizeDomain(analysis?.domain ?? paper?.domain);
  const sections = useMemo(() => {
    if (!analysis) return [];

    return (Object.keys(SECTION_META) as SectionKey[]).map((sectionKey) => ({
      sectionKey,
      title: SECTION_META[sectionKey],
      result: normalizeSectionSynthesis(analysis.synthesis_output?.[sectionKey]),
      chunks: getSectionChunks(analysis.inferred_structure, sectionKey),
      equations: sectionKey === "methods" ? getMethodsEquations(analysis.inferred_structure) : null,
    }));
  }, [analysis]);

  return (
    <ArchivistShell
      activeTopNav="pipeline"
      activeDomain={normalizedDomain}
      searchPlaceholder="Search archives..."
    >
      <div className="mx-auto max-w-5xl p-8">
        <header className="mb-10">
          <div className="flex items-start justify-between gap-6">
            <div>
              <div className="mb-2 flex items-center gap-3">
                <span className="bg-[#5dd9d8]/10 px-2 py-0.5 font-[family-name:var(--font-label)] text-xs font-semibold tracking-widest text-[#5dd9d8]">
                  {paper?.id.slice(0, 12).toUpperCase() ?? "PAPER"}
                </span>
                {analysis && (
                  <div className="flex items-center gap-1.5 text-[#66dd8b]">
                    <CheckCircle2 className="h-3.5 w-3.5" />
                    <span className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.14em]">
                      Verified Extraction
                    </span>
                  </div>
                )}
              </div>
              <h1 className="text-3xl font-extrabold leading-tight tracking-tight text-[#e2e2e6]">
                {paper?.title ?? "Loading paper..."}
              </h1>
            </div>
            <div className="text-right">
              <div className="mb-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#afcbd8]">
                Domain Confidence
              </div>
              <div className="text-2xl font-bold text-[#5dd9d8]">
                {formatConfidence(analysis?.domain_confidence ?? paper?.domain_confidence)}
              </div>
              <div className="font-[family-name:var(--font-label)] text-[10px] uppercase text-[#bcc9c8]">
                {formatDomainLabel(normalizedDomain)}
              </div>
            </div>
          </div>
          <div className="mt-6 flex items-center gap-1 border-b border-[#3d4949]/20 pb-3">
            {[
              { key: "overview" as const, label: "Overview", icon: BookOpen },
              { key: "qa" as const, label: "Q&A", icon: Bot },
              { key: "experiment" as const, label: "Experiment", icon: FlaskConical },
            ].map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.key}
                  onClick={() => setActiveTab(tab.key)}
                  className={clsx(
                    "inline-flex cursor-pointer items-center gap-2 rounded-sm px-4 py-2 text-sm transition-colors",
                    activeTab === tab.key
                      ? "border-b-2 border-[#e2e2e6] text-[#e2e2e6]"
                      : "text-[#879392] hover:text-[#e2e2e6]"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  {tab.label}
                </button>
              );
            })}
          </div>
        </header>

        {loadState === "loading" && (
          <div className="flex min-h-[40vh] items-center justify-center">
            <div className="inline-flex items-center gap-3 text-[#879392]">
              <Loader2 className="h-5 w-5 animate-spin" />
              Loading paper analysis...
            </div>
          </div>
        )}

        {loadState === "idle" && (
          <div className="rounded-sm border border-[#3d4949]/20 bg-[#1a1c1f] p-8">
            <h2 className="mb-2 text-xl font-bold text-[#e2e2e6]">No analysis stored yet</h2>
            <p className="mb-6 text-sm leading-relaxed text-[#bcc9c8]">
              This paper is uploaded, but the structured analysis has not been generated yet.
            </p>
            <button
              onClick={handleRunAnalysis}
              className="inline-flex cursor-pointer items-center gap-2 rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] px-5 py-3 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.18em] text-[#002f2f]"
            >
              <Rocket className="h-4 w-4" />
              Run Analysis
            </button>
          </div>
        )}

        {loadState === "error" && (
          <div className="rounded-sm border border-[#93000a] bg-[#93000a]/10 p-6 text-[#ffdad6]">
            {error ?? "Something went wrong while loading this paper."}
          </div>
        )}

        {loadState === "running" && (
          <div className="rounded-sm border border-[#3d4949]/20 bg-[#1a1c1f] p-8">
            <div className="inline-flex items-center gap-3 text-[#5dd9d8]">
              <Loader2 className="h-5 w-5 animate-spin" />
              Recomputing sections, synthesis, and reproducibility...
            </div>
          </div>
        )}

        {loadState === "ready" && analysis && activeTab === "overview" && (
          <div className="space-y-12">
            <RepositoryLinksPanel repositories={analysis.repository_info?.repositories} />

            {sections.map((section) => (
              <SectionBlock
                key={section.sectionKey}
                title={section.title}
                result={section.result}
                chunks={section.chunks}
                equations={section.equations}
              />
            ))}

            {normalizedDomain === "ml" && (
              <div className="relative overflow-hidden rounded-lg border border-[#5dd9d8]/10 bg-[#333538]/30 p-8">
                <div className="mb-6 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.2em] text-[#5dd9d8]">
                  ML Systems Metadata
                </div>
                <div className="grid grid-cols-1 gap-8 md:grid-cols-3">
                  <div>
                    <span className="mb-2 block font-[family-name:var(--font-label)] text-[9px] uppercase tracking-[0.18em] text-[#afcbd8]">
                      Model Architecture
                    </span>
                    <div className="text-xl font-bold text-[#e2e2e6]">
                      {analysis.model_architecture ?? analysis.ml_details?.model_architecture ?? "Not inferred"}
                    </div>
                  </div>
                  <div>
                    <span className="mb-2 block font-[family-name:var(--font-label)] text-[9px] uppercase tracking-[0.18em] text-[#afcbd8]">
                      Dataset Source
                    </span>
                    <div className="text-xl font-bold text-[#e2e2e6]">
                      {analysis.dataset ?? analysis.ml_details?.datasets?.[0] ?? "Not inferred"}
                    </div>
                  </div>
                  <div>
                    <span className="mb-2 block font-[family-name:var(--font-label)] text-[9px] uppercase tracking-[0.18em] text-[#afcbd8]">
                      Training Objective
                    </span>
                    <div className="text-xl font-bold text-[#e2e2e6]">
                      {analysis.training_objective ?? analysis.ml_details?.training_objective ?? "Not inferred"}
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {paper && activeTab === "qa" && (
          <div className="overflow-hidden rounded-2xl border border-[#3d4949]/20 bg-[#1a1c1f]">
            <div className="border-b border-[#3d4949]/20 px-6 py-4">
              <div className="text-lg font-semibold text-[#e2e2e6]">Paper Q&amp;A</div>
              <div className="mt-1 text-sm text-[#879392]">
                Ask focused questions about this paper and get grounded answers with cited sources.
              </div>
            </div>
            <div className="h-[70vh] min-h-[540px]">
              <QAChat paperId={paper.id} domain={normalizedDomain} />
            </div>
          </div>
        )}

        {paper && activeTab === "experiment" && (
          <div className="rounded-2xl border border-[#3d4949]/20 bg-[#1a1c1f] p-8">
            <div className="mb-4 inline-flex items-center gap-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#5dd9d8]">
              <FlaskConical className="h-4 w-4" />
              Experiment Path
            </div>
            <h2 className="mb-2 text-2xl font-bold text-[#e2e2e6]">Open the Experiment Lab for this paper</h2>
            <p className="mb-6 max-w-2xl text-sm leading-relaxed text-[#bcc9c8]">
              Reproducibility scoring, repository support, and scaffold generation live in the experiment workspace.
            </p>
            <div className="flex flex-wrap gap-4">
              <Link
                href={`/experiments?paperId=${paper.id}`}
                className="inline-flex items-center gap-2 rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] px-5 py-3 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.16em] text-[#002f2f]"
              >
                <FlaskConical className="h-4 w-4" />
                Open Experiment Lab
              </Link>
              <button
                onClick={() => setActiveTab("overview")}
                className="inline-flex items-center gap-2 rounded-sm border border-[#3d4949]/30 px-5 py-3 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.16em] text-[#e2e2e6]"
              >
                Back to Overview
              </button>
            </div>
            {analysis && (
              <RepositoryLinksPanel
                repositories={analysis.repository_info?.repositories}
                title="Repository Links"
                className="mt-8"
              />
            )}
          </div>
        )}
      </div>

      {analysis && activeTab === "overview" && (
        <div className="fixed bottom-6 left-1/2 z-50 flex -translate-x-1/2 items-center gap-1 rounded-lg border border-[#3d4949]/30 bg-[#37393d]/90 p-1.5 shadow-2xl backdrop-blur-md">
          <button className="rounded-sm px-4 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.14em] transition-colors hover:bg-[#333538]">
            Compare
          </button>
          <div className="mx-1 h-4 w-px bg-[#3d4949]/40" />
          <button className="rounded-sm px-4 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.14em] transition-colors hover:bg-[#333538]">
            Annotate
          </button>
          <div className="mx-1 h-4 w-px bg-[#3d4949]/40" />
          <Link
            href={`/experiments?paperId=${id}`}
            className="inline-flex items-center gap-2 rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] px-4 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.14em] text-[#002f2f]"
          >
            <FlaskConical className="h-3.5 w-3.5" />
            Execute Scaffold
          </Link>
        </div>
      )}
    </ArchivistShell>
  );
}
