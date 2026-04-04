"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Activity,
  ArrowUpRight,
  Bot,
  CheckCircle2,
  ExternalLink,
  Loader2,
  Search,
} from "lucide-react";
import clsx from "clsx";
import ArchivistShell from "@/components/archivist/ArchivistShell";
import RepositoryLinksPanel from "@/components/repositories/RepositoryLinksPanel";
import { getAnalysis, getPapers } from "@/lib/api-client";
import { formatDomainLabel, normalizeDomain } from "@/lib/domain-utils";
import type {
  AnalysisChunk,
  MethodsStructure,
  Paper,
  PaperAnalysis,
  SectionSynthesisResult,
  SynthesisOutput,
} from "@/lib/types";

interface PaperWithAnalysis extends Paper {
  latestAnalysis?: PaperAnalysis | null;
}

type SectionKey = "key_ideas" | "methods" | "results" | "discussion";

const SECTION_LABELS: Record<SectionKey, string> = {
  key_ideas: "Key Ideas",
  methods: "Methods",
  results: "Results",
  discussion: "Discussion",
};

function normalizeSection(value: SynthesisOutput[keyof SynthesisOutput]): SectionSynthesisResult | null {
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
  structure: PaperAnalysis["inferred_structure"] | null | undefined,
  sectionKey: SectionKey
): AnalysisChunk[] {
  const value = structure?.[sectionKey];
  if (sectionKey === "methods" && value && !Array.isArray(value)) {
    return (value as MethodsStructure).chunks ?? [];
  }
  return Array.isArray(value) ? value : [];
}

function getAllChunks(analysis?: PaperAnalysis | null): AnalysisChunk[] {
  if (!analysis) return [];
  return (Object.keys(SECTION_LABELS) as SectionKey[]).flatMap((sectionKey) =>
    getSectionChunks(analysis.inferred_structure, sectionKey)
  );
}

function getEvidenceCount(analysis?: PaperAnalysis | null) {
  if (!analysis?.synthesis_output) return 0;
  return (Object.keys(SECTION_LABELS) as SectionKey[]).reduce((sum, key) => {
    const section = normalizeSection(analysis.synthesis_output?.[key]);
    return sum + (section?.evidence_chunk_count ?? 0);
  }, 0);
}

function getAverageChunkConfidence(analysis?: PaperAnalysis | null) {
  const chunks = getAllChunks(analysis).filter((chunk) => typeof chunk.confidence === "number");
  if (!chunks.length) return null;
  return chunks.reduce((sum, chunk) => sum + (chunk.confidence ?? 0), 0) / chunks.length;
}

function getAnalysisStatus(analysis?: PaperAnalysis | null) {
  if (!analysis) return "No analysis";
  const status = analysis.analysis_status?.status;
  if (status === "partial_failure") return "Partial review";
  if (status === "failed") return "Failed";
  return "Ready";
}

function AdminSummaryRail({
  papers,
  selectedPaper,
}: {
  papers: PaperWithAnalysis[];
  selectedPaper: PaperWithAnalysis | null;
}) {
  const analyzedCount = papers.filter((paper) => paper.latestAnalysis).length;
  const totalRepoCount = papers.reduce(
    (sum, paper) => sum + (paper.latestAnalysis?.repository_info?.repositories?.length ?? 0),
    0
  );
  const avgConfidence = useMemo(() => {
    const values = papers
      .map((paper) => getAverageChunkConfidence(paper.latestAnalysis))
      .filter((value): value is number => typeof value === "number");
    if (!values.length) return null;
    return values.reduce((sum, value) => sum + value, 0) / values.length;
  }, [papers]);

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[#3d4949]/20 p-6">
        <h2 className="text-sm font-bold text-[#e2e2e6]">QA Diagnostics</h2>
        <p className="mt-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.16em] text-[#879392]">
          Admin-Only Grounded Retrieval
        </p>
      </div>

      <div className="flex-1 space-y-6 overflow-y-auto p-4">
        <div className="rounded-sm border border-[#3d4949]/10 bg-[#333538]/30 p-4">
          <div className="mb-4 flex items-center justify-between">
            <span className="font-[family-name:var(--font-label)] text-[10px] uppercase text-[#879392]">
              Corpus Coverage
            </span>
            <span className="text-[10px] font-bold text-[#66dd8b]">
              {analyzedCount}/{papers.length}
            </span>
          </div>
          <div className="space-y-4 text-[10px]">
            <div className="flex items-center justify-between">
              <span className="text-[#909094]">Analyzed papers</span>
              <span className="font-bold text-[#e2e2e6]">{analyzedCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#909094]">Repo links</span>
              <span className="font-bold text-[#e2e2e6]">{totalRepoCount}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-[#909094]">Avg chunk confidence</span>
              <span className="font-bold text-[#e2e2e6]">
                {avgConfidence == null ? "--" : `${Math.round(avgConfidence * 100)}%`}
              </span>
            </div>
          </div>
        </div>

        {selectedPaper && (
          <div className="rounded-sm border border-[#3d4949]/10 bg-[#1e2023] p-4">
            <div className="mb-2 text-[10px] font-[family-name:var(--font-label)] uppercase tracking-[0.16em] text-[#879392]">
              Selected Paper
            </div>
            <div className="text-sm font-semibold text-[#e2e2e6]">{selectedPaper.title}</div>
            <div className="mt-2 flex flex-wrap gap-2">
              <span className="rounded-sm bg-[#334d58] px-2 py-0.5 text-[9px] font-bold uppercase text-[#a1bdca]">
                {formatDomainLabel(selectedPaper.latestAnalysis?.domain ?? selectedPaper.domain)}
              </span>
              <span className="rounded-sm bg-[#25a55a]/10 px-2 py-0.5 text-[9px] font-bold uppercase text-[#66dd8b]">
                {getAnalysisStatus(selectedPaper.latestAnalysis)}
              </span>
            </div>
            <div className="mt-4 space-y-2 text-[10px]">
              <div className="flex items-center justify-between">
                <span className="text-[#909094]">Evidence chunks</span>
                <span className="font-bold text-[#e2e2e6]">{getEvidenceCount(selectedPaper.latestAnalysis)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-[#909094]">Repository links</span>
                <span className="font-bold text-[#e2e2e6]">
                  {selectedPaper.latestAnalysis?.repository_info?.repositories?.length ?? 0}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default function AdminPage() {
  const [papers, setPapers] = useState<PaperWithAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState("");
  const [selectedPaperId, setSelectedPaperId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const response = await getPapers();
        const rows = response.items;
        const analyses = await Promise.all(
          rows.map(async (paper) => {
            try {
              const analysis = await getAnalysis(paper.id);
              return [paper.id, analysis] as const;
            } catch {
              return [paper.id, null] as const;
            }
          })
        );

        if (cancelled) return;

        const analysisMap = new Map(analyses);
        const hydrated = rows.map((paper) => ({
          ...paper,
          latestAnalysis: analysisMap.get(paper.id) ?? null,
        }));

        setPapers(hydrated);
        setSelectedPaperId(hydrated[0]?.id ?? null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load admin diagnostics");
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  const filteredPapers = useMemo(() => {
    const safeQuery = query.trim().toLowerCase();
    if (!safeQuery) return papers;
    return papers.filter((paper) => {
      const haystack = [
        paper.title,
        paper.id,
        paper.latestAnalysis?.domain,
        paper.domain,
        paper.latestAnalysis?.repository_info?.repositories?.map((repo) => repo.url).join(" "),
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();

      return haystack.includes(safeQuery);
    });
  }, [papers, query]);

  const selectedPaper = useMemo(() => {
    return filteredPapers.find((paper) => paper.id === selectedPaperId) ?? filteredPapers[0] ?? null;
  }, [filteredPapers, selectedPaperId]);

  useEffect(() => {
    if (!filteredPapers.length) {
      setSelectedPaperId(null);
      return;
    }
    if (!selectedPaperId || !filteredPapers.some((paper) => paper.id === selectedPaperId)) {
      setSelectedPaperId(filteredPapers[0].id);
    }
  }, [filteredPapers, selectedPaperId]);

  return (
    <ArchivistShell
      activeTopNav="health"
      activeDomain={selectedPaper?.latestAnalysis?.domain ?? selectedPaper?.domain ?? "general"}
      rightRail={<AdminSummaryRail papers={papers} selectedPaper={selectedPaper} />}
      searchPlaceholder="Search traces..."
    >
      <div className="p-8">
        <header className="mb-10">
          <div className="mb-4 flex items-center gap-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.16em] text-[#5dd9d8]">
            <span className="rounded bg-[#5dd9d8]/10 px-2 py-0.5">Admin Route</span>
            <span className="text-[#879392]">/</span>
            <span>Paper-Backed QA Diagnostics</span>
          </div>
          <h1 className="mb-2 text-3xl font-extrabold tracking-tight text-[#e2e2e6]">
            QA &amp; Retrieval Diagnostics
          </h1>
          <p className="max-w-3xl text-sm text-[#afcbd8]">
            Internal diagnostics only. This page uses stored paper analyses, evidence chunks, reproducibility scores,
            and all discovered repository links to help improve retrieval and answer quality.
          </p>
        </header>

        <section className="mb-8 grid grid-cols-1 gap-6 xl:grid-cols-[1.2fr_1.8fr]">
          <div className="rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-6">
            <label className="mb-3 block font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
              Filter Papers
            </label>
            <div className="relative">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#879392]" />
              <input
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search by title, id, domain, or repository..."
                className="w-full rounded-sm border border-[#3d4949]/20 bg-[#111316] py-2 pl-10 pr-4 text-sm text-[#e2e2e6] outline-none placeholder:text-[#879392] focus:border-[#5dd9d8]/40"
              />
            </div>
          </div>

          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            {[
              {
                label: "Tracked Papers",
                value: loading ? "--" : papers.length,
              },
              {
                label: "Analyses Ready",
                value: loading ? "--" : papers.filter((paper) => paper.latestAnalysis).length,
              },
              {
                label: "Visible Rows",
                value: loading ? "--" : filteredPapers.length,
              },
            ].map((metric) => (
              <div key={metric.label} className="rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-6">
                <div className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
                  {metric.label}
                </div>
                <div className="mt-2 text-3xl font-bold text-[#e2e2e6]">{metric.value}</div>
              </div>
            ))}
          </div>
        </section>

        <section className="overflow-hidden rounded-lg border border-[#3d4949]/20 bg-[#1e2023]">
          <div className="border-b border-[#3d4949]/20 bg-[#282a2d]/50 p-4">
            <div className="font-[family-name:var(--font-label)] text-xs uppercase tracking-[0.18em] text-[#879392]">
              All Papers and Diagnostics
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="bg-[#1a1c1f] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
                  <th className="px-6 py-4 font-medium">Paper</th>
                  <th className="px-4 py-4 font-medium">Status</th>
                  <th className="px-4 py-4 font-medium">Evidence</th>
                  <th className="px-4 py-4 font-medium">Chunk Conf.</th>
                  <th className="px-4 py-4 font-medium">Repos</th>
                  <th className="px-4 py-4 font-medium">Actions</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#3d4949]/10">
                {loading ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-sm text-[#879392]">
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading paper diagnostics...
                      </span>
                    </td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-sm text-[#ffb4ab]">
                      {error}
                    </td>
                  </tr>
                ) : filteredPapers.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-6 py-12 text-center text-sm text-[#879392]">
                      No papers match this filter.
                    </td>
                  </tr>
                ) : (
                  filteredPapers.map((paper) => {
                    const averageConfidence = getAverageChunkConfidence(paper.latestAnalysis);
                    const evidenceCount = getEvidenceCount(paper.latestAnalysis);
                    const repoCount = paper.latestAnalysis?.repository_info?.repositories?.length ?? 0;
                    const isSelected = selectedPaper?.id === paper.id;

                    return (
                      <tr
                        key={paper.id}
                        className={clsx(
                          "cursor-pointer transition-colors hover:bg-[#282a2d]",
                          isSelected && "bg-[#282a2d]"
                        )}
                        onClick={() => setSelectedPaperId(paper.id)}
                      >
                        <td className="px-6 py-4">
                          <div className="text-sm font-bold text-[#e2e2e6]">{paper.title}</div>
                          <div className="mt-1 flex flex-wrap items-center gap-2">
                            <span className="rounded-sm bg-[#334d58] px-1.5 py-0.5 text-[9px] font-bold uppercase text-[#a1bdca]">
                              {formatDomainLabel(paper.latestAnalysis?.domain ?? paper.domain)}
                            </span>
                            <span className="font-[family-name:var(--font-label)] text-[10px] text-[#879392]">
                              {paper.id}
                            </span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-sm text-[#e2e2e6]">{getAnalysisStatus(paper.latestAnalysis)}</td>
                        <td className="px-4 py-4 text-sm text-[#e2e2e6]">{evidenceCount}</td>
                        <td className="px-4 py-4 text-sm text-[#e2e2e6]">
                          {averageConfidence == null ? "--" : `${Math.round(averageConfidence * 100)}%`}
                        </td>
                        <td className="px-4 py-4 text-sm text-[#e2e2e6]">{repoCount}</td>
                        <td className="px-4 py-4">
                          <div className="flex flex-wrap gap-2">
                            <Link
                              href={`/papers/${paper.id}`}
                              className="inline-flex items-center gap-1 rounded-sm border border-[#3d4949]/20 px-3 py-2 text-[10px] font-[family-name:var(--font-label)] uppercase tracking-[0.16em] text-[#e2e2e6] hover:border-[#5dd9d8]/30 hover:text-[#5dd9d8]"
                              onClick={(event) => event.stopPropagation()}
                            >
                              Open
                              <ArrowUpRight className="h-3.5 w-3.5" />
                            </Link>
                            {paper.latestAnalysis?.repository_info?.repositories?.[0] && (
                              <a
                                href={paper.latestAnalysis.repository_info.repositories[0].url}
                                target="_blank"
                                rel="noreferrer"
                                className="inline-flex items-center gap-1 rounded-sm border border-[#5dd9d8]/20 px-3 py-2 text-[10px] font-[family-name:var(--font-label)] uppercase tracking-[0.16em] text-[#5dd9d8] hover:bg-[#5dd9d8]/5"
                                onClick={(event) => event.stopPropagation()}
                              >
                                <ExternalLink className="h-3.5 w-3.5" />
                                Open Repo
                              </a>
                            )}
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        {selectedPaper && (
          <section className="mt-8 space-y-6">
            <div className="rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-6">
              <div className="mb-4 flex items-start justify-between gap-4">
                <div>
                  <div className="mb-2 inline-flex items-center gap-2 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#5dd9d8]">
                    <Activity className="h-4 w-4" />
                    Selected Diagnostic View
                  </div>
                  <h2 className="text-2xl font-bold text-[#e2e2e6]">{selectedPaper.title}</h2>
                  <p className="mt-1 text-sm text-[#afcbd8]">
                    Stored retrieval-related diagnostics derived from the latest paper analysis.
                  </p>
                </div>
                <div className="flex flex-wrap gap-3">
                  <Link
                    href={`/papers/${selectedPaper.id}`}
                    className="inline-flex items-center gap-2 rounded-sm border border-[#3d4949]/20 px-4 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.16em] text-[#e2e2e6] hover:border-[#5dd9d8]/30 hover:text-[#5dd9d8]"
                  >
                    Open Paper
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </Link>
                  <Link
                    href={`/experiments?paperId=${selectedPaper.id}`}
                    className="inline-flex items-center gap-2 rounded-sm border border-[#3d4949]/20 px-4 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.16em] text-[#e2e2e6] hover:border-[#5dd9d8]/30 hover:text-[#5dd9d8]"
                  >
                    Open Experiment
                    <ArrowUpRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              </div>

              {!selectedPaper.latestAnalysis ? (
                <div className="rounded-sm border border-dashed border-[#3d4949]/20 bg-[#111316] p-6 text-sm text-[#879392]">
                  No analysis is stored for this paper yet, so there are no QA diagnostics to inspect.
                </div>
              ) : (
                <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                  <div className="space-y-4">
                    <h3 className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
                      Section Diagnostics
                    </h3>
                    {(Object.keys(SECTION_LABELS) as SectionKey[]).map((sectionKey) => {
                      const section = normalizeSection(selectedPaper.latestAnalysis?.synthesis_output?.[sectionKey]);
                      return (
                        <div key={sectionKey} className="rounded-sm border border-[#3d4949]/20 bg-[#111316] p-4">
                          <div className="mb-2 flex items-center justify-between">
                            <span className="text-sm font-semibold text-[#e2e2e6]">{SECTION_LABELS[sectionKey]}</span>
                            <span className="rounded-sm bg-[#334d58] px-2 py-0.5 text-[9px] font-bold uppercase text-[#a1bdca]">
                              {section?.confidence ?? "none"}
                            </span>
                          </div>
                          <div className="grid grid-cols-2 gap-3 text-[10px]">
                            <div className="rounded-sm bg-[#1a1c1f] p-3">
                              <div className="text-[#879392]">Evidence chunks</div>
                              <div className="mt-1 text-lg font-bold text-[#e2e2e6]">
                                {section?.evidence_chunk_count ?? 0}
                              </div>
                            </div>
                            <div className="rounded-sm bg-[#1a1c1f] p-3">
                              <div className="text-[#879392]">Review score</div>
                              <div className="mt-1 text-lg font-bold text-[#e2e2e6]">
                                {section?.review_score ?? 0}
                              </div>
                            </div>
                          </div>
                        </div>
                      );
                    })}
                  </div>

                  <div className="space-y-4">
                    <h3 className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
                      Evidence Preview
                    </h3>
                    {getAllChunks(selectedPaper.latestAnalysis)
                      .slice(0, 4)
                      .map((chunk, index) => (
                        <div key={chunk.id ?? `chunk-${index}`} className="rounded-sm border border-[#3d4949]/20 bg-[#111316] p-4">
                          <div className="mb-2 flex items-center justify-between text-[10px]">
                            <span className="text-[#879392]">
                              {chunk.section_name ?? "Unknown section"} #{chunk.chunk_index ?? index + 1}
                            </span>
                            <span className="text-[#66dd8b]">
                              {typeof chunk.confidence === "number" ? `${Math.round(chunk.confidence * 100)}%` : "--"}
                            </span>
                          </div>
                          <p className="text-sm leading-relaxed text-[#bcc9c8]">
                            {(chunk.summary ?? chunk.text ?? "No chunk text available.").slice(0, 240)}
                            {(chunk.summary ?? chunk.text ?? "").length > 240 ? "..." : ""}
                          </p>
                        </div>
                      ))}
                  </div>
                </div>
              )}
            </div>

            {selectedPaper.latestAnalysis && (
              <RepositoryLinksPanel repositories={selectedPaper.latestAnalysis.repository_info?.repositories} />
            )}
          </section>
        )}
      </div>
    </ArchivistShell>
  );
}
