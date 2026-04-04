"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import clsx from "clsx";
import {
  CheckCircle2,
  Code2,
  Database,
  FlaskConical,
  FolderCode,
  Loader2,
  MonitorUp,
  Rocket,
} from "lucide-react";
import toast from "react-hot-toast";
import ArchivistShell from "@/components/archivist/ArchivistShell";
import RepositoryLinksPanel from "@/components/repositories/RepositoryLinksPanel";
import { generateExperiment, getAnalysis, getPapers } from "@/lib/api-client";
import { formatDomainLabel, normalizeDomain } from "@/lib/domain-utils";
import type { AnalysisChunk, ExperimentResult, Paper, PaperAnalysis } from "@/lib/types";

function toHundredScale(value?: number | null) {
  if (typeof value !== "number") return null;
  return value <= 1 ? value * 100 : value;
}

function toTenScale(value?: number | null) {
  if (typeof value !== "number") return null;
  return value <= 1 ? value * 10 : value;
}

function getPreviewChunks(analysis: PaperAnalysis | null): AnalysisChunk[] {
  if (!analysis?.inferred_structure) return [];

  const sections = [
    analysis.inferred_structure.key_ideas,
    Array.isArray(analysis.inferred_structure.methods)
      ? analysis.inferred_structure.methods
      : analysis.inferred_structure.methods?.chunks,
    analysis.inferred_structure.results,
  ];

  return sections.flatMap((section) => section ?? []).slice(0, 2);
}

function ExperimentDiagnosticsRail({
  papers,
  selectedPaperId,
  analysis,
  experiment,
}: {
  papers: Paper[];
  selectedPaperId: string | null;
  analysis: PaperAnalysis | null;
  experiment: ExperimentResult | null;
}) {
  const previewChunks = getPreviewChunks(analysis);
  const analysisConfidence =
    typeof analysis?.reproducibility?.methodology_completeness === "number"
      ? toHundredScale(analysis.reproducibility.methodology_completeness)
      : null;

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-[#3d4949]/20 p-6">
        <h2 className="text-sm font-bold text-[#e2e2e6]">QA Diagnostics</h2>
        <p className="mt-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.16em] text-[#879392]">
          Grounded Retrieval
        </p>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        <div className="rounded-sm border border-[#3d4949]/20 bg-[#1e2023] p-4">
          <div className="mb-3 flex items-center justify-between">
            <span className="font-[family-name:var(--font-label)] text-[10px] uppercase text-[#879392]">
              Active Model
            </span>
            <span className="text-[10px] font-bold text-[#66dd8b]">
              {experiment?.generation_status === "completed" ? "Stable" : "Watching"}
            </span>
          </div>
          <div className="text-[11px] font-medium text-[#5dd9d8]">Grounded Retrieval Engine v2.4</div>
          <div className="mt-3 space-y-3">
            <div className="flex items-center justify-between text-[10px]">
              <span className="text-[#879392]">Methodology Analysis</span>
              <span className="text-[#66dd8b]">
                {analysisConfidence == null ? "--" : analysisConfidence.toFixed(0)}%
              </span>
            </div>
            <div className="h-1 w-full overflow-hidden rounded-full bg-[#333538]">
              <div className="h-full bg-[#66dd8b]" style={{ width: `${analysisConfidence ?? 0}%` }} />
            </div>
          </div>
        </div>

        <div className="space-y-3 border-t border-[#3d4949]/10 pt-4">
          <h3 className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#afcbd8]">
            Extraction Ranks
          </h3>
          {previewChunks.length > 0 ? (
            previewChunks.map((chunk, index) => (
              <div key={chunk.id ?? `rail-chunk-${index}`} className="rounded-sm bg-[#333538]/40 p-3">
                <div className="mb-2 flex items-center justify-between">
                  <span className="text-[10px] uppercase text-[#879392]">Extraction Rank #{index + 1}</span>
                  <span className="text-xs font-bold text-[#66dd8b]">
                    {(chunk.confidence ?? 0.9).toFixed(2)}
                  </span>
                </div>
                <p className="text-[11px] italic leading-relaxed text-[#afcbd8]">
                  "{(chunk.summary ?? chunk.text ?? "Structured extraction available.").slice(0, 110)}
                  {(chunk.summary ?? chunk.text ?? "").length > 110 ? "..." : ""}"
                </p>
              </div>
            ))
          ) : (
            <div className="rounded-sm bg-[#333538]/40 p-3 text-[10px] text-[#879392]">
              Analysis evidence will appear here after extraction.
            </div>
          )}
        </div>

        <div className="space-y-2 border-t border-[#3d4949]/10 pt-4">
          <div className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
            Paper Context
          </div>
        {papers.map((paper) => (
          <Link
            key={paper.id}
            href={`/experiments?paperId=${paper.id}`}
            className={clsx(
              "block rounded-sm border p-3 transition-colors",
              paper.id === selectedPaperId
                ? "border-[#5dd9d8]/30 bg-[#333538] text-[#e2e2e6]"
                : "border-[#3d4949]/20 bg-[#1e2023] text-[#879392] hover:border-[#5dd9d8]/20 hover:text-[#e2e2e6]"
            )}
          >
            <div className="text-xs font-semibold">{paper.title}</div>
            <div className="mt-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.14em]">
              {paper.id.slice(0, 12)}
            </div>
          </Link>
        ))}
        </div>
      </div>

      <div className="border-t border-[#3d4949]/20 bg-[#1a1c1f] p-4">
        <div className="flex items-center justify-between text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
          <span>Total Grounding Chunks</span>
          <span className="font-bold text-[#e2e2e6]">{previewChunks.length || "--"}</span>
        </div>
        <div className="mt-2 flex items-center justify-between text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
          <span>Source Fidelity</span>
          <span className="font-bold text-[#66dd8b]">
            {experiment?.validation?.errors?.length ? "Needs review" : "Excellent"}
          </span>
        </div>
      </div>
    </div>
  );
}

function ExperimentLabPage() {
  const searchParams = useSearchParams();
  const requestedPaperId = searchParams.get("paperId");
  const [papers, setPapers] = useState<Paper[]>([]);
  const [selectedPaper, setSelectedPaper] = useState<Paper | null>(null);
  const [analysis, setAnalysis] = useState<PaperAnalysis | null>(null);
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      setLoading(true);
      setError(null);
      try {
        const paperResponse = await getPapers();
        const items = paperResponse.items;
        const fallbackPaper = items.find((paper) => paper.id === requestedPaperId) ?? items[0] ?? null;
        if (!fallbackPaper) {
          if (!cancelled) {
            setPapers([]);
            setSelectedPaper(null);
            setAnalysis(null);
          }
          return;
        }

        const analysisData = await getAnalysis(fallbackPaper.id).catch(() => null);
        if (!cancelled) {
          setPapers(items);
          setSelectedPaper(fallbackPaper);
          setAnalysis(analysisData);
          setExperiment(null);
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load experiment lab");
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
  }, [requestedPaperId]);

  async function handleGenerate() {
    if (!selectedPaper) return;
    setGenerating(true);
    try {
      const result = await generateExperiment(selectedPaper.id, analysis?.domain ?? undefined);
      setExperiment(result);
      if (result.generation_status === "completed") {
        toast.success("Experiment scaffold generated");
      } else {
        toast.error(result.error_message ?? "Scaffold generation failed");
      }
    } catch (generationError) {
      toast.error(generationError instanceof Error ? generationError.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }

  const repro = analysis?.reproducibility ?? null;
  const overallScore = toHundredScale(repro?.overall_score);
  const scaledReproScore = toTenScale(repro?.overall_score);
  const activeDomain = normalizeDomain(analysis?.domain ?? selectedPaper?.domain);

  const artifactRows = useMemo(() => {
    if (!analysis) return [];
    const primaryRepo = analysis.repository_info?.primary_repo;
    return [
      {
        icon: FolderCode,
        label: "Source Repository",
        value: primaryRepo ? primaryRepo.replace("https://", "") : "No primary repo linked",
        status: primaryRepo ? "available" : "missing",
      },
      {
        icon: Database,
        label: "Dataset Availability",
        value:
          repro?.dataset_available == null
            ? "Unknown"
            : repro.dataset_available
              ? "Public or discoverable"
              : "Not confirmed",
        status: repro?.dataset_available ? "available" : "pending",
      },
      {
        icon: MonitorUp,
        label: "Code Availability",
        value: repro?.code_available ? "Repository discovered" : "No code repo discovered",
        status: repro?.code_available ? "available" : "pending",
      },
    ];
  }, [analysis, repro]);

  return (
    <ArchivistShell
      activeTopNav="systems"
      activeDomain={activeDomain}
      searchPlaceholder="Search archive..."
    >
      <div className="min-h-screen bg-[#111316] p-8">
        <header className="mb-10">
          <div className="mb-2 flex items-center gap-2 font-[family-name:var(--font-label)] text-xs uppercase tracking-[0.2em] text-[#5dd9d8]">
            <FlaskConical className="h-4 w-4" />
            <span>Active Research Lab</span>
          </div>
          <h1 className="mb-2 text-3xl font-bold tracking-tight text-[#e2e2e6]">
            Reproducibility &amp; Experiment Lab
          </h1>
          <p className="max-w-2xl text-sm leading-relaxed text-[#afcbd8]">
            Use current analysis outputs to inspect reproducibility, repository support, and generate experiment scaffolds.
          </p>
        </header>

        {loading ? (
          <div className="flex items-center gap-3 rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-8 text-[#879392]">
            <Loader2 className="h-5 w-5 animate-spin" />
            Loading experiment lab...
          </div>
        ) : error ? (
          <div className="rounded-lg border border-[#93000a] bg-[#93000a]/10 p-6 text-[#ffdad6]">{error}</div>
        ) : !selectedPaper ? (
          <div className="rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-8 text-[#879392]">
            No papers available yet. Upload and analyze a paper first.
          </div>
        ) : !analysis ? (
          <div className="rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-8">
            <h2 className="mb-2 text-xl font-semibold text-[#e2e2e6]">{selectedPaper.title}</h2>
            <p className="mb-6 text-sm text-[#bcc9c8]">
              This paper does not have an analysis yet. Run analysis from the paper page before using the experiment lab.
            </p>
            <Link
              href={`/papers/${selectedPaper.id}`}
              className="inline-flex items-center gap-2 rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] px-5 py-3 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-[0.16em] text-[#002f2f]"
            >
              <Rocket className="h-4 w-4" />
              Open Paper Analysis
            </Link>
          </div>
        ) : (
          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 border-l border-[#5dd9d8]/30 bg-[#1a1c1f] p-6 lg:col-span-8">
              <div className="mb-8 flex items-start justify-between gap-4">
                <div>
                  <h2 className="mb-1 font-[family-name:var(--font-label)] text-xs uppercase tracking-[0.18em] text-[#879392]">
                    Reproduction Score
                  </h2>
                  <div className="text-5xl font-bold tracking-tighter text-[#5dd9d8]">
                    {overallScore == null ? "--" : overallScore.toFixed(1)}
                    <span className="ml-1 text-lg text-[#afcbd8]">/100</span>
                  </div>
                </div>
                <div className="flex gap-4">
                  <div className="text-right">
                    <span className="block text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
                      Paper Domain
                    </span>
                    <span className="font-bold text-[#66dd8b]">{formatDomainLabel(activeDomain)}</span>
                  </div>
                  <div className="h-10 w-px bg-[#3d4949]/30" />
                  <div className="text-right">
                    <span className="block text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
                      Artifact Delta
                    </span>
                    <span className="font-bold text-[#ffb4ab]">
                      {repro?.code_available ? "Low" : "Needs review"}
                    </span>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                {[
                  { label: "Methodology Completeness", value: repro?.methodology_completeness },
                  { label: "Result Reproducibility", value: repro?.result_reproducibility },
                  { label: "Artifact Availability", value: repro?.artifact_availability },
                ].map((metric) => {
                  const scaled = toHundredScale(metric.value);
                  return (
                    <div key={metric.label} className="space-y-2">
                      <div className="flex justify-between text-[10px] font-[family-name:var(--font-label)] uppercase">
                        <span className="text-[#bcc9c8]">{metric.label}</span>
                        <span className="text-[#e2e2e6]">{scaled == null ? "--" : `${Math.round(scaled)}%`}</span>
                      </div>
                      <div className="h-1 overflow-hidden rounded-full bg-[#333538]">
                        <div className="h-full bg-[#5dd9d8]" style={{ width: `${scaled ?? 0}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            <div className="col-span-12 flex flex-col gap-6 lg:col-span-4">
              <div className="bg-[#282a2d] p-6">
                <h3 className="mb-4 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#bcc9c8]">
                  Methodology Analysis
                </h3>
                <div className="mb-2 flex items-center gap-4">
                  <CheckCircle2 className="h-5 w-5 text-[#66dd8b]" />
                  <span className="text-sm font-medium text-[#e2e2e6]">
                    {analysis.analysis_status?.status === "success" ? "Analysis Stored" : "Needs review"}
                  </span>
                </div>
                <p className="text-xs leading-relaxed text-[#afcbd8]">
                  {repro?.summary ?? "Structured analysis exists for this paper and can be used to generate scaffolds."}
                </p>
              </div>
            </div>

            <div className="col-span-12 grid grid-cols-1 gap-6 md:grid-cols-3">
              {artifactRows.map((row) => (
                <div key={row.label} className="border border-[#3d4949]/10 bg-[#1e2023] p-5">
                  <div className="mb-4 flex items-center gap-3">
                    <row.icon className="h-5 w-5 text-[#5dd9d8]" />
                    <h4 className="font-[family-name:var(--font-label)] text-xs uppercase tracking-[0.18em]">
                      {row.label}
                    </h4>
                  </div>
                  <div className="text-xs leading-relaxed text-[#e2e2e6]">{row.value}</div>
                  <div className="mt-3 inline-flex rounded-sm bg-[#334d58] px-2 py-0.5 text-[10px] font-bold uppercase text-[#a1bdca]">
                    {row.status}
                  </div>
                </div>
              ))}
            </div>

            <RepositoryLinksPanel
              repositories={analysis.repository_info?.repositories}
              title="Discovered Repository Links"
              className="col-span-12"
            />

            <div className="col-span-12 rounded-lg border border-[#3d4949]/20 bg-[#0c0e11] p-8">
              <div className="mb-8 flex flex-col gap-6 lg:flex-row lg:items-center lg:justify-between">
                <div>
                  <h3 className="mb-1 text-xl font-bold text-[#e2e2e6]">Experiment Scaffold Generation</h3>
                  <p className="text-sm text-[#bcc9c8]">
                    Generate a scaffold using the current stored analysis for {selectedPaper.title}.
                  </p>
                </div>
                <button
                  onClick={handleGenerate}
                  disabled={generating}
                  className="inline-flex cursor-pointer items-center gap-2 rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] px-6 py-3 font-bold text-[#002f2f] disabled:opacity-50"
                >
                  {generating ? <Loader2 className="h-4 w-4 animate-spin" /> : <Rocket className="h-4 w-4" />}
                  Generate Experiment Scaffold
                </button>
              </div>

              <div className="overflow-x-auto">
                <table className="w-full text-left text-xs">
                  <thead>
                    <tr className="border-b border-[#3d4949]/20 font-[family-name:var(--font-label)] uppercase tracking-[0.18em] text-[#879392]">
                      <th className="pb-4 pr-4 font-medium">Scaffold Entity</th>
                      <th className="px-4 pb-4 font-medium">Status</th>
                      <th className="px-4 pb-4 font-medium">Artifact Path</th>
                      <th className="pb-4 pl-4 text-right font-medium">Validation</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[#3d4949]/10">
                    {[
                      {
                        label: "Experiment Artifact",
                        status: experiment?.generation_status ?? "pending",
                        path: experiment?.artifact_path ?? "--",
                        validation:
                          experiment?.generation_status === "completed"
                            ? "Verified"
                            : experiment?.validation?.errors?.length
                              ? "Missing Keys"
                              : "Pending",
                      },
                      {
                        label: "Recommended Action",
                        status: experiment ? "generated" : "pending",
                        path: experiment?.recommended_action ?? "--",
                        validation: experiment ? "Ready" : "Waiting",
                      },
                      {
                        label: "Repository Matches",
                        status: analysis.repository_info?.repositories?.length ? "linked" : "pending",
                        path: `${analysis.repository_info?.repositories?.length ?? 0} discovered`,
                        validation: analysis.repository_info?.repositories?.length ? "Linked" : "None",
                      },
                    ].map((row) => (
                      <tr key={row.label}>
                        <td className="py-4 pr-4">
                          <div className="flex items-center gap-3">
                            <div className="flex h-8 w-8 items-center justify-center bg-[#282a2d]">
                              <FlaskConical className="h-4 w-4 text-[#5dd9d8]" />
                            </div>
                            <span className="font-medium text-[#e2e2e6]">{row.label}</span>
                          </div>
                        </td>
                        <td className="px-4 py-4 text-[#bcc9c8]">{row.status}</td>
                        <td className="px-4 py-4 font-[family-name:var(--font-label)] text-[#879392]">{row.path}</td>
                        <td className="py-4 pl-4 text-right">
                          <span className="bg-[#25a55a]/10 px-2 py-0.5 text-[10px] font-bold uppercase text-[#66dd8b]">
                            {row.validation}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}
      </div>
    </ArchivistShell>
  );
}

export default function ExperimentsPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#111316]" />}>
      <ExperimentLabPage />
    </Suspense>
  );
}
