"use client";

import { Suspense, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import clsx from "clsx";
import {
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  Code2,
  Loader2,
  Sparkles,
} from "lucide-react";
import ArchivistShell from "@/components/archivist/ArchivistShell";
import { getAnalysis, getPapers } from "@/lib/api-client";
import { DOMAIN_ORDER, formatDomainLabel, formatDomainShortLabel, normalizeDomain } from "@/lib/domain-utils";
import type { Paper, PaperAnalysis } from "@/lib/types";

interface PaperWithAnalysis extends Paper {
  latestAnalysis?: PaperAnalysis | null;
}

function formatDate(value?: string | null) {
  if (!value) return "Unknown";
  const date = new Date(value);
  return `${date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })}\n${date.toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" })}`;
}

function formatReproScore(analysis?: PaperAnalysis | null) {
  const raw = analysis?.reproducibility?.overall_score;
  if (typeof raw !== "number") return "--";
  const scaled = raw <= 1 ? raw * 10 : raw;
  return scaled.toFixed(1);
}

function getAnalysisStatus(analysis?: PaperAnalysis | null) {
  if (!analysis) {
    return {
      label: "Ready to Analyze",
      detail: "No analysis stored yet",
      tone: "text-[#879392]",
    };
  }

  const status = analysis.analysis_status?.status;
  if (status === "partial_failure") {
    return {
      label: "Partial Review",
      detail: analysis.analysis_status?.message ?? "Some sections need verification",
      tone: "text-[#ffb4ab]",
    };
  }
  if (status === "failed") {
    return {
      label: "Failed",
      detail: analysis.analysis_status?.message ?? "Analysis failed",
      tone: "text-[#ffb4ab]",
    };
  }
  return {
    label: "Synthesized",
    detail: analysis.analysis_status?.message ?? "Sections and scores available",
    tone: "text-[#66dd8b]",
  };
}

function PapersDashboardPage() {
  const searchParams = useSearchParams();
  const activeDomain = normalizeDomain(searchParams.get("domain"));
  const [papers, setPapers] = useState<PaperWithAnalysis[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const perPage = 10;

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
        const analysisMap = new Map(analyses);
        if (!cancelled) {
          setPapers(rows.map((paper) => ({ ...paper, latestAnalysis: analysisMap.get(paper.id) ?? null })));
        }
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : "Failed to load papers");
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
    if (!searchParams.get("domain")) return papers;
    return papers.filter((paper) => normalizeDomain(paper.domain) === activeDomain);
  }, [papers, activeDomain, searchParams]);

  const domainCounts = useMemo(() => {
    return DOMAIN_ORDER.map((domain) => ({
      domain,
      count: papers.filter((paper) => normalizeDomain(paper.domain) === domain).length,
    }));
  }, [papers]);

  const analyzedCount = papers.filter((paper) => paper.latestAnalysis).length;
  const currentPageItems = filteredPapers.slice((page - 1) * perPage, page * perPage);
  const totalPages = Math.max(1, Math.ceil(filteredPapers.length / perPage));

  useEffect(() => {
    setPage(1);
  }, [activeDomain]);

  return (
    <ArchivistShell
      activeTopNav="pipeline"
      activeDomain={activeDomain}
      searchPlaceholder="Search archive..."
    >
      <div className="p-8">
        <header className="mb-10">
          <div className="mb-8 flex items-end justify-between">
            <div>
              <h1 className="mb-2 text-3xl font-extrabold uppercase tracking-tight text-[#e2e2e6]">
                Papers Dashboard
              </h1>
              <p className="font-[family-name:var(--font-label)] text-sm uppercase tracking-[0.18em] text-[#879392]">
                Archive coverage: {loading ? "Loading..." : `${analyzedCount} analyzed of ${papers.length} papers`}
              </p>
            </div>
            <div className="hidden gap-4 lg:flex">
              <div className="rounded-sm border-l-2 border-[#5dd9d8] bg-[#1a1c1f] px-4 py-2">
                <div className="font-[family-name:var(--font-label)] text-[10px] uppercase text-[#879392]">
                  Active Domain
                </div>
                <div className="text-lg font-bold text-[#5dd9d8]">{formatDomainLabel(activeDomain)}</div>
              </div>
              <div className="rounded-sm border-l-2 border-[#66dd8b] bg-[#1a1c1f] px-4 py-2">
                <div className="font-[family-name:var(--font-label)] text-[10px] uppercase text-[#879392]">
                  Tracked Papers
                </div>
                <div className="text-lg font-bold text-[#66dd8b]">{loading ? "--" : papers.length}</div>
              </div>
            </div>
          </div>

          <div className="mb-10 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
            {domainCounts.slice(0, 4).map(({ domain, count }) => (
              <div
                key={domain}
                className="group relative overflow-hidden rounded-lg border border-[#3d4949]/10 bg-[#1a1c1f] p-5"
              >
                <div className="mb-2 font-[family-name:var(--font-label)] text-xs uppercase text-[#bcc9c8]">
                  {formatDomainLabel(domain)}
                </div>
                <div className="text-3xl font-bold text-[#e2e2e6]">{loading ? "--" : count}</div>
                <div className="mt-2 text-[10px] font-[family-name:var(--font-label)] text-[#5dd9d8]">
                  {domain === activeDomain ? "Current filter" : "Tracked in corpus"}
                </div>
              </div>
            ))}
          </div>
        </header>

        <div className="overflow-hidden rounded-sm border border-[#3d4949]/10 bg-[#1e2023] shadow-xl">
          <div className="flex items-center justify-between border-b border-[#3d4949]/20 bg-[#282a2d]/50 p-4">
            <div className="flex items-center gap-4">
              <span className="font-[family-name:var(--font-label)] text-xs uppercase tracking-[0.18em] text-[#879392]">
                Archive Manifest
              </span>
              <div className="flex rounded-sm border border-[#3d4949]/30">
                <Link
                  href="/papers"
                  className={clsx(
                    "px-3 py-1 text-[10px] font-bold uppercase",
                    !searchParams.get("domain")
                      ? "bg-[#333538] text-[#5dd9d8]"
                      : "text-[#879392] hover:bg-[#1a1c1f]"
                  )}
                >
                  All
                </Link>
                {DOMAIN_ORDER.map((domain) => (
                  <Link
                    key={domain}
                    href={`/papers?domain=${domain}`}
                    className={clsx(
                      "px-3 py-1 text-[10px] font-bold uppercase",
                      activeDomain === domain && searchParams.get("domain")
                        ? "bg-[#333538] text-[#5dd9d8]"
                        : "text-[#879392] hover:bg-[#1a1c1f]"
                    )}
                  >
                    {formatDomainShortLabel(domain)}
                  </Link>
                ))}
              </div>
            </div>
            <div className="text-[10px] font-[family-name:var(--font-label)] uppercase text-[#879392]">
              {loading ? "Loading..." : `${filteredPapers.length} visible papers`}
            </div>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-left">
              <thead>
                <tr className="bg-[#1a1c1f] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
                  <th className="px-6 py-4 font-medium">Title &amp; ID</th>
                  <th className="px-4 py-4 font-medium">Domain &amp; Conf.</th>
                  <th className="px-4 py-4 font-medium">Analysis Status</th>
                  <th className="px-4 py-4 font-medium text-center">Repr. Score</th>
                  <th className="px-4 py-4 font-medium">Code</th>
                  <th className="px-4 py-4 font-medium">Created At</th>
                  <th className="px-4 py-4" />
                </tr>
              </thead>
              <tbody className="divide-y divide-[#3d4949]/10">
                {loading ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-sm text-[#879392]">
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Loading papers and latest analyses...
                      </span>
                    </td>
                  </tr>
                ) : error ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-sm text-[#ffb4ab]">
                      {error}
                    </td>
                  </tr>
                ) : currentPageItems.length === 0 ? (
                  <tr>
                    <td colSpan={7} className="px-6 py-12 text-center text-sm text-[#879392]">
                      No papers available for this filter.
                    </td>
                  </tr>
                ) : (
                  currentPageItems.map((paper) => {
                    const status = getAnalysisStatus(paper.latestAnalysis);
                    const repositoryCount = paper.latestAnalysis?.repository_info?.repositories?.length ?? 0;
                    return (
                      <tr key={paper.id} className="group transition-colors hover:bg-[#282a2d]">
                        <td className="px-6 py-4">
                          <div className="mb-1 text-sm font-bold text-[#e2e2e6] group-hover:text-[#5dd9d8]">
                            {paper.title}
                          </div>
                          <div className="font-[family-name:var(--font-label)] text-[10px] text-[#879392]">
                            {paper.id}
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className="mb-1 flex items-center gap-2">
                            <span className="rounded-sm bg-[#334d58] px-1.5 py-0.5 text-[9px] font-bold uppercase text-[#a1bdca]">
                              {formatDomainShortLabel(paper.domain)}
                            </span>
                            <span className="text-[10px] font-bold text-[#66dd8b]">
                              {Math.round((paper.domain_confidence ?? 0) * 100)}%
                            </span>
                          </div>
                          <div className="h-1 w-16 overflow-hidden rounded-full bg-[#333538]">
                            <div
                              className="h-full bg-[#66dd8b]"
                              style={{ width: `${Math.round((paper.domain_confidence ?? 0) * 100)}%` }}
                            />
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          <div className={`mb-1 flex items-center gap-2 text-xs ${status.tone}`}>
                            <Sparkles className="h-3.5 w-3.5" />
                            <span>{status.label}</span>
                          </div>
                          <div className="font-[family-name:var(--font-label)] text-[9px] uppercase text-[#879392]">
                            {status.detail}
                          </div>
                        </td>
                        <td className="px-4 py-4 text-center">
                          <div className="font-[family-name:var(--font-label)] text-xl font-bold text-[#e2e2e6]">
                            {formatReproScore(paper.latestAnalysis)}
                          </div>
                        </td>
                        <td className="px-4 py-4">
                          {repositoryCount > 0 ? (
                            <span className="inline-flex items-center gap-1 text-[10px] text-[#5dd9d8]">
                              <Code2 className="h-3.5 w-3.5" />
                              {repositoryCount} link{repositoryCount === 1 ? "" : "s"}
                            </span>
                          ) : (
                            <span className="text-[10px] uppercase text-[#879392]">None</span>
                          )}
                        </td>
                        <td className="whitespace-pre-line px-4 py-4 font-[family-name:var(--font-label)] text-[10px] uppercase text-[#879392]">
                          {formatDate(paper.created_at)}
                        </td>
                        <td className="px-4 py-4 text-right">
                          <Link
                            href={`/papers/${paper.id}`}
                            className="inline-flex items-center gap-2 rounded-sm border border-[#3d4949]/30 px-3 py-2 text-[10px] font-[family-name:var(--font-label)] uppercase tracking-[0.14em] text-[#e2e2e6] transition-colors hover:border-[#5dd9d8]/30 hover:text-[#5dd9d8]"
                          >
                            Open
                            <ArrowUpRight className="h-3.5 w-3.5" />
                          </Link>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div className="flex items-center justify-between border-t border-[#3d4949]/10 bg-[#1a1c1f]/30 px-6 py-4">
            <div className="font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.14em] text-[#879392]">
              Showing {filteredPapers.length === 0 ? 0 : (page - 1) * perPage + 1}-
              {Math.min(page * perPage, filteredPapers.length)} of {filteredPapers.length} entities
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setPage((current) => Math.max(1, current - 1))}
                disabled={page === 1}
                className="flex h-8 w-8 items-center justify-center rounded-sm border border-[#3d4949]/30 text-[#879392] disabled:opacity-40 cursor-pointer hover:bg-[#333538]"
              >
                <ChevronLeft className="h-4 w-4" />
              </button>
              <div className="flex h-8 min-w-8 items-center justify-center rounded-sm border border-[#5dd9d8]/20 bg-[#333538] px-3 text-[10px] font-bold text-[#5dd9d8]">
                {page}
              </div>
              <button
                onClick={() => setPage((current) => Math.min(totalPages, current + 1))}
                disabled={page >= totalPages}
                className="flex h-8 w-8 items-center justify-center rounded-sm border border-[#3d4949]/30 text-[#879392] disabled:opacity-40 cursor-pointer hover:bg-[#333538]"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </ArchivistShell>
  );
}

export default function PapersPage() {
  return (
    <Suspense fallback={<div className="min-h-screen bg-[#111316]" />}>
      <PapersDashboardPage />
    </Suspense>
  );
}
