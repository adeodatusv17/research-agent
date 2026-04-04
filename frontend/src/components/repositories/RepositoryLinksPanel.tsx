"use client";

import { useMemo } from "react";
import { Code2, ExternalLink } from "lucide-react";
import type { RepoEntry } from "@/lib/types";

interface RepositoryLinksPanelProps {
  repositories?: RepoEntry[] | null;
  title?: string;
  emptyLabel?: string;
  className?: string;
}

function formatSourceLabel(source: RepoEntry["source"]) {
  switch (source) {
    case "paper_link":
      return "Paper";
    case "paperswithcode":
      return "PWC";
    case "github_search":
      return "Search";
    default:
      return "Repo";
  }
}

export default function RepositoryLinksPanel({
  repositories,
  title = "Repository Matches",
  emptyLabel = "No repository links were discovered for this paper yet.",
  className = "",
}: RepositoryLinksPanelProps) {
  const uniqueRepositories = useMemo(() => {
    const seen = new Set<string>();
    return (repositories ?? []).filter((repo) => {
      if (!repo.url || seen.has(repo.url)) return false;
      seen.add(repo.url);
      return true;
    });
  }, [repositories]);

  return (
    <section className={`rounded-lg border border-[#3d4949]/20 bg-[#1a1c1f] p-6 ${className}`.trim()}>
      <div className="mb-4 flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-[#e2e2e6]">{title}</h3>
          <div className="mt-1 font-[family-name:var(--font-label)] text-[10px] uppercase tracking-[0.18em] text-[#879392]">
            {uniqueRepositories.length} discovered link{uniqueRepositories.length === 1 ? "" : "s"}
          </div>
        </div>
        <Code2 className="h-5 w-5 text-[#5dd9d8]" />
      </div>

      {uniqueRepositories.length === 0 ? (
        <div className="rounded-sm border border-dashed border-[#3d4949]/20 bg-[#111316] p-4 text-sm text-[#879392]">
          {emptyLabel}
        </div>
      ) : (
        <div className="space-y-3">
          {uniqueRepositories.map((repo, index) => (
            <div
              key={`${repo.url}-${index}`}
              className="flex flex-col gap-3 rounded-sm border border-[#3d4949]/20 bg-[#111316] p-4"
            >
              <div className="flex flex-wrap items-center gap-2">
                <span className="rounded-sm bg-[#334d58] px-2 py-0.5 text-[9px] font-bold uppercase text-[#a1bdca]">
                  {formatSourceLabel(repo.source)}
                </span>
                <span className="rounded-sm bg-[#25a55a]/10 px-2 py-0.5 text-[9px] font-bold uppercase text-[#66dd8b]">
                  {Math.round(repo.confidence * 100)}% match
                </span>
              </div>
              <div className="break-all text-sm text-[#e2e2e6]">{repo.url}</div>
              <div className="flex flex-wrap gap-2">
                <a
                  href={repo.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-2 rounded-sm border border-[#5dd9d8]/30 px-3 py-2 font-[family-name:var(--font-label)] text-[10px] font-bold uppercase tracking-[0.16em] text-[#5dd9d8] transition-colors hover:bg-[#5dd9d8]/5"
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open Link
                </a>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
