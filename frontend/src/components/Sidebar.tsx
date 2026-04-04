"use client";

import { useState, useEffect, useCallback } from "react";
import { useRouter, usePathname } from "next/navigation";
import Link from "next/link";
import {
  BookOpen,
  Search,
  FlaskConical,
  ChevronRight,
  Plus,
  PanelLeft,
  ChevronDown,
} from "lucide-react";
import clsx from "clsx";
import { getPapers } from "@/lib/api-client";
import type { Paper, PaperDomain } from "@/lib/types";
import { Skeleton } from "@/components/ui/Skeleton";
import UploadModal from "@/features/ingestion/UploadModal";

interface SidebarProps {
  onPaperUploaded?: (paperId: string) => void;
}

const DOMAIN_ORDER: PaperDomain[] = ["ml", "theory", "systems", "security", "networks", "general"];

function normalizeDomain(domain?: string | null): PaperDomain {
  const safeDomain = (domain ?? "general").toLowerCase();
  return DOMAIN_ORDER.includes(safeDomain as PaperDomain) ? (safeDomain as PaperDomain) : "general";
}

function formatDomainLabel(domain: PaperDomain) {
  return domain === "ml" ? "ML" : domain.charAt(0).toUpperCase() + domain.slice(1);
}

function DomainBadge({ domain }: { domain?: string | null }) {
  const safeDomain = normalizeDomain(domain);
  return (
    <span className="rounded-full border border-white/10 bg-bg-hover px-2 py-0.5 text-[11px] text-gray-500">
      {formatDomainLabel(safeDomain)}
    </span>
  );
}

export default function Sidebar({ onPaperUploaded }: SidebarProps) {
  const router = useRouter();
  const pathname = usePathname();
  const [papers, setPapers] = useState<Paper[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploadOpen, setUploadOpen] = useState(false);
  const [activeFilter, setActiveFilter] = useState<PaperDomain | "all">("all");
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const [collapsed, setCollapsed] = useState<boolean>(() => {
    if (typeof window === "undefined") return false;
    return localStorage.getItem("sidebar-collapsed") === "true";
  });

  const toggleCollapsed = useCallback(() => {
    setCollapsed((prev) => {
      const next = !prev;
      localStorage.setItem("sidebar-collapsed", String(next));
      return next;
    });
  }, []);

  const loadPapers = useCallback(async () => {
    try {
      const data = await getPapers();
      setPapers(data.items);
    } catch {
      setPapers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadPapers();
  }, [loadPapers]);

  const activePaperId = pathname.match(/\/papers\/([^/]+)/)?.[1];

  function toggleDomain(domain: PaperDomain) {
    setCollapsedGroups((prev) => ({ ...prev, [domain]: !prev[domain] }));
  }

  function handleUploaded(paperId: string) {
    setUploadOpen(false);
    loadPapers();
    router.push(`/papers/${paperId}`);
    onPaperUploaded?.(paperId);
  }

  const filteredPapers = papers.filter((paper) => {
    const domain = normalizeDomain(paper.domain);
    return activeFilter === "all" ? true : domain === activeFilter;
  });

  const groupedPapers = filteredPapers.reduce<Record<PaperDomain, Paper[]>>(
    (acc, paper) => {
      const domain = normalizeDomain(paper.domain);
      acc[domain].push(paper);
      return acc;
    },
    {
      ml: [],
      theory: [],
      systems: [],
      security: [],
      networks: [],
      general: [],
    }
  );

  const visibleDomains = DOMAIN_ORDER.filter((domain) => groupedPapers[domain].length > 0);

  return (
    <>
      <aside
        className={clsx(
          "fixed left-0 top-0 z-40 flex h-screen flex-col overflow-hidden border-r border-bg-border bg-bg-surface transition-all duration-200 ease-in-out",
          collapsed ? "w-[64px]" : "w-[300px]"
        )}
      >
        <div
          className={clsx(
            "flex flex-shrink-0 items-center border-b border-bg-border",
            collapsed ? "justify-center px-0 py-4" : "justify-between gap-3 px-4 py-4"
          )}
        >
          {!collapsed && (
            <div className="flex min-w-0 items-center gap-2.5">
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md border border-bg-border bg-bg-hover">
                <FlaskConical className="h-3.5 w-3.5 text-text-secondary" />
              </div>
              <div className="min-w-0">
                <div className="truncate text-sm font-semibold text-text-primary leading-none">PaperAgent</div>
                <div className="mt-0.5 text-[10px] text-text-muted">Replication System</div>
              </div>
            </div>
          )}
          <button
            onClick={toggleCollapsed}
            title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
            className={clsx(
              "flex items-center justify-center rounded-md text-text-muted transition-all duration-150 hover:bg-bg-hover hover:text-text-primary cursor-pointer",
              collapsed ? "h-9 w-9" : "h-7 w-7 shrink-0"
            )}
          >
            <PanelLeft className={clsx("h-4 w-4 transition-transform duration-200", collapsed && "rotate-180")} />
          </button>
        </div>

        <div className={clsx("flex-shrink-0 border-b border-bg-border", collapsed ? "px-2 py-3" : "px-3 py-3")}>
          <div className={clsx("space-y-2", collapsed && "space-y-1")}>
            <button
              onClick={() => setUploadOpen(true)}
              title="Upload Paper"
              className={clsx("btn-primary w-full text-xs py-2", collapsed ? "justify-center px-0" : "justify-start px-3")}
            >
              <Plus className="h-3.5 w-3.5 shrink-0" />
              {!collapsed && "Upload Paper"}
            </button>
            <Link
              href="/search"
              title="Semantic Search"
              className={clsx(
                "btn-ghost w-full text-xs",
                collapsed ? "justify-center px-0" : "justify-start",
                pathname === "/search" && "bg-bg-hover text-text-primary"
              )}
            >
              <Search className="h-3.5 w-3.5 shrink-0" />
              {!collapsed && "Semantic Search"}
            </Link>
          </div>

          {!collapsed && (
            <div className="mt-3 flex flex-wrap gap-2">
              <button
                onClick={() => setActiveFilter("all")}
                className={clsx(
                  "rounded-full border px-3 py-1 text-xs transition-colors cursor-pointer",
                  activeFilter === "all"
                    ? "border-white/20 bg-bg-hover text-text-primary"
                    : "border-white/10 text-gray-500 hover:border-white/20 hover:text-text-primary"
                )}
              >
                All
              </button>
              {DOMAIN_ORDER.map((domain) => (
                <button
                  key={domain}
                  onClick={() => setActiveFilter(domain)}
                  className={clsx(
                    "rounded-full border px-3 py-1 text-xs transition-colors cursor-pointer",
                    activeFilter === domain
                      ? "border-white/20 bg-bg-hover text-text-primary"
                      : "border-white/10 text-gray-500 hover:border-white/20 hover:text-text-primary"
                  )}
                >
                  {formatDomainLabel(domain)}
                </button>
              ))}
            </div>
          )}
        </div>

        <div className="min-w-0 flex-1 overflow-y-auto px-2 py-2">
          {!collapsed && (
            <div className="px-2 py-1.5 text-[10px] font-semibold uppercase tracking-wider text-text-muted">
              Papers ({filteredPapers.length})
            </div>
          )}

          {loading ? (
            <div className="mt-1 space-y-2 px-2">
              {[1, 2, 3].map((i) => (
                <Skeleton key={i} className={clsx("rounded-lg", collapsed ? "h-9 w-9" : "h-12 w-full")} />
              ))}
            </div>
          ) : filteredPapers.length === 0 && !collapsed ? (
            <div className="px-4 py-8 text-center">
              <BookOpen className="mx-auto mb-2 h-6 w-6 text-text-muted" />
              <p className="text-xs text-text-muted">No papers for this filter</p>
            </div>
          ) : (
            <nav className="space-y-1">
              {visibleDomains.map((domain) => {
                const isGroupCollapsed = collapsedGroups[domain] ?? false;

                return (
                  <div key={domain} className="space-y-1">
                    {!collapsed && (
                      <button
                        onClick={() => toggleDomain(domain)}
                        className="flex w-full items-center justify-between rounded-lg px-2 py-2 text-left text-sm text-gray-500 transition-colors hover:bg-bg-hover hover:text-text-primary cursor-pointer"
                      >
                        <span>{formatDomainLabel(domain)}</span>
                        <ChevronDown className={clsx("h-4 w-4 transition-transform", isGroupCollapsed && "-rotate-90")} />
                      </button>
                    )}

                    {(!isGroupCollapsed || collapsed) &&
                      groupedPapers[domain].map((paper) => {
                        const isActive = activePaperId === paper.id;
                        const safeDomain = normalizeDomain(paper.domain);

                        return (
                          <Link
                            key={paper.id}
                            href={`/papers/${paper.id}`}
                            title={paper.title}
                            className={clsx(
                              "group relative flex items-center gap-2 rounded-lg border transition-all duration-150 cursor-pointer",
                              collapsed ? "justify-center p-2" : "px-2 py-2.5",
                              isActive
                                ? "border-bg-border bg-bg-hover text-text-primary"
                                : "border-transparent text-text-secondary hover:border-white/10 hover:bg-bg-hover hover:text-text-primary"
                            )}
                          >
                            <span
                              className={clsx(
                                "absolute left-0 top-1/2 h-8 w-0.5 -translate-y-1/2 rounded-r-full transition-opacity",
                                isActive ? "bg-accent-cyan opacity-100" : "opacity-0"
                              )}
                            />
                            <BookOpen
                              className={clsx(
                                "shrink-0",
                                collapsed ? "h-4 w-4" : "h-3.5 w-3.5",
                                isActive ? "text-text-primary" : "text-text-muted group-hover:text-text-secondary"
                              )}
                            />
                            {!collapsed && (
                              <>
                                <div className="min-w-0 flex-1">
                                  <div className="truncate text-xs leading-snug">{paper.title}</div>
                                  <div className="mt-1">
                                    <DomainBadge domain={safeDomain} />
                                  </div>
                                </div>
                                {isActive && <ChevronRight className="h-3 w-3 shrink-0 text-text-muted" />}
                              </>
                            )}
                          </Link>
                        );
                      })}
                  </div>
                );
              })}
            </nav>
          )}
        </div>

        {!collapsed && (
          <div className="flex-shrink-0 border-t border-bg-border px-4 py-3">
            <p className="text-[10px] text-text-muted">Research Replication Agent</p>
          </div>
        )}
      </aside>

      <div className={clsx("flex-shrink-0 transition-all duration-200 ease-in-out", collapsed ? "w-[64px]" : "w-[300px]")} />

      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploaded={handleUploaded}
      />
    </>
  );
}
