"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import clsx from "clsx";
import {
  Bell,
  BookCopy,
  BrainCircuit,
  FolderKanban,
  FunctionSquare,
  Globe2,
  Plus,
  Search,
  Server,
  Settings,
  Shield,
} from "lucide-react";
import UploadModal from "@/features/ingestion/UploadModal";
import { DOMAIN_ORDER, formatDomainLabel, normalizeDomain } from "@/lib/domain-utils";
import type { PaperDomain } from "@/lib/types";

interface ArchivistShellProps {
  children: React.ReactNode;
  rightRail?: React.ReactNode;
  activeTopNav?: "pipeline" | "systems" | "health";
  activeDomain?: PaperDomain | null;
  searchPlaceholder?: string;
}

const TOP_NAV = [
  { key: "pipeline" as const, label: "Pipeline", href: "/papers" },
  { key: "systems" as const, label: "Systems", href: "/experiments" },
];

function getDomainIcon(domain: PaperDomain) {
  switch (domain) {
    case "ml":
      return BrainCircuit;
    case "theory":
      return FunctionSquare;
    case "systems":
      return Server;
    case "security":
      return Shield;
    case "networks":
      return Globe2;
    default:
      return BookCopy;
  }
}

export default function ArchivistShell({
  children,
  rightRail,
  activeTopNav = "pipeline",
  activeDomain,
  searchPlaceholder = "Search archive...",
}: ArchivistShellProps) {
  const pathname = usePathname();
  const router = useRouter();
  const [searchValue, setSearchValue] = useState("");
  const [uploadOpen, setUploadOpen] = useState(false);

  const selectedDomain = useMemo(() => {
    return normalizeDomain(activeDomain);
  }, [activeDomain]);

  function navigateSearch() {
    const query = searchValue.trim();
    if (!query) return;
    router.push(`/search?query=${encodeURIComponent(query)}`);
  }

  function handleUploaded(paperId: string) {
    setUploadOpen(false);
    router.push(`/papers/${paperId}`);
  }

  return (
    <>
      <div className="min-h-screen bg-[#111316] text-[#e2e2e6]">
        <nav className="fixed top-0 z-50 flex h-14 w-full items-center justify-between border-b border-[#3d4949]/20 bg-[#111316] px-6">
          <div className="flex items-center gap-8">
            <Link
              href="/papers"
              className="text-lg font-bold uppercase tracking-tight text-[#e2e2e6] transition-colors hover:text-white"
            >
              Digital Archivist
            </Link>
            <div className="hidden items-center gap-6 md:flex">
              {TOP_NAV.map((item) => (
                <Link
                  key={item.key}
                  href={item.href}
                  className={clsx(
                    "border-b-2 px-2 py-1 text-sm transition-colors",
                    activeTopNav === item.key
                      ? "border-[#5dd9d8] text-[#5dd9d8]"
                      : "border-transparent text-[#909094] hover:bg-[#333538] hover:text-[#e2e2e6]"
                  )}
                >
                  {item.label}
                </Link>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-4">
            <div className="relative hidden md:block">
              <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-[#879392]" />
              <input
                value={searchValue}
                onChange={(event) => setSearchValue(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    navigateSearch();
                  }
                }}
                className="w-72 rounded-sm border border-[#3d4949]/20 bg-[#1a1c1f] py-1.5 pl-9 pr-4 text-xs text-[#e2e2e6] outline-none placeholder:text-[#879392] focus:border-[#5dd9d8]/40"
                placeholder={searchPlaceholder}
                type="text"
              />
            </div>
            <button className="rounded-sm p-2 text-[#909094] transition-colors hover:bg-[#333538] hover:text-[#e2e2e6] cursor-pointer">
              <Bell className="h-4 w-4" />
            </button>
            <button className="rounded-sm p-2 text-[#909094] transition-colors hover:bg-[#333538] hover:text-[#e2e2e6] cursor-pointer">
              <Settings className="h-4 w-4" />
            </button>
            <div className="flex h-8 w-8 items-center justify-center rounded-full border border-[#3d4949]/30 bg-[#1a1c1f] text-xs font-bold text-[#5dd9d8]">
              DA
            </div>
          </div>
        </nav>

        <aside className="fixed left-0 top-14 z-40 flex h-[calc(100vh-3.5rem)] w-64 flex-col border-r border-[#3d4949]/20 bg-[#1a1c1f] py-4">
          <div className="mb-8 px-6">
            <div className="mb-1 flex items-center gap-3">
              <div className="flex h-8 w-8 items-center justify-center rounded-sm bg-[#333538]">
                <FolderKanban className="h-4 w-4 text-[#5dd9d8]" />
              </div>
              <div>
                <div className="font-[family-name:var(--font-label)] text-sm font-bold uppercase tracking-widest text-[#e2e2e6]">
                  Corpus
                </div>
                <div className="text-[10px] tracking-normal text-[#879392]">
                  Intellectual Discovery
                </div>
              </div>
            </div>
          </div>

          <nav className="flex-1 space-y-1">
            {DOMAIN_ORDER.map((domain) => {
              const Icon = getDomainIcon(domain);
              const isActive = selectedDomain === domain;
              const href = pathname.startsWith("/papers") ? `/papers?domain=${domain}` : `/papers?domain=${domain}`;
              return (
                <Link
                  key={domain}
                  href={href}
                  className={clsx(
                    "flex items-center gap-4 px-6 py-3 font-[family-name:var(--font-label)] text-xs uppercase tracking-widest transition-all",
                    isActive
                      ? "border-l-4 border-[#5dd9d8] bg-[#333538] font-bold text-[#5dd9d8]"
                      : "text-[#909094] hover:bg-[#111316] hover:text-[#e2e2e6]"
                  )}
                >
                  <Icon className="h-4 w-4" />
                  <span>{formatDomainLabel(domain)}</span>
                </Link>
              );
            })}
          </nav>

          <div className="mt-auto px-4">
            <button
              onClick={() => setUploadOpen(true)}
              className="w-full rounded-sm bg-gradient-to-br from-[#5dd9d8] to-[#00a1a1] py-3 font-[family-name:var(--font-label)] text-xs font-bold uppercase tracking-widest text-[#002f2f] transition-transform hover:opacity-95 active:translate-y-0.5 cursor-pointer"
            >
              <span className="inline-flex items-center gap-2">
                <Plus className="h-4 w-4" />
                New Ingestion
              </span>
            </button>
          </div>
        </aside>

        <main
          className={clsx(
            "ml-64 pt-14 min-h-screen",
            rightRail ? "mr-80" : ""
          )}
        >
          {children}
        </main>

        {rightRail && (
          <aside className="fixed right-0 top-14 z-40 h-[calc(100vh-3.5rem)] w-80 border-l border-[#3d4949]/20 bg-[#1a1c1f]/90 backdrop-blur-md">
            {rightRail}
          </aside>
        )}
      </div>

      <UploadModal
        open={uploadOpen}
        onClose={() => setUploadOpen(false)}
        onUploaded={handleUploaded}
      />
    </>
  );
}
