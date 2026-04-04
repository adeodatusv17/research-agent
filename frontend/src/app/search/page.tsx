"use client";

import { useState, useRef } from "react";
import { Search, BookOpen, ChevronRight, Loader2 } from "lucide-react";
import Link from "next/link";
import clsx from "clsx";
import AppShell from "@/components/AppShell";
import { searchPapers } from "@/lib/api-client";
import type { SearchResultItem } from "@/lib/types";

function ResultCard({ item }: { item: SearchResultItem }) {
  return (
    <Link
      href={`/papers/${item.paper_id}`}
      className="card p-4 group cursor-pointer hover:border-accent-indigo/30 transition-all duration-200 block"
    >
      <div className="flex items-start gap-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 mb-1.5 flex-wrap">
            {item.section_name && (
              <span className="text-[10px] font-medium text-accent-indigo bg-accent-indigo/10 border border-accent-indigo/20 rounded px-1.5 py-0.5">
                {item.section_name}
              </span>
            )}
            {item.subsection_name && (
              <span className="text-[10px] text-text-muted">{item.subsection_name}</span>
            )}
            {item.page_number && (
              <span className="text-[10px] text-text-muted">p.{item.page_number}</span>
            )}
            <span className="text-[10px] font-mono text-accent-cyan ml-auto">
              {Math.round(item.score * 100)}% match
            </span>
          </div>
          <p className="text-xs text-text-secondary leading-relaxed line-clamp-3">{item.content}</p>
          <p className="text-[10px] text-text-muted mt-2 flex items-center gap-1">
            <BookOpen className="w-3 h-3" />
            {item.title || item.paper_id}
          </p>
        </div>
        <ChevronRight className="w-4 h-4 text-text-muted group-hover:text-accent-indigo transition-colors flex-shrink-0 mt-0.5" />
      </div>
    </Link>
  );
}

export default function SearchPage() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResultItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [searched, setSearched] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleSearch() {
    const q = query.trim();
    if (!q || loading) return;
    setLoading(true);
    setSearched(true);
    try {
      const data = await searchPapers(q);
      setResults(data.results);
    } catch {
      setResults([]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <AppShell>
      <div className="max-w-3xl mx-auto px-8 py-10">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-xl font-semibold text-text-primary mb-1">Semantic Search</h1>
          <p className="text-sm text-text-secondary">
            Search across all uploaded papers using vector similarity.
          </p>
        </div>

        {/* Search input */}
        <div className="relative mb-8">
          <div className="flex gap-2">
            <div className="relative flex-1">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-text-muted pointer-events-none" />
              <input
                ref={inputRef}
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleSearch()}
                placeholder="Search across all papers… (e.g. 'attention mechanism' or 'batch normalization')"
                className="input-field pl-9 w-full"
              />
            </div>
            <button
              onClick={handleSearch}
              disabled={!query.trim() || loading}
              className={clsx(
                "btn-primary px-4 flex-shrink-0",
                (!query.trim() || loading) && "opacity-50 cursor-not-allowed"
              )}
            >
              {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
            </button>
          </div>
        </div>

        {/* Results */}
        {loading && (
          <div className="flex items-center gap-2 text-xs text-text-muted py-4">
            <Loader2 className="w-3.5 h-3.5 animate-spin text-accent-indigo" />
            Searching…
          </div>
        )}

        {!loading && searched && results.length === 0 && (
          <div className="card p-8 text-center">
            <Search className="w-8 h-8 text-text-muted mx-auto mb-3" />
            <p className="text-sm font-medium text-text-primary mb-1">No results found</p>
            <p className="text-xs text-text-muted">
              Try different keywords or upload more papers.
            </p>
          </div>
        )}

        {!loading && results.length > 0 && (
          <div className="space-y-2">
            <p className="text-xs text-text-muted mb-3">{results.length} results</p>
            {results.map((item) => (
              <ResultCard key={item.chunk_id} item={item} />
            ))}
          </div>
        )}

        {!searched && (
          <div className="text-center py-12">
            <Search className="w-10 h-10 text-text-muted mx-auto mb-3 opacity-40" />
            <p className="text-xs text-text-muted">Enter a query to search across paper content</p>
          </div>
        )}
      </div>
    </AppShell>
  );
}
