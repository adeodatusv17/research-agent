import type {
  ConversationTurn,
  PaperListResponse,
  Paper,
  UploadResponse,
  PaperAnalysis,
  QAResponse,
  ReproducibilityResult,
  ExperimentResult,
  SearchResponse,
} from "./types";

const BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  (process.env.NODE_ENV === "development" ? "http://localhost:8000" : "/api");

export interface RequestMetadata {
  requestId: string;
}

function formatApiErrorDetail(detail: unknown): string {
  if (typeof detail === "string" && detail.trim()) {
    return detail;
  }
  if (Array.isArray(detail) && detail.length > 0) {
    return detail
      .map((item) => {
        if (typeof item === "string") return item;
        if (item && typeof item === "object") {
          const record = item as Record<string, unknown>;
          const loc = Array.isArray(record.loc) ? record.loc.join(".") : "request";
          const msg = typeof record.msg === "string" ? record.msg : JSON.stringify(record);
          return `${loc}: ${msg}`;
        }
        return String(item);
      })
      .join("; ");
  }
  if (detail && typeof detail === "object") {
    try {
      return JSON.stringify(detail);
    } catch {
      return String(detail);
    }
  }
  return "";
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = formatApiErrorDetail(err.detail);
    throw new Error(detail || `Request failed: ${res.status}`);
  }
  return res.json();
}

// Papers
export const getPapers = () => req<PaperListResponse>("/papers/");
export const getPaper = (id: string) => req<Paper>(`/papers/${id}`);

export const uploadPaper = async (file: File): Promise<UploadResponse> => {
  const form = new FormData();
  form.append("file", file);
  const res = await fetch(`${BASE}/papers/upload`, { method: "POST", body: form });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    const detail = formatApiErrorDetail(err.detail);
    throw new Error(detail || `Upload failed: ${res.status}`);
  }
  return res.json();
};

// Analysis
export const analyzePaper = (id: string) =>
  req<PaperAnalysis>(`/papers/${id}/analyze`, { method: "POST" });

export const getAnalysis = (id: string) =>
  req<PaperAnalysis>(`/papers/${id}/analysis`);

// QA
export const askQuestion = (
  id: string,
  query: string,
  metadata?: RequestMetadata,
  recentTurns?: ConversationTurn[],
) =>
  req<QAResponse>(`/papers/${id}/qa`, {
    method: "POST",
    headers: metadata?.requestId ? { "X-Request-Id": metadata.requestId } : undefined,
    body: JSON.stringify({ query, recent_turns: recentTurns ?? [] }),
  });

// Reproducibility
export const getReproducibility = (id: string) =>
  req<ReproducibilityResult>(`/reproducibility/${id}`);

// Experiments
export const generateExperiment = (id: string, domain?: string) =>
  req<ExperimentResult>(
    `/experiments/${id}/generate${domain ? `?domain=${encodeURIComponent(domain)}` : ""}`,
    { method: "POST" }
  );

// Search
export const searchPapers = (query: string) =>
  req<SearchResponse>(`/papers/search?query=${encodeURIComponent(query)}`);
