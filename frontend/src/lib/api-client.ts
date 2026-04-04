import type {
  PaperListResponse,
  Paper,
  UploadResponse,
  PaperAnalysis,
  QAResponse,
  ReproducibilityResult,
  ExperimentResult,
  SearchResponse,
} from "./types";

const BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api";

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail ?? `Request failed: ${res.status}`);
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
    throw new Error(err.detail ?? `Upload failed: ${res.status}`);
  }
  return res.json();
};

// Analysis
export const analyzePaper = (id: string) =>
  req<PaperAnalysis>(`/papers/${id}/analyze`, { method: "POST" });

export const getAnalysis = (id: string) =>
  req<PaperAnalysis>(`/papers/${id}/analysis`);

// QA
export const askQuestion = (id: string, query: string) =>
  req<QAResponse>(`/papers/${id}/qa`, {
    method: "POST",
    body: JSON.stringify({ query }),
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
