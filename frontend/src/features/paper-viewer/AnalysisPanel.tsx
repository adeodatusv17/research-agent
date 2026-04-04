"use client";

import { FolderGit2 } from "lucide-react";
import type {
  AnalysisChunk,
  AnalysisStatus,
  EquationCollection,
  MLPaperDetails,
  MethodsStructure,
  PaperAnalysis,
  PaperDomain,
  SectionSynthesisResult,
  SynthesisOutput,
  SystemsPaperDetails,
  TheoryPaperDetails,
} from "@/lib/types";
import DomainBlock from "@/features/paper-viewer/DomainBlock";
import SynthesisSection from "@/features/paper-viewer/SynthesisSection";

interface AnalysisPanelProps {
  analysis: PaperAnalysis;
}

const SECTION_META = {
  key_ideas: "Key Ideas",
  methods: "Methods & Approach",
  results: "Results & Evaluation",
  discussion: "Discussion & Limitations",
} as const;

type SectionKey = keyof typeof SECTION_META;

function normalizeDomain(domain?: PaperDomain | null): PaperDomain {
  return domain ?? "general";
}

function normalizeChunk(chunk: AnalysisChunk, index: number) {
  const fullText = (chunk.text ?? chunk.summary ?? "").trim();
  const summary = (chunk.summary ?? fullText).trim();
  return {
    id: chunk.id ?? `chunk-${index}`,
    text: fullText,
    summary,
    role: chunk.role ?? "other",
    importance: chunk.importance ?? 0,
    confidence: chunk.confidence ?? 0,
    source: chunk.source ?? "extracted",
  };
}

function averageImportance(chunks: ReturnType<typeof normalizeChunk>[]) {
  if (chunks.length === 0) return 0;
  return chunks.reduce((sum, chunk) => sum + chunk.importance, 0) / chunks.length;
}

function normalizeSectionSynthesis(
  value: SynthesisOutput[keyof SynthesisOutput]
): SectionSynthesisResult | null {
  if (!value) return null;
  if (typeof value === "string") {
    return {
      synthesis: value.trim(),
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
  return {
    synthesis: (value.synthesis ?? "").trim(),
    confidence: value.confidence ?? "low",
    warning: value.warning ?? null,
    fabrication_flagged: value.fabrication_flagged ?? false,
    retrieval_rounds: value.retrieval_rounds ?? 0,
    rewrite_rounds: value.rewrite_rounds ?? 0,
    review_score: value.review_score ?? 0,
    review_issues: value.review_issues ?? [],
    evidence_chunk_count: value.evidence_chunk_count ?? 0,
  };
}

function getSectionChunks(
  inferredStructure: PaperAnalysis["inferred_structure"],
  key: SectionKey
): AnalysisChunk[] {
  const sectionValue = inferredStructure?.[key];
  if (key === "methods" && sectionValue && !Array.isArray(sectionValue)) {
    return (sectionValue as MethodsStructure).chunks ?? [];
  }
  return (Array.isArray(sectionValue) ? sectionValue : []) ?? [];
}

function getMethodsEquations(
  inferredStructure: PaperAnalysis["inferred_structure"]
): EquationCollection | null {
  const methodsValue = inferredStructure?.methods;
  if (!methodsValue || Array.isArray(methodsValue)) {
    return null;
  }
  return (methodsValue as MethodsStructure).equations ?? null;
}

function buildMlDetails(analysis: PaperAnalysis): MLPaperDetails | null {
  const details = analysis.ml_details ?? null;
  if (details) return details;

  const losses = [
    analysis.losses?.primary,
    ...(analysis.losses?.auxiliary ?? []),
  ].filter((value): value is string => Boolean(value));
  const optimizers = [
    analysis.optimizers?.primary,
    ...(analysis.optimizers?.baseline ?? []),
    analysis.optimizer ?? undefined,
  ].filter((value, index, arr): value is string => Boolean(value) && arr.indexOf(value) === index);
  const datasets = [analysis.dataset ?? undefined].filter((value): value is string => Boolean(value));

  if (
    !analysis.model_architecture &&
    losses.length === 0 &&
    optimizers.length === 0 &&
    !analysis.training_objective &&
    datasets.length === 0
  ) {
    return null;
  }

  return {
    model_architecture: analysis.model_architecture ?? null,
    losses,
    optimizers,
    training_objective: analysis.training_objective ?? null,
    datasets,
  };
}

function buildTheoryDetails(analysis: PaperAnalysis): TheoryPaperDetails | null {
  return analysis.theory_details ?? analysis.theory ?? null;
}

function buildSystemsDetails(analysis: PaperAnalysis): SystemsPaperDetails | null {
  return analysis.systems_details ?? analysis.systems ?? null;
}

export default function AnalysisPanel({ analysis }: AnalysisPanelProps) {
  const domain = normalizeDomain(analysis?.domain);
  const inferredStructure = analysis?.inferred_structure;
  const synthesisOutput = analysis?.synthesis_output;
  const analysisStatus = analysis?.analysis_status;
  const methodsEquations = getMethodsEquations(inferredStructure);

  const sectionEntries = (Object.keys(SECTION_META) as SectionKey[])
    .map((key) => {
      const rawChunks = getSectionChunks(inferredStructure, key);
      const chunks = rawChunks
        .map(normalizeChunk)
        .filter((chunk) => chunk.text.length > 0);
      const sectionSynthesis = normalizeSectionSynthesis(synthesisOutput?.[key]);

      return {
        key,
        title: SECTION_META[key],
        sectionSynthesis,
        chunks,
        equations: key === "methods" ? methodsEquations : null,
      };
    })
    .filter(
      (entry) =>
        entry.chunks.length > 0 ||
        Boolean(entry.sectionSynthesis?.synthesis) ||
        Boolean(entry.sectionSynthesis?.warning)
    );

  const mlDetails = buildMlDetails(analysis);
  const theoryDetails = buildTheoryDetails(analysis);
  const systemsDetails = buildSystemsDetails(analysis);
  const repos = analysis?.repository_info?.repositories ?? [];
  const primaryRepo = analysis?.repository_info?.primary_repo;

  const structureMissing =
    (!inferredStructure && !synthesisOutput) ||
    sectionEntries.length === 0;

  return (
    <div className="animate-fade-in p-6">
      <section>
        <div className="rounded-xl border border-white/10 bg-bg-surface p-4 shadow-lg shadow-black/20">
          <div className="flex items-center gap-3">
            <div className="rounded-full border border-white/10 bg-bg-hover px-3 py-1 text-sm text-gray-300">
              {domain === "ml" ? "ML" : domain.charAt(0).toUpperCase() + domain.slice(1)}
            </div>
            <div className="text-sm text-gray-500">
              Confidence {Math.round((analysis?.domain_confidence ?? 0) * 100)}%
            </div>
          </div>
        </div>
      </section>

      {structureMissing ? (
        <section className="mt-8">
          <h2 className="text-base font-semibold text-white">Analysis Overview</h2>
          <div className="mt-4 rounded-xl border border-white/10 bg-bg-card p-5 shadow-lg shadow-black/20">
            <p className="text-sm leading-relaxed text-gray-300">
              {analysisStatus?.message ?? "Analysis structure unavailable for this paper."}
              {" "}Try re-analyzing or ask a question below.
            </p>
          </div>
        </section>
      ) : (
        sectionEntries.map((section) => (
          <SynthesisSection
            key={section.key}
            title={section.title}
            sectionResult={section.sectionSynthesis}
            chunks={section.chunks}
            equations={section.equations}
          />
        ))
      )}

      {domain === "ml" && mlDetails && (
        <DomainBlock
          title="Training Details"
          items={[
            { label: "Model Architecture", value: mlDetails.model_architecture },
            { label: "Losses", value: mlDetails.losses ?? [] },
            { label: "Optimizers", value: mlDetails.optimizers ?? [] },
            { label: "Training Objective", value: mlDetails.training_objective },
            { label: "Datasets", value: mlDetails.datasets ?? [] },
          ]}
        />
      )}

      {domain === "theory" && theoryDetails && (
        <DomainBlock
          title="Formal Results"
          items={[
            { label: "Theorems", value: theoryDetails.theorems ?? [] },
            { label: "Proofs", value: theoryDetails.proofs ?? [] },
            { label: "Complexity Claims", value: theoryDetails.complexity_claims ?? [] },
          ]}
        />
      )}

      {domain === "systems" && systemsDetails && (
        <DomainBlock
          title="System Details"
          items={[
            { label: "System Components", value: systemsDetails.system_components ?? [] },
            { label: "Performance Claims", value: systemsDetails.performance_claims ?? [] },
            { label: "Benchmarks", value: systemsDetails.benchmarks ?? [] },
          ]}
        />
      )}

      {repos.length > 0 && (
        <section className="mt-8">
          <h2 className="text-base font-semibold text-white">Code Repositories</h2>
          <div className="mt-4 space-y-3">
            {repos.map((repo) => {
              const isPrimary = repo.url === primaryRepo;
              return (
                <a
                  key={repo.url}
                  href={repo.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between gap-3 rounded-xl border border-white/10 bg-bg-card p-4 shadow-lg shadow-black/20 transition-transform duration-150 hover:-translate-y-[2px] hover:border-white/20"
                >
                  <div className="flex min-w-0 items-center gap-3">
                    <FolderGit2 className="h-4 w-4 shrink-0 text-gray-500" />
                    <div className="min-w-0">
                      <p className="truncate text-sm text-gray-300">{repo.url.replace("https://", "")}</p>
                      <p className="text-sm text-gray-500">
                        {repo.source}
                        {isPrimary ? " • Primary" : ""}
                      </p>
                    </div>
                  </div>
                  <div className="text-sm text-gray-500">{Math.round(repo.confidence * 100)}%</div>
                </a>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}
