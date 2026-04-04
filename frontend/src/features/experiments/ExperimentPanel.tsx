"use client";

import { useState } from "react";
import {
  Loader2,
  FlaskConical,
  GitBranch,
  CheckCircle,
  AlertTriangle,
  ChevronDown,
  ChevronUp,
  ExternalLink,
  Terminal,
} from "lucide-react";
import { generateExperiment } from "@/lib/api-client";
import type { PaperAnalysis, ExperimentResult } from "@/lib/types";
import { ScoreBar, OverallScoreDial } from "@/components/ui/ScoreBar";
import toast from "react-hot-toast";
import clsx from "clsx";

interface ExperimentPanelProps {
  paperId: string;
  analysis: PaperAnalysis;
}

const SCORE_LABELS: { key: keyof NonNullable<PaperAnalysis["reproducibility"]>; label: string }[] = [
  { key: "hyperparameter_completeness", label: "Hyperparameter Completeness" },
  { key: "training_detail_score", label: "Training Detail Coverage" },
  { key: "evaluation_protocol_score", label: "Evaluation Protocol" },
];

function ReproducibilitySection({ repro }: { repro: NonNullable<PaperAnalysis["reproducibility"]> }) {
  return (
    <div className="card p-5 space-y-5">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <OverallScoreDial score={repro.overall_score} />
        <div className="flex gap-4">
          <div className="text-center">
            <div className={clsx("text-xs font-semibold", repro.dataset_available ? "text-status-success" : "text-status-error")}>
              {repro.dataset_available == null ? "Unknown" : repro.dataset_available ? "Public" : "Unavailable"}
            </div>
            <div className="text-[10px] text-text-muted mt-0.5">Dataset</div>
          </div>
          <div className="text-center">
            <div className={clsx("text-xs font-semibold", repro.code_available ? "text-status-success" : "text-status-warning")}>
              {repro.code_available ? "Available" : "Not Found"}
            </div>
            <div className="text-[10px] text-text-muted mt-0.5">Code</div>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {SCORE_LABELS.map(({ key, label }) => {
          const val = repro[key] as number;
          return <ScoreBar key={key} value={val} label={label} />;
        })}
      </div>

      {repro.summary && (
        <p className="text-xs text-text-secondary leading-relaxed border-t border-bg-border pt-3">
          {repro.summary}
        </p>
      )}
    </div>
  );
}

export default function ExperimentPanel({ paperId, analysis }: ExperimentPanelProps) {
  const [generating, setGenerating] = useState(false);
  const [result, setResult] = useState<ExperimentResult | null>(null);
  const [showValidation, setShowValidation] = useState(false);

  const repro = analysis.reproducibility;

  async function handleGenerate() {
    setGenerating(true);
    try {
      const data = await generateExperiment(paperId, analysis.domain ?? undefined);
      setResult(data);
      if (data.generation_status === "completed") {
        toast.success("Experiment scaffold generated!");
      } else {
        toast.error(data.error_message ?? "Generation failed");
      }
    } catch (err: unknown) {
      toast.error(err instanceof Error ? err.message : "Generation failed");
    } finally {
      setGenerating(false);
    }
  }

  return (
    <div className="p-6 space-y-5 animate-fade-in">
      {(analysis.domain ?? "general") === "ml" && (
        <div>
          <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Reproducibility Assessment</h3>
          {repro ? (
            <ReproducibilitySection repro={repro} />
          ) : (
            <div className="card p-5 text-center">
              <p className="text-xs text-text-muted">No reproducibility score yet - run analysis first.</p>
            </div>
          )}
        </div>
      )}

      <div>
        <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-3">Experiment Scaffold Generation</h3>

        {!result ? (
          <div className="card p-6 text-center space-y-4">
            <div className="w-12 h-12 rounded-xl bg-accent-violet/15 border border-accent-violet/25 flex items-center justify-center mx-auto shadow-glow-indigo/20">
              <FlaskConical className="w-5 h-5 text-accent-violet" />
            </div>
            <div>
              <p className="text-sm font-medium text-text-primary">Generate scaffold from analyzed structure</p>
              <p className="text-xs text-text-muted mt-1">
                ML papers produce training code; non-ML papers produce a reproducibility runbook scaffold.
              </p>
            </div>
            <button
              onClick={handleGenerate}
              disabled={generating}
              className={clsx("btn-primary mx-auto", generating && "opacity-60 cursor-not-allowed")}
            >
              {generating ? (
                <>
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <FlaskConical className="w-3.5 h-3.5" />
                  Generate Scaffold
                </>
              )}
            </button>
          </div>
        ) : (
          <div className="space-y-4 animate-slide-up">
            <div
              className={clsx(
                "card p-4 flex items-center gap-3",
                result.generation_status === "completed" ? "border-status-success/20" : "border-status-error/20"
              )}
            >
              {result.generation_status === "completed" ? (
                <CheckCircle className="w-5 h-5 text-status-success flex-shrink-0" />
              ) : (
                <AlertTriangle className="w-5 h-5 text-status-error flex-shrink-0" />
              )}
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-text-primary">
                  {result.generation_status === "completed" ? "Scaffold generated" : "Generation failed"}
                </p>
                <p className="text-xs text-text-muted font-mono truncate mt-0.5">{result.artifact_path}</p>
              </div>
            </div>

            {result.recommended_action && (
              <div className="flex items-center gap-2 text-xs">
                <Terminal className="w-3.5 h-3.5 text-accent-cyan flex-shrink-0" />
                <span className="text-text-secondary">Recommendation:</span>
                <span className="font-mono text-accent-cyan">{result.recommended_action}</span>
              </div>
            )}

            {result.primary_repo && (
              <a
                href={result.primary_repo}
                target="_blank"
                rel="noopener noreferrer"
                className="card p-3 flex items-center gap-2 hover:border-accent-indigo/30 transition-all cursor-pointer group"
              >
                <GitBranch className="w-3.5 h-3.5 text-accent-indigo flex-shrink-0" />
                <span className="text-xs text-text-secondary group-hover:text-text-primary transition-colors truncate flex-1">
                  {result.primary_repo.replace("https://", "")}
                </span>
                <ExternalLink className="w-3 h-3 text-text-muted group-hover:text-accent-indigo transition-colors flex-shrink-0" />
              </a>
            )}

            {(result.validation?.warnings?.length > 0 || result.validation?.errors?.length > 0) && (
              <div>
                <button
                  onClick={() => setShowValidation(!showValidation)}
                  className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-text-primary cursor-pointer transition-colors"
                >
                  <AlertTriangle className="w-3.5 h-3.5 text-status-warning" />
                  Validation ({result.validation.errors.length} errors, {result.validation.warnings.length} warnings)
                  {showValidation ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                </button>
                {showValidation && (
                  <div className="mt-2 space-y-1">
                    {result.validation.errors.map((e, i) => (
                      <p key={i} className="text-xs text-status-error bg-status-error/8 border border-status-error/20 rounded px-2 py-1">
                        {e}
                      </p>
                    ))}
                    {result.validation.warnings.map((w, i) => (
                      <p key={i} className="text-xs text-status-warning bg-status-warning/8 border border-status-warning/20 rounded px-2 py-1">
                        {w}
                      </p>
                    ))}
                  </div>
                )}
              </div>
            )}

            <button onClick={handleGenerate} disabled={generating} className="btn-secondary text-xs">
              {generating ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <FlaskConical className="w-3.5 h-3.5" />}
              Regenerate
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
