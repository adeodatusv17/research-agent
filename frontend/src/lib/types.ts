export type PaperDomain =
  | "ml"
  | "theory"
  | "systems"
  | "security"
  | "networks"
  | "general";

export type ChunkRole =
  | "problem"
  | "idea"
  | "method"
  | "algorithm"
  | "equation"
  | "formula"
  | "theory"
  | "evaluation"
  | "implementation"
  | "discussion"
  | "other";

export interface Paper {
  id: string;
  title: string;
  domain?: PaperDomain | null;
  domain_confidence?: number | null;
  source_type: "pdf";
  source_url: string | null;
  pdf_storage_path: string | null;
  created_at: string;
}

export interface PaperListResponse {
  items: Paper[];
}

export interface UploadResponse {
  paper_id: string;
}

export interface AnalysisChunk {
  id?: string;
  text?: string;
  summary?: string;
  role?: ChunkRole | string;
  importance?: number;
  confidence?: number;
  source?: "extracted" | "inferred";
  section_name?: string | null;
  chunk_index?: number;
}

export interface EquationItem {
  id?: string | null;
  chunk_index?: number | null;
  latex?: string | null;
  description?: string | null;
  text?: string | null;
  source?: "extracted" | "llm_generated" | null;
}

export interface EquationCollection {
  source?: "extracted" | "llm_generated" | null;
  items?: EquationItem[] | null;
}

export interface MethodsStructure {
  chunks?: AnalysisChunk[] | null;
  equations?: EquationCollection | null;
}

export interface InferredStructure {
  key_ideas?: AnalysisChunk[] | null;
  methods?: AnalysisChunk[] | MethodsStructure | null;
  results?: AnalysisChunk[] | null;
  discussion?: AnalysisChunk[] | null;
}

export type SectionConfidence = "high" | "medium" | "low";

export interface SectionSynthesisResult {
  synthesis: string;
  confidence: SectionConfidence;
  warning?: string | null;
  fabrication_flagged: boolean;
  retrieval_rounds: number;
  rewrite_rounds: number;
  review_score: number;
  review_issues: string[];
  evidence_chunk_count: number;
}

export interface SynthesisOutput {
  key_ideas?: SectionSynthesisResult | string | null;
  methods?: SectionSynthesisResult | string | null;
  results?: SectionSynthesisResult | string | null;
  discussion?: SectionSynthesisResult | string | null;
}

export interface AnalysisStatus {
  status: "success" | "partial_failure" | "failed";
  message?: string | null;
  successful_sections: string[];
  failed_sections: string[];
}

export interface ReproducibilityResult {
  overall_score: number;
  artifact_availability?: number;
  methodology_completeness?: number;
  result_reproducibility?: number;
  dataset_available?: boolean | null;
  code_available?: boolean;
  hyperparameter_completeness?: number;
  training_detail_score?: number;
  evaluation_protocol_score?: number;
  summary?: string;
  evidence?: Record<string, unknown>;
}

export interface MLPaperDetails {
  model_architecture?: string | null;
  losses?: string[];
  optimizers?: string[];
  training_objective?: string | null;
  datasets?: string[];
}

export interface TheoryPaperDetails {
  theorems?: string[];
  proofs?: string[];
  complexity_claims?: string[];
}

export interface SystemsPaperDetails {
  system_components?: string[];
  performance_claims?: string[];
  benchmarks?: string[];
}

export interface ArchitectureInfo {
  proposed: string[];
  baseline: string[];
}

export interface LossInfo {
  primary: string | null;
  auxiliary: string[];
  baseline: string[];
  inferred: boolean;
  confidence: number;
}

export interface OptimizerInfo {
  primary: string | null;
  baseline: string[];
}

export interface RepoEntry {
  url: string;
  source: "paper_link" | "paperswithcode" | "github_search";
  confidence: number;
}

export interface RepositoryInfo {
  repositories: RepoEntry[];
  primary_repo: string | null;
  extracted_urls?: string[];
  search_title?: string;
}

export interface PaperAnalysis {
  id: string;
  paper_id: string;
  domain?: PaperDomain | null;
  domain_confidence?: number | null;
  inferred_structure?: InferredStructure | null;
  synthesis_output?: SynthesisOutput | null;
  analysis_status?: AnalysisStatus | null;
  synthesis_generated_at?: string | null;
  reproducibility?: ReproducibilityResult | null;

  ml_details?: MLPaperDetails | null;
  theory_details?: TheoryPaperDetails | null;
  systems_details?: SystemsPaperDetails | null;

  // Legacy/backward-compatible ML fields
  model_architecture?: string | null;
  architectures?: ArchitectureInfo | null;
  dataset?: string | null;
  loss_function?: string | null;
  losses?: LossInfo | null;
  training_objective?: string | null;
  optimizer?: string | null;
  optimizers?: OptimizerInfo | null;
  training_details?: Record<string, unknown> | null;
  evaluation_metrics?: string[] | null;
  contributions?: string[] | null;
  theory?: TheoryPaperDetails | null;
  systems?: SystemsPaperDetails | null;

  repository_info: RepositoryInfo;
  created_at: string;
}

export interface QASource {
  paper_id: string;
  section_name: string | null;
  subsection_name: string | null;
  page_number: number | null;
  content: string;
  content_snippet: string;
  score: number;
}

export interface AnswerTiers {
  evidence_backed: string[];
  inferred_from_evidence: string[];
  general_background: string[];
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export interface QAResponse {
  request_id?: string;
  answer: string;
  sources: QASource[];
  answer_tiers?: AnswerTiers;
  equations?: EquationCollection | null;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: QASource[];
  equations?: EquationCollection | null;
  timestamp: number;
}

export interface SearchResultItem {
  chunk_id: string;
  paper_id: string;
  chunk_index: number;
  section_name: string | null;
  subsection_name: string | null;
  page_number: number | null;
  title: string;
  content: string;
  score: number;
}

export interface SearchResponse {
  results: SearchResultItem[];
}

export interface ExperimentValidation {
  errors: string[];
  warnings: string[];
}

export interface ExperimentResult {
  experiment_id: string;
  artifact_path: string;
  generation_status: "completed" | "failed" | "pending";
  recommended_action: "use_generated_scaffold" | "use_primary_repo" | string;
  primary_repo: string | null;
  repositories: RepoEntry[];
  validation: ExperimentValidation;
  error_message?: string;
}

export type AsyncState<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "success"; data: T }
  | { status: "error"; message: string };
