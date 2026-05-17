from typing import Literal, TypedDict


ConfidenceLabel = Literal["low", "medium", "high"]
EvaluationBand = Literal["insufficient", "needs_review", "sufficient", "strong"]


class PlannerSubtask(TypedDict, total=False):
    subtask_id: str
    title: str
    question: str
    evidence_types: list[str]
    priority: str


class ExecutionPlan(TypedDict, total=False):
    strategy: str
    complexity: str
    decomposition_rationale: str
    required_evidence: list[str]
    missing_information: list[str]
    subtasks: list[PlannerSubtask]


class RetrievalParameters(TypedDict, total=False):
    section_top_k: int
    subsection_top_k: int
    semantic_top_k: int
    final_top_k: int
    include_references: bool
    disable_subsection_filter: bool
    expanded_scope: bool


class RetrievalAttempt(TypedDict, total=False):
    attempt: int
    query: str
    reason: str
    expanded_scope: bool
    missing_required_fields: list[str]


class EvidenceDiagnostics(TypedDict, total=False):
    evidence_density: float
    section_coverage: float
    contradiction_ratio: float
    retrieval_diversity: float
    retrieval_strength: float
    missing_required_fields: list[str]
    covered_evidence: dict[str, bool]
    weak_signals: list[str]
    sufficient: bool
    should_retry: bool


class GroundedClaim(TypedDict, total=False):
    claim_id: str
    claim_text: str
    tier: str
    supporting_chunk_ids: list[str]
    confidence: float
    evidence_coverage: float
    contradiction_count: int
    support_strength: float
    support_status: str


class AnswerTiers(TypedDict, total=False):
    evidence_backed: list[str]
    inferred_from_evidence: list[str]
    general_background: list[str]


class VerificationReport(TypedDict, total=False):
    used_llm: bool
    supported_claim_ratio: float
    unsupported_claim_ids: list[str]
    weak_claim_ids: list[str]
    issues: list[str]
    status: str


class CritiqueReport(TypedDict, total=False):
    used_llm: bool
    issues: list[str]
    revision_focus: list[str]
    should_revise: bool
    severity: str


class EvaluationReport(TypedDict, total=False):
    grounding_quality: float
    evidence_coverage: float
    reasoning_depth: float
    scientific_rigor: float
    critique_usefulness: float
    contradiction_handling: float
    completeness: float
    overall_status: EvaluationBand
    summary: list[str]


class QAState(TypedDict, total=False):
    db: object
    query: str
    active_query: str
    query_type: str
    paper_id: str | None
    query_analysis: dict
    query_embedding: list[float]
    analysis_hits: dict
    orchestration_level: int
    should_plan: bool
    should_verify: bool
    should_critique: bool
    retry_budget: int
    revision_budget: int
    retry_count: int
    revision_count: int
    execution_plan: ExecutionPlan
    retrieval_parameters: RetrievalParameters
    retrieval_attempts: list[RetrievalAttempt]
    retrieved_sections: list[dict]
    selected_sections: list[dict]
    retrieved_subsections: list[dict]
    retrieved_chunks: list[dict]
    filtered_chunks: list[dict]
    retrieval_confidence: float
    evidence_diagnostics: EvidenceDiagnostics
    context: str
    answer: str
    answer_tiers: AnswerTiers
    grounded_claims: list[GroundedClaim]
    final_confidence: float
    verifier_report: VerificationReport
    critic_report: CritiqueReport
    evaluation_report: EvaluationReport
    sources: list[dict]
    execution_trace: list[str]
