import logging
import uuid
from time import perf_counter

from sqlalchemy.orm import Session

from research_agent.services.paper_analysis_service import get_structured_answer_if_available
from research_agent.services.qa_orchestration_service import resolve_conversational_query
from research_agent.tools.embedder import get_tokenizer
from research_agent.tools.gemini_client import generate_answer


logger = logging.getLogger(__name__)
MAX_CONTEXT_TOKENS = 2000

CODE_INTENT_KEYWORDS = [
    "code",
    "implement",
    "write",
    "example",
    "snippet",
    "how to build",
    "python",
]

def _has_code_intent(query: str) -> bool:
    q = query.lower()
    return any(keyword in q for keyword in CODE_INTENT_KEYWORDS)


def generate_answer_from_chunks(
    query: str,
    paper_id: uuid.UUID,
    filtered_chunks: list[dict],
) -> dict[str, object]:
    tokenizer = get_tokenizer()
    context_parts: list[str] = []
    current_tokens = 0
    used_chunks: list[dict] = []

    for chunk in filtered_chunks:
        section_name = chunk.get("section_name") or "unknown"
        subsection_name = chunk.get("subsection_name")
        location = f"Section: {section_name}"
        if subsection_name:
            location += f" | Subsection: {subsection_name}"
        part = f"[Paper {chunk['paper_id']} | {location} | Score {chunk['score']:.4f}]\n{chunk['content']}"
        part_tokens = len(
            tokenizer(
                part,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                verbose=False,
            )["input_ids"]
        )

        if current_tokens + part_tokens > MAX_CONTEXT_TOKENS:
            break

        context_parts.append(part)
        current_tokens += part_tokens
        used_chunks.append(chunk)

    context = "\n\n".join(context_parts)
    prompt = (
        "You are a knowledgeable research assistant.\n\n"
        "Use the provided research paper context to answer the question as thoroughly as possible.\n"
        "If the user asks for code or an implementation, generate complete, working Python code based on "
        "the architecture, equations, and methods described in the context. Do not refuse to generate code.\n"
        "If specific implementation details are not in the context, use your knowledge of the described "
        "architecture to fill in reasonable defaults.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{query}"
    )

    logger.info(
        "rag_generation query=%s paper_id=%s number_of_chunks_used=%s retrieved_chunk_ids=%s similarity_scores=%s",
        query,
        paper_id,
        len(used_chunks),
        [chunk["chunk_id"] for chunk in used_chunks],
        [round(float(chunk["score"]), 4) for chunk in used_chunks],
    )

    answer = generate_answer(prompt)
    return {
        "context": context,
        "answer": answer,
        "sources": [
            {
                "paper_id": chunk["paper_id"],
                "section_name": chunk.get("section_name"),
                "subsection_name": chunk.get("subsection_name"),
                "page_number": chunk.get("page_number"),
                "content": chunk["content"],
                "content_snippet": chunk["content"][:400],
                "score": chunk["score"],
            }
            for chunk in used_chunks
        ],
    }


def answer_question(
    db: Session,
    paper_id: uuid.UUID,
    query: str,
    *,
    recent_turns: list[dict[str, str]] | None = None,
    request_id: str | None = None,
) -> dict[str, object]:
    effective_request_id = request_id or str(uuid.uuid4())
    started_at = perf_counter()
    conversation_resolution = resolve_conversational_query(query, recent_turns or [])
    retrieval_query = str(conversation_resolution.get("rewritten_query") or query).strip() or query
    logger.info(
        "qa_request_started request_id=%s paper_id=%s query=%s followup_rewrite=%s retrieval_query=%s",
        effective_request_id,
        paper_id,
        query,
        bool(conversation_resolution.get("is_follow_up")),
        retrieval_query,
    )
    # Skip the structured shortcut for code/implementation requests — it only returns
    # a brief summary like "Proposed: Conformer; Baselines: Transformer" which is not
    # useful when the user explicitly asks for code or a detailed explanation.
    if not _has_code_intent(query) and not conversation_resolution.get("is_follow_up"):
        structured_answer = get_structured_answer_if_available(db, paper_id, query)
        if structured_answer is not None:
            logger.info(
                "structured_analysis_shortcut request_id=%s paper_id=%s query=%s analysis_hits=%s",
                effective_request_id,
                paper_id,
                query,
                list(structured_answer.get("analysis_hits", {}).keys()),
            )
            payload = {
                "answer": structured_answer["answer"],
                "sources": structured_answer["sources"],
                "orchestration_level": 0,
                "query_type": "structured_shortcut",
                "execution_plan": {
                    "strategy": "structured_shortcut",
                    "complexity": "simple",
                    "decomposition_rationale": "Direct answer from stored structured analysis.",
                    "required_evidence": [],
                    "missing_information": [],
                    "subtasks": [],
                },
                "evidence_diagnostics": {
                    "evidence_density": 1.0,
                    "section_coverage": 1.0,
                    "contradiction_ratio": 0.0,
                    "retrieval_diversity": 1.0,
                    "retrieval_strength": 1.0,
                    "missing_required_fields": [],
                    "covered_evidence": {},
                    "weak_signals": [],
                    "sufficient": True,
                    "should_retry": False,
                },
                "answer_tiers": {
                    "evidence_backed": [structured_answer["answer"]],
                    "inferred_from_evidence": [],
                    "general_background": [],
                },
                "equations": {"source": None, "items": []},
                "grounded_claims": [],
                "retrieval_confidence": 1.0,
                "final_confidence": 1.0,
                "verifier_report": {
                    "used_llm": False,
                    "supported_claim_ratio": 1.0,
                    "unsupported_claim_ids": [],
                    "weak_claim_ids": [],
                    "issues": [],
                    "status": "pass",
                },
                "critic_report": {
                    "used_llm": False,
                    "issues": [],
                    "revision_focus": [],
                    "should_revise": False,
                    "severity": "low",
                },
                "evaluation_report": {
                    "grounding_quality": 1.0,
                    "evidence_coverage": 1.0,
                    "reasoning_depth": 1.0,
                    "scientific_rigor": 1.0,
                    "critique_usefulness": 0.0,
                    "contradiction_handling": 1.0,
                    "completeness": 1.0,
                    "overall_status": "strong",
                    "summary": ["Answered directly from structured analysis."],
                },
                "retrieval_attempts": [],
                "execution_trace": ["structured_shortcut"],
            }
            logger.info(
                "qa_request_completed request_id=%s paper_id=%s orchestration_level=0 duration_ms=%s",
                effective_request_id,
                paper_id,
                round((perf_counter() - started_at) * 1000, 2),
            )
            return payload

    from research_agent.agents.graphs.research_qa_graph import build_research_qa_graph

    graph = build_research_qa_graph()
    try:
        final_state = graph.invoke(
            {
                "db": db,
                "query": query,
                "active_query": retrieval_query,
                "query_type": "",
                "formula_mode": False,
                "paper_id": paper_id,
                "recent_turns": recent_turns or [],
                "query_embedding": [],
                "analysis_hits": {},
                "orchestration_level": 1,
                "should_plan": False,
                "should_verify": False,
                "should_critique": False,
                "retry_budget": 0,
                "revision_budget": 0,
                "retry_count": 0,
                "revision_count": 0,
                "execution_plan": {},
                "retrieval_parameters": {},
                "retrieval_attempts": [],
                "retrieved_sections": [],
                "selected_sections": [],
                "retrieved_subsections": [],
                "retrieved_chunks": [],
                "filtered_chunks": [],
                "retrieval_confidence": 0.0,
                "evidence_diagnostics": {},
                "context": "",
                "answer": "",
                "answer_tiers": {},
                "equations": {},
                "grounded_claims": [],
                "final_confidence": 0.0,
                "verifier_report": {},
                "critic_report": {},
                "evaluation_report": {},
                "sources": [],
                "execution_trace": [],
            }
        )
    except Exception:
        logger.exception(
            "qa_request_failed request_id=%s paper_id=%s duration_ms=%s",
            effective_request_id,
            paper_id,
            round((perf_counter() - started_at) * 1000, 2),
        )
        raise
    payload = {
        "answer": final_state["answer"],
        "sources": final_state["sources"],
        "query_type": final_state.get("query_type"),
        "orchestration_level": final_state.get("orchestration_level"),
        "execution_plan": final_state.get("execution_plan"),
        "retrieval_confidence": final_state.get("retrieval_confidence"),
        "evidence_diagnostics": final_state.get("evidence_diagnostics"),
        "answer_tiers": final_state.get("answer_tiers"),
        "equations": final_state.get("equations"),
        "grounded_claims": final_state.get("grounded_claims"),
        "final_confidence": final_state.get("final_confidence"),
        "verifier_report": final_state.get("verifier_report"),
        "critic_report": final_state.get("critic_report"),
        "evaluation_report": final_state.get("evaluation_report"),
        "retrieval_attempts": final_state.get("retrieval_attempts"),
        "execution_trace": final_state.get("execution_trace"),
    }
    logger.info(
        "qa_request_completed request_id=%s paper_id=%s orchestration_level=%s duration_ms=%s retrieval_confidence=%s final_confidence=%s",
        effective_request_id,
        paper_id,
        final_state.get("orchestration_level"),
        round((perf_counter() - started_at) * 1000, 2),
        final_state.get("retrieval_confidence"),
        final_state.get("final_confidence"),
    )
    return payload
