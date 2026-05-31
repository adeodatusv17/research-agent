import logging

from research_agent.services.retrieval_service import balanced_select_chunks


logger = logging.getLogger(__name__)


def rerank_chunks_node(state: dict) -> dict:
    retrieval_parameters = state.get("retrieval_parameters", {})
    filtered_chunks = balanced_select_chunks(
        state.get("retrieved_chunks", []),
        selected_sections=state.get("selected_sections", []),
        query_intent=state.get("query_type", "method"),
        top_k=int(retrieval_parameters.get("final_top_k", 12)),
        formula_mode=bool(retrieval_parameters.get("formula_mode", False)),
    )
    retrieval_confidence = (
        sum(
            0.7 * float(chunk.get("score", 0.0)) + 0.3 * float(chunk.get("importance", 0.0))
            for chunk in filtered_chunks
        )
        / len(filtered_chunks)
        if filtered_chunks
        else 0.0
    )

    logger.info(
        "qa_graph_node=rerank_chunks paper_id=%s filtered_chunks=%s top_sections=%s retrieval_confidence=%s top_chunk_summary=%s",
        state.get("paper_id"),
        len(filtered_chunks),
        [section["section_name"] for section in state.get("selected_sections", [])],
        round(retrieval_confidence, 4),
        [
            {
                "section": chunk.get("section_name"),
                "subsection": chunk.get("subsection_name"),
                "semantic_score": round(float(chunk["score"]), 4),
                "role": chunk.get("role"),
                "importance": round(float(chunk.get("importance", 0.0)), 4),
            }
            for chunk in filtered_chunks[:5]
        ],
    )
    return {
        **state,
        "filtered_chunks": filtered_chunks,
        "retrieval_confidence": retrieval_confidence,
        "execution_trace": [*state.get("execution_trace", []), f"rerank_chunks:{len(filtered_chunks)}"],
    }
