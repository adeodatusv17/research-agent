import logging

from research_agent.services.rag_service import generate_answer_from_chunks


logger = logging.getLogger(__name__)


def generate_answer_node(state: dict) -> dict:
    answer_payload = generate_answer_from_chunks(
        query=state["query"],
        paper_id=state["paper_id"],
        filtered_chunks=state.get("filtered_chunks", []),
    )

    logger.info(
        "qa_graph_node=generate_answer paper_id=%s sources=%s",
        state.get("paper_id"),
        len(answer_payload["sources"]),
    )
    return {
        **state,
        "context": answer_payload["context"],
        "answer": answer_payload["answer"],
        "sources": answer_payload["sources"],
    }
