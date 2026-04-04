import logging

from research_agent.services.retrieval_service import semantic_retrieve_sections
from research_agent.tools.embedder import generate_embedding


logger = logging.getLogger(__name__)


def retrieve_sections_node(state: dict) -> dict:
    db = state["db"]
    query = state["query"]
    paper_id = state["paper_id"]
    query_type = state.get("query_type", "method")
    query_embedding = generate_embedding(query)
    selected_sections = semantic_retrieve_sections(
        db,
        query_embedding,
        paper_id,
        query_type=query_type,
    )

    logger.info(
        "qa_graph_node=retrieve_sections paper_id=%s query_type=%s sections_selected=%s section_names=%s",
        paper_id,
        query_type,
        len(selected_sections),
        [section["section_name"] for section in selected_sections],
    )
    return {
        **state,
        "query_embedding": query_embedding,
        "retrieved_sections": selected_sections,
        "selected_sections": selected_sections,
    }
