import logging

from research_agent.services.retrieval_service import semantic_retrieve_subsections


logger = logging.getLogger(__name__)


def retrieve_subsections_node(state: dict) -> dict:
    db = state["db"]
    paper_id = state["paper_id"]
    query_embedding = state["query_embedding"]
    section_names = [section["section_name"] for section in state.get("selected_sections", [])]

    retrieved_subsections = semantic_retrieve_subsections(
        db,
        query_embedding,
        paper_id,
        section_names=section_names,
    )

    logger.info(
        "qa_graph_node=retrieve_subsections paper_id=%s subsection_count=%s subsection_names=%s",
        paper_id,
        len(retrieved_subsections),
        [subsection.get("subsection_name") for subsection in retrieved_subsections],
    )
    return {
        **state,
        "retrieved_subsections": retrieved_subsections,
    }
