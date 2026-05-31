import logging

from research_agent.services.retrieval_service import semantic_retrieve_subsections


logger = logging.getLogger(__name__)


def retrieve_subsections_node(state: dict) -> dict:
    db = state["db"]
    paper_id = state["paper_id"]
    query_embedding = state["query_embedding"]
    section_names = [section["section_name"] for section in state.get("selected_sections", [])]
    retrieval_parameters = state.get("retrieval_parameters", {})

    retrieved_subsections = semantic_retrieve_subsections(
        db,
        query_embedding,
        paper_id,
        section_names=section_names,
        subsection_top_k=int(retrieval_parameters.get("subsection_top_k", 6)),
        formula_mode=bool(retrieval_parameters.get("formula_mode", False)),
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
        "execution_trace": [*state.get("execution_trace", []), f"retrieve_subsections:{len(retrieved_subsections)}"],
    }
