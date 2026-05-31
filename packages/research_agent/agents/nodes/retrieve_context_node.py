import logging

from research_agent.services.retrieval_service import semantic_retrieve_chunks


logger = logging.getLogger(__name__)


def retrieve_context_node(state: dict) -> dict:
    db = state["db"]
    paper_id = state["paper_id"]
    query_embedding = state["query_embedding"]
    retrieval_parameters = state.get("retrieval_parameters", {})
    section_names = [section["section_name"] for section in state.get("selected_sections", [])]
    subsection_names = [
        subsection.get("subsection_name") for subsection in state.get("retrieved_subsections", [])
    ]
    if retrieval_parameters.get("disable_subsection_filter"):
        subsection_names = []

    retrieved_chunks = semantic_retrieve_chunks(
        db,
        query_embedding=query_embedding,
        paper_id=paper_id,
        section_names=section_names or None,
        subsection_names=subsection_names or None,
        query_intent=state.get("query_type", "method"),
        semantic_top_k=int(retrieval_parameters.get("semantic_top_k", 18)),
        formula_mode=bool(retrieval_parameters.get("formula_mode", False)),
        debug_query=state.get("active_query") or state.get("query"),
    )

    logger.info(
        "qa_graph_node=retrieve_chunks paper_id=%s semantic_chunks_retrieved=%s section_filter=%s subsection_filter=%s",
        paper_id,
        len(retrieved_chunks),
        section_names,
        subsection_names,
    )
    return {
        **state,
        "retrieved_chunks": retrieved_chunks,
        "execution_trace": [*state.get("execution_trace", []), f"retrieve_context:{len(retrieved_chunks)}"],
    }
