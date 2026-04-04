import logging

from research_agent.services.retrieval_service import classify_query


logger = logging.getLogger(__name__)


def query_analysis_node(state: dict) -> dict:
    query = state["query"].strip()
    keywords = [token.lower() for token in query.replace("?", " ").split() if token.strip()]
    query_type = classify_query(query)

    logger.info(
        "qa_graph_node=query_analysis paper_id=%s query=%s keyword_count=%s query_type=%s",
        state.get("paper_id"),
        query,
        len(keywords),
        query_type,
    )
    return {
        **state,
        "query_type": query_type,
        "query": query,
        "query_analysis": {
            "keywords": keywords,
            "length": len(query),
            "query_type": query_type,
        },
    }
