from research_agent.services.qa_orchestration_service import analyze_query_step


def query_analysis_node(state: dict) -> dict:
    return analyze_query_step(state)
