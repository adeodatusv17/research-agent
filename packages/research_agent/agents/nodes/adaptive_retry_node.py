from research_agent.services.qa_orchestration_service import adaptive_retry_step


def adaptive_retry_node(state: dict) -> dict:
    return adaptive_retry_step(state)
