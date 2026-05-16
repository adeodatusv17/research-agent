from research_agent.services.qa_orchestration_service import verify_answer_step


def verify_answer_node(state: dict) -> dict:
    return verify_answer_step(state)
