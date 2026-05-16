from research_agent.services.qa_orchestration_service import critique_answer_step


def critique_answer_node(state: dict) -> dict:
    return critique_answer_step(state)
