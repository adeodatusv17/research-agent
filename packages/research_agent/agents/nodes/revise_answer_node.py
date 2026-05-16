from research_agent.services.qa_orchestration_service import revise_answer_step


def revise_answer_node(state: dict) -> dict:
    return revise_answer_step(state)
