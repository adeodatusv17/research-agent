from research_agent.services.qa_orchestration_service import generate_grounded_answer_step


def generate_answer_node(state: dict) -> dict:
    return generate_grounded_answer_step(state)
