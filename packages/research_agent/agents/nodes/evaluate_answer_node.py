from research_agent.services.qa_orchestration_service import evaluate_answer_step


def evaluate_answer_node(state: dict) -> dict:
    return evaluate_answer_step(state)
