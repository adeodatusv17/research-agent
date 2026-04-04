from research_agent.services.experiment_generation_service import generate_code_step


def generate_code_node(state: dict) -> dict:
    return generate_code_step(state)
