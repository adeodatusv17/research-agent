from research_agent.services.experiment_generation_service import generate_config_step


def generate_config_node(state: dict) -> dict:
    return generate_config_step(state)
