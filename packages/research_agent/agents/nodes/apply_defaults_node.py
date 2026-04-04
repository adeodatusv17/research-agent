from research_agent.services.experiment_generation_service import apply_defaults_step


def apply_defaults_node(state: dict) -> dict:
    return apply_defaults_step(state)
