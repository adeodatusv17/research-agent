from research_agent.services.experiment_generation_service import load_analysis_step


def load_analysis_node(state: dict) -> dict:
    return load_analysis_step(state)
