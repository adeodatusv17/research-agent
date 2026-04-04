from research_agent.services.experiment_generation_service import store_metadata_step


def store_metadata_node(state: dict) -> dict:
    return store_metadata_step(state)
