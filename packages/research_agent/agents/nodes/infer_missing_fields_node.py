from research_agent.services.experiment_generation_service import infer_missing_fields_step


def infer_missing_fields_node(state: dict) -> dict:
    return infer_missing_fields_step(state)
