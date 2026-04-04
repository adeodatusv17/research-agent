from research_agent.services.experiment_generation_service import validate_artifact_step


def validate_artifact_node(state: dict) -> dict:
    return validate_artifact_step(state)
