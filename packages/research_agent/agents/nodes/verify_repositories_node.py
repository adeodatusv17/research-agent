from research_agent.services.experiment_generation_service import verify_repositories_step


def verify_repositories_node(state: dict) -> dict:
    return verify_repositories_step(state)
