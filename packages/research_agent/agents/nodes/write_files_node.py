from research_agent.services.experiment_generation_service import write_files_step


def write_files_node(state: dict) -> dict:
    return write_files_step(state)
