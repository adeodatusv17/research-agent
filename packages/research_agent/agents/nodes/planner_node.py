from research_agent.services.qa_orchestration_service import planner_step


def planner_node(state: dict) -> dict:
    return planner_step(state)
