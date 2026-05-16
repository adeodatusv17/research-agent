from research_agent.services.qa_orchestration_service import evidence_diagnostics_step


def evidence_diagnostics_node(state: dict) -> dict:
    return evidence_diagnostics_step(state)
