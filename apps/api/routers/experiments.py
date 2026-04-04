import uuid

from fastapi import APIRouter

from research_agent.services.experiment_generation_service import generate_experiment

router = APIRouter()


@router.post("/{paper_id}/generate")
def generate_experiment_route(paper_id: uuid.UUID, domain: str | None = None) -> dict[str, object]:
    return generate_experiment(str(paper_id), domain=domain)
