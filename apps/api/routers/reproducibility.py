import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.domain.models.reproducibility_score import ReproducibilityScore


router = APIRouter()


@router.get("/{paper_id}")
def get_reproducibility_score(paper_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, object]:
    score = (
        db.query(ReproducibilityScore)
        .filter(ReproducibilityScore.paper_id == paper_id)
        .order_by(ReproducibilityScore.created_at.desc())
        .first()
    )
    if score is None:
        raise HTTPException(status_code=404, detail="Reproducibility score not found")

    return {
        "paper_id": str(paper_id),
        "dataset_available": score.dataset_available,
        "code_available": score.code_available,
        "hyperparameter_completeness": score.hyperparameter_completeness,
        "training_detail_score": score.training_detail_score,
        "evaluation_protocol_score": score.evaluation_protocol_score,
        "overall_score": score.overall_score,
        "summary": score.summary,
        "evidence": score.evidence,
        "created_at": score.created_at.isoformat() if score.created_at else None,
    }
