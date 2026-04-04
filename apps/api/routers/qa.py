import uuid

from pydantic import BaseModel
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.services.rag_service import answer_question


router = APIRouter(prefix="/papers")


class QARequest(BaseModel):
    query: str


@router.post("/{paper_id}/qa")
def ask_question(
    paper_id: uuid.UUID,
    payload: QARequest,
    db: Session = Depends(get_db),
) -> dict[str, object]:
    return answer_question(db, paper_id, payload.query)
