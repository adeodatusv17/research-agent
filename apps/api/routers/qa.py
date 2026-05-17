import uuid

from pydantic import BaseModel
from fastapi import APIRouter, Depends, Header
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
    x_request_id: str | None = Header(default=None),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    request_id = x_request_id or str(uuid.uuid4())
    response = answer_question(db, paper_id, payload.query, request_id=request_id)
    return {
        **response,
        "request_id": request_id,
    }
