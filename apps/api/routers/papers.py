from fastapi import APIRouter, Depends
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.domain.models.paper import Paper


router = APIRouter()


@router.get("/")
def list_papers(db: Session = Depends(get_db)) -> dict[str, list]:
    papers = db.scalars(select(Paper).order_by(Paper.created_at.desc())).all()
    return {
        "items": [
            {
                "id": str(p.id),
                "title": p.title,
                "domain": (p.domain or "general"),
                "domain_confidence": p.domain_confidence,
                "source_type": p.source_type,
                "source_url": p.source_url,
                "pdf_storage_path": p.pdf_storage_path,
                "created_at": p.created_at.isoformat() if p.created_at else None,
            }
            for p in papers
        ]
    }
