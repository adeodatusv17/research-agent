from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.tools.embedder import generate_embedding
from research_agent.tools.vector_store import semantic_search


router = APIRouter(prefix="/papers")


@router.get("/search")
def search_papers(query: str, db: Session = Depends(get_db)) -> dict[str, object]:
    query_embedding = generate_embedding(query, task="query")
    results = semantic_search(db, query_embedding, top_k=20)
    return {"results": results}
