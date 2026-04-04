import shutil
import uuid
import logging
import traceback
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.domain.models.paper import Paper
from research_agent.services.paper_indexing_service import index_paper_document
from research_agent.tools.embedder import (
    get_embedding_model_name,
    get_max_embed_tokens,
)
from research_agent.tools.pdf_parser import extract_title_from_pages, parse_pdf
from research_agent.tools.pdf_text_extractor import extract_text_from_pdf


router = APIRouter(prefix="/papers")

STORAGE_DIR = Path("storage/papers")
logger = logging.getLogger(__name__)


@router.post("/upload")
async def upload_pdf(file: UploadFile = File(...), db: Session = Depends(get_db)) -> dict[str, str]:
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    paper_id = uuid.uuid4()
    safe_name = file.filename or f"{paper_id}.pdf"
    destination = STORAGE_DIR / f"{paper_id}_{Path(safe_name).name}"
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)

    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    stage = "paper_record"
    section_segments: list[dict[str, str | int | None]] = []
    indexed_sections: list[dict[str, str | int | None]] = []
    indexed_subsections: list[dict[str, str | int | None]] = []
    chunks: list[dict[str, int | str | None]] = []
    max_chunk_tokens = 0

    try:
        pages = parse_pdf(str(destination))
        extracted_title = extract_title_from_pages(pages) or Path(safe_name).stem

        paper = Paper(
            id=paper_id,
            title=extracted_title,
            source_type="pdf",
            source_url=None,
            pdf_storage_path=str(destination),
        )
        db.add(paper)
        db.flush()

        stage = "pdf_extraction"
        text = extract_text_from_pdf(str(destination))

        stage = "indexing"
        index_result = index_paper_document(db, paper, pages or text)
        section_segments = list(index_result["section_segments"])
        indexed_sections = list(index_result["indexed_sections"])
        indexed_subsections = list(index_result["indexed_subsections"])
        chunks = list(index_result["chunks"])
        max_chunk_tokens = int(index_result["max_chunk_tokens"])

        logger.info(
            "chunking_complete paper_id=%s number_of_sections=%s number_of_subsections=%s number_of_chunk_segments=%s number_of_chunks=%s max_chunk_tokens=%s domain=%s domain_confidence=%s embedding_model=%s",
            paper_id,
            len(indexed_sections),
            len(indexed_subsections),
            len(section_segments),
            len(chunks),
            max_chunk_tokens,
            paper.domain,
            paper.domain_confidence,
            get_embedding_model_name(),
        )

        stage = "commit"
        db.commit()

        logger.info(
            "upload_complete paper_id=%s sections_created=%s subsections_created=%s chunks_created=%s max_chunk_tokens=%s embedding_count=%s embedding_model=%s",
            paper_id,
            len(indexed_sections),
            len(indexed_subsections),
            len(chunks),
            max_chunk_tokens,
            len(chunks),
            get_embedding_model_name(),
        )
    except Exception as exc:
        db.rollback()
        if destination.exists():
            destination.unlink(missing_ok=True)
        failed_chunk_index = None
        if chunks and stage == "indexing":
            for index, chunk in enumerate(chunks):
                if int(chunk["token_count"]) > get_max_embed_tokens():
                    failed_chunk_index = index
                    break

        logger.error(
            "upload_failed paper_id=%s stage=%s number_of_chunks=%s max_chunk_tokens=%s "
            "embedding_model=%s chunk_index_on_failure=%s error_type=%s traceback=%s",
            paper_id,
            stage,
            len(chunks),
            max_chunk_tokens,
            get_embedding_model_name(),
            failed_chunk_index,
            type(exc).__name__,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed during {stage}: {type(exc).__name__}",
        ) from exc

    return {"paper_id": str(paper.id)}


@router.get("/{paper_id}")
def get_paper(paper_id: uuid.UUID, db: Session = Depends(get_db)) -> dict[str, object]:
    paper = db.get(Paper, paper_id)
    if paper is None:
        raise HTTPException(status_code=404, detail="Paper not found")

    return {
        "id": str(paper.id),
        "title": paper.title,
        "domain": paper.domain or "general",
        "domain_confidence": paper.domain_confidence,
        "source_type": paper.source_type,
        "source_url": paper.source_url,
        "pdf_storage_path": paper.pdf_storage_path,
        "created_at": paper.created_at.isoformat() if paper.created_at else None,
    }
