import logging
import uuid

from sqlalchemy import delete
from sqlalchemy.orm import Session

from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_analysis import PaperAnalysis
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.domain.models.paper_repository import PaperRepository
from research_agent.domain.models.paper_section import PaperSection
from research_agent.domain.models.paper_subsection import PaperSubsection
from research_agent.domain.models.reproducibility_score import ReproducibilityScore
from research_agent.services.chunk_structure import prepare_chunks_for_indexing
from research_agent.services.chunking_service import split_sections_into_chunks
from research_agent.services.domain_detector import detect_domain
from research_agent.services.section_parser_service import (
    build_section_index_entries,
    build_subsection_index_entries,
    parse_sections,
)
from research_agent.tools.embedder import (
    generate_embeddings,
    get_embedding_model_name,
    normalize_texts_for_embedding,
)
from research_agent.tools.vector_store import store_chunks, store_sections, store_subsections


logger = logging.getLogger(__name__)


def reset_paper_indices(db: Session, paper_id: uuid.UUID) -> None:
    db.execute(delete(PaperChunk).where(PaperChunk.paper_id == paper_id))
    db.execute(delete(PaperSection).where(PaperSection.paper_id == paper_id))
    db.execute(delete(PaperSubsection).where(PaperSubsection.paper_id == paper_id))
    db.execute(delete(PaperAnalysis).where(PaperAnalysis.paper_id == paper_id))
    db.execute(delete(PaperRepository).where(PaperRepository.paper_id == paper_id))
    db.execute(delete(ReproducibilityScore).where(ReproducibilityScore.paper_id == paper_id))


def index_paper_document(
    db: Session,
    paper: Paper,
    document: str | list[dict],
    *,
    replace_existing: bool = False,
) -> dict[str, object]:
    section_segments = parse_sections(document)
    indexed_sections = build_section_index_entries(section_segments)
    indexed_subsections = build_subsection_index_entries(section_segments)

    section_embedding_texts, section_token_counts = (
        normalize_texts_for_embedding([str(section["content"]) for section in indexed_sections])
        if indexed_sections
        else ([], [])
    )
    for section, token_count in zip(indexed_sections, section_token_counts, strict=False):
        section["token_count"] = token_count

    subsection_embedding_texts, subsection_token_counts = (
        normalize_texts_for_embedding([str(subsection["content"]) for subsection in indexed_subsections])
        if indexed_subsections
        else ([], [])
    )
    for subsection, token_count in zip(indexed_subsections, subsection_token_counts, strict=False):
        subsection["token_count"] = token_count

    raw_chunks = split_sections_into_chunks(section_segments)
    chunks = prepare_chunks_for_indexing(raw_chunks)
    max_chunk_tokens = max((int(chunk["token_count"]) for chunk in chunks), default=0)

    top_chunk_texts = [
        str(chunk["content"])
        for chunk in sorted(
            chunks,
            key=lambda row: (-(int(row.get("token_count") or 0)), int(row.get("chunk_index") or 0)),
        )[:20]
    ]
    domain_info = detect_domain(top_chunk_texts)
    paper.domain = str(domain_info.get("domain") or "general")
    paper.domain_confidence = float(domain_info.get("confidence") or 0.0)

    if replace_existing:
        reset_paper_indices(db, paper.id)

    chunk_embeddings = generate_embeddings([str(chunk["content"]) for chunk in chunks]) if chunks else []
    section_embeddings = generate_embeddings(section_embedding_texts) if indexed_sections else []
    subsection_embeddings = generate_embeddings(subsection_embedding_texts) if indexed_subsections else []

    store_sections(db, paper.id, indexed_sections, section_embeddings)
    store_subsections(db, paper.id, indexed_subsections, subsection_embeddings)
    store_chunks(db, paper.id, chunks, chunk_embeddings)

    logger.info(
        "paper_indexing_complete paper_id=%s sections=%s subsections=%s raw_chunks=%s kept_chunks=%s max_chunk_tokens=%s domain=%s domain_confidence=%s embedding_model=%s replace_existing=%s",
        paper.id,
        len(indexed_sections),
        len(indexed_subsections),
        len(raw_chunks),
        len(chunks),
        max_chunk_tokens,
        paper.domain,
        paper.domain_confidence,
        get_embedding_model_name(),
        replace_existing,
    )

    return {
        "section_segments": section_segments,
        "indexed_sections": indexed_sections,
        "indexed_subsections": indexed_subsections,
        "raw_chunks": raw_chunks,
        "chunks": chunks,
        "max_chunk_tokens": max_chunk_tokens,
    }
