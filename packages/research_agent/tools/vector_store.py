import uuid

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.domain.models.paper_section import PaperSection
from research_agent.domain.models.paper_subsection import PaperSubsection


def store_chunks(
    db: Session,
    paper_id,
    chunks: list[dict[str, int | str | None]],
    embeddings: list[list[float]],
) -> list[PaperChunk]:
    if len(chunks) != len(embeddings):
        raise ValueError("chunks and embeddings must have the same length")

    records = [
        PaperChunk(
            paper_id=paper_id,
            chunk_index=int(chunk["chunk_index"]),
            section_name=str(chunk["section_name"]) if chunk.get("section_name") else None,
            subsection_name=str(chunk["subsection_name"]) if chunk.get("subsection_name") else None,
            content=str(chunk["content"]),
            token_count=int(chunk["token_count"]),
            embedding=embedding,
        )
        for chunk, embedding in zip(chunks, embeddings, strict=False)
    ]

    db.add_all(records)
    db.flush()
    return records


def store_sections(
    db: Session,
    paper_id,
    sections: list[dict[str, str | int | None]],
    embeddings: list[list[float]],
) -> list[PaperSection]:
    if len(sections) != len(embeddings):
        raise ValueError("sections and embeddings must have the same length")

    records = [
        PaperSection(
            paper_id=paper_id,
            section_name=str(section["section_name"]),
            section_heading=str(section["section_heading"]) if section.get("section_heading") else None,
            section_order=int(section["section_order"]),
            content=str(section["content"]),
            token_count=int(section["token_count"]) if section.get("token_count") else None,
            embedding=embedding,
        )
        for section, embedding in zip(sections, embeddings, strict=False)
    ]

    db.add_all(records)
    db.flush()
    return records


def store_subsections(
    db: Session,
    paper_id,
    subsections: list[dict[str, str | int | None]],
    embeddings: list[list[float]],
) -> list[PaperSubsection]:
    if len(subsections) != len(embeddings):
        raise ValueError("subsections and embeddings must have the same length")

    records = [
        PaperSubsection(
            paper_id=paper_id,
            section_name=str(subsection["section_name"]),
            subsection_name=str(subsection["subsection_name"]) if subsection.get("subsection_name") else None,
            page_number=int(subsection["page_number"]) if subsection.get("page_number") else None,
            content=str(subsection["content"]),
            token_count=int(subsection["token_count"]) if subsection.get("token_count") else None,
            embedding=embedding,
        )
        for subsection, embedding in zip(subsections, embeddings, strict=False)
    ]

    db.add_all(records)
    db.flush()
    return records


def semantic_search(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID | None = None,
    section_names: list[str] | None = None,
    subsection_names: list[str | None] | None = None,
    top_k: int = 20,
) -> list[dict]:
    subsection_page_number = (
        select(PaperSubsection.page_number)
        .where(
            PaperSubsection.paper_id == PaperChunk.paper_id,
            PaperSubsection.section_name == PaperChunk.section_name,
            or_(
                and_(
                    PaperChunk.subsection_name.is_(None),
                    PaperSubsection.subsection_name.is_(None),
                ),
                PaperSubsection.subsection_name == PaperChunk.subsection_name,
            ),
        )
        .limit(1)
        .scalar_subquery()
    )

    distance = PaperChunk.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            PaperChunk.id,
            PaperChunk.paper_id,
            PaperChunk.chunk_index,
            PaperChunk.section_name,
            PaperChunk.subsection_name,
            subsection_page_number.label("page_number"),
            Paper.title,
            PaperChunk.content,
            (1 - distance).label("score"),
        )
        .join(Paper, Paper.id == PaperChunk.paper_id)
    )

    if paper_id is not None:
        statement = statement.where(PaperChunk.paper_id == paper_id)

    if section_names:
        statement = statement.where(PaperChunk.section_name.in_(section_names))

    if subsection_names:
        statement = statement.where(
            or_(
                PaperChunk.subsection_name.in_([name for name in subsection_names if name is not None]),
                and_(PaperChunk.subsection_name.is_(None), None in subsection_names),
            )
        )

    statement = statement.order_by(distance).limit(top_k)

    rows = db.execute(statement).all()
    return [
        {
            "chunk_id": str(row.id),
            "paper_id": str(row.paper_id),
            "chunk_index": int(row.chunk_index),
            "section_name": row.section_name,
            "subsection_name": row.subsection_name,
            "page_number": row.page_number,
            "title": row.title,
            "content": row.content,
            "score": float(row.score),
        }
        for row in rows
    ]


def semantic_search_sections(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    top_k: int = 6,
) -> list[dict]:
    distance = PaperSection.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            PaperSection.id,
            PaperSection.paper_id,
            PaperSection.section_name,
            PaperSection.section_heading,
            PaperSection.section_order,
            PaperSection.content,
            (1 - distance).label("score"),
        )
        .where(PaperSection.paper_id == paper_id)
        .order_by(distance)
        .limit(top_k)
    )

    rows = db.execute(statement).all()
    return [
        {
            "section_id": str(row.id),
            "paper_id": str(row.paper_id),
            "section_name": row.section_name,
            "section_heading": row.section_heading,
            "section_order": int(row.section_order),
            "content": row.content,
            "score": float(row.score),
        }
        for row in rows
    ]


def semantic_search_subsections(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID,
    section_names: list[str] | None = None,
    top_k: int = 8,
) -> list[dict]:
    distance = PaperSubsection.embedding.cosine_distance(query_embedding)
    statement = (
        select(
            PaperSubsection.id,
            PaperSubsection.paper_id,
            PaperSubsection.section_name,
            PaperSubsection.subsection_name,
            PaperSubsection.page_number,
            PaperSubsection.content,
            (1 - distance).label("score"),
        )
        .where(PaperSubsection.paper_id == paper_id)
    )

    if section_names:
        statement = statement.where(PaperSubsection.section_name.in_(section_names))

    statement = statement.order_by(distance).limit(top_k)
    rows = db.execute(statement).all()
    return [
        {
            "subsection_id": str(row.id),
            "paper_id": str(row.paper_id),
            "section_name": row.section_name,
            "subsection_name": row.subsection_name,
            "page_number": row.page_number,
            "content": row.content,
            "score": float(row.score),
        }
        for row in rows
    ]
