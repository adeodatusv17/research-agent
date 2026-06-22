import uuid

from sqlalchemy import and_, or_, select
from sqlalchemy.orm import Session

from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.domain.models.paper_section import PaperSection
from research_agent.domain.models.paper_subsection import PaperSubsection
from research_agent.domain.models.paper_table import PaperTable


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


def store_tables(
    db: Session,
    paper_id,
    tables: list[dict[str, object]],
    embeddings: list[list[float]],
) -> list[PaperTable]:
    if len(tables) != len(embeddings):
        raise ValueError("tables and embeddings must have the same length")

    records = [
        PaperTable(
            paper_id=paper_id,
            table_index=int(table["table_index"]),
            table_label=str(table["table_label"]) if table.get("table_label") else None,
            caption=str(table["caption"]) if table.get("caption") else None,
            section_name=str(table["section_name"]) if table.get("section_name") else None,
            subsection_name=str(table["subsection_name"]) if table.get("subsection_name") else None,
            page_number=int(table["page_number"]) if table.get("page_number") else None,
            raw_table_text=str(table["raw_table_text"]),
            normalized_table_text=str(table["normalized_table_text"]),
            table_type=str(table["table_type"]) if table.get("table_type") else None,
            metric_names=list(table.get("metric_names") or []),
            dataset_names=list(table.get("dataset_names") or []),
            model_names=list(table.get("model_names") or []),
            linked_chunk_indexes=list(table.get("linked_chunk_indexes") or []),
            token_count=int(table["token_count"]) if table.get("token_count") else None,
            embedding=embedding,
        )
        for table, embedding in zip(tables, embeddings, strict=False)
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


def fetch_neighbor_chunks(
    db: Session,
    *,
    paper_id: uuid.UUID | str,
    section_name: str | None,
    subsection_name: str | None,
    chunk_indexes: list[int],
) -> list[dict]:
    if not chunk_indexes:
        return []
    if isinstance(paper_id, str):
        paper_id = uuid.UUID(paper_id)

    statement = (
        select(
            PaperChunk.id,
            PaperChunk.paper_id,
            PaperChunk.chunk_index,
            PaperChunk.section_name,
            PaperChunk.subsection_name,
            PaperChunk.content,
        )
        .where(
            PaperChunk.paper_id == paper_id,
            PaperChunk.chunk_index.in_(chunk_indexes),
        )
        .order_by(PaperChunk.chunk_index.asc())
    )

    if section_name is None:
        statement = statement.where(PaperChunk.section_name.is_(None))
    else:
        statement = statement.where(PaperChunk.section_name == section_name)

    if subsection_name is None:
        statement = statement.where(PaperChunk.subsection_name.is_(None))
    else:
        statement = statement.where(PaperChunk.subsection_name == subsection_name)

    rows = db.execute(statement).all()
    return [
        {
            "chunk_id": str(row.id),
            "paper_id": str(row.paper_id),
            "chunk_index": int(row.chunk_index),
            "section_name": row.section_name,
            "subsection_name": row.subsection_name,
            "content": row.content,
        }
        for row in rows
    ]


def semantic_search_tables(
    db: Session,
    query_embedding: list[float],
    paper_id: uuid.UUID | None = None,
    top_k: int = 10,
) -> list[dict]:
    distance = PaperTable.embedding.cosine_distance(query_embedding)
    statement = select(
        PaperTable.id,
        PaperTable.paper_id,
        PaperTable.table_index,
        PaperTable.table_label,
        PaperTable.caption,
        PaperTable.section_name,
        PaperTable.subsection_name,
        PaperTable.page_number,
        PaperTable.raw_table_text,
        PaperTable.normalized_table_text,
        PaperTable.table_type,
        PaperTable.metric_names,
        PaperTable.dataset_names,
        PaperTable.model_names,
        PaperTable.linked_chunk_indexes,
        (1 - distance).label("score"),
    )

    if paper_id is not None:
        statement = statement.where(PaperTable.paper_id == paper_id)

    statement = statement.order_by(distance).limit(top_k)
    rows = db.execute(statement).all()
    return [
        {
            "table_id": str(row.id),
            "paper_id": str(row.paper_id),
            "table_index": int(row.table_index),
            "table_label": row.table_label,
            "caption": row.caption,
            "section_name": row.section_name,
            "subsection_name": row.subsection_name,
            "page_number": row.page_number,
            "raw_table_text": row.raw_table_text,
            "normalized_table_text": row.normalized_table_text,
            "table_type": row.table_type,
            "metric_names": row.metric_names or [],
            "dataset_names": row.dataset_names or [],
            "model_names": row.model_names or [],
            "linked_chunk_indexes": row.linked_chunk_indexes or [],
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
