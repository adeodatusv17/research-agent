import argparse
import logging
import sys
import uuid
from pathlib import Path

from sqlalchemy import select


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGES_DIR = PROJECT_ROOT / "packages"
if str(PACKAGES_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGES_DIR))

from research_agent.domain.models.paper import Paper
from research_agent.infrastructure.db.session import SessionLocal
from research_agent.services.paper_analysis_service import analyze_paper
from research_agent.services.paper_indexing_service import index_paper_document
from research_agent.tools.pdf_parser import parse_pdf
from research_agent.tools.pdf_text_extractor import extract_text_from_pdf


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def reanalyze_paper(paper_id: str) -> None:
    paper_uuid = uuid.UUID(paper_id)
    with SessionLocal() as db:
        paper = db.get(Paper, paper_uuid)
        if paper is None:
            raise ValueError(f"Paper not found: {paper_id}")
        if not paper.pdf_storage_path:
            raise ValueError(f"Paper {paper_id} does not have a stored PDF path")

        pdf_path = Path(paper.pdf_storage_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"Stored PDF not found for paper {paper_id}: {pdf_path}")

        logger.info("reanalyze_paper_start paper_id=%s title=%s", paper_id, paper.title)
        try:
            pages = parse_pdf(str(pdf_path))
            text = extract_text_from_pdf(str(pdf_path))
            index_result = index_paper_document(db, paper, pages or text, replace_existing=True)
            db.commit()
            analyze_paper(db, paper_uuid)
            logger.info(
                "reanalyze_paper_complete paper_id=%s kept_chunks=%s raw_chunks=%s domain=%s",
                paper_id,
                len(index_result["chunks"]),
                len(index_result["raw_chunks"]),
                paper.domain,
            )
        except Exception:
            db.rollback()
            logger.exception("reanalyze_paper_failed paper_id=%s", paper_id)
            raise


def reanalyze_all() -> None:
    with SessionLocal() as db:
        paper_ids = [str(paper_id) for paper_id in db.scalars(select(Paper.id).order_by(Paper.created_at)).all()]

    succeeded = 0
    failed = 0
    for paper_id in paper_ids:
        try:
            reanalyze_paper(paper_id)
            succeeded += 1
        except Exception:
            failed += 1
            logger.exception("reanalyze_all_item_failed paper_id=%s", paper_id)

    summary = f"Reanalysis complete: {succeeded} succeeded, {failed} failed."
    logger.info(summary)
    print(summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reanalyze papers after chunk quality updates.")
    parser.add_argument("--paper-id", help="Reanalyze a single paper by UUID.")
    parser.add_argument("--all", action="store_true", help="Reanalyze all papers.")
    args = parser.parse_args()

    if args.paper_id:
        reanalyze_paper(args.paper_id)
        return
    if args.all:
        reanalyze_all()
        return
    parser.error("Provide --paper-id <uuid> or --all.")


if __name__ == "__main__":
    main()
