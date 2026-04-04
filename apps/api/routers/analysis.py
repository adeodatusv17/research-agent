import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from apps.api.dependencies import get_db
from research_agent.domain.models.paper_analysis import PaperAnalysis
from research_agent.domain.models.paper_chunk import PaperChunk
from research_agent.domain.models.paper_repository import PaperRepository
from research_agent.domain.models.reproducibility_score import ReproducibilityScore
from research_agent.services.paper_analysis_service import analyze_paper, build_domain_view


router = APIRouter(prefix="/papers")
SECTION_KEYS = ("key_ideas", "methods", "results", "discussion")


def _hydrate_inferred_structure(
    db: Session,
    paper_id: uuid.UUID,
    inferred_structure: dict | list | None,
) -> dict:
    if not isinstance(inferred_structure, dict):
        return {}

    chunk_rows = db.scalars(
        select(PaperChunk).where(PaperChunk.paper_id == paper_id).order_by(PaperChunk.chunk_index)
    ).all()
    if not chunk_rows:
        return inferred_structure

    by_id = {str(chunk.id): chunk for chunk in chunk_rows}
    by_index = {int(chunk.chunk_index): chunk for chunk in chunk_rows}
    hydrated: dict[str, list] = {}

    def hydrate_items(items) -> list[dict]:
        if not isinstance(items, list):
            return []
        hydrated_items: list[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue

            chunk = None
            chunk_id = item.get("id")
            chunk_index = item.get("chunk_index")
            if chunk_id is not None:
                chunk = by_id.get(str(chunk_id))
            if chunk is None and chunk_index is not None:
                try:
                    chunk = by_index.get(int(chunk_index))
                except (TypeError, ValueError):
                    chunk = None

            full_text = str(item.get("text") or "").strip()
            if chunk is not None and (not full_text or full_text == str(item.get("summary") or "").strip()):
                full_text = str(chunk.content).strip()

            hydrated_items.append(
                {
                    **item,
                    "id": str(chunk.id) if chunk is not None else item.get("id"),
                    "text": full_text or str(item.get("summary") or "").strip(),
                    "source": item.get("source") or ("extracted" if chunk is not None else "inferred"),
                }
            )
        return hydrated_items

    for key, items in inferred_structure.items():
        if key == "methods" and isinstance(items, dict):
            equations = items.get("equations") if isinstance(items.get("equations"), dict) else {}
            hydrated[key] = {
                "chunks": hydrate_items(items.get("chunks")),
                "equations": {
                    "source": equations.get("source"),
                    "items": hydrate_items(equations.get("items")),
                },
            }
            continue

        hydrated[key] = hydrate_items(items)

    return hydrated


def _normalize_section_synthesis(value) -> dict | None:
    if value is None:
        return None
    if isinstance(value, str):
        return {
            "synthesis": value.strip(),
            "confidence": "high",
            "warning": None,
            "fabrication_flagged": False,
            "retrieval_rounds": 0,
            "rewrite_rounds": 0,
            "review_score": 0,
            "review_issues": [],
            "evidence_chunk_count": 0,
        }
    if isinstance(value, dict):
        return {
            "synthesis": str(value.get("synthesis") or "").strip(),
            "confidence": str(value.get("confidence") or "low"),
            "warning": str(value.get("warning")).strip() if value.get("warning") else None,
            "fabrication_flagged": bool(value.get("fabrication_flagged")),
            "retrieval_rounds": int(value.get("retrieval_rounds") or 0),
            "rewrite_rounds": int(value.get("rewrite_rounds") or 0),
            "review_score": int(value.get("review_score") or 0),
            "review_issues": [
                str(issue).strip()
                for issue in (value.get("review_issues") or [])
                if str(issue).strip()
            ],
            "evidence_chunk_count": int(value.get("evidence_chunk_count") or 0),
        }
    return None


def _normalize_synthesis_output(synthesis_output) -> tuple[dict | None, dict]:
    if not isinstance(synthesis_output, dict):
        return None, {
            "status": "failed",
            "message": "No section synthesis is stored for this paper.",
            "successful_sections": [],
            "failed_sections": list(SECTION_KEYS),
        }

    normalized: dict[str, dict | None] = {}
    for key in SECTION_KEYS:
        normalized[key] = _normalize_section_synthesis(synthesis_output.get(key))

    successful_sections = [key for key, value in normalized.items() if isinstance(value, dict)]
    failed_sections = [key for key in SECTION_KEYS if key not in successful_sections]
    if failed_sections and successful_sections:
        status = "partial_failure"
        message = "Some section agents failed."
    elif failed_sections and not successful_sections:
        status = "failed"
        message = "All section agents failed."
    else:
        status = "success"
        message = None
    return normalized, {
        "status": status,
        "message": message,
        "successful_sections": successful_sections,
        "failed_sections": failed_sections,
    }


@router.post("/{paper_id}/analyze")
def analyze_paper_route(paper_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    try:
        return analyze_paper(db, paper_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{paper_id}/analysis")
def get_analysis(paper_id: uuid.UUID, db: Session = Depends(get_db)) -> dict:
    """Return the most recent stored analysis for a paper without re-running the LLM pipeline."""
    analysis = (
        db.query(PaperAnalysis)
        .filter(PaperAnalysis.paper_id == paper_id)
        .order_by(PaperAnalysis.created_at.desc())
        .first()
    )
    if analysis is None:
        raise HTTPException(
            status_code=404,
            detail="No analysis found for this paper. Run POST /papers/{paper_id}/analyze first.",
        )

    # Fetch associated data
    repositories = (
        db.scalars(
            select(PaperRepository)
            .where(PaperRepository.paper_id == paper_id)
            .order_by(PaperRepository.confidence.desc())
        ).all()
    )
    reproducibility = (
        db.query(ReproducibilityScore)
        .filter(ReproducibilityScore.paper_id == paper_id)
        .order_by(ReproducibilityScore.created_at.desc())
        .first()
    )
    domain = (analysis.domain or "general").lower()
    inferred_structure = _hydrate_inferred_structure(db, paper_id, analysis.inferred_structure)
    synthesis_output, analysis_status = _normalize_synthesis_output(analysis.synthesis_output)
    domain_view = build_domain_view(db, paper_id, domain, inferred_structure)
    ml_view = domain_view.get("ml") if isinstance(domain_view.get("ml"), dict) else {}

    result: dict = {
        "id": str(analysis.id),
        "paper_id": str(paper_id),
        "domain": domain,
        "inferred_structure": inferred_structure,
        "analysis_status": analysis_status,
        "synthesis_output": synthesis_output,
        "synthesis_generated_at": (
            analysis.synthesis_generated_at.isoformat() if analysis.synthesis_generated_at else None
        ),
        "model_architecture": analysis.model_architecture or ml_view.get("model_architecture"),
        "architectures": analysis.architectures or ml_view.get("architectures"),
        "dataset": analysis.dataset or ml_view.get("dataset"),
        "loss_function": analysis.loss_function or ml_view.get("loss_function"),
        "losses": analysis.losses or ml_view.get("losses"),
        "training_objective": analysis.training_objective or ml_view.get("training_objective"),
        "optimizer": analysis.optimizer or ml_view.get("optimizer"),
        "optimizers": analysis.optimizers or ml_view.get("optimizers"),
        "training_details": analysis.training_details or ml_view.get("training_details"),
        "evaluation_metrics": analysis.evaluation_metrics or ml_view.get("evaluation_metrics"),
        "contributions": analysis.contributions or ml_view.get("contributions"),
        "created_at": analysis.created_at.isoformat() if analysis.created_at else None,
        "repository_info": {
            "repositories": [
                {
                    "url": repo.repo_url,
                    "source": repo.source,
                    "confidence": repo.confidence,
                }
                for repo in repositories
            ],
            "primary_repo": repositories[0].repo_url if repositories else None,
        },
    }

    if reproducibility is not None:
        result["reproducibility"] = {
            "dataset_available": reproducibility.dataset_available,
            "code_available": reproducibility.code_available,
            "hyperparameter_completeness": reproducibility.hyperparameter_completeness,
            "training_detail_score": reproducibility.training_detail_score,
            "evaluation_protocol_score": reproducibility.evaluation_protocol_score,
            "overall_score": reproducibility.overall_score,
            "summary": reproducibility.summary,
            "evidence": reproducibility.evidence,
        }
    else:
        result["reproducibility"] = None

    if "theory" in domain_view:
        result["theory"] = domain_view["theory"]
    if "systems" in domain_view:
        result["systems"] = domain_view["systems"]

    return result
