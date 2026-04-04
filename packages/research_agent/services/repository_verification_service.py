import base64
import logging
import os
import re
import uuid
from difflib import SequenceMatcher
from urllib.parse import urlparse

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from research_agent.domain.models.paper import Paper
from research_agent.domain.models.paper_analysis import PaperAnalysis
from research_agent.domain.models.paper_repository import PaperRepository
from research_agent.services.repository_discovery_service import discover_repositories


logger = logging.getLogger(__name__)

GITHUB_REPO_PATTERN = re.compile(r"^https?://github\.com/([^/]+)/([^/]+)/?$", re.IGNORECASE)
HIGH_CONFIDENCE_REPO_THRESHOLD = 0.75
MEDIUM_CONFIDENCE_REPO_THRESHOLD = 0.55
FILENAMEISH_TITLE_PATTERN = re.compile(r"(^\d{4}\.\d{5}(v\d+)?$)|(^[0-9a-f-]{20,}$)", re.IGNORECASE)


def _github_headers(accept: str = "application/vnd.github+json") -> dict[str, str]:
    headers = {"Accept": accept}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _repo_slug_from_url(url: str) -> tuple[str, str] | None:
    match = GITHUB_REPO_PATTERN.match(url.strip())
    if not match:
        return None
    return match.group(1), match.group(2)


def _looks_like_filename_title(title: str | None) -> bool:
    if not title:
        return True
    normalized = title.strip()
    return bool(FILENAMEISH_TITLE_PATTERN.match(normalized)) or "_" in normalized or len(normalized.split()) <= 1


def _extract_architecture_name(analysis: PaperAnalysis) -> str:
    architectures = analysis.architectures if isinstance(analysis.architectures, dict) else {}
    proposed = architectures.get("proposed", []) if isinstance(architectures, dict) else []
    if proposed:
        return str(proposed[0])
    return analysis.model_architecture or ""


def _fetch_github_json(path: str) -> dict | None:
    try:
        response = httpx.get(
            f"https://api.github.com{path}",
            timeout=10.0,
            headers=_github_headers(),
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        logger.warning("github_api_request_failed path=%s", path)
        return None


def _fetch_github_readme(owner: str, repo: str) -> str:
    payload = _fetch_github_json(f"/repos/{owner}/{repo}/readme")
    if not payload:
        return ""
    content = payload.get("content")
    encoding = payload.get("encoding")
    if not content or encoding != "base64":
        return ""
    try:
        return base64.b64decode(content).decode("utf-8", errors="ignore")
    except Exception:
        logger.exception("github_readme_decode_failed repo=%s/%s", owner, repo)
        return ""


def _fetch_github_contents(owner: str, repo: str) -> list[str]:
    payload = _fetch_github_json(f"/repos/{owner}/{repo}/contents")
    if not isinstance(payload, list):
        return []
    return [str(item.get("name", "")).lower() for item in payload]


def _fetch_repo_page(owner: str, repo: str) -> str:
    try:
        response = httpx.get(
            f"https://github.com/{owner}/{repo}",
            timeout=10.0,
            headers=_github_headers("text/html"),
        )
        response.raise_for_status()
        return response.text
    except Exception:
        logger.warning("github_repo_page_fetch_failed repo=%s/%s", owner, repo)
        return ""


def _fetch_raw_readme(owner: str, repo: str) -> str:
    for branch in ["main", "master"]:
        for filename in ["README.md", "readme.md", "README.rst"]:
            try:
                response = httpx.get(
                    f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}",
                    timeout=10.0,
                )
                if response.status_code == 200:
                    return response.text
            except Exception:
                logger.warning("github_raw_readme_fetch_failed repo=%s/%s branch=%s", owner, repo, branch)
    return ""


def _parse_stars_from_html(html: str) -> int:
    match = re.search(r"([\d,]+)\s+stars", html, re.IGNORECASE)
    if not match:
        return 0
    try:
        return int(match.group(1).replace(",", ""))
    except ValueError:
        return 0


def _score_repository(
    repo_url: str,
    paper_title: str,
    architecture_name: str,
    base_confidence: float,
) -> dict:
    slug = _repo_slug_from_url(repo_url)
    if slug is None:
        return {
            "url": repo_url,
            "trust_score": round(base_confidence, 4),
            "verification_signals": [],
            "stars": 0,
            "readme_excerpt": "",
        }

    owner, repo = slug
    repo_payload = _fetch_github_json(f"/repos/{owner}/{repo}") or {}
    contents = _fetch_github_contents(owner, repo)
    readme = _fetch_github_readme(owner, repo)
    repo_page_html = ""
    if not readme:
        readme = _fetch_raw_readme(owner, repo)
    if not readme or not contents:
        repo_page_html = _fetch_repo_page(owner, repo)
    readme_lower = readme.lower()
    repo_name = repo.replace("-", " ").replace("_", " ")
    stars = int(repo_payload.get("stargazers_count") or 0) or _parse_stars_from_html(repo_page_html)

    verification_signals: list[str] = []
    title_similarity = _similarity(repo_name, paper_title)
    architecture_similarity = _similarity(repo_name, architecture_name)
    trust_score = float(base_confidence)

    if title_similarity >= 0.55:
        trust_score += 0.18
        verification_signals.append("repo_name_similar_to_paper_title")
    if architecture_name and architecture_similarity >= 0.55:
        trust_score += 0.12
        verification_signals.append("repo_name_mentions_architecture")
    if paper_title and paper_title.lower() in readme_lower:
        trust_score += 0.2
        verification_signals.append("readme_mentions_paper_title")
    if architecture_name and architecture_name.lower() in readme_lower:
        trust_score += 0.12
        verification_signals.append("readme_mentions_architecture")
    if "train.py" in contents or "training" in readme_lower or "train.py" in repo_page_html.lower():
        trust_score += 0.08
        verification_signals.append("training_entrypoint_present")
    if (
        "requirements.txt" in contents
        or "environment.yml" in contents
        or "pyproject.toml" in contents
        or "requirements.txt" in repo_page_html.lower()
    ):
        trust_score += 0.05
        verification_signals.append("environment_spec_present")
    if stars > 50:
        trust_score += min(0.15, stars / 5000.0)
        verification_signals.append("community_signal_present")

    trust_score = round(min(1.0, trust_score), 4)
    return {
        "url": repo_url,
        "trust_score": trust_score,
        "verification_signals": verification_signals,
        "stars": stars,
        "readme_excerpt": readme[:600],
        "description": repo_payload.get("description") or "",
        "default_branch": repo_payload.get("default_branch") or "",
    }


def _load_stored_repositories(db: Session, paper_id: uuid.UUID) -> list[dict]:
    rows = (
        db.execute(
            select(PaperRepository).where(PaperRepository.paper_id == paper_id).order_by(PaperRepository.confidence.desc())
        )
        .scalars()
        .all()
    )
    return [{"url": row.repo_url, "source": row.source, "confidence": float(row.confidence)} for row in rows]


def _recommend_action(best_score: float) -> str:
    if best_score >= HIGH_CONFIDENCE_REPO_THRESHOLD:
        return "clone_existing_repo"
    if best_score >= MEDIUM_CONFIDENCE_REPO_THRESHOLD:
        return "review_repo_then_generate"
    return "use_generated_scaffold"


def get_repository_recommendation(db: Session, paper_id: uuid.UUID, paper: Paper, analysis: PaperAnalysis) -> dict:
    repositories = _load_stored_repositories(db, paper_id)
    paper_text = ""
    if not repositories and paper.pdf_storage_path:
        try:
            from research_agent.tools.pdf_text_extractor import extract_text_from_pdf

            paper_text = extract_text_from_pdf(paper.pdf_storage_path)
        except Exception:
            logger.exception("paper_text_reextract_failed paper_id=%s", paper_id)

    if not repositories:
        discovery = discover_repositories(
            {
                "model_architecture": analysis.model_architecture,
                "architectures": analysis.architectures,
            },
            paper_text,
            paper.title,
        )
        repositories = discovery.get("repositories", [])

    architecture_name = _extract_architecture_name(analysis)
    search_title = architecture_name if _looks_like_filename_title(paper.title) and architecture_name else paper.title
    deduped_repositories: dict[str, dict] = {}
    for repo in repositories:
        url = str(repo["url"]).rstrip("/")
        current = deduped_repositories.get(url)
        if current is None or float(repo.get("confidence", 0.0)) > float(current.get("confidence", 0.0)):
            deduped_repositories[url] = {**repo, "url": url}

    verified_repositories = []
    for repo in list(deduped_repositories.values())[:5]:
        verification = _score_repository(
            repo_url=str(repo["url"]),
            paper_title=search_title,
            architecture_name=architecture_name,
            base_confidence=float(repo.get("confidence", 0.0)),
        )
        verified_repositories.append(
            {
                "url": repo["url"],
                "source": repo.get("source", "unknown"),
                "confidence": float(repo.get("confidence", 0.0)),
                **verification,
            }
        )

    verified_repositories.sort(key=lambda item: item.get("trust_score", 0.0), reverse=True)
    top_repo = verified_repositories[0] if verified_repositories else None
    recommended_action = _recommend_action(float(top_repo.get("trust_score", 0.0)) if top_repo else 0.0)
    return {
        "repositories": verified_repositories,
        "primary_repo": top_repo["url"] if top_repo else None,
        "recommended_action": recommended_action,
        "search_title": search_title,
    }
