import logging
import os
import re
from difflib import SequenceMatcher
from urllib.parse import quote

import httpx


logger = logging.getLogger(__name__)

GITHUB_URL_PATTERN = re.compile(r"https?://github\.com/[^\s)\]]+", re.IGNORECASE)
URL_PATTERN = re.compile(r"https?://[^\s)\]]+", re.IGNORECASE)
FILENAMEISH_TITLE_PATTERN = re.compile(r"(^\d{4}\.\d{5}(v\d+)?$)|(^[0-9a-f-]{20,}$)", re.IGNORECASE)


def _github_headers() -> dict[str, str]:
    headers = {"Accept": "application/vnd.github+json"}
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _text_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left.lower(), right.lower()).ratio()


def _normalize_url(url: str) -> str:
    return url.rstrip(".,);]")


def _looks_like_filename_title(title: str | None) -> bool:
    if not title:
        return True
    normalized = title.strip()
    return bool(FILENAMEISH_TITLE_PATTERN.match(normalized)) or "_" in normalized or len(normalized.split()) <= 1


def _extract_title_from_text(paper_text: str) -> str | None:
    for line in paper_text.splitlines():
        candidate = " ".join(line.split()).strip()
        if not candidate:
            continue
        if len(candidate) < 12 or len(candidate) > 200:
            continue
        lowered = candidate.lower()
        if lowered in {"abstract", "introduction"}:
            continue
        if "@" in candidate or "{" in candidate:
            continue
        if sum(1 for char in candidate if char.isalpha()) < 10:
            continue
        return candidate
    return None


def _resolve_search_title(paper_title: str, paper_text: str, architecture_name: str | None) -> str:
    extracted_title = _extract_title_from_text(paper_text)
    if extracted_title and (_looks_like_filename_title(paper_title) or len(extracted_title) > len(paper_title)):
        return extracted_title
    if paper_title and not _looks_like_filename_title(paper_title):
        return paper_title
    return architecture_name or paper_title


def _extract_github_urls(paper_text: str) -> list[str]:
    return sorted({_normalize_url(match.group(0)) for match in GITHUB_URL_PATTERN.finditer(paper_text)})


def _extract_urls(paper_text: str) -> list[str]:
    return sorted({_normalize_url(match.group(0)) for match in URL_PATTERN.finditer(paper_text)})


def _score_repo_candidate(url: str, source: str, paper_title: str, architecture_name: str | None) -> float:
    lowered_url = url.lower()
    repo_name = lowered_url.rstrip("/").split("/")[-1].replace("-", " ").replace("_", " ")
    title_score = _text_similarity(repo_name, paper_title)
    architecture_score = _text_similarity(repo_name, architecture_name or "")
    source_bonus = {
        "paper_link": 0.35,
        "paperswithcode": 0.25,
        "github_search": 0.15,
    }.get(source, 0.0)
    return round(min(1.0, 0.25 + title_score * 0.45 + architecture_score * 0.2 + source_bonus), 4)


def _search_paperswithcode(paper_title: str) -> list[dict]:
    try:
        query = quote(paper_title)
        response = httpx.get(
            f"https://paperswithcode.com/api/v1/papers/?q={query}",
            timeout=10.0,
            headers={"Accept": "application/json"},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        logger.exception("paperswithcode_search_failed title=%s", paper_title)
        return []

    results: list[dict] = []
    for paper in payload.get("results", [])[:3]:
        repo_url = paper.get("repository", {}).get("url")
        if repo_url:
            results.append({"url": repo_url, "source": "paperswithcode"})
    return results


def _search_github(paper_title: str, architecture_name: str | None) -> list[dict]:
    queries = [paper_title]
    if architecture_name and architecture_name.lower() not in paper_title.lower():
        queries.append(f"{paper_title} {architecture_name}")

    results: list[dict] = []
    for query in queries[:2]:
        try:
            response = httpx.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 3},
                timeout=10.0,
                headers=_github_headers(),
            )
            response.raise_for_status()
            payload = response.json()
        except Exception:
            logger.exception("github_search_failed query=%s", query)
            continue

        for item in payload.get("items", [])[:3]:
            results.append(
                {
                    "url": item.get("html_url"),
                    "source": "github_search",
                    "stars": item.get("stargazers_count", 0),
                    "description": item.get("description") or "",
                }
            )
    return results


def discover_repositories(paper_analysis: dict, paper_text: str, paper_title: str) -> dict:
    architecture_name = paper_analysis.get("model_architecture") or ", ".join(
        (paper_analysis.get("architectures") or {}).get("proposed", [])[:1]
    )
    search_title = _resolve_search_title(paper_title, paper_text, architecture_name)

    candidates: list[dict] = []
    mentioned_github_urls = _extract_github_urls(paper_text)
    for url in mentioned_github_urls:
        candidates.append(
            {
                "url": url,
                "source": "paper_link",
                "confidence": _score_repo_candidate(url, "paper_link", search_title, architecture_name),
            }
        )

    for candidate in _search_paperswithcode(search_title):
        if candidate.get("url"):
            candidates.append(
                {
                    "url": candidate["url"],
                    "source": candidate["source"],
                    "confidence": _score_repo_candidate(
                        candidate["url"], candidate["source"], search_title, architecture_name
                    ),
                }
            )

    for candidate in _search_github(search_title, architecture_name):
        if candidate.get("url"):
            confidence = _score_repo_candidate(
                candidate["url"], candidate["source"], search_title, architecture_name
            )
            confidence = min(1.0, confidence + min(0.15, float(candidate.get("stars", 0)) / 10000.0))
            candidates.append(
                {
                    "url": candidate["url"],
                    "source": candidate["source"],
                    "confidence": round(confidence, 4),
                }
            )

    deduped: dict[str, dict] = {}
    for candidate in candidates:
        url = _normalize_url(candidate["url"])
        existing = deduped.get(url)
        if existing is None or float(candidate["confidence"]) > float(existing["confidence"]):
            deduped[url] = {
                "url": url,
                "source": candidate["source"],
                "confidence": float(candidate["confidence"]),
            }

    repositories = sorted(deduped.values(), key=lambda repo: repo["confidence"], reverse=True)
    return {
        "repositories": repositories,
        "primary_repo": repositories[0]["url"] if repositories else None,
        "extracted_urls": _extract_urls(paper_text),
        "search_title": search_title,
    }
