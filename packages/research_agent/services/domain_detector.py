from collections import Counter

from research_agent.services.chunk_structure import infer_chunk_role


DomainLabel = str

DOMAIN_KEYWORDS: dict[DomainLabel, list[str]] = {
    "ml": [
        "neural",
        "training",
        "dataset",
        "loss",
        "optimizer",
        "inference",
        "accuracy",
        "transformer",
        "embedding",
    ],
    "theory": [
        "theorem",
        "lemma",
        "proof",
        "corollary",
        "bound",
        "complexity",
        "approximation",
    ],
    "systems": [
        "throughput",
        "latency",
        "distributed",
        "deployment",
        "cluster",
        "fault tolerance",
        "scalability",
        "runtime",
    ],
    "security": [
        "attack",
        "threat",
        "adversary",
        "vulnerability",
        "encryption",
        "malware",
        "intrusion",
        "privacy",
    ],
    "networks": [
        "network",
        "routing",
        "packet",
        "bandwidth",
        "protocol",
        "tcp",
        "wireless",
        "congestion",
    ],
}


def _keyword_score(texts: list[str], keywords: list[str]) -> float:
    corpus = "\n".join(texts).lower()
    if not corpus.strip():
        return 0.0
    return sum(corpus.count(keyword) for keyword in keywords) / max(1, len(keywords))


def detect_domain(top_chunk_texts: list[str]) -> dict[str, float | str]:
    texts = [text for text in top_chunk_texts if text and text.strip()]
    if not texts:
        return {"domain": "general", "confidence": 0.2}

    role_counter: Counter[str] = Counter()
    for text in texts:
        role, _ = infer_chunk_role(text, None, None)
        role_counter[role] += 1

    scores: dict[str, float] = {"general": 0.25}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = _keyword_score(texts, keywords)

    # Lightweight role priors to stabilize classification.
    scores["theory"] += 0.25 * (role_counter.get("theory", 0) / max(1, len(texts)))
    scores["systems"] += 0.18 * (role_counter.get("implementation", 0) / max(1, len(texts)))
    scores["security"] += 0.12 * (role_counter.get("discussion", 0) / max(1, len(texts)))
    scores["ml"] += 0.2 * (role_counter.get("evaluation", 0) / max(1, len(texts)))

    domain, best = max(scores.items(), key=lambda item: item[1])
    sorted_scores = sorted(scores.values(), reverse=True)
    second = sorted_scores[1] if len(sorted_scores) > 1 else 0.0
    margin = max(0.0, best - second)
    confidence = max(0.2, min(1.0, 0.35 + margin))

    # Guardrail: if signal is weak, return general.
    if best < 0.4:
        return {"domain": "general", "confidence": 0.35}
    return {"domain": domain, "confidence": round(confidence, 4)}
