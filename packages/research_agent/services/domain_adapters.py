import re
from collections.abc import Iterable


ML_ARCHITECTURE_CANDIDATES = [
    "Mask-Conformer",
    "Transformer",
    "Conformer",
    "CNN",
    "RNN",
    "LSTM",
    "GRU",
    "GAN",
    "Diffusion",
    "Autoencoder",
    "Graph Neural Network",
]
ML_LOSS_CANDIDATES = [
    "CrossEntropy",
    "BCEWithLogits",
    "MSE",
    "CTC",
    "RNNT",
    "Hinge",
]
ML_OPTIMIZER_CANDIDATES = ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad"]
ML_DATASET_CANDIDATES = [
    "ImageNet",
    "COCO",
    "CIFAR-10",
    "CIFAR-100",
    "MNIST",
    "LibriSpeech",
    "SQuAD",
    "WMT",
]
ML_METRIC_CANDIDATES = ["accuracy", "f1", "precision", "recall", "bleu", "rouge", "wer", "latency"]
ML_ARCHITECTURE_ALIASES = {
    "Mask-Conformer": [r"\bmask[\s-]?conformer\b"],
    "Conformer": [r"\bconformer\b"],
    "Transformer": [r"\btransformer\b", r"\btransducer\b"],
    "CNN": [r"\bcnn\b", r"\bconvolutional neural network\b"],
    "RNN": [r"\brnn\b", r"\brecurrent neural network\b"],
    "LSTM": [r"\blstm\b"],
    "GRU": [r"\bgru\b"],
    "GAN": [r"\bgan\b", r"\bgenerative adversarial network\b"],
    "Diffusion": [r"\bdiffusion\b"],
    "Autoencoder": [r"\bautoencoder\b"],
    "Graph Neural Network": [r"\bgraph neural network\b", r"\bgnn\b"],
}
PROPOSAL_PATTERNS = [
    r"\bwe propose\b",
    r"\bwe present(?:ed)?\b",
    r"\bwe introduce(?:d)?\b",
    r"\bour proposed\b",
    r"\bour model\b",
    r"\bnovel\b",
    r"\bnamed\b",
    r"\bcalled\b",
]
BASELINE_PATTERNS = [
    r"\bbaseline\b",
    r"\bprior work\b",
    r"\bprevious best\b",
    r"\bprevious work\b",
    r"\bcompared with\b",
    r"\bcompared to\b",
    r"\boutperforming\b",
    r"\brelative to\b",
    r"\bsimilar approaches\b",
    r"\bcontemporary work\b",
]
SPECIFICITY_BONUS = {
    "Mask-Conformer": 0.45,
    "Conformer": 0.3,
    "Transformer": 0.05,
}
ARCHITECTURE_SUPPRESSION = {
    "Mask-Conformer": {"Conformer", "Transformer"},
    "Conformer": {"Transformer"},
}


def _flatten_chunk_texts(chunks: Iterable[dict]) -> list[str]:
    texts: list[str] = []
    for chunk in chunks:
        content = str(chunk.get("content") or chunk.get("content_excerpt") or "").strip()
        if content:
            texts.append(content)
    return texts


def _pick_candidates(corpus: str, candidates: list[str], *, case_sensitive: bool = False) -> list[str]:
    if not case_sensitive:
        lowered = corpus.lower()
        found = [candidate for candidate in candidates if candidate.lower() in lowered]
    else:
        found = [candidate for candidate in candidates if candidate in corpus]
    # stable unique order
    deduped: list[str] = []
    seen: set[str] = set()
    for item in found:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _extract_sentence_candidates(texts: list[str], pattern: str, *, limit: int = 6) -> list[str]:
    rx = re.compile(pattern, re.IGNORECASE)
    collected: list[str] = []
    for text in texts:
        for match in rx.finditer(text):
            snippet = match.group(0).strip()
            if snippet and snippet not in collected:
                collected.append(snippet)
            if len(collected) >= limit:
                return collected
    return collected


def _iter_chunk_texts(chunks: Iterable[dict]) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    for chunk in chunks:
        content = str(chunk.get("content") or chunk.get("text") or chunk.get("content_excerpt") or "").strip()
        if not content:
            continue
        items.append(
            {
                "content": content,
                "role": str(chunk.get("role") or ""),
                "section_name": str(chunk.get("section_name") or ""),
            }
        )
    return items


def _matches_any_pattern(text: str, patterns: list[str]) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns)


def _extract_ml_architectures(chunks: list[dict], inferred_structure: dict | None = None) -> tuple[list[str], list[str], list[str]]:
    chunk_items = _iter_chunk_texts(chunks)
    priority_texts: list[dict[str, str]] = []

    def collect_priority_items(items) -> None:
        if not isinstance(items, list):
            return
        for item in items:
            if not isinstance(item, dict):
                continue
            content = str(item.get("text") or item.get("summary") or "").strip()
            if not content:
                continue
            priority_texts.append(
                {
                    "content": content,
                    "role": str(item.get("role") or ""),
                    "section_name": str(item.get("section_name") or ""),
                }
            )

    if isinstance(inferred_structure, dict):
        collect_priority_items(inferred_structure.get("key_ideas"))
        methods_value = inferred_structure.get("methods")
        if isinstance(methods_value, dict):
            collect_priority_items(methods_value.get("chunks"))
        else:
            collect_priority_items(methods_value)

    scored_items = priority_texts + chunk_items
    if not scored_items:
        return [], [], []

    mention_scores = {candidate: 0.0 for candidate in ML_ARCHITECTURE_CANDIDATES}
    proposed_scores = {candidate: 0.0 for candidate in ML_ARCHITECTURE_CANDIDATES}
    baseline_scores = {candidate: 0.0 for candidate in ML_ARCHITECTURE_CANDIDATES}

    for item in scored_items:
        content = item["content"]
        lowered = content.lower()
        role = item["role"]
        section_name = item["section_name"].lower()
        proposal_context = _matches_any_pattern(lowered, PROPOSAL_PATTERNS)
        baseline_context = _matches_any_pattern(lowered, BASELINE_PATTERNS)
        context_boost = 1.0
        if role in {"idea", "method", "algorithm"}:
            context_boost += 0.35
        if section_name in {"abstract", "introduction", "method"}:
            context_boost += 0.25

        for candidate, patterns in ML_ARCHITECTURE_ALIASES.items():
            if not any(re.search(pattern, lowered, re.IGNORECASE) for pattern in patterns):
                continue
            mention_scores[candidate] += context_boost + SPECIFICITY_BONUS.get(candidate, 0.0)
            if proposal_context:
                proposed_scores[candidate] += 1.6 + context_boost + SPECIFICITY_BONUS.get(candidate, 0.0)
            if baseline_context:
                baseline_scores[candidate] += 1.2 + context_boost

    scored_all = [
        candidate
        for candidate in ML_ARCHITECTURE_CANDIDATES
        if mention_scores[candidate] > 0
    ]
    scored_all.sort(
        key=lambda candidate: (
            proposed_scores[candidate] * 2.0 + mention_scores[candidate] - baseline_scores[candidate],
            mention_scores[candidate],
        ),
        reverse=True,
    )

    proposed = [
        candidate
        for candidate in scored_all
        if proposed_scores[candidate] > 0 and proposed_scores[candidate] >= baseline_scores[candidate]
    ]
    if not proposed and scored_all:
        proposed = [scored_all[0]]

    filtered_proposed: list[str] = []
    suppressed: set[str] = set()
    for candidate in proposed:
        if candidate in suppressed:
            continue
        filtered_proposed.append(candidate)
        suppressed.update(ARCHITECTURE_SUPPRESSION.get(candidate, set()))

    baseline = [
        candidate
        for candidate in scored_all
        if candidate not in filtered_proposed and baseline_scores[candidate] > proposed_scores[candidate]
    ]

    return filtered_proposed[:3], baseline[:3], scored_all[:5]


def ml_adapter(chunks: list[dict], inferred_structure: dict | None = None) -> dict:
    texts = _flatten_chunk_texts(chunks)
    corpus = "\n".join(texts)

    proposed_architectures, baseline_architectures, architecture_mentions = _extract_ml_architectures(
        chunks,
        inferred_structure=inferred_structure,
    )
    datasets = _pick_candidates(corpus, ML_DATASET_CANDIDATES)
    losses = _pick_candidates(corpus, ML_LOSS_CANDIDATES)
    optimizers = _pick_candidates(corpus, ML_OPTIMIZER_CANDIDATES)
    metrics = _pick_candidates(corpus, ML_METRIC_CANDIDATES)

    primary_arch = proposed_architectures[0] if proposed_architectures else None
    primary_loss = losses[0] if losses else None
    primary_optimizer = optimizers[0] if optimizers else None
    primary_dataset = datasets[0] if datasets else None

    contribution_items = []
    if isinstance(inferred_structure, dict):
        for idea in inferred_structure.get("key_ideas", []):
            summary = str(idea.get("summary") if isinstance(idea, dict) else idea).strip()
            if summary:
                contribution_items.append(summary)
    if not contribution_items:
        contribution_items = _extract_sentence_candidates(
            texts,
            r"(we propose[^.]*\.|our method[^.]*\.|we introduce[^.]*\.)",
            limit=5,
        )

    training_summary = []
    if primary_loss:
        training_summary.append(f"Optimizes {primary_loss}")
    if primary_optimizer:
        training_summary.append(f"with {primary_optimizer}")
    if primary_dataset:
        training_summary.append(f"on {primary_dataset}")
    training_objective = " ".join(training_summary).strip() or None

    return {
        "model_architecture": primary_arch,
        "architectures": {"proposed": proposed_architectures, "baseline": baseline_architectures},
        "dataset": primary_dataset,
        "datasets": datasets[:5],
        "loss_function": primary_loss,
        "losses": {
            "primary": primary_loss,
            "auxiliary": losses[1:4],
            "baseline": [],
            "inferred": True if primary_loss else False,
            "confidence": 0.6 if primary_loss else 0.0,
        },
        "training_objective": training_objective,
        "optimizer": primary_optimizer,
        "optimizers": {"primary": primary_optimizer, "baseline": optimizers[1:4]},
        "training_details": {
            "signals_detected": {
                "architectures": architecture_mentions,
                "datasets": datasets[:5],
                "losses": losses[:5],
                "optimizers": optimizers[:5],
            }
        },
        "evaluation_metrics": [metric.upper() if metric == "wer" else metric for metric in metrics[:8]],
        "contributions": contribution_items[:6],
    }


def theory_adapter(chunks: list[dict]) -> dict:
    texts = _flatten_chunk_texts(chunks)
    theorems = _extract_sentence_candidates(texts, r"(theorem\s+\d+[^.]*\.|theorem[^.]*\.)", limit=8)
    proofs = _extract_sentence_candidates(texts, r"(proof[^.]*\.)", limit=8)
    complexity_claims = _extract_sentence_candidates(texts, r"(O\([^)]*\)[^.]*\.)", limit=8)
    return {
        "theorems": theorems,
        "proofs": proofs,
        "complexity_claims": complexity_claims,
    }


def systems_adapter(chunks: list[dict]) -> dict:
    texts = _flatten_chunk_texts(chunks)
    components = _extract_sentence_candidates(
        texts,
        r"(component[^.]*\.|module[^.]*\.|pipeline[^.]*\.|architecture[^.]*\.)",
        limit=10,
    )
    performance = _extract_sentence_candidates(
        texts,
        r"((latency|throughput|speedup|overhead|efficiency)[^.]*\.)",
        limit=10,
    )
    benchmarks = _extract_sentence_candidates(
        texts,
        r"(benchmark[^.]*\.|evaluation[^.]*\.|workload[^.]*\.)",
        limit=10,
    )
    return {
        "system_components": components,
        "performance_claims": performance,
        "benchmarks": benchmarks,
    }


def derive_domain_fields(domain: str, chunks: list[dict], inferred_structure: dict | None = None) -> dict:
    normalized = (domain or "general").lower()
    if normalized == "ml":
        return {"ml": ml_adapter(chunks, inferred_structure=inferred_structure)}
    if normalized == "theory":
        return {"theory": theory_adapter(chunks)}
    if normalized in {"systems", "security", "networks"}:
        return {"systems": systems_adapter(chunks)}
    return {}
