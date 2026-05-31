import os
import logging
from typing import Literal

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

_MODEL_NAME = "BAAI/bge-base-en-v1.5"
_MAX_EMBED_TOKENS = 400
_EMBEDDING_DIMENSION = 768
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
BGE_DOCUMENT_PREFIX = "Represent this passage for retrieval: "
_model: SentenceTransformer | None = None
logger = logging.getLogger(__name__)
EmbeddingTask = Literal["query", "document"]


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        hf_token = os.getenv("HF_TOKEN")
        try:
            _model = SentenceTransformer(_MODEL_NAME, token=hf_token, local_files_only=True)
        except Exception:
            _model = SentenceTransformer(_MODEL_NAME, token=hf_token)
    return _model


def get_tokenizer():
    return _get_model().tokenizer


def _token_ids_for_text(text: str) -> list[int]:
    tokenizer = get_tokenizer()
    return tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
        verbose=False,
    )["input_ids"]


def get_max_embed_tokens() -> int:
    return _MAX_EMBED_TOKENS


def get_embedding_model_name() -> str:
    return _MODEL_NAME


def get_embedding_dimension() -> int:
    return _EMBEDDING_DIMENSION


def _get_prefix_for_task(task: EmbeddingTask) -> str:
    if "bge-" not in _MODEL_NAME.lower():
        return ""
    if task == "query":
        return BGE_QUERY_PREFIX
    return BGE_DOCUMENT_PREFIX


def _prepare_text_for_embedding(text: str, task: EmbeddingTask) -> str:
    prefix = _get_prefix_for_task(task)
    return f"{prefix}{text}".strip() if prefix else text


def normalize_text_for_embedding(text: str, task: EmbeddingTask = "document") -> tuple[str, int]:
    tokenizer = get_tokenizer()
    prefix = _get_prefix_for_task(task)
    prefix_token_count = len(_token_ids_for_text(prefix)) if prefix else 0
    token_ids = _token_ids_for_text(text)
    original_count = len(token_ids)

    if original_count + prefix_token_count <= _MAX_EMBED_TOKENS:
        return text, original_count + prefix_token_count

    available_tokens = max(1, _MAX_EMBED_TOKENS - prefix_token_count)
    trimmed_ids = token_ids[:available_tokens]
    normalized = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    while len(_token_ids_for_text(_prepare_text_for_embedding(normalized, task))) > _MAX_EMBED_TOKENS and trimmed_ids:
        trimmed_ids = trimmed_ids[:-1]
        normalized = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    logger.warning(
        "embedding_text_trimmed original_tokens=%s trimmed_tokens=%s",
        original_count,
        len(trimmed_ids) + prefix_token_count,
    )
    return normalized, len(trimmed_ids) + prefix_token_count


def normalize_texts_for_embedding(
    texts: list[str],
    task: EmbeddingTask = "document",
) -> tuple[list[str], list[int]]:
    normalized_texts: list[str] = []
    token_counts: list[int] = []

    for text in texts:
        normalized_text, token_count = normalize_text_for_embedding(text, task=task)
        normalized_texts.append(normalized_text)
        token_counts.append(token_count)

    return normalized_texts, token_counts


def generate_embedding(text: str, task: EmbeddingTask = "query") -> list[float]:
    return generate_embeddings([text], task=task)[0]


def generate_embeddings(texts: list[str], task: EmbeddingTask = "document") -> list[list[float]]:
    if not texts:
        return []
    normalized_texts, token_counts = normalize_texts_for_embedding(texts, task=task)
    for index, token_count in enumerate(token_counts):
        if token_count > _MAX_EMBED_TOKENS:
            raise ValueError(
                f"text at index {index} exceeds max token length after normalization: "
                f"{token_count} > {_MAX_EMBED_TOKENS}"
            )
    prepared_texts = [_prepare_text_for_embedding(text, task) for text in normalized_texts]
    embeddings = _get_model().encode(
        prepared_texts,
        batch_size=min(32, len(texts)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
