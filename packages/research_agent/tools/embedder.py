import os
import logging

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer


load_dotenv()

_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MAX_EMBED_TOKENS = 400
_model: SentenceTransformer | None = None
logger = logging.getLogger(__name__)


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


def normalize_text_for_embedding(text: str) -> tuple[str, int]:
    tokenizer = get_tokenizer()
    token_ids = _token_ids_for_text(text)
    original_count = len(token_ids)

    if original_count <= _MAX_EMBED_TOKENS:
        return text, original_count

    trimmed_ids = token_ids[:_MAX_EMBED_TOKENS]
    normalized = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    while len(_token_ids_for_text(normalized)) > _MAX_EMBED_TOKENS and trimmed_ids:
        trimmed_ids = trimmed_ids[:-1]
        normalized = tokenizer.decode(trimmed_ids, skip_special_tokens=True).strip()

    logger.warning(
        "embedding_text_trimmed original_tokens=%s trimmed_tokens=%s",
        original_count,
        len(trimmed_ids),
    )
    return normalized, len(trimmed_ids)


def normalize_texts_for_embedding(texts: list[str]) -> tuple[list[str], list[int]]:
    normalized_texts: list[str] = []
    token_counts: list[int] = []

    for text in texts:
        normalized_text, token_count = normalize_text_for_embedding(text)
        normalized_texts.append(normalized_text)
        token_counts.append(token_count)

    return normalized_texts, token_counts


def generate_embedding(text: str) -> list[float]:
    return generate_embeddings([text])[0]


def generate_embeddings(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    normalized_texts, token_counts = normalize_texts_for_embedding(texts)
    for index, token_count in enumerate(token_counts):
        if token_count > _MAX_EMBED_TOKENS:
            raise ValueError(
                f"text at index {index} exceeds max token length after normalization: "
                f"{token_count} > {_MAX_EMBED_TOKENS}"
            )
    embeddings = _get_model().encode(
        normalized_texts,
        batch_size=min(32, len(texts)),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return embeddings.tolist()
