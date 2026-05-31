import json
import logging
import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


load_dotenv()

logger = logging.getLogger(__name__)

MODEL_NAME = "gemini-2.5-flash"
REASONING_MODEL_CANDIDATES = [
    os.getenv("GEMINI_REASONING_MODEL_1", "gemini-2.5-flash-lite"),
    os.getenv("GEMINI_REASONING_MODEL_2", "gemini-3-flash"),
    os.getenv("GEMINI_REASONING_MODEL_3", "gemini-3.1-flash-lite"),
    os.getenv("GEMINI_REASONING_MODEL_4", "gemma-3-27b-it"),
]
PRIMARY_MODEL_MAX_RETRIES = int(os.getenv("GEMINI_PRIMARY_MAX_RETRIES", "1"))
FALLBACK_MODEL_MAX_RETRIES = int(os.getenv("GEMINI_FALLBACK_MAX_RETRIES", "2"))
PRIMARY_RATE_LIMIT_DELAY_SECONDS = float(os.getenv("GEMINI_PRIMARY_RATE_LIMIT_DELAY_SECONDS", "3.0"))
FALLBACK_RATE_LIMIT_DELAY_SECONDS = float(os.getenv("GEMINI_FALLBACK_RATE_LIMIT_DELAY_SECONDS", "5.0"))


def _get_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY must be set")
    return genai.Client(api_key=api_key)


def _should_fallback(exc: Exception) -> bool:
    message = str(exc).lower()
    fallback_markers = [
        "quota",
        "rate limit",
        "resource exhausted",
        "429",
        "500",
        "404",
        "not found",
        "internal",
        "unavailable",
        "deadline exceeded",
    ]
    return any(marker in message for marker in fallback_markers)


def _is_rate_limit_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in ["429", "resource exhausted", "quota", "rate limit", "exhausted"])


def _generate_content(
    prompt: str,
    model: str,
    response_mime_type: str | None = None,
    *,
    max_retries: int = 3,
    base_delay: float = 10.0,
) -> str:
    client = _get_client()
    config = (
        types.GenerateContentConfig(response_mime_type=response_mime_type)
        if response_mime_type
        else None
    )
    

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return response.text or ""
        except Exception as exc:
            if attempt == max_retries - 1:
                raise

            if _is_rate_limit_error(exc):
                delay = base_delay * (2 ** attempt)
                logger.warning("gemini_rate_limit_exceeded model=%s attempt=%s waiting=%ss", model, attempt + 1, delay)
                time.sleep(delay)
            else:
                raise

    raise RuntimeError(f"Failed to generate content after {max_retries} attempts")


def generate_text_with_fallback(prompt: str) -> str:
    last_exception: Exception | None = None
    for model in REASONING_MODEL_CANDIDATES:
        try:
            text = _generate_content(
                prompt,
                model,
                max_retries=FALLBACK_MODEL_MAX_RETRIES,
                base_delay=FALLBACK_RATE_LIMIT_DELAY_SECONDS,
            )
            logger.info("fallback_text_model_success model=%s", model)
            return text
        except Exception as exc:
            last_exception = exc
            logger.warning("fallback_text_model_failed model=%s error=%s", model, type(exc).__name__)
            if not _should_fallback(exc):
                raise
            time.sleep(0.5)
    if last_exception is not None:
        raise last_exception
    return ""


def generate_answer(prompt: str) -> str:
    try:
        return _generate_content(
            prompt,
            MODEL_NAME,
            max_retries=PRIMARY_MODEL_MAX_RETRIES,
            base_delay=PRIMARY_RATE_LIMIT_DELAY_SECONDS,
        )
    except Exception as exc:
        if not _should_fallback(exc):
            raise
        logger.warning("primary_text_model_failed_falling_back model=%s error=%s", MODEL_NAME, type(exc).__name__)
        return generate_text_with_fallback(prompt)


def generate_json(prompt: str) -> dict:
    try:
        text = _generate_content(
            prompt,
            MODEL_NAME,
            response_mime_type="application/json",
            max_retries=PRIMARY_MODEL_MAX_RETRIES,
            base_delay=PRIMARY_RATE_LIMIT_DELAY_SECONDS,
        )
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        logger.warning("primary_json_model_returned_invalid_json model=%s", MODEL_NAME)
        return generate_json_with_reasoning_fallback(prompt)
    except Exception as exc:
        if not _should_fallback(exc):
            raise
        logger.warning("primary_json_model_failed_falling_back model=%s error=%s", MODEL_NAME, type(exc).__name__)
        return generate_json_with_reasoning_fallback(prompt)


def generate_json_with_reasoning_fallback(prompt: str) -> dict:
    last_exception: Exception | None = None
    for model in REASONING_MODEL_CANDIDATES:
        try:
            text = _generate_content(
                prompt,
                model,
                response_mime_type="application/json",
                max_retries=FALLBACK_MODEL_MAX_RETRIES,
                base_delay=FALLBACK_RATE_LIMIT_DELAY_SECONDS,
            )
            logger.info("reasoning_model_success model=%s", model)
            return json.loads(text or "{}")
        except json.JSONDecodeError as exc:
            last_exception = exc
            logger.warning("reasoning_model_returned_invalid_json model=%s", model)
            time.sleep(0.5)
        except Exception as exc:
            last_exception = exc
            logger.warning("reasoning_model_failed model=%s error=%s", model, type(exc).__name__)
            if not _should_fallback(exc):
                raise
            time.sleep(0.5)
    if last_exception is not None:
        raise last_exception
    return {}


def generate_text(prompt: str) -> str:
    return generate_answer(prompt)
