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


def _generate_content(
    prompt: str,
    model: str,
    response_mime_type: str | None = None,
) -> str:
    client = _get_client()
    config = (
        types.GenerateContentConfig(response_mime_type=response_mime_type)
        if response_mime_type
        else None
    )
    
    max_retries = 3
    base_delay = 10.0
    
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(model=model, contents=prompt, config=config)
            return response.text or ""
        except Exception as exc:
            if attempt == max_retries - 1:
                raise
            
            exc_str = str(exc).lower()
            if "429" in exc_str or "resource" in exc_str or "exhausted" in exc_str or "quota" in exc_str:
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
            text = _generate_content(prompt, model)
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
        return _generate_content(prompt, MODEL_NAME)
    except Exception as exc:
        if not _should_fallback(exc):
            raise
        logger.warning("primary_text_model_failed_falling_back model=%s error=%s", MODEL_NAME, type(exc).__name__)
        return generate_text_with_fallback(prompt)


def generate_json(prompt: str) -> dict:
    try:
        text = _generate_content(prompt, MODEL_NAME, response_mime_type="application/json")
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
            text = _generate_content(prompt, model, response_mime_type="application/json")
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
