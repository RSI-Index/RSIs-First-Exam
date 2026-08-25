"""RL data generation utilities — litellm-backed LLM client and helpers."""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from tqdm import tqdm

import litellm
litellm.suppress_debug_info = True

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

DEFAULT_MODEL = "gemini/gemini-3.1-pro-preview"
MAX_RETRIES = 5


def _resolve_extra_body() -> Optional[Dict[str, Any]]:
    """Parse $LITELLM_EXTRA_BODY_JSON (used to thread provider-specific OpenAI
    extensions like ``chat_template_kwargs={"enable_thinking": False}`` through
    to vLLM's /v1/chat/completions endpoint).

    Returns the parsed dict, or ``None`` when the var is unset/blank/invalid.
    Read every call so callers can change it at runtime if they want.
    """
    raw = os.environ.get("LITELLM_EXTRA_BODY_JSON")
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.warning(
            "Ignoring malformed LITELLM_EXTRA_BODY_JSON (%s): %s", exc, raw
        )
        return None
    if not isinstance(parsed, dict):
        logger.warning(
            "Ignoring LITELLM_EXTRA_BODY_JSON: expected a JSON object, got %s",
            type(parsed).__name__,
        )
        return None
    return parsed


def chat_completion_batch(
    messages: List[List[Dict[str, str]]],
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    num_completions: int = 1,
    max_concurrency: int = 64,
    show_progress: bool = True,
) -> List[Any]:
    """Submit multiple chat completion requests concurrently via litellm."""

    if model is None:
        model = DEFAULT_MODEL

    extra_body = _resolve_extra_body()

    def _one_with_retry(idx: int, msgs: List[Dict[str, str]]):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=msgs,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    n=num_completions,
                )
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                resp = litellm.completion(**kwargs)
                return resp
            except litellm.exceptions.AuthenticationError as e:
                logger.error(f"[req {idx}] Auth error (not retrying): {e}")
                raise
            except Exception as e:
                last_error = e
                error_str = str(e)
                if attempt < MAX_RETRIES - 1:
                    if "rate" in error_str.lower():
                        wait_time = min(2 ** (attempt + 2), 30)
                    elif "timeout" in error_str.lower():
                        wait_time = 2
                    else:
                        wait_time = 2 ** attempt
                    logger.warning(f"[req {idx}] attempt {attempt+1}/{MAX_RETRIES} failed: {type(e).__name__}: {str(e)[:200]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[req {idx}] all {MAX_RETRIES} attempts failed: {type(e).__name__}: {str(e)[:200]}")
                    raise last_error

    results: List[Any] = [None] * len(messages)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one_with_retry, i, m): i
            for i, m in enumerate(messages)
        }

        pbar = tqdm(
            total=len(messages),
            disable=not show_progress,
            dynamic_ncols=True,
            desc="Processing",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        try:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None
                finally:
                    pbar.update(1)
        finally:
            pbar.close()

    failed_indices = [i for i, r in enumerate(results) if r is None]
    if failed_indices:
        logger.warning(f"Failed requests: {failed_indices}")

    return results


def chat_completion_batch_with_tools(
    messages: List[List[Dict[str, Any]]],
    tools: List[Dict[str, Any]],
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int = 1024,
    max_concurrency: int = 64,
    show_progress: bool = True,
) -> List[Any]:
    """Submit multiple tool-calling chat completion requests concurrently via litellm."""

    if model is None:
        model = DEFAULT_MODEL

    extra_body = _resolve_extra_body()

    def _one_with_retry(idx: int, msgs: List[Dict[str, Any]]):
        last_error = None
        for attempt in range(MAX_RETRIES):
            try:
                kwargs: Dict[str, Any] = dict(
                    model=model,
                    messages=msgs,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if extra_body is not None:
                    kwargs["extra_body"] = extra_body
                resp = litellm.completion(**kwargs)
                return resp
            except litellm.exceptions.AuthenticationError as e:
                logger.error(f"[req {idx}] Auth error (not retrying): {e}")
                raise
            except Exception as e:
                last_error = e
                error_str = str(e)
                if attempt < MAX_RETRIES - 1:
                    if "rate" in error_str.lower():
                        wait_time = min(2 ** (attempt + 2), 30)
                    elif "timeout" in error_str.lower():
                        wait_time = 2
                    else:
                        wait_time = 2 ** attempt
                    logger.warning(f"[req {idx}] attempt {attempt+1}/{MAX_RETRIES} failed: {type(e).__name__}: {str(e)[:200]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"[req {idx}] all {MAX_RETRIES} attempts failed: {type(e).__name__}: {str(e)[:200]}")
                    raise last_error

    results: List[Any] = [None] * len(messages)

    with ThreadPoolExecutor(max_workers=max_concurrency) as pool:
        future_to_idx = {
            pool.submit(_one_with_retry, i, m): i
            for i, m in enumerate(messages)
        }

        pbar = tqdm(
            total=len(messages),
            disable=not show_progress,
            dynamic_ncols=True,
            desc="Processing (tools)",
            unit="req",
            miniters=1,
            file=sys.stdout,
        )
        try:
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                try:
                    results[idx] = fut.result()
                except Exception:
                    results[idx] = None
                finally:
                    pbar.update(1)
        finally:
            pbar.close()

    failed_indices = [i for i, r in enumerate(results) if r is None]
    if failed_indices:
        logger.warning(f"Failed tool-calling requests: {failed_indices}")

    return results


# ---------------------------------------------------------------------------
# Python code extraction helpers (unchanged from endless-terminals)
# ---------------------------------------------------------------------------

def parse_python_code(code: str) -> str:
    """Extract raw Python code from an LLM response, stripping markdown fences."""
    fence_regex = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL | re.IGNORECASE)
    match = fence_regex.search(code)
    if match:
        snippet = match.group(1)
    else:
        snippet = code
    return textwrap.dedent(snippet).rstrip()


def check_python_code(code: str) -> bool:
    """Check if the Python code compiles successfully."""
    try:
        compile(code, "<string>", "exec")
        return True
    except SyntaxError:
        return False
    except Exception:
        return False
