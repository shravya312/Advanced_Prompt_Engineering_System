"""Gemini REST client using urllib (works with Windows system SSL store)."""

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Optional

_RETRYABLE_HTTP_CODES = {429, 500, 502, 503, 504}
_DEFAULT_FALLBACK_CHAIN = [
    "gemini-3.8-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3-flash-preview",
    "gemini-flash-latest",
]


class GeminiResponse:
    def __init__(self, text: str, model: Optional[str] = None):
        self.text = text
        self.model = model


class GeminiModel:
    """Drop-in wrapper: generate_content(prompt) -> response with .text"""

    def __init__(self, api_key: str, model: Optional[str] = None):
        if not api_key:
            raise ValueError("GEMINI_API_KEY is required")
        self._api_key = api_key
        self._primary_model = model or os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
        self._model_chain = self._build_model_chain(self._primary_model)
        self._last_model_used: Optional[str] = None

    @property
    def model_name(self) -> str:
        return self._last_model_used or self._primary_model

    @staticmethod
    def _build_model_chain(primary: str) -> List[str]:
        env_chain = os.getenv("GEMINI_FALLBACK_MODELS", "")
        if env_chain.strip():
            models = [m.strip() for m in env_chain.split(",") if m.strip()]
        else:
            models = [primary] + [m for m in _DEFAULT_FALLBACK_CHAIN if m != primary]
        seen = set()
        ordered: List[str] = []
        for name in models:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        return ordered

    @staticmethod
    def _http_code_from_error(exc: BaseException) -> Optional[int]:
        match = re.search(r"HTTP (\d{3})", str(exc))
        if match:
            return int(match.group(1))
        return None

    def _call_model(self, model: str, prompt: str, timeout: int) -> GeminiResponse:
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{model}:generateContent?key={self._api_key}"
        )
        payload = json.dumps(
            {"contents": [{"parts": [{"text": prompt}]}]}
        ).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Gemini API HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Gemini API connection failed: {exc}") from exc

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected Gemini response: {data}") from exc

        self._last_model_used = model
        return GeminiResponse(text, model=model)

    @staticmethod
    def _is_retryable_error(exc: RuntimeError) -> bool:
        message = str(exc).lower()
        if "timed out" in message or "timeout" in message:
            return True
        code = GeminiModel._http_code_from_error(exc)
        return code in _RETRYABLE_HTTP_CODES

    def generate_content(self, prompt: str, timeout: Optional[int] = None) -> GeminiResponse:
        if timeout is None:
            timeout = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "180"))
        last_error: Optional[Exception] = None
        max_retries = int(os.getenv("GEMINI_MAX_RETRIES", "2"))

        for model in self._model_chain:
            for attempt in range(max_retries + 1):
                try:
                    return self._call_model(model, prompt, timeout)
                except RuntimeError as exc:
                    last_error = exc
                    if self._is_retryable_error(exc) and attempt < max_retries:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                    code = self._http_code_from_error(exc)
                    if code in (404, 400):
                        break
                    if code in _RETRYABLE_HTTP_CODES:
                        break
                    raise

        if last_error:
            raise last_error
        raise RuntimeError("Gemini API call failed with no response.")
