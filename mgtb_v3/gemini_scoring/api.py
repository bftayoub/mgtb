from __future__ import annotations

import os
import random
import threading
import time
from collections import deque
from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .config import ModelQuota
from .prompt import SYSTEM_INSTRUCTION, serialized_contents
from .schema import JUDGE_RESPONSE_SCHEMA


PINNED_GOOGLE_GENAI_VERSION = "2.20.0"


def sdk_version() -> str:
    try:
        return version("google-genai")
    except PackageNotFoundError:
        return "not-installed"


def conservative_token_estimate(text: str) -> int:
    # UTF-8 bytes are a deliberately high upper bound for normal Gemini text tokens.
    return max(1, len(text.encode("utf-8")))


class RateLimiter:
    def __init__(self, quota: ModelQuota, requests_today: int = 0):
        self.quota = quota
        self.requests_today = int(requests_today)
        self._requests: deque[float] = deque()
        self._tokens: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def acquire(self, estimated_tokens: int) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                while self._requests and now - self._requests[0] >= 60:
                    self._requests.popleft()
                while self._tokens and now - self._tokens[0][0] >= 60:
                    self._tokens.popleft()
                if self.requests_today >= self.quota.rpd:
                    raise RuntimeError(f"daily request budget exhausted for {self.quota.model}")
                used_tokens = sum(value for _, value in self._tokens)
                if len(self._requests) < self.quota.rpm and used_tokens + estimated_tokens <= self.quota.tpm:
                    self._requests.append(now)
                    self._tokens.append((now, estimated_tokens))
                    self.requests_today += 1
                    return
                waits = []
                if len(self._requests) >= self.quota.rpm:
                    waits.append(60 - (now - self._requests[0]))
                if used_tokens + estimated_tokens > self.quota.tpm and self._tokens:
                    waits.append(60 - (now - self._tokens[0][0]))
            time.sleep(max(0.05, min(waits or [1.0])))


def _usage(response: Any) -> dict[str, int]:
    usage = getattr(response, "usage_metadata", None)
    return {
        "input_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "output_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "thinking_tokens": int(getattr(usage, "thoughts_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }


def status_code(exc: BaseException) -> int | None:
    for name in ("code", "status_code"):
        value = getattr(exc, name, None)
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            pass
    return None


def retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("retry-after") or headers.get("Retry-After")
    try:
        return max(0.0, float(value)) if value is not None else None
    except (TypeError, ValueError):
        return None


def retry_delay(exc: BaseException, attempt: int) -> float:
    header = retry_after(exc)
    if header is not None:
        return header
    return min(120.0, 2.0 ** min(attempt, 6)) + random.SystemRandom().uniform(0.0, 1.0)


class GeminiClient:
    def __init__(self, quota: ModelQuota, temperature: float, max_output_tokens: int, limiter: RateLimiter):
        if sdk_version() != PINNED_GOOGLE_GENAI_VERSION:
            raise RuntimeError(
                f"google-genai=={PINNED_GOOGLE_GENAI_VERSION} is required; found {sdk_version()}"
            )
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        from google import genai

        self._types = __import__("google.genai.types", fromlist=["types"])
        self._client = genai.Client(api_key=api_key)
        self.quota = quota
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.limiter = limiter

    def generate(self, payload: dict[str, Any]) -> tuple[str, dict[str, int], float]:
        contents = serialized_contents(payload)
        self.limiter.acquire(conservative_token_estimate(contents + SYSTEM_INSTRUCTION))
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.quota.model,
            contents=contents,
            config=self._types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
                response_mime_type="application/json",
                response_json_schema=JUDGE_RESPONSE_SCHEMA,
                thinking_config=self._types.ThinkingConfig(thinking_level=self.quota.thinking_level),
                automatic_function_calling=self._types.AutomaticFunctionCallingConfig(disable=True),
            ),
        )
        latency = time.perf_counter() - started
        text = getattr(response, "text", None)
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Gemini returned no textual structured response")
        return text, _usage(response), latency
