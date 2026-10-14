"""Client HTTP du chatbot : POST https://{base_url}/usecases/{use_case_id}/invoke."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from .config import ChatbotConfig
from .dataset import TestCase

logger = logging.getLogger(__name__)

OUTPUT_FIELD = "output"
RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}


@dataclass
class ChatbotResponse:
    status_code: int | None
    latency_s: float | None
    raw: Any = None  # JSON (ou texte) renvoyé par le chatbot
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    @property
    def actual_output(self) -> str | None:
        if not self.ok or self.raw is None:
            return None
        value = self.raw.get(OUTPUT_FIELD) if isinstance(self.raw, dict) else self.raw
        if value is None or isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code in RETRYABLE_STATUS
    return isinstance(exc, (httpx.TransportError, httpx.TimeoutException))


class ChatbotClient:
    def __init__(self, config: ChatbotConfig, transport: httpx.AsyncBaseTransport | None = None):
        self.config = config
        self._transport = transport
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ChatbotClient":
        self._client = httpx.AsyncClient(
            headers=self.config.headers(),
            timeout=self.config.timeout,
            verify=self.config.verify_ssl,
            transport=self._transport,
        )
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        if self._client:
            await self._client.aclose()

    async def _post(self, payload: dict[str, Any]) -> httpx.Response:
        assert self._client is not None, "Utiliser `async with ChatbotClient(...)`"
        response = await self._client.post(self.config.invoke_url, json=payload)
        response.raise_for_status()
        return response

    async def invoke(self, case: TestCase) -> ChatbotResponse:
        start = time.perf_counter()
        response: httpx.Response | None = None
        try:
            async for attempt in AsyncRetrying(
                stop=stop_after_attempt(self.config.max_retries + 1),
                wait=wait_exponential(multiplier=1, min=1, max=20),
                retry=retry_if_exception(_is_retryable),
                reraise=True,
            ):
                with attempt:
                    response = await self._post(case.payload())
        except httpx.HTTPStatusError as exc:
            body = exc.response.text[:500]
            return ChatbotResponse(
                status_code=exc.response.status_code,
                latency_s=time.perf_counter() - start,
                error=f"HTTP {exc.response.status_code}: {body}",
            )
        except Exception as exc:  # erreurs réseau, timeouts...
            return ChatbotResponse(
                status_code=None,
                latency_s=time.perf_counter() - start,
                error=f"{type(exc).__name__}: {exc}",
            )

        latency = time.perf_counter() - start
        try:
            raw = response.json()
        except ValueError:
            raw = response.text
        return ChatbotResponse(status_code=response.status_code, latency_s=latency, raw=raw)

    async def invoke_all(self, cases: list[TestCase]) -> list[ChatbotResponse]:
        semaphore = asyncio.Semaphore(self.config.concurrency)
        done = 0

        async def _run(case: TestCase) -> ChatbotResponse:
            nonlocal done
            async with semaphore:
                result = await self.invoke(case)
            done += 1
            level = logging.INFO if result.ok else logging.WARNING
            logger.log(
                level,
                "[%d/%d] ligne %d -> %s (%.2fs)",
                done, len(cases), case.row, "OK" if result.ok else result.error, result.latency_s or 0,
            )
            return result

        return await asyncio.gather(*(_run(c) for c in cases))
