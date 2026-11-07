"""HTTP helpers: retry with backoff on transient failures, and client-side rate limiting."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


def backoff_seconds(attempt: int, base: float = 1.0, cap: float = 30.0) -> float:
    """Exponential backoff: 1s, 2s, 4s, ... capped."""
    return min(cap, base * 2 ** (attempt - 1))


def get_with_retry(
    client: httpx.Client,
    url: str,
    *,
    params: dict | None = None,
    max_attempts: int = 5,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """GET with retries on network errors, 429 and 5xx. Other statuses are returned as-is."""
    for attempt in range(1, max_attempts + 1):
        try:
            resp = client.get(url, params=params)
        except httpx.TransportError as exc:
            if attempt == max_attempts:
                raise
            delay = backoff_seconds(attempt)
            log.warning("GET %s failed (%s); retry %d in %.1fs", url, exc, attempt, delay)
            sleep(delay)
            continue

        if resp.status_code not in RETRYABLE_STATUS:
            return resp
        if attempt == max_attempts:
            resp.raise_for_status()

        retry_after = resp.headers.get("Retry-After", "")
        delay = float(retry_after) if retry_after.isdigit() else backoff_seconds(attempt)
        log.warning("GET %s -> %d; retry %d in %.1fs", url, resp.status_code, attempt, delay)
        sleep(delay)

    raise AssertionError("unreachable")


class RateLimiter:
    """Enforces a minimum interval between calls (GLEIF allows 60 requests/minute)."""

    def __init__(
        self,
        min_interval: float,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.min_interval = min_interval
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None

    def wait(self) -> None:
        now = self._clock()
        if self._last is not None:
            remaining = self.min_interval - (now - self._last)
            if remaining > 0:
                self._sleep(remaining)
                now += remaining
        self._last = now
