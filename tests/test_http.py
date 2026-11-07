from __future__ import annotations

import httpx
import pytest

from supplygraph.http import RateLimiter, backoff_seconds, get_with_retry


def client_with(responses: list) -> tuple[httpx.Client, list]:
    calls: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        item = responses[len(calls) - 1]
        if isinstance(item, Exception):
            raise item
        return item

    return httpx.Client(transport=httpx.MockTransport(handler)), calls


def test_retries_429_then_succeeds_honouring_retry_after() -> None:
    client, calls = client_with([
        httpx.Response(429, headers={"Retry-After": "3"}),
        httpx.Response(200, json={"ok": True}),
    ])
    sleeps: list[float] = []
    resp = get_with_retry(client, "https://x.test/a", sleep=sleeps.append)
    assert resp.status_code == 200
    assert len(calls) == 2
    assert sleeps == [3.0]


def test_retries_transport_errors() -> None:
    client, calls = client_with([httpx.ConnectError("boom"), httpx.Response(200)])
    resp = get_with_retry(client, "https://x.test/a", sleep=lambda _: None)
    assert resp.status_code == 200
    assert len(calls) == 2


def test_does_not_retry_404() -> None:
    client, calls = client_with([httpx.Response(404)])
    assert get_with_retry(client, "https://x.test/a", sleep=lambda _: None).status_code == 404
    assert len(calls) == 1


def test_gives_up_after_max_attempts() -> None:
    client, calls = client_with([httpx.Response(503)] * 3)
    with pytest.raises(httpx.HTTPStatusError):
        get_with_retry(client, "https://x.test/a", max_attempts=3, sleep=lambda _: None)
    assert len(calls) == 3


def test_backoff_is_exponential_and_capped() -> None:
    assert [backoff_seconds(n) for n in (1, 2, 3)] == [1.0, 2.0, 4.0]
    assert backoff_seconds(20) == 30.0


def test_rate_limiter_spaces_calls() -> None:
    now = [100.0]
    sleeps: list[float] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(1.0, clock=lambda: now[0], sleep=sleep)
    limiter.wait()  # first call: no wait
    now[0] += 0.25
    limiter.wait()  # 0.25s elapsed -> wait 0.75s
    now[0] += 5.0
    limiter.wait()  # plenty elapsed -> no wait
    assert sleeps == [pytest.approx(0.75)]
