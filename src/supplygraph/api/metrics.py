"""Prometheus metrics for the API: request traffic plus data freshness and volume.

The data gauges are computed at scrape time from the warehouse, so they report what the
server is actually serving. Staleness is the main operational signal: the pipeline CronJob
can fail silently, but `supplygraph_data_age_seconds{source="risk_scores"}` keeps growing.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from datetime import UTC, datetime

import duckdb
from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily, InfoMetricFamily
from prometheus_client.registry import Collector

from supplygraph import __version__
from supplygraph.agent.actions import ActionStore
from supplygraph.warehouse import table_exists

log = logging.getLogger(__name__)

# (source label, query returning the newest timestamp of that source)
FRESHNESS = {
    "risk_scores": ("risk_config", "SELECT max(scored_at) FROM risk_config"),
    "sanctions": ("sanctions", "SELECT max(TRY_CAST(last_seen AS TIMESTAMP)) FROM sanctions"),
    "news": ("news_events", "SELECT max(published_at) FROM news_events"),
}
COUNTS = {
    "companies": "SELECT count(*) FROM companies",
    "companies_at_risk": "SELECT count(*) FROM company_risk WHERE risk_score > 0",
    "sanctioned_companies": "SELECT count(DISTINCT lei) FROM company_sanction_matches",
    "review_queue": "SELECT count(*) FROM er_candidates WHERE decision = 'review'",
    "supply_links": "SELECT count(*) FROM supply_links",
    "watchlist": "SELECT count(*) FROM watchlist",
}


def _age_seconds(value: object, now: datetime) -> float | None:
    if not isinstance(value, datetime):
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return max(0.0, (now - value).total_seconds())


class DataCollector(Collector):
    def __init__(self, connect: Callable[[], duckdb.DuckDBPyConnection],
                 store: ActionStore) -> None:
        self.connect = connect
        self.store = store

    def collect(self) -> Iterator:
        up = GaugeMetricFamily("supplygraph_warehouse_up",
                               "1 if the warehouse could be read at scrape time")
        try:
            con = self.connect()
        except Exception as exc:  # a missing or locked warehouse must not break /metrics
            log.warning("metrics: warehouse unavailable: %s", exc)
            up.add_metric([], 0)
            yield up
            return
        try:
            up.add_metric([], 1)
            yield up
            rows = GaugeMetricFamily("supplygraph_rows", "Rows / entities in the warehouse",
                                     labels=["kind"])
            for kind, sql in COUNTS.items():
                table = sql.split(" FROM ")[1].split()[0]
                if table_exists(con, table):
                    rows.add_metric([kind], float(con.execute(sql).fetchone()[0]))
            yield rows
            age = GaugeMetricFamily("supplygraph_data_age_seconds",
                                    "Seconds since the newest record of each source",
                                    labels=["source"])
            now = datetime.now(UTC)
            for source, (table, sql) in FRESHNESS.items():
                if table_exists(con, table):
                    seconds = _age_seconds(con.execute(sql).fetchone()[0], now)
                    if seconds is not None:
                        age.add_metric([source], seconds)
            yield age
        finally:
            con.close()
        actions = GaugeMetricFamily("supplygraph_actions", "Analyst actions by status",
                                    labels=["status"])
        counts: dict[str, int] = {"proposed": 0, "approved": 0, "rejected": 0}
        for a in self.store.actions():
            counts[a.status] = counts.get(a.status, 0) + 1
        for status, n in counts.items():
            actions.add_metric([status], float(n))
        yield actions
        info = InfoMetricFamily("supplygraph_build", "SupplyGraph version")
        info.add_metric([], {"version": __version__})
        yield info


class Metrics:
    def __init__(self, connect: Callable[[], duckdb.DuckDBPyConnection],
                 store: ActionStore) -> None:
        self.registry = CollectorRegistry()
        self.requests = Counter("supplygraph_http_requests_total", "HTTP requests",
                                ["method", "route", "status"], registry=self.registry)
        self.latency = Histogram("supplygraph_http_request_duration_seconds",
                                 "HTTP request latency", ["method", "route"],
                                 buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 30),
                                 registry=self.registry)
        self.registry.register(DataCollector(connect, store))

    def observe(self, method: str, route: str, status: int, started: float) -> None:
        # Route templates ("/api/companies/{lei}"), never raw paths: bounded label cardinality.
        self.requests.labels(method, route, str(status)).inc()
        self.latency.labels(method, route).observe(time.perf_counter() - started)
