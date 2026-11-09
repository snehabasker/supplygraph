# 0012. Observability: data freshness first, then traffic

**Status:** Accepted (M8)

## Context
The likeliest production failure is not a crash. It is a nightly pipeline that keeps failing:
a source changed format, a quality gate blocks, a download times out. Meanwhile the server
keeps happily serving week-old risk scores. Request metrics alone would show a healthy service.

## Decision
- **Probes.** `/healthz` is liveness: the process answers. `/readyz` is readiness: the warehouse
  opens and has companies. A fresh deployment without data stays unready instead of serving
  errors.
- **`/metrics` (Prometheus).** Data gauges are computed at scrape time from the warehouse:
  - `supplygraph_data_age_seconds{source=risk_scores|sanctions|news}`;
  - `supplygraph_rows{kind}`;
  - `supplygraph_actions{status}`;
  - `supplygraph_warehouse_up`.

  Traffic is measured with `supplygraph_http_requests_total` and
  `supplygraph_http_request_duration_seconds`. Their route label is the route *template*, so
  label cardinality stays bounded: LEIs never become labels.
- **Alerts** (optional PrometheusRule):
  - data stale for more than 30 h (one missed nightly run plus margin);
  - warehouse unreadable;
  - more than 5 % server errors;
  - proposals piling up with no analyst deciding.
- **Logs.** One JSON object per line (`SUPPLYGRAPH_LOG_FORMAT=json`, set in the image),
  including `extra=` fields, for any log collector.

## Alternatives considered
- **Pipeline metrics through a Pushgateway.** That adds a component. The freshness gauge
  already turns every pipeline failure mode into one symptom an alert can watch.
- **OpenTelemetry tracing.** There is little to trace in a single-process service. Revisit if
  the agent's Claude calls need latency or cost tracing.

## Consequences
- One alert, `SupplyGraphDataStale`, covers pipeline crashes, blocked quality gates and a
  suspended CronJob alike. The pipeline's logs, and the failed Job, say which it was.
- Scrapes cost a few small DuckDB queries. At a 30 s interval that is negligible.
