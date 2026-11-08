"""SEC EDGAR ingestion: the latest annual report (10-K / 20-F) of each filer in the seeds.

Fair-access rules (https://www.sec.gov/os/accessing-edgar-data): at most 10 requests/second,
and a User-Agent that identifies you with a contact email — www.sec.gov answers 403 otherwise.
Filings are immutable, so a document already on disk is never downloaded again.
"""

from __future__ import annotations

import csv
import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import httpx

from supplygraph.http import RateLimiter, get_with_retry
from supplygraph.ingest.gleif import is_valid_lei

log = logging.getLogger(__name__)

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:0>10}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{document}"
ANNUAL_FORMS = ("10-K", "20-F", "40-F")
_EMAIL = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")


class SecConfigError(RuntimeError):
    pass


def check_user_agent(user_agent: str | None) -> str:
    if not user_agent or not _EMAIL.search(user_agent):
        raise SecConfigError(
            "SEC_USER_AGENT must name you and include a contact email, e.g. "
            "'SupplyGraph research you@example.com' (SEC fair-access policy; www.sec.gov "
            "returns 403 without it). Set it in .env."
        )
    return user_agent


@dataclass(frozen=True)
class Filer:
    lei: str
    cik: str  # no leading zeros, as in EDGAR archive paths
    name: str


@dataclass(frozen=True)
class FilingRef:
    filer_lei: str
    cik: str
    filer_name: str
    entity_name: str  # as registered with the SEC
    form: str
    accession: str
    filing_date: str
    report_date: str | None
    url: str
    path: str  # relative to the SEC raw dir


def read_filers(path: Path) -> list[Filer]:
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    filers, bad = [], []
    for row in rows:
        lei, cik, name = row["lei"].strip(), row["cik"].strip(), row["name"].strip()
        if not is_valid_lei(lei) or not cik.isdigit():
            bad.append(row)
            continue
        filers.append(Filer(lei, str(int(cik)), name))
    if bad:
        raise ValueError(f"Invalid rows in {path}: {bad}")
    return filers


def latest_annual(submissions: dict, forms: tuple[str, ...] = ANNUAL_FORMS) -> dict | None:
    """The most recent original annual report (amendments such as 10-K/A are skipped: they
    often contain only the changed part)."""
    recent = (submissions.get("filings") or {}).get("recent") or {}
    best = None
    for i, form in enumerate(recent.get("form", [])):
        if form not in forms:
            continue
        row = {key: values[i] for key, values in recent.items() if isinstance(values, list)}
        if best is None or row["filingDate"] > best["filingDate"]:
            best = row
    return best


class SecApi(Protocol):
    def submissions(self, cik: str) -> dict: ...
    def document(self, url: str) -> bytes: ...


class SecClient:
    def __init__(
        self, user_agent: str, *, min_interval: float = 0.15, client: httpx.Client | None = None
    ) -> None:
        self._client = client or httpx.Client(
            headers={"User-Agent": check_user_agent(user_agent),
                     "Accept-Encoding": "gzip, deflate"},
            timeout=60.0,
            follow_redirects=True,
        )
        self._limiter = RateLimiter(min_interval)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> SecClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _get(self, url: str) -> httpx.Response:
        self._limiter.wait()
        resp = get_with_retry(self._client, url)
        if resp.status_code == 403:
            raise SecConfigError(f"SEC refused {url} (403): check SEC_USER_AGENT")
        resp.raise_for_status()
        return resp

    def submissions(self, cik: str) -> dict:
        return self._get(SUBMISSIONS_URL.format(cik=cik)).json()

    def document(self, url: str) -> bytes:
        return self._get(url).content


def filing_ref(filer: Filer, submissions: dict, row: dict) -> FilingRef:
    accession = row["accessionNumber"]
    document = row["primaryDocument"]
    url = ARCHIVE_URL.format(cik=filer.cik, accession_nodash=accession.replace("-", ""),
                             document=document)
    return FilingRef(
        filer_lei=filer.lei, cik=filer.cik, filer_name=filer.name,
        entity_name=submissions.get("name") or filer.name, form=row["form"],
        accession=accession, filing_date=row["filingDate"],
        report_date=row.get("reportDate") or None, url=url,
        path=f"{filer.cik}/{accession}_{document}",
    )


def ingest_filings(api: SecApi, filers: list[Filer], out_dir: Path) -> list[FilingRef]:
    """Download each filer's latest annual report and write `filings.jsonl` (the manifest)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    refs = []
    for filer in filers:
        submissions = api.submissions(filer.cik)
        row = latest_annual(submissions)
        if row is None:
            log.warning("No annual report found for %s (CIK %s)", filer.name, filer.cik)
            continue
        ref = filing_ref(filer, submissions, row)
        target = out_dir / ref.path
        if target.exists() and target.stat().st_size > 0:
            log.info("SEC %s %s %s already on disk", filer.name, ref.form, ref.accession)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(api.document(ref.url))
            tmp.replace(target)  # atomic: a crash never leaves a half-written filing
            log.info("SEC %s %s %s -> %s", filer.name, ref.form, ref.filing_date, target)
        refs.append(ref)
    with (out_dir / "filings.jsonl").open("w", encoding="utf-8") as fh:
        for ref in refs:
            fh.write(json.dumps(asdict(ref), ensure_ascii=False) + "\n")
    return refs


def read_manifest(out_dir: Path) -> list[FilingRef]:
    path = out_dir / "filings.jsonl"
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [FilingRef(**json.loads(line)) for line in fh if line.strip()]
