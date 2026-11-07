"""OpenSanctions ingestion: bulk download of the consolidated sanctions list.

Data: https://www.opensanctions.org/datasets/sanctions/  (CC BY-NC 4.0, non-commercial use)
"""

from __future__ import annotations

import logging
from pathlib import Path

import httpx

log = logging.getLogger(__name__)


def download(url: str, dest: Path, user_agent: str, client: httpx.Client | None = None) -> Path:
    """Stream the CSV to disk atomically (write to .part, then rename) so a failed
    download never leaves a truncated file that later steps would silently load."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    owns_client = client is None
    client = client or httpx.Client(
        headers={"User-Agent": user_agent}, timeout=120.0, follow_redirects=True
    )
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            # Content-Length counts encoded bytes; only compare when the body isn't compressed.
            encoded = resp.headers.get("Content-Encoding", "identity") != "identity"
            expected = None if encoded else resp.headers.get("Content-Length")
            written = 0
            with tmp.open("wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
                    written += len(chunk)
        if expected is not None and expected.isdigit() and int(expected) != written:
            raise OSError(f"Truncated download: expected {expected} bytes, got {written}")
        tmp.replace(dest)
        log.info("Downloaded %s (%.1f MB)", dest, written / 1e6)
        return dest
    finally:
        if tmp.exists():
            tmp.unlink()
        if owns_client:
            client.close()
