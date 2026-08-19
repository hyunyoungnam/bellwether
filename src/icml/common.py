"""Shared helpers: paths, HTTP with retries, JSONL I/O.

Stdlib only by design — the whole pipeline must run with a bare `python3`.
"""
from __future__ import annotations

import gzip
import json
import os
import random
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Iterable, Iterator

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"

FOCUS_YEAR = int(os.environ.get("ICML_YEAR", "2026"))
# The atlas describes ONE corpus and makes no cross-year claims (product decision,
# 2026-07-29). Feeds for 2023-2025 are confirmed to exist and can be re-fetched
# with `icml.collect --years 2023 2024 2025` if that scope ever changes.
TREND_YEARS = [FOCUS_YEAR]

# Verified working 2026-07-28. See CLAUDE.md > Data sources.
def feed_url(year: int) -> str:
    return f"https://icml.cc/static/virtual/data/icml-{year}-orals-posters.json"


def paper_page(year: int, event_id: int) -> str:
    return f"https://icml.cc/virtual/{year}/poster/{event_id}"

UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


def ensure_dirs() -> None:
    for p in (RAW, RAW / "abstracts", INTERIM, PROCESSED, REPORTS):
        p.mkdir(parents=True, exist_ok=True)


def fetch(url: str, *, timeout: int = 60, retries: int = 4) -> bytes:
    """GET with exponential backoff. Raises on final failure."""
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": UA,
                    "Accept": "text/html,application/json,*/*",
                    "Accept-Encoding": "gzip",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return body
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
            last = exc
            code = getattr(exc, "code", None)
            if code in (404, 410):  # not transient
                raise
            time.sleep((2**attempt) * 0.7 + random.random() * 0.4)
    raise RuntimeError(f"fetch failed after {retries} attempts: {url}") from last


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with tmp.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)  # atomic: never leave a half-written dataset
    return n


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def dump_json(path: Path, obj: Any, *, indent: int = 2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=indent)
    tmp.replace(path)
