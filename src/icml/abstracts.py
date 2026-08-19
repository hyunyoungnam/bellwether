"""Stage 2 — scrape abstracts from the ICML virtual site.

One HTTP request per paper (~6.8k), so this stage is built to be interrupted
and resumed at will:

  * every fetched abstract is appended to data/raw/abstracts/abstracts.jsonl
  * on start we load the set of already-fetched event ids and skip them
  * failures are recorded too, so a retry pass only touches what actually broke

    python3 -m icml.abstracts                # resume until complete
    python3 -m icml.abstracts --limit 50     # smoke test
    python3 -m icml.abstracts --retry-failed # re-attempt recorded failures
"""
from __future__ import annotations

import argparse
import html
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .collect import latest_snapshot
from .common import FOCUS_YEAR, RAW, ensure_dirs, fetch, load_json, paper_page, read_jsonl

STORE = RAW / "abstracts" / "abstracts.jsonl"

_ABSTRACT_RE = re.compile(r'<div class="abstract-content">(.*?)</div>\s*</div>', re.S)
_ABSTRACT_FALLBACK = re.compile(r'class="abstract-text"[^>]*>(.*?)</div>', re.S)
_TAG_RE = re.compile(r"<[^>]+>")


def extract_abstract(page: str) -> str | None:
    """Pull the abstract text out of a virtual-site paper page."""
    for pattern in (_ABSTRACT_RE, _ABSTRACT_FALLBACK):
        m = pattern.search(page)
        if not m:
            continue
        text = html.unescape(_TAG_RE.sub(" ", m.group(1)))
        text = re.sub(r"\s+", " ", text).strip()
        # Guard against matching an empty shell or a CSS block.
        if len(text) >= 40 and "{" not in text[:80]:
            return text
    return None


def load_done() -> tuple[set[int], set[int]]:
    """Return (ok_ids, failed_ids) already recorded on disk."""
    ok: set[int] = set()
    failed: set[int] = set()
    for row in read_jsonl(STORE):
        eid = row.get("event_id")
        if eid is None:
            continue
        (ok if row.get("abstract") else failed).add(eid)
    return ok, failed - ok


class Writer:
    """Thread-safe append-only JSONL writer, flushed per record."""

    def __init__(self, path: Path):
        self.fh = path.open("a", encoding="utf-8")
        self.lock = threading.Lock()

    def write(self, row: dict) -> None:
        with self.lock:
            self.fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape ICML paper abstracts (resumable).")
    ap.add_argument("--limit", type=int, default=0, help="stop after N new fetches (0 = all)")
    ap.add_argument("--workers", type=int, default=6, help="concurrent requests (keep <= 8, be polite)")
    ap.add_argument("--delay", type=float, default=0.15, help="per-worker sleep between requests, seconds")
    ap.add_argument("--retry-failed", action="store_true", help="also re-attempt previously failed ids")
    ap.add_argument("--year", type=int, default=FOCUS_YEAR)
    args = ap.parse_args()

    ensure_dirs()
    snap = latest_snapshot(args.year)
    if snap is None:
        raise SystemExit("no feed snapshot found — run `python3 -m icml.collect` first")

    results = load_json(snap)["results"]
    ok, failed = load_done()

    skip = ok if args.retry_failed else (ok | failed)
    todo = [r for r in results if r["id"] not in skip]
    if args.limit:
        todo = todo[: args.limit]

    print(f"snapshot={snap.name} total={len(results)} done={len(ok)} failed={len(failed)} todo={len(todo)}")
    if not todo:
        print("nothing to do — abstract cache is complete")
        return 0

    writer = Writer(STORE)
    counter = {"ok": 0, "err": 0}
    start = time.time()
    lock = threading.Lock()

    def work(rec: dict) -> None:
        eid = rec["id"]
        url = paper_page(args.year, eid)
        row = {"event_id": eid, "url": url}
        try:
            page = fetch(url, timeout=45).decode("utf-8", errors="replace")
            abstract = extract_abstract(page)
            row["abstract"] = abstract
            if abstract is None:
                row["error"] = "abstract-not-found"
        except Exception as exc:  # noqa: BLE001 — record and move on
            row["abstract"] = None
            row["error"] = f"{type(exc).__name__}: {exc}"[:200]

        writer.write(row)
        with lock:
            key = "ok" if row.get("abstract") else "err"
            counter[key] += 1
            n = counter["ok"] + counter["err"]
            if n % 100 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1e-6)
                eta = (len(todo) - n) / max(rate, 1e-6)
                print(f"  {n}/{len(todo)}  ok={counter['ok']} err={counter['err']}  "
                      f"{rate:.1f}/s  eta={eta/60:.1f}m", flush=True)
        time.sleep(args.delay)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    except KeyboardInterrupt:
        print("\ninterrupted — progress is saved, re-run to resume")
    finally:
        writer.close()

    print(f"done: +{counter['ok']} abstracts, {counter['err']} failures in {(time.time()-start)/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
