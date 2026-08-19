"""Full-text phase 2 — download arXiv PDFs for resolved papers.

Resumable and polite, same shape as the abstract scraper: one PDF per file on
disk, so an interrupted run costs nothing. Safe to start while
`arxiv_resolve` is still running — it re-reads the resolution file each run and
picks up whatever is newly available.

    python3 -m icml.arxiv_fetch                # resume until complete
    python3 -m icml.arxiv_fetch --limit 20     # smoke test

Output: data/raw/pdf/<arxiv_base>.pdf  +  data/raw/pdf/_fetch_log.jsonl
"""
from __future__ import annotations

import argparse
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import RAW, ensure_dirs, read_jsonl

RESOLVED = RAW / "arxiv" / "resolved.jsonl"
PDF_DIR = RAW / "pdf"
LOG = PDF_DIR / "_fetch_log.jsonl"

UA = "icml-atlas/0.1 (academic research; bulk full-text for a research landscape)"
MIN_PDF_BYTES = 8_000  # anything smaller is an error page, not a paper


def pdf_path(base: str) -> Path:
    return PDF_DIR / f"{base}.pdf"


def fetch_pdf(base: str, timeout: int = 120) -> bytes:
    url = f"https://arxiv.org/pdf/{base}"
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "application/pdf"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="Download arXiv PDFs (resumable).")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4, help="keep low — arXiv is a shared resource")
    ap.add_argument("--delay", type=float, default=1.2, help="per-worker sleep between downloads")
    args = ap.parse_args()

    ensure_dirs()
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    if not RESOLVED.exists():
        raise SystemExit("no resolved.jsonl — run `python3 -m icml.arxiv_resolve` first")

    # Deduplicate by arxiv_base: two ICML entries can resolve to one preprint.
    targets: dict[str, dict] = {}
    for r in read_jsonl(RESOLVED):
        if r.get("arxiv_base"):
            targets.setdefault(r["arxiv_base"], r)

    todo = [b for b in targets if not pdf_path(b).exists()]
    if args.limit:
        todo = todo[: args.limit]

    have = len(targets) - len([b for b in targets if not pdf_path(b).exists()])
    print(f"resolved={len(targets)} on-disk={have} todo={len(todo)}")
    if not todo:
        print("nothing to do — PDF cache complete for currently resolved papers")
        return 0

    lock = threading.Lock()
    logfh = LOG.open("a", encoding="utf-8")
    counter = {"ok": 0, "err": 0, "bytes": 0}
    start = time.time()

    def work(base: str) -> None:
        row = {"arxiv_base": base}
        try:
            data = fetch_pdf(base)
            if not data.startswith(b"%PDF") or len(data) < MIN_PDF_BYTES:
                raise ValueError(f"not a pdf ({len(data)} bytes)")
            tmp = pdf_path(base).with_suffix(".pdf.tmp")
            tmp.write_bytes(data)
            tmp.replace(pdf_path(base))  # atomic: never leave a truncated PDF
            row |= {"ok": True, "bytes": len(data)}
        except Exception as exc:  # noqa: BLE001
            row |= {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:200]}

        with lock:
            logfh.write(json.dumps(row) + "\n")
            logfh.flush()
            counter["ok" if row.get("ok") else "err"] += 1
            counter["bytes"] += row.get("bytes", 0)
            n = counter["ok"] + counter["err"]
            if n % 50 == 0 or n == len(todo):
                rate = n / max(time.time() - start, 1e-6)
                print(f"  {n}/{len(todo)}  ok={counter['ok']} err={counter['err']}  "
                      f"{counter['bytes']/1e9:.1f} GB  {rate:.1f}/s  "
                      f"eta={(len(todo)-n)/max(rate,1e-6)/60:.0f}m", flush=True)
        time.sleep(args.delay)

    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(work, todo))
    except KeyboardInterrupt:
        print("\ninterrupted — progress saved, re-run to resume")
    finally:
        logfh.close()

    print(f"done: +{counter['ok']} PDFs ({counter['bytes']/1e9:.1f} GB), "
          f"{counter['err']} failures in {(time.time()-start)/60:.1f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
