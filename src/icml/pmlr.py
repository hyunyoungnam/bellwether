"""Camera-ready full text from the official PMLR proceedings.

Preferred over arXiv wherever a volume exists, and not because the text is
cleaner — measured, it is not; both are two-column PDFs and the camera-ready
actually carries slightly more line-break hyphens. The reasons are coverage and
simplicity:

    ICML 2025   PMLR v267   3,314 / 3,339 papers matched on title  (99.3%)
    ICML 2026   arXiv       4,765 / 6,637                          (71.8%)

and the arXiv route needs a 472k-record OAI index, offline title matching, and a
rate-limited fetch, none of which PMLR needs. arXiv stays as the bridge for the
current year, until its proceedings volume is published.

    python3 -m icml.pmlr index   --volume v267 --year 2025
    python3 -m icml.pmlr fetch   --year 2025
    .venv/bin/python -m icml.pmlr text --year 2025

Outputs: data/raw/pmlr/v<vol>_index.json, data/raw/pmlr/pdf/<event_id>.pdf,
         data/interim/fulltext_pmlr_<year>.jsonl
"""
from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.request
from pathlib import Path

from .common import INTERIM, RAW, dump_json, load_json, read_jsonl
from .corpus import Corpus

PMLR = RAW / "pmlr"
PDFS = PMLR / "pdf"
BASE = "https://proceedings.mlr.press"
UA = {"User-Agent": "icml-research-tool/1.0 (paper metadata; contact via repo)"}


def norm_title(s: str) -> str:
    """Match key: accents folded, everything but letters and digits dropped."""
    s = unicodedata.normalize("NFKD", s or "").lower()
    return re.sub(r"[^a-z0-9]", "", s)


def fetch_index(volume: str) -> list[dict]:
    """Every paper in a volume: title, camera-ready PDF, landing page."""
    req = urllib.request.Request(f"{BASE}/{volume}/", headers=UA)
    with urllib.request.urlopen(req, timeout=120) as fh:
        html = fh.read().decode("utf-8", "replace")
    rows = []
    for block in re.split(r'<div class="paper">', html)[1:]:
        t = re.search(r'<p class="title">(.*?)</p>', block, re.S)
        pdf = re.search(rf'href="(https://raw\.githubusercontent\.com/mlresearch/{volume}/[^"]+\.pdf)"',
                        block)
        page = re.search(rf'href="({BASE}/{volume}/[^"]+\.html)"', block)
        if t and pdf:
            rows.append({"title": re.sub(r"<[^>]+>", "", t.group(1)).strip(),
                         "pdf": pdf.group(1),
                         "page": page.group(1) if page else None})
    return rows


def match(volume_rows: list[dict], corpus: Corpus) -> list[dict]:
    """Attach our event_id to each PMLR entry, by normalised title."""
    by_title = {norm_title(r["title"]): r for r in volume_rows}
    out = []
    for p in corpus.read_papers():
        hit = by_title.get(norm_title(p["title"]))
        if hit:
            out.append({"event_id": p["event_id"], "title": p["title"], **hit})
    return out


def fetch_pdfs(matched: list[dict], delay: float, limit: int) -> tuple[int, int]:
    PDFS.mkdir(parents=True, exist_ok=True)
    got = skipped = 0
    todo = [m for m in matched if not (PDFS / f"{m['event_id']}.pdf").exists()]
    have = len(matched) - len(todo)          # count BEFORE --limit truncates it
    if limit:
        todo = todo[:limit]
    print(f"{len(matched):,} matched · {have:,} already on disk · {len(todo):,} to fetch")
    for i, m in enumerate(todo, 1):
        dest = PDFS / f"{m['event_id']}.pdf"
        try:
            req = urllib.request.Request(m["pdf"], headers=UA)
            with urllib.request.urlopen(req, timeout=120) as fh:
                data = fh.read()
            # A transient failure must never be recorded as a result — an empty
            # file would be skipped forever by the resume check.
            if len(data) < 2000:
                raise OSError(f"suspiciously small ({len(data)} bytes)")
            dest.write_bytes(data)
            got += 1
        except Exception as exc:  # noqa: BLE001
            skipped += 1
            print(f"  fail {m['event_id']}: {type(exc).__name__}", flush=True)
        if i % 200 == 0:
            print(f"  {i}/{len(todo)}  ok={got} fail={skipped}", flush=True)
        time.sleep(delay)
    return got, skipped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("index", "fetch", "text"):
        s = sub.add_parser(name)
        s.add_argument("--year", type=int, required=True)
        s.add_argument("--venue", default="ICML")
        if name == "index":
            s.add_argument("--volume", required=True)
        if name == "fetch":
            s.add_argument("--delay", type=float, default=0.25)
            s.add_argument("--limit", type=int, default=0)
        if name == "text":
            s.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    corpus = Corpus(args.venue, args.year)
    matched_path = PMLR / f"matched_{corpus.key}.json"

    if args.cmd == "index":
        rows = fetch_index(args.volume)
        dump_json(PMLR / f"{args.volume}_index.json", rows)
        m = match(rows, corpus)
        dump_json(matched_path, m)
        n = len(corpus.read_papers())
        print(f"{args.volume}: {len(rows):,} papers in the volume · "
              f"{len(m):,}/{n:,} matched to {corpus.key} ({len(m)/max(n,1):.1%})")
        return 0

    if args.cmd == "fetch":
        got, bad = fetch_pdfs(load_json(matched_path), args.delay, args.limit)
        print(f"downloaded {got:,}, failed {bad:,}")
        return 0

    # text: same section splitter the arXiv path uses, so downstream sees one shape
    from concurrent.futures import ProcessPoolExecutor
    from .pdf_extract import extract
    matched = {m["event_id"]: m for m in load_json(matched_path)}
    files = [p for p in sorted(PDFS.glob("*.pdf")) if int(p.stem) in matched]
    out = INTERIM / f"fulltext_pmlr_{args.year}.jsonl"
    done = {r["event_id"] for r in read_jsonl(out)} if out.exists() else set()
    files = [f for f in files if int(f.stem) not in done]
    print(f"{len(files):,} PDFs to parse ({len(done):,} already done)")
    ok = 0
    with out.open("a", encoding="utf-8") as fh, ProcessPoolExecutor(args.workers) as pool:
        for i, (f, row) in enumerate(zip(files, pool.map(extract, files)), 1):
            row["event_id"] = int(f.stem)
            row["source"] = "pmlr"
            ok += bool(row.get("ok"))
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if i % 300 == 0:
                fh.flush()
                print(f"  {i}/{len(files)}  ok={ok}", flush=True)
    print(f"wrote {out.name}: {ok:,} parsed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
