#!/usr/bin/env python3
"""Print pipeline state and the next command to run.

First thing to run in a session: answers "what exists, what's stale, what next"
without inferring it from the filesystem.

    python3 scripts/status.py
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from icml.common import FOCUS_YEAR, INTERIM, PROCESSED, RAW, REPORTS  # noqa: E402

OK, MISS, WARN = "OK  ", "--  ", "!!  "
VENV = ".venv/bin/python"


def age(p: Path) -> str:
    d = dt.datetime.now() - dt.datetime.fromtimestamp(p.stat().st_mtime)
    if d.days:
        return f"{d.days}d ago"
    if d.seconds >= 3600:
        return f"{d.seconds // 3600}h ago"
    return f"{max(d.seconds // 60, 1)}m ago"


def count_lines(p: Path) -> int:
    with p.open("rb") as fh:
        return sum(1 for _ in fh)


def size(p: Path) -> str:
    n = p.stat().st_size
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f}{unit}" if unit == "B" else f"{n:.1f}{unit}"
        n /= 1024
    return ""


def main() -> int:
    nxt: list[str] = []
    print(f"ICML {FOCUS_YEAR} — find papers in my field, and what is new in them\n")

    # ---- corpus ----
    print("corpus")
    snaps = sorted(RAW.glob(f"virtual_feed_{FOCUS_YEAR}_*.json"))
    if snaps:
        n = json.loads(snaps[-1].read_text())["count"]
        print(f"   {OK}feed: {n:,} events   {snaps[-1].name}  ({age(snaps[-1])})")
    else:
        print(f"   {MISS}feed: missing")
        nxt.append("PYTHONPATH=src python3 -m icml.collect")

    store = RAW / "abstracts" / "abstracts.jsonl"
    if store.exists():
        latest: dict[int, dict] = {}
        for line in store.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                latest[r["event_id"]] = r
        ok = sum(1 for r in latest.values() if r.get("abstract"))
        print(f"   {OK}abstracts: {ok:,} ({ok/max(len(latest),1):.1%}), "
              f"{len(latest)-ok} failed  ({age(store)})")
    else:
        print(f"   {MISS}abstracts: not scraped")
        nxt.append("PYTHONPATH=src python3 -m icml.abstracts")

    papers = PROCESSED / "papers.jsonl"
    if papers.exists():
        print(f"   {OK}papers.jsonl: {count_lines(papers):,} papers  ({age(papers)})")
    else:
        print(f"   {MISS}papers.jsonl: missing")
        nxt.append("PYTHONPATH=src python3 -m icml.normalize")

    # ---- full text ----
    print("\nfull text (enrichment only — ~71% coverage, never for corpus claims)")
    resolved = RAW / "arxiv" / "resolved.jsonl"
    if resolved.exists():
        rows = [json.loads(l) for l in resolved.open() if l.strip()]
        m = sum(1 for r in rows if r.get("arxiv_id"))
        print(f"   {OK}arxiv matched: {m:,}/{len(rows):,} ({m/max(len(rows),1):.1%})")
    else:
        print(f"   {MISS}arxiv match: not run")
        nxt.append("PYTHONPATH=src python3 -m icml.arxiv_harvest && "
                   "PYTHONPATH=src python3 -m icml.arxiv_match")

    pdfs = len(list((RAW / "pdf").glob("*.pdf"))) if (RAW / "pdf").exists() else 0
    print(f"   {OK if pdfs else MISS}pdfs: {pdfs:,}")
    ft = INTERIM / "fulltext.jsonl"
    if ft.exists():
        print(f"   {OK}sectioned text: {count_lines(ft):,} ({size(ft)})  ({age(ft)})")
    elif pdfs:
        nxt.append(f"PYTHONPATH=src {VENV} -m icml.pdf_extract")

    # ---- extraction ----
    print("\nextraction (two passes, never merged)")
    for src, label in (("abstract", "census — valid for corpus claims"),
                       ("fulltext", "enrichment — per-paper only")):
        p = INTERIM / f"facts_{src}.jsonl"
        if p.exists():
            rows = [json.loads(l) for l in p.open() if l.strip()]
            ok = sum(1 for r in rows if r.get("ok"))
            flag = sum(1 for r in rows if r.get("flag"))
            print(f"   {OK}{src}: {ok:,} ok, {len(rows)-ok} bad, {flag} flagged"
                  f"  ({age(p)})   {label}")
        else:
            print(f"   {MISS}{src}: not run   {label}")
            nxt.append(f"PYTHONPATH=src {VENV} -m icml.extract_facts --source {src}")

    # ---- retrieval assets ----
    print("\nretrieval assets")
    embs = sorted(PROCESSED.glob("emb_*.npy"))
    if embs:
        print(f"   {OK}embeddings: {embs[-1].name} ({size(embs[-1])})")
    else:
        print(f"   {MISS}embeddings: not built")
        nxt.append(f"PYTHONPATH=src {VENV} -m icml.landscape")

    alias = ROOT / "config" / "term_aliases.json"
    print(f"   {OK if alias.exists() else MISS}term aliases: "
          f"{'present' if alias.exists() else 'not built'}")
    if not alias.exists():
        nxt.append("PYTHONPATH=src python3 -m icml.normalize_terms")

    land = PROCESSED / "landscape.json"
    if land.exists():
        d = json.loads(land.read_text())
        print(f"   {OK}landscape: {len(d['points']):,} points, "
              f"{d['stats']['clusters']} regions  ({age(land)})")

    # ---- to build ----
    print("\nto build (see CLAUDE.md)")
    # No novelty.json: spans live inside facts_*.jsonl and are inlined by
    # icml.site. Listing it here told every session to build a file that should
    # not exist.
    for name, desc in [("topics.json", "level-3 multi-label topic tags"),
                       ("neighbors.json", "precomputed semantic neighbours")]:
        p = PROCESSED / name
        print(f"   {OK if p.exists() else MISS}{name}: "
              f"{size(p) if p.exists() else 'not built'} — {desc}")

    pages = list(REPORTS.glob("*.html"))
    print(f"   {OK if pages else MISS}interface: "
          f"{', '.join(p.name for p in pages) if pages else 'not built'}")

    print()
    if nxt:
        print("next:")
        seen = set()
        for c in nxt:
            if c not in seen:
                seen.add(c)
                print(f"   {c}")
    else:
        print("data and interface built — open work is Problem 3 in CLAUDE.md:\n"
              "  showing a selected set so a reader can tell what to open.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
