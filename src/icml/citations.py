"""Intra-corpus citation edges from the HTML route's references.

The references the HTML parser preserves are strings ("Bhat et al. (2025)
Bhat, V., ... 3d cavla: Leveraging depth and 3d context ..."), and the only
citations this product needs are the ones between its own six editions —
A cites B where both are among our 29,669 papers. So matching is title
containment: a corpus paper's normalised title appearing inside a reference
string is that paper being cited.

Matching is deliberately strict, like icml.arxiv_match: a wrong edge silently
attaches one paper's lineage to another. Short titles are the hazard — a
normalised title must be >= 25 characters to participate, and a title that
matches implausibly many references is dropped as generic.

    python3 -m icml.citations                     # all fulltext files on disk

Output: data/processed/citations.json
    {"edges": [[citing_gid, cited_gid], ...], "papers": N,
     "note": ...}   — sorted, deduplicated; self-edges excluded.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

from .common import INTERIM, PROCESSED, dump_json, load_json, read_jsonl
from .corpus import available

_NORM = re.compile(r"[^a-z0-9]+")

# a title matched in more references than this is a generic phrase, not a
# citation ("deep learning", survey-ish titles) — measured, not guessed:
# real titles land far below it
MAX_HITS = 200
MIN_TITLE = 25


def norm(s: str) -> str:
    return _NORM.sub(" ", (s or "").lower()).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true", help="print sample edges")
    args = ap.parse_args()

    u = load_json(PROCESSED / "union.json")
    gid = {(k, e): i for i, (k, e) in enumerate(zip(u["keys"], u["eids"]))}

    # gid -> normalised title, filtered for distinctiveness
    titles: dict[int, str] = {}
    titletxt: dict[int, str] = {}
    for c in available():
        for p in c.read_papers():
            g = gid.get((c.key, p["event_id"]))
            if g is None:
                continue
            t = norm(p["title"])
            if len(t) >= MIN_TITLE:
                titles[g] = " " + t + " "   # word-boundary containment
                titletxt[g] = p["title"]

    # collect references per citing gid, from every fulltext file that has them
    refs_of: dict[int, list[str]] = {}
    src_files = sorted(INTERIM.glob("fulltext*.jsonl"))
    for f in src_files:
        # arxiv_base -> event_id bridge for this file's corpus
        key = f.stem.replace("fulltext_html_", "").replace("fulltext_", "").replace("_", "-")
        res = Path("data/raw/arxiv") / (f"resolved_{key}.jsonl" if key else "resolved.jsonl")
        if not res.exists():
            res = Path("data/raw/arxiv/resolved.jsonl")
            key = "icml-2026"
        to_gid = {}
        for r in read_jsonl(res):
            if r.get("arxiv_base"):
                g = gid.get((key, r["event_id"]))
                if g is not None:
                    to_gid.setdefault(r["arxiv_base"], g)
        n_here = 0
        for row in read_jsonl(f):
            if not row.get("ok") or not row.get("references"):
                continue
            g = to_gid.get(row.get("arxiv_base"))
            if g is None:
                continue
            refs_of.setdefault(g, row["references"])
            n_here += 1
        if n_here:
            print(f"{f.name}: references for {n_here:,} papers")

    # token -> ref locations, so each title only scans plausible candidates
    ref_norm: dict[tuple[int, int], str] = {}
    tok_ix: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for g, refs in refs_of.items():
        for j, r in enumerate(refs):
            n = " " + norm(r) + " "
            ref_norm[(g, j)] = n
            for t in set(n.split()):
                if len(t) >= 4:
                    tok_ix[t].append((g, j))

    edges: set[tuple[int, int]] = set()
    generic = 0
    for g2, t in titles.items():
        toks = [w for w in t.split() if len(w) >= 4]
        if not toks:
            continue
        anchor = min(toks, key=lambda w: len(tok_ix.get(w, ())))
        cands = tok_ix.get(anchor, ())
        hits = [(cg, j) for (cg, j) in cands if t in ref_norm[(cg, j)]]
        if len(hits) > MAX_HITS:
            generic += 1
            continue
        for cg, _j in hits:
            if cg != g2:
                edges.add((cg, g2))

    out = sorted(edges)
    citing = len({a for a, _ in out})
    cited = len({b for _, b in out})
    dump_json(PROCESSED / "citations.json", {
        "note": "intra-corpus citation edges: [citing gid, cited gid]; from the "
                "HTML route's references, matched by >=25-char normalised title "
                "containment; generic titles (> %d reference hits) excluded" % MAX_HITS,
        "sources": [f.name for f in src_files],
        "papers_with_refs": len(refs_of),
        "edges": [list(e) for e in out],
    }, indent=None)
    print(f"\n{len(out):,} edges — {citing:,} papers cite into the corpus, "
          f"{cited:,} papers are cited; {generic} generic titles excluded")

    if args.report:
        import random
        random.seed(1)
        for cg, tg in random.sample(out, min(10, len(out))):
            j = next(j2 for (g3, j2), n in ref_norm.items()
                     if g3 == cg and titles[tg] in n)
            print(f"\n  cites -> {titletxt[tg][:70]}")
            print(f"  ref:     {refs_of[cg][j][:110]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
