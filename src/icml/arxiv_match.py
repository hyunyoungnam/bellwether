"""Full-text phase 1b — match ICML papers to the local arXiv index, offline.

Replaces the per-paper API resolver (`arxiv_resolve`), which drew HTTP 429s and
projected 8+ hours. This runs against `oai_index.jsonl` with no network at all,
so it finishes in seconds and can be re-run freely after tuning the threshold.

Matching is deliberately strict. Attaching the wrong paper's full text is far
worse than having none: a bad match silently injects another paper's methods
into this paper's extracted ideas. Exact normalised-title equality is accepted;
otherwise a high token-overlap is required, and near-ties are rejected.

    python3 -m icml.arxiv_match
    python3 -m icml.arxiv_match --threshold 0.9 --report

Output: data/raw/arxiv/resolved.jsonl  (same schema as the old resolver)
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict

from .common import PROCESSED, RAW, ensure_dirs, read_jsonl

INDEX = RAW / "arxiv" / "oai_index.jsonl"
OUT = RAW / "arxiv" / "resolved.jsonl"

_NORM = re.compile(r"[^a-z0-9]+")
STOP = {"the", "and", "for", "with", "via", "from", "into", "using", "based", "その"}


def norm(s: str) -> str:
    return _NORM.sub(" ", (s or "").lower()).strip()


def toks(s: str) -> frozenset[str]:
    return frozenset(t for t in norm(s).split() if len(t) > 2 and t not in STOP)


def main() -> int:
    ap = argparse.ArgumentParser(description="Match ICML papers to the local arXiv index.")
    ap.add_argument("--threshold", type=float, default=0.85,
                    help="minimum Jaccard token overlap for a non-exact match")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="reject if the runner-up is within this of the best (ambiguous)")
    ap.add_argument("--report", action="store_true", help="show sample matches and rejects")
    args = ap.parse_args()

    ensure_dirs()
    if not INDEX.exists():
        raise SystemExit("no oai_index.jsonl — run `python3 -m icml.arxiv_harvest` first")

    papers = list(read_jsonl(PROCESSED / "papers.jsonl"))

    # Exact-title map plus an inverted index on rare-ish tokens, so each paper
    # only scores against plausible candidates rather than the whole corpus.
    exact: dict[str, dict] = {}
    postings: dict[str, list[int]] = defaultdict(list)
    index: list[dict] = []
    for row in read_jsonl(INDEX):
        i = len(index)
        index.append(row)
        exact.setdefault(norm(row["title"]), row)
        for t in toks(row["title"]):
            postings[t].append(i)

    print(f"index: {len(index):,} arXiv records, {len(exact):,} distinct titles")
    print(f"papers: {len(papers):,}")

    results = []
    n_exact = n_fuzzy = n_none = n_ambig = 0
    samples: list[str] = []

    for p in papers:
        title = p["title"]
        key = norm(title)
        row = {"event_id": p["event_id"], "title": title}

        hit = exact.get(key)
        if hit:
            row |= {"arxiv_base": hit["arxiv_base"], "arxiv_id": hit["arxiv_base"],
                    "arxiv_title": hit["title"], "published": hit.get("date"),
                    "score": 1.0, "match": "exact"}
            results.append(row)
            n_exact += 1
            continue

        pt = toks(title)
        if not pt:
            row |= {"arxiv_id": None, "score": 0.0, "match": "no-tokens"}
            results.append(row)
            n_none += 1
            continue

        # Candidate generation: any index row sharing a token with this title.
        counts: dict[int, int] = defaultdict(int)
        for t in pt:
            for i in postings.get(t, ()):
                counts[i] += 1
        # Jaccard can't exceed shared/len(pt), so prune before scoring.
        cands = [i for i, c in counts.items() if c / len(pt) >= args.threshold]

        scored = []
        for i in cands:
            it = toks(index[i]["title"])
            j = len(pt & it) / len(pt | it) if it else 0.0
            scored.append((j, i))
        scored.sort(reverse=True)

        if scored and scored[0][0] >= args.threshold:
            best_score, bi = scored[0]
            runner = scored[1][0] if len(scored) > 1 else 0.0
            if runner >= best_score - args.margin and index[scored[1][1]]["arxiv_base"] != index[bi]["arxiv_base"]:
                row |= {"arxiv_id": None, "score": round(best_score, 3), "match": "ambiguous",
                        "rejected_title": index[bi]["title"]}
                n_ambig += 1
            else:
                hit = index[bi]
                row |= {"arxiv_base": hit["arxiv_base"], "arxiv_id": hit["arxiv_base"],
                        "arxiv_title": hit["title"], "published": hit.get("date"),
                        "score": round(best_score, 3), "match": "fuzzy"}
                n_fuzzy += 1
                if len(samples) < 8:
                    samples.append(f"  {best_score:.2f}  {title[:52]}\n        -> {hit['title'][:52]}")
        else:
            row |= {"arxiv_id": None, "score": round(scored[0][0], 3) if scored else 0.0,
                    "match": "none"}
            n_none += 1
        results.append(row)

    with OUT.open("w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    total = len(results)
    matched = n_exact + n_fuzzy
    print(f"\nmatched {matched:,}/{total:,} ({matched/total:.1%})")
    print(f"  exact     {n_exact:,}")
    print(f"  fuzzy     {n_fuzzy:,}  (>= {args.threshold})")
    print(f"  ambiguous {n_ambig:,}  (rejected: runner-up within {args.margin})")
    print(f"  no match  {n_none:,}")
    if args.report and samples:
        print("\nsample fuzzy matches:")
        print("\n".join(samples))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
