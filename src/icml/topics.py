"""Level-3 topic tags — multi-label, the level the author taxonomy is missing.

The author-declared taxonomy stops at 70 subareas. The level a researcher
actually works at ("autonomous driving") is absent: those 40 papers sit in 7
different area->subarea buckets, so no single-bucket scheme can retrieve them.

Two properties this must have, both deliberate:

* **Multi-label.** A driving paper is also computer vision, also video
  generation. One bucket per paper would rebuild the scattering it fixes.
* **Traceable membership.** Every paper in a topic is there for a stated reason:
  either an EXPLICIT extracted field said so, or it is semantically near the
  topic's centre and is labelled as such. The two are never merged, so the
  interface can show "12 papers say this, 9 more look like it".

Signals, in the trust order set out in CLAUDE.md: extracted `domain` and `task`
first, then datasets as domain anchors, then embedding expansion.

    .venv/bin/python -m icml.topics
    .venv/bin/python -m icml.topics --min-papers 6 --expand-threshold 0.62

Output: data/processed/topics.json
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from .common import INTERIM, PROCESSED, ROOT, dump_json, load_json, read_jsonl
from .normalize_terms import basic

FACTS = INTERIM / "facts_abstract.jsonl"
OUT = PROCESSED / "topics.json"

# The vocabulary rules live in taxonomy.py — one home, so a guard added in one
# place cannot be missing in another. That is how `null` once shipped as a topic
# holding 15 papers: topics.py filtered it and setview.py did not.
from .taxonomy import TOO_GENERIC, NULL_LITERALS, canon, is_label   # noqa: F401


def main() -> int:
    ap = argparse.ArgumentParser(description="Build level-3 multi-label topic tags.")
    ap.add_argument("--min-papers", type=int, default=8,
                    help="explicit members before a topic exists")
    ap.add_argument("--expand-threshold", type=float, default=0.62,
                    help="absolute cosine floor for a semantic candidate")
    ap.add_argument("--calibrate-pct", type=float, default=70.0,
                    help="candidate must beat this percentile of self-declared members")
    ap.add_argument("--max-expand", type=int, default=200,
                    help="cap on semantic additions per topic")
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    if not FACTS.exists():
        raise SystemExit("no facts_abstract.jsonl — run the census extraction first")

    import numpy as np

    papers = [p for p in read_jsonl(PROCESSED / "papers.jsonl") if p.get("abstract")]
    index = {p["event_id"]: i for i, p in enumerate(papers)}
    byid = {p["event_id"]: p for p in papers}

    cache = PROCESSED / f"emb_{len(papers)}_{args.model.replace('/', '_')}.npy"
    emb = None
    if cache.exists():
        emb = np.load(cache).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    else:
        print("no embedding cache — explicit membership only, no semantic expansion")

    # ---- explicit membership from extracted fields -------------------------
    members: dict[str, set[int]] = defaultdict(set)
    origin: dict[str, Counter] = defaultdict(Counter)
    datasets_of: dict[str, Counter] = defaultdict(Counter)
    scored = 0

    for r in read_jsonl(FACTS):
        if not r.get("ok"):
            continue
        eid = r["event_id"]
        if eid not in index:
            continue
        scored += 1
        f = r["facts"]
        ds = [d["name"] for d in (f.get("datasets") or [])]

        cands: list[tuple[str, str]] = []
        if f.get("domain"):
            cands.append((canon(f["domain"]), "domain"))
        for t in f.get("tasks") or []:
            cands.append((canon(t["name"]), "task"))

        for label, src in cands:
            if not label or len(label) < 4 or label in TOO_GENERIC:
                continue
            if label in NULL_LITERALS:
                continue
            members[label].add(eid)
            origin[label][src] += 1
            datasets_of[label].update(ds)

    topics = [t for t, s in members.items() if len(s) >= args.min_papers]
    dropped = len(members) - len(topics)
    topics.sort(key=lambda t: -len(members[t]))
    print(f"{scored} papers scored — {len(topics)} topics with >= {args.min_papers} "
          f"explicit members ({dropped} candidate labels below threshold)")

    # ---- semantic expansion ------------------------------------------------
    out = []
    for label in topics:
        explicit = sorted(members[label])
        expanded: list[list] = []
        if emb is not None and len(explicit) >= 3:
            rows = [index[e] for e in explicit]
            centroid = emb[rows].mean(0)
            centroid /= np.linalg.norm(centroid).clip(1e-9)
            sims = emb @ centroid
            have = set(rows)

            # Self-calibrating bar: a candidate must sit at least as close to the
            # centre as a typical SELF-DECLARED member of this topic. A single
            # global threshold cannot work — topics differ in how tight they are,
            # and a fixed 0.62 admitted the top-200 for every topic, which is not
            # expansion, just noise.
            own = np.sort(sims[rows])[::-1]
            bar = max(float(np.percentile(own, args.calibrate_pct)),
                      args.expand_threshold)
            # Never let guesses outnumber self-declarations: a topic may at most
            # double. Diffuse topics ("healthcare" spans 33 subareas) have a
            # centroid that means little, and without this cap they absorbed the
            # top-200 regardless of the bar.
            cap = min(args.max_expand, len(explicit))

            order = np.argsort(-sims)
            for j in order:
                if len(expanded) >= cap:
                    break
                if j in have:
                    continue
                s = float(sims[j])
                if s < bar:
                    break
                expanded.append([papers[j]["event_id"], round(s, 4)])

        # Scatter is measured over EXPLICIT members only: expanded ones are
        # "looks similar", and counting them would inflate the very number used
        # to argue the author taxonomy scatters this topic.
        subareas = Counter(
            f"{byid[i].get('area')} → {byid[i].get('subarea')}"
            for i in explicit if byid[i].get("subarea"))

        out.append({
            "label": label,
            "explicit": explicit,
            "expanded": expanded,
            "n_explicit": len(explicit),
            "n_expanded": len(expanded),
            "origin": dict(origin[label]),
            "top_datasets": [d for d, _ in datasets_of[label].most_common(5)],
            # The payoff: how badly the author taxonomy scatters this topic.
            "subareas_spanned": len(subareas),
            "top_subareas": subareas.most_common(4),
        })

    out.sort(key=lambda t: -(t["n_explicit"] + t["n_expanded"]))
    dump_json(OUT, {
        "method": ("multi-label level-3 tags. EXPLICIT membership = the paper's "
                   "extracted domain/task names the topic. EXPANDED = cosine to "
                   "the topic centroid over title+abstract embeddings. The two are "
                   "kept separate; expanded members are similar, not self-declared."),
        "params": {"min_papers": args.min_papers,
                   "expand_threshold": args.expand_threshold,
                   "max_expand": args.max_expand},
        "coverage": {
            "papers_scored": scored,
            "topics": len(out),
            "candidate_labels_dropped": dropped,
            "papers_with_any_topic": len({i for t in out for i in t["explicit"]}),
        },
        "topics": out,
    }, indent=None)

    covered = len({i for t in out for i in t["explicit"]})
    print(f"wrote {OUT.name}: {len(out)} topics, "
          f"{covered:,}/{scored:,} papers have >=1 explicit tag ({covered/max(scored,1):.0%})")
    print("\nlargest topics (explicit + expanded, subareas they span):")
    for t in out[:12]:
        print(f"   {t['n_explicit']:4d} +{t['n_expanded']:<4d} across {t['subareas_spanned']:2d} subareas   {t['label'][:44]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
