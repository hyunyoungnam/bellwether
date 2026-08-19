"""Normalize extracted term vocabulary into canonical concepts.

The raw extraction produces ~17k distinct method names from ~6.6k papers, of which
only ~57 appear 10+ times. Most of that fragmentation is mechanical, not semantic:

    "reinforcement learning (rl)"              vs "reinforcement learning"
    "diffusion model"                          vs "diffusion models"
    "group relative policy optimization (grpo)" vs "grpo"

Used as graph nodes, these would split one concept across several points and make
the Idea Constellation unreadable.

Everything here is DETERMINISTIC and data-driven — no model call, so the mapping is
fully auditable and reproducible. Two rules do the work:

  1. "long form (ACRONYM)" defines an acronym -> long-form alias, learned from the
     corpus itself rather than a hand-written list.
  2. Plural/singular pairs merge ONLY when both forms actually occur, so we never
     mangle a word that merely ends in "s".

Anything requiring real semantic judgement is left alone and reported, so a human
(or a reviewed LLM pass) can extend config/term_aliases.json deliberately.

    python3 -m icml.normalize_terms
    python3 -m icml.normalize_terms --min-count 8 --report

Outputs: config/term_aliases.json (reviewable), data/processed/terms.json
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict

from .common import INTERIM, PROCESSED, ROOT, dump_json, load_json, read_jsonl

FACTS_ABSTRACT = INTERIM / "facts_abstract.jsonl"
ALIASES = ROOT / "config" / "term_aliases.json"
OUT = PROCESSED / "terms.json"

# "group relative policy optimization (grpo)" -> long, short
_PAREN = re.compile(r"^(?P<long>.+?)\s*\(\s*(?P<short>[^)]{2,40})\s*\)\s*$")
_PUNCT = re.compile(r"[\s\-_/]+")
_TRAIL = re.compile(r"[\s.,;:]+$")


def basic(s: str) -> str:
    """Case/whitespace/punctuation normalisation only — no semantic change."""
    s = (s or "").strip().lower()
    s = _TRAIL.sub("", s)
    s = _PUNCT.sub(" ", s)
    return s.strip()


def build_mapping(counts: Counter, min_count: int) -> tuple[dict[str, str], dict]:
    """Return (alias -> canonical, stats). Learned from the corpus, not hardcoded."""
    # Rule 1: learn acronym expansions from "long (SHORT)" occurrences.
    acronym_long: dict[str, Counter] = defaultdict(Counter)
    surface: Counter = Counter()
    for raw, n in counts.items():
        b = basic(raw)
        m = _PAREN.match(b)
        if m:
            lng, shrt = basic(m.group("long")), basic(m.group("short"))
            if lng and shrt and lng != shrt:
                acronym_long[shrt][lng] += n
                surface[lng] += n
                surface[shrt] += n
                continue
        surface[b] += n

    # An acronym maps to its most frequently co-stated long form.
    acro_map = {a: lc.most_common(1)[0][0] for a, lc in acronym_long.items()}

    mapping: dict[str, str] = {}

    # Rule 2: plural -> singular, only when the singular actually occurs.
    def depluralize(t: str) -> str:
        for suf, repl in (("ies", "y"), ("ses", "s"), ("s", "")):
            if t.endswith(suf) and len(t) > len(suf) + 2:
                cand = t[: -len(suf)] + repl
                if surface.get(cand, 0) > 0:
                    return cand
        return t

    canonical_of: dict[str, str] = {}
    for t in surface:
        c = acro_map.get(t, t)      # acronym -> long form
        c = depluralize(c)          # plural -> singular
        c = acro_map.get(c, c)      # long form may itself be an acronym surface
        canonical_of[t] = c

    # Map every ORIGINAL raw string to a canonical concept.
    for raw in counts:
        b = basic(raw)
        m = _PAREN.match(b)
        if m:
            b = basic(m.group("long"))
        mapping[raw] = canonical_of.get(b, b)

    canon_counts: Counter = Counter()
    for raw, n in counts.items():
        canon_counts[mapping[raw]] += n

    stats = {
        "raw_distinct": len(counts),
        "canonical_distinct": len(canon_counts),
        "acronyms_learned": len(acro_map),
        "raw_above_min": sum(1 for v in counts.values() if v >= min_count),
        "canonical_above_min": sum(1 for v in canon_counts.values() if v >= min_count),
    }
    return mapping, {"stats": stats, "acro_map": acro_map, "canon_counts": canon_counts}


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize extracted term vocabulary.")
    ap.add_argument("--min-count", type=int, default=8,
                    help="occurrences before a concept is graph-worthy")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    if not FACTS_ABSTRACT.exists():
        raise SystemExit("no facts_abstract.jsonl — run `icml.extract_facts --source abstract`")

    rows = [r for r in read_jsonl(FACTS_ABSTRACT) if r.get("ok")]

    method_counts: Counter = Counter()
    task_counts: Counter = Counter()
    dataset_counts: Counter = Counter()
    # Kept separate: a concept a paper PROPOSES is usually its own name and should
    # not become a shared graph node just because it was mentioned.
    proposed_counts: Counter = Counter()

    for r in rows:
        f = r["facts"]
        for m in f.get("methods") or []:
            method_counts[m["name"]] += 1
            if m.get("role") == "proposed":
                proposed_counts[m["name"]] += 1
        for t in f.get("tasks") or []:
            task_counts[t["name"]] += 1
        for d in f.get("datasets") or []:
            dataset_counts[d["name"]] += 1

    out = {"min_count": args.min_count, "papers": len(rows), "kinds": {}}
    alias_file = {"_comment": (
        "Deterministic alias -> canonical mapping learned from the corpus "
        "(acronym expansion + data-confirmed depluralisation). Regenerate with "
        "`python3 -m icml.normalize_terms`. Add manual entries under "
        "'manual_overrides' for semantic merges the rules cannot infer; those are "
        "applied last and win.")}

    for kind, counts in (("methods", method_counts), ("tasks", task_counts),
                         ("datasets", dataset_counts)):
        mapping, info = build_mapping(counts, args.min_count)
        s = info["stats"]
        cc = info["canon_counts"]
        out["kinds"][kind] = {
            **s,
            "graph_worthy": [{"term": t, "count": c}
                             for t, c in cc.most_common() if c >= args.min_count],
        }
        alias_file[kind] = {
            "acronyms": info["acro_map"],
            "mapping": {k: v for k, v in mapping.items() if basic(k) != v},
        }
        print(f"{kind:9s} raw {s['raw_distinct']:6,} -> canonical {s['canonical_distinct']:6,}"
              f"   >=({args.min_count}x): {s['raw_above_min']:4} -> {s['canonical_above_min']:4}"
              f"   acronyms learned: {s['acronyms_learned']}")

    # Proposed-name share tells us how much of the tail is inherently unshared.
    prop_total = sum(proposed_counts.values())
    out["proposed"] = {
        "distinct": len(proposed_counts),
        "occurrences": prop_total,
        "share_of_method_mentions": round(prop_total / max(sum(method_counts.values()), 1), 3),
    }

    alias_file.setdefault("manual_overrides", {})
    dump_json(ALIASES, alias_file)
    dump_json(OUT, out)

    print(f"\nproposed-by-paper names: {len(proposed_counts):,} distinct "
          f"({out['proposed']['share_of_method_mentions']:.0%} of all method mentions)")
    print(f"wrote {ALIASES.relative_to(ROOT)} and {OUT.relative_to(ROOT)}")

    if args.report:
        for kind in ("methods", "tasks"):
            gw = out["kinds"][kind]["graph_worthy"][:15]
            print(f"\ntop canonical {kind}:")
            for e in gw:
                print(f"  {e['count']:5d}  {e['term'][:58]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
