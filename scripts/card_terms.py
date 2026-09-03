"""Precompute the agent card's NAME fields — the axis it counts on.

Reads every corpus's abstract-pass facts, canonicalizes each name through
config/term_aliases.json (acronym expansion, then up to two mapping hops so
learned chains resolve), and writes data/processed/card_terms.json:
{corpus_key: {event_id: {p,b,t,d,dom}}} — proposes / builds_on / tasks /
data / domain. Deterministic; rerun after re-learning aliases
(`python3 -m icml.normalize_terms`) or re-extracting abstracts.

    python3 scripts/card_terms.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ALIASES = json.loads((ROOT / "config/term_aliases.json").read_text())
MAN = ALIASES.get("manual_overrides") or {}

_DASH = re.compile(r"[‐-―−]|--+")
_GLOSS = re.compile(r"\s*\(([A-Za-z][A-Za-z0-9\-/]{1,17})\)\s*")

MAP = {"icml-2026": "facts_abstract.jsonl",
       "icml-2025": "facts_abstract_2025.jsonl",
       "neurips-2024": "facts_abstract_neurips_2024.jsonl",
       "neurips-2025": "facts_abstract_neurips_2025.jsonl",
       "iclr-2025": "facts_abstract_iclr_2025.jsonl",
       "iclr-2026": "facts_abstract_iclr_2026.jsonl"}


def canon(name: str, kind: str) -> str:
    k = ALIASES.get(kind) or {}
    s = _GLOSS.sub(" ", _DASH.sub("-", name or "").strip()).strip()
    low = re.sub(r"\s+", " ", s.lower())
    low = (k.get("acronyms") or {}).get(low, low)
    mapping = k.get("mapping") or {}
    for _ in range(2):                       # follow learned chains
        nxt = mapping.get(low)
        if not nxt or nxt == low:
            break
        low = nxt
    return MAN.get(low, low) if isinstance(MAN, dict) else low


def dedup(seq):
    out = []
    for x in seq:
        if x and x not in out:
            out.append(x)
    return out


def main() -> int:
    out = {}
    raw, can = set(), set()
    for key, fn in MAP.items():
        d = {}
        for line in open(ROOT / "data/interim" / fn, encoding="utf-8"):
            r = json.loads(line)
            f2 = r.get("facts") or {}
            if not f2:
                continue
            meth = f2.get("methods") or []
            prop = dedup(canon(m["name"], "methods") for m in meth
                         if m.get("role") == "proposed" and m.get("name"))
            bo = dedup(canon(m["name"], "methods") for m in meth
                       if m.get("role") == "building-block" and m.get("name"))
            tasks = dedup(canon(t["name"], "tasks")
                          for t in f2.get("tasks") or [] if t.get("name"))
            ds = dedup(canon(x["name"] if isinstance(x, dict) else x, "datasets")
                       for x in f2.get("datasets") or []
                       if (x.get("name") if isinstance(x, dict) else x))
            for m in meth:
                if m.get("name"):
                    raw.add(m["name"].lower())
                    can.add(canon(m["name"], "methods"))
            row = {}
            if prop:
                row["p"] = prop[:4]
            if bo:
                row["b"] = bo[:6]
            if tasks:
                row["t"] = tasks[:4]
            if ds:
                row["d"] = ds[:5]
            if f2.get("domain") and f2["domain"] != "null":
                row["dom"] = f2["domain"]
            if row:
                d[str(r["event_id"])] = row
        out[key] = d
    p = ROOT / "data/processed/card_terms.json"
    p.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"method surfaces {len(raw):,} -> canonical {len(can):,} "
          f"(merged {len(raw) - len(can):,}) -> {p.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
