"""How much of an answer is anchored — the number we were not measuring.

Pass rates say what happened to the anchors that exist. They say nothing about
the anchors that were never written, and a model that stops citing scores a
perfect 0/0. The provenance literature calls this the coverage half of the
problem (TRACER reports precision AND recall against reference annotations);
this is our version of it, computed over the conversations on disk.

Two coverage figures, both mechanical:

  figures   how many of the numbers an answer prints were placed in what the
            turn's tools returned. Since the automatic pass this is checked
            for every number, so the ratio is a property of the ANSWER, not
            of the agent's willingness to declare anything.
  papers    how many sentences that name a paper of ours carry a quote anchor.
            A sentence "names a paper" when it contains the title of a paper
            cited elsewhere in the same answer — a mechanical test, no
            judgement about whether the sentence makes a claim.

Run:  PYTHONPATH=src python3 scripts/audit/anchor_coverage.py [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

CHATS = ROOT / "data" / "chats"
_SENT = re.compile(r"[^.!?\n]+[.!?\n]|[^.!?\n]+$")


def title_words(t: str) -> str:
    """The distinctive head of a title — enough to spot it in a sentence."""
    head = re.split(r"[:：]", t)[0]
    return head.strip().lower()


def turn_coverage(turn: dict) -> dict:
    segs = turn.get("segs", [])
    v = turn.get("verified") or {}
    cited = {}
    for s in segs:
        if s.get("t") == "c" and s.get("title"):
            cited[s["gid"]] = title_words(s["title"])

    named = anchored = 0
    for i, s in enumerate(segs):
        if s.get("t") != "p":
            continue
        # a sentence is "anchored" when a cite chip follows it before the next
        # sentence break — the chip sits between segments, so look ahead
        nxt = segs[i + 1] if i + 1 < len(segs) else None
        has_chip = bool(nxt and nxt.get("t") == "c")
        for m in _SENT.finditer(s.get("s", "")):
            sent = m.group(0)
            low = sent.lower()
            if not any(t and len(t) > 12 and t in low for t in cited.values()):
                continue
            named += 1
            # the chip lands after the last sentence of the segment
            if has_chip and m.end() >= len(s.get("s", "").rstrip()):
                anchored += 1
    return {
        "figures_checked": v.get("fchecked", 0),
        "figures_placed": v.get("fpassed", 0),
        "quotes_checked": v.get("checked", 0),
        "quotes_passed": v.get("passed", 0),
        "sentences_naming_a_paper": named,
        "of_those_anchored": anchored,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    tot = {k: 0 for k in ("figures_checked", "figures_placed", "quotes_checked",
                          "quotes_passed", "sentences_naming_a_paper",
                          "of_those_anchored")}
    turns = 0
    rows = []
    for f in sorted(CHATS.glob("*.json")):
        try:
            d = json.loads(f.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for t in d.get("turns", []):
            c = turn_coverage(t)
            if not any(c.values()):
                continue
            turns += 1
            rows.append({"chat": d.get("title", "")[:38], **c})
            for k in tot:
                tot[k] += c[k]

    if a.json:
        print(json.dumps({"turns": turns, "total": tot, "rows": rows},
                         ensure_ascii=False, indent=1))
        return 0

    pct = lambda n, d: f"{n/d:.0%}" if d else "—"
    print(f"{turns} turns on disk\n")
    print(f"  figures placed in a tool result   "
          f"{tot['figures_placed']}/{tot['figures_checked']}  "
          f"{pct(tot['figures_placed'], tot['figures_checked'])}")
    print(f"  quotes verified                   "
          f"{tot['quotes_passed']}/{tot['quotes_checked']}  "
          f"{pct(tot['quotes_passed'], tot['quotes_checked'])}")
    print(f"  sentences naming a paper, anchored "
          f"{tot['of_those_anchored']}/{tot['sentences_naming_a_paper']}  "
          f"{pct(tot['of_those_anchored'], tot['sentences_naming_a_paper'])}")
    print("\n(coverage is not a pass rate: the last line is how often a "
          "sentence that\n names one of our papers also carries its quote)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
