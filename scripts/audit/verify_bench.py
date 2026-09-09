"""What the verifier actually does, measured — not assumed.

The product's one hard promise is that a quote shown as the paper's words was
matched against that paper before display. That promise is a function, and a
function can regress: relax the matcher to subsequence matching and every check
still passes while the guarantee is gone.

So this measures both directions on real sentences:

  accept — the paper's own extracted sentences must verify (a false reject
           silently starves the cards)
  reject — the same sentences, edited the way a model edits them (a swapped
           word, a spliced pair, a dropped middle) must NOT verify

Run:  PYTHONPATH=src python3 scripts/audit/verify_bench.py [N]
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from bellwether.mcp import Store          # noqa: E402
from bellwether.verify import Verifier    # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
FLOOR_ACCEPT = 0.98      # the paper's own sentences
CEIL_REJECT = 0.02       # edited ones that slip through

SWAPS = [("increases", "decreases"), ("improves", "worsens"),
         ("higher", "lower"), ("more", "less"), ("with", "without"),
         ("outperforms", "underperforms"), ("reduces", "raises"),
         ("all", "none"), ("first", "second"), ("can", "cannot")]


def edits(s: str, other: str) -> list[tuple[str, str]]:
    """The three ways a quote stops being the paper's words."""
    out = []
    for a, b in SWAPS:                       # one word turned into its opposite
        if f" {a} " in f" {s} ":
            out.append(("swapped word", s.replace(a, b, 1)))
            break
    w = s.split()
    if len(w) > 12:                          # the middle quietly dropped
        out.append(("dropped clause", " ".join(w[:5] + w[-5:])))
    ow = other.split()
    if len(w) > 8 and len(ow) > 8:           # two papers spliced into one quote
        out.append(("spliced", " ".join(w[: len(w) // 2] + ow[len(ow) // 2:])))
    return out


def main() -> int:
    store = Store()
    ver = Verifier(store)
    total = len(store.union["keys"])
    rng = random.Random(20260909)

    sents: list[tuple[int, str]] = []
    tried = 0
    while len(sents) < N and tried < N * 12:
        tried += 1
        gid = rng.randrange(total)
        sp = store.span_entry(gid)
        if not sp:
            continue
        for s in (sp[1], sp[2], sp[3]):
            if s and len(s) >= 40:
                sents.append((gid, s))
                break

    ok = sum(1 for g, s in sents if ver.role(g, s) is not None)
    acc = ok / max(len(sents), 1)

    slipped: list[tuple[str, int, str]] = []
    n_ed = 0
    by_kind: dict[str, list[int]] = {}
    for i, (gid, s) in enumerate(sents):
        other = sents[(i + 1) % len(sents)][1]
        for kind, bad in edits(s, other):
            n_ed += 1
            hit = ver.role(gid, bad) is not None
            by_kind.setdefault(kind, [0, 0])
            by_kind[kind][0] += 1
            if hit:
                by_kind[kind][1] += 1
                slipped.append((kind, gid, bad[:70]))
    rej = 1 - (len(slipped) / max(n_ed, 1))

    print(f"sentences  {len(sents)} from {len({g for g, _ in sents})} papers")
    print(f"accept     {ok}/{len(sents)} = {acc:.1%}   (floor {FLOOR_ACCEPT:.0%})")
    print(f"reject     {n_ed - len(slipped)}/{n_ed} = {rej:.1%}   "
          f"(ceiling on slips {CEIL_REJECT:.0%})")
    for kind, (n, bad) in sorted(by_kind.items()):
        print(f"  {kind:<16} {n - bad}/{n} rejected")
    for kind, gid, txt in slipped[:5]:
        print(f"  SLIPPED [{kind}] gid {gid}: {txt}…")

    bad = acc < FLOOR_ACCEPT or (1 - rej) > CEIL_REJECT
    print("FAIL: the verifier moved" if bad else "PASS")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
