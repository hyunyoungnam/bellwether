"""Ground truth for the layer-1 questions, computed from the corpus — nothing
hand-authored. Writes truth.json and questions_resolved.json (the T4/T6
placeholders filled with the picked paper's title/abstract).

    PYTHONPATH=src python scripts/eval/make_truth.py
"""
from __future__ import annotations

import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.mcp import Store, _VENUE  # noqa: E402

SEED = 3
MIN_L = 60          # a limitation sentence worth asking about
MIN_CITES = 8       # a paper with enough in-corpus citations to grade recall


def main() -> int:
    S = Store()
    qs = json.load((HERE / "questions.json").open(encoding="utf-8"))
    truth, resolved = {}, []
    rng = random.Random(SEED)

    def papers_of(key):
        out = []
        for eid, rec in sorted(S.papers(key).items()):
            g = S.gid_by.get((key, int(eid)))
            if g is not None:
                out.append((g, rec))
        return out

    # deterministic picks per corpus: papers with a stated limitation (T4)
    # and papers with enough in-corpus citations (T6)
    pick_L, pick_C = {}, {}
    for key in S.union["corpora"]:
        ps = papers_of(key)
        withL = [(g, r) for g, r in ps
                 if (S.span_entry(g) or [None, None])[1]
                 and len(S.span_entry(g)[1]) >= MIN_L]
        withC = [(g, r) for g, r in ps
                 if len(S.cites_out.get(g, ())) >= MIN_CITES and r.get("abstract")]
        rng2 = random.Random(SEED)
        rng2.shuffle(withL)
        rng2.shuffle(withC)
        pick_L[key], pick_C[key] = withL, withC

    for q in qs["questions"]:
        item = dict(q)
        t = q["type"]
        if t == "exhaustive":
            ph = q["phrase"].lower()
            hits = [{"gid": g, "title": r["title"]} for g, r in papers_of(q["corpus"])
                    if ph in (r.get("title", "") + " " + r.get("abstract", "")).lower()]
            truth[q["id"]] = {"type": t, "titles": hits, "n": len(hits)}
        elif t == "paper":
            g, r = pick_L[q["pick"]["corpus"]][q["pick"]["index"]]
            venue, year = q["pick"]["corpus"].rsplit("-", 1)
            item["q"] = q["q"].format(title=r["title"],
                                      venue=f"{_VENUE.get(venue, venue)} {year}")
            truth[q["id"]] = {"type": t, "gid": g, "title": r["title"],
                              "limitation": S.span_entry(g)[1]}
        elif t == "related":
            g, r = pick_C[q["pick"]["corpus"]][q["pick"]["index"]]
            item["q"] = q["q"].format(abstract=r["abstract"])
            cited = [{"gid": c, "title": S.brief(c)["title"]}
                     for c in S.cites_out.get(g, ())]
            truth[q["id"]] = {"type": t, "gid": g, "title": r["title"],
                              "cited": cited, "n": len(cited)}
        elif t == "scope":
            truth[q["id"]] = {"type": t, "honest": "NeurIPS 2026 accepted list "
                              "not public at freeze — any count/list is fabricated"}
        else:
            truth[q["id"]] = {"type": t, "human": True}
        item.pop("pick", None)
        resolved.append(item)
        rng.random()

    (HERE / "truth.json").write_text(
        json.dumps(truth, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "questions_resolved.json").write_text(
        json.dumps({"questions": resolved}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    for qid, tr in truth.items():
        extra = tr.get("n", "")
        print(f"{qid}: {tr['type']} {extra} {tr.get('title', '')[:60]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
