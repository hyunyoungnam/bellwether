"""Blue-ocean candidates: (field, technique) pairs the corpus has not yet
seen together, ranked by whether the field's own stated failures match what
the technique claims to fix elsewhere — and VALIDATED by time.

This is literature-based discovery (Swanson 1986; Tshitoyan et al. 2019) on
a fully-held corpus: fields are the papers' declared domains (card_terms
`dom`), techniques are what papers build on or propose (`b` + `p`, alias-
normalised), a cell (F, T) is the count of papers in F using T.

Validation is temporal link prediction. Editions <= 2025 are the world as
it was; the two 2026 editions are the future we already hold. A cell empty
in the past and filled in 2026 is a "first application" — the truth set.
Every scorer ranks the past's empty cells, and precision@k says how many of
its top picks the future actually filled:

  random     the base rate
  pa         preferential attachment, size(F) * size(T) — popularity
  cf         collaborative filtering: fields that share techniques with F
             already use T, techniques that share fields with T already
             serve F — the classical link-prediction baseline
  lim        Bellwether's instrument: cosine between F's failure-term
             profile (its papers' limitation sentences) and T's claim-term
             profile (key_change/result sentences of papers using T)
  lim_cf     lim x cf (both min-max normalised) — evidence + structure

If lim (or lim_cf) does not beat pa and cf, the instrument is decoration
and must not ship as a tool. The numbers are printed, not assumed.

    python3 -m icml.blue_ocean --eval          # retrospective validation
    python3 -m icml.blue_ocean --emit [--scorer lim_cf] [--top 200]
                                               # candidates over ALL editions
                                               # -> data/processed/blue_ocean.json
                                               # (the pre-registrable prediction)
"""
from __future__ import annotations

import argparse
import math
import random
from collections import Counter, defaultdict

from bellwether.mcp import RDATA, Store, _GAP_STOP, _TOK
from .common import PROCESSED, dump_json, load_json

TRAIN = ["neurips-2024", "icml-2025", "neurips-2025", "iclr-2025"]
TEST = ["icml-2026", "iclr-2026"]
MIN_F = 12        # papers declaring the domain, within the training editions
MIN_T = 8         # papers using the technique (any paper), within training
KS = (10, 25, 50, 100)

# A "field" here is an APPLICATION domain — where a technique lands, not the
# ML subfield it comes from. Declared domains that name ML itself are not
# oceans to cross and are excluded; three obvious spellings are merged.
NOT_A_FIELD = {
    "computer vision", "natural language processing", "large language models",
    "reinforcement learning", "machine learning", "artificial intelligence",
    "deep learning", "multimodal large language models", "text-to-image generation",
    "video generation", "image generation", "3d vision", "video understanding",
    "federated learning", "generative models", "representation learning",
}
FIELD_ALIAS = {"drug design": "drug discovery",
               "software development": "software engineering",
               "security": "cybersecurity"}

# Model product names are not techniques: "robotics <- llama-2" says a field
# ran an LLM, not that a method crossed over. Dropped from T.
import re as _re
NOT_A_TECHNIQUE = _re.compile(
    r"^(gpt|chatgpt|llama|gemma|gemini|qwen|claude|mistral|mixtral|deepseek|phi|"
    r"vicuna|palm|o1|o3|grok|falcon|yi|internlm|baichuan|glm|command)"
    r"([\s\-_.]|\d|$)")


def terms_of(text: str) -> set[str]:
    """Failure/claim terms the way gap_scan extracts them: content unigrams
    (>= 5 chars) and bigrams, register words removed."""
    toks = [w for w in _TOK.findall(text.lower()) if w not in _GAP_STOP]
    out = {w for w in toks if len(w) >= 5}
    out |= {f"{u} {v}" for u, v in zip(toks, toks[1:])}
    return out


def load_rows(store: Store):
    ct = load_json(PROCESSED / "card_terms.json")
    spans = {k: load_json(RDATA / f"spans_{k}.json") for k in store.union["corpora"]}
    rows = []
    for key, per in ct.items():
        for eid, v in per.items():
            g = store.gid_by.get((key, int(eid)))
            if g is None:
                continue
            sp = spans[key].get(str(g)) or [None] * 4
            dom = (v.get("dom") or "").strip().lower() or None
            dom = FIELD_ALIAS.get(dom, dom)
            if dom in NOT_A_FIELD:
                dom = None
            rows.append({
                "g": g, "key": key, "dom": dom,
                "techs": {t for t in (v.get("b") or []) + (v.get("p") or [])
                          if not NOT_A_TECHNIQUE.match(t)},
                "L": sp[1] or "", "KR": " ".join(x for x in (sp[2], sp[3]) if x),
            })
    return rows


class World:
    """Everything a scorer may know, built from one set of editions only."""

    def __init__(self, rows, keys):
        inset = [r for r in rows if r["key"] in keys]
        self.rows = [r for r in inset if r["dom"]]
        fcount = Counter(r["dom"] for r in self.rows)
        # technique vocabulary from EVERY paper of these editions — a technique
        # is established by its use anywhere, not only where a domain is declared
        tcount = Counter(t for r in inset for t in r["techs"])
        self.F = sorted(f for f, c in fcount.items() if c >= MIN_F)
        self.T = sorted(t for t, c in tcount.items() if c >= MIN_T)
        self.fset, self.tset = set(self.F), set(self.T)
        self.nF, self.nT = fcount, tcount
        self.cell: Counter = Counter()
        self.techs_of_F: dict[str, set] = defaultdict(set)
        self.fields_of_T: dict[str, set] = defaultdict(set)
        for r in self.rows:
            if r["dom"] not in self.fset:
                continue
            for t in r["techs"] & self.tset:
                self.cell[(r["dom"], t)] += 1
                self.techs_of_F[r["dom"]].add(t)
                self.fields_of_T[t].add(r["dom"])

    # ---- profiles for the limitation-match scorer
    def profiles(self):
        fdf: dict[str, Counter] = defaultdict(Counter)   # failure terms per F
        tdf: dict[str, Counter] = defaultdict(Counter)   # claim terms per T
        fn, tn = Counter(), Counter()
        gdf: Counter = Counter()                          # document freq, all
        for r in self.rows:
            lt = terms_of(r["L"]) if r["L"] else set()
            kt = terms_of(r["KR"]) if r["KR"] else set()
            for w in lt | kt:
                gdf[w] += 1
            if r["dom"] in self.fset and lt:
                fn[r["dom"]] += 1
                for w in lt:
                    fdf[r["dom"]][w] += 1
            if kt:
                for t in r["techs"] & self.tset:
                    tn[t] += 1
                    for w in kt:
                        tdf[t][w] += 1
        n = max(1, len(self.rows))
        # terms in >2% of sentences ("datasets", "real-world", "neural") say
        # nothing about a match; terms in <3 are noise
        idf = {w: math.log(n / c) for w, c in gdf.items() if 3 <= c <= n * 0.02}

        def vec(df: Counter, cnt: int):
            v = {w: (c / cnt) * idf[w] for w, c in df.items() if w in idf}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            return {w: x / norm for w, x in v.items()}
        self.vF = {f: vec(fdf[f], fn[f]) for f in self.F if fn[f]}
        self.vT = {t: vec(tdf[t], tn[t]) for t in self.T if tn[t]}
        self.fdf, self.tdf = fdf, tdf

    def empty_cells(self):
        return [(f, t) for f in self.F for t in self.T if self.cell[(f, t)] == 0]

    # ---- scorers -------------------------------------------------------
    def s_pa(self, f, t):
        return self.nF[f] * self.nT[t]

    def s_cf(self, f, t):
        def jac(a, b):
            return len(a & b) / len(a | b) if (a or b) else 0.0
        tf, ft = self.techs_of_F, self.fields_of_T
        s = sum(jac(tf[f], tf[f2]) for f2 in ft[t] if f2 != f)
        s += sum(jac(ft[t], ft[t2]) for t2 in tf[f] if t2 != t)
        return s

    def s_lim(self, f, t):
        a, b = self.vF.get(f), self.vT.get(t)
        if not a or not b:
            return 0.0
        if len(a) > len(b):
            a, b = b, a
        return sum(x * b[w] for w, x in a.items() if w in b)

    def matched_terms(self, f, t, k=6):
        a, b = self.vF.get(f, {}), self.vT.get(t, {})
        return [w for w, _ in sorted(((w, x * b[w]) for w, x in a.items() if w in b),
                                     key=lambda p: -p[1])[:k]]


def _minmax(vals):
    lo, hi = min(vals), max(vals)
    return [(v - lo) / (hi - lo) if hi > lo else 0.0 for v in vals]


SCORERS = ("random", "pa", "cf", "lim", "lim_cf", "lim_spec")


def rank_all(world: World, cells, seed=0):
    """Every scorer's ranking of the same cells, best first."""
    pa = [world.s_pa(f, t) for f, t in cells]
    cf = [world.s_cf(f, t) for f, t in cells]
    lim = [world.s_lim(f, t) for f, t in cells]
    lim_cf = [x * y for x, y in zip(_minmax(lim), _minmax(cf))]
    # lim discounted by how many fields already use T — the reader asked for
    # oceans, not the technique everyone is already sailing
    lim_spec = [x / math.log(2 + len(world.fields_of_T[t]))
                for x, (f, t) in zip(lim, cells)]
    rnd = list(range(len(cells)))
    random.Random(seed).shuffle(rnd)
    order = lambda s: [c for _, c in sorted(zip(s, cells), key=lambda p: -p[0])]
    return {"random": [cells[i] for i in rnd], "pa": order(pa), "cf": order(cf),
            "lim": order(lim), "lim_cf": order(lim_cf), "lim_spec": order(lim_spec)}


def average_precision(order, truth) -> float:
    hits, s = 0, 0.0
    for i, c in enumerate(order, 1):
        if c in truth:
            hits += 1
            s += hits / i
    return s / max(1, len(truth))


def evaluate(store: Store, rows, verbose=True):
    past = World(rows, TRAIN)
    past.profiles()
    future = World(rows, TEST)          # only its cell counts are used
    cells = past.empty_cells()
    truth = {c for c in cells if future.cell[c] > 0}
    base = len(truth) / max(1, len(cells))
    ranks = rank_all(past, cells)
    res = {"fields": len(past.F), "techniques": len(past.T),
           "empty_cells": len(cells), "filled_in_2026": len(truth),
           "base_rate": round(base, 4), "precision_at": {}}
    res["average_precision"] = {}
    for name, order in ranks.items():
        res["precision_at"][name] = {
            str(k): round(sum(1 for c in order[:k] if c in truth) / k, 3) for k in KS}
        res["average_precision"][name] = round(average_precision(order, truth), 4)
    if verbose:
        print(f"fields {len(past.F)} x techniques {len(past.T)} -> "
              f"{len(cells):,} empty cells, {len(truth):,} filled in 2026 "
              f"(base rate {base:.3%})")
        print(f"{'scorer':9}" + "".join(f"{'P@'+str(k):>8}" for k in KS)
              + "      AP   lift@50")
        for name in SCORERS:
            p = res["precision_at"][name]
            lift = p["50"] / base if base else 0
            print(f"{name:9}" + "".join(f"{p[str(k)]:>8.3f}" for k in KS)
                  + f"  {res['average_precision'][name]:6.3f}   {lift:5.1f}x")
        # the hits, as evidence: what we would have said, and who then did it
        print("\nlim_spec top-25 (field <- technique : the 2026 paper that did it)")
        fut_rows = [r for r in rows if r["key"] in TEST and r["dom"]]
        for f, t in ranks["lim_spec"][:25]:
            hit = [r for r in fut_rows if r["dom"] == f and t in r["techs"]]
            mark = "HIT " if hit else "    "
            who = f" : {store.brief(hit[0]['g'])['title'][:70]}" if hit else ""
            print(f"  {mark}{f} <- {t}{who}")
    return res, ranks, truth, past


def emit(store: Store, rows, scorer: str, top: int, validation: dict):
    """Two tiers, each with its own measured precision: `established` (lim —
    the pairs most likely to be filled within a year) and `novel` (lim_spec —
    techniques few fields use yet; half the hit rate, that is what an ocean
    costs). The reader gets both numbers and both lists."""
    world = World(rows, TRAIN + TEST)
    world.profiles()
    cells = world.empty_cells()
    ranks = rank_all(world, cells)
    by_dom = defaultdict(list)
    by_tech = defaultdict(list)
    for r in world.rows:
        if r["dom"] in world.fset and r["L"]:
            by_dom[r["dom"]].append(r)
        if r["KR"]:
            for t in r["techs"] & world.tset:
                by_tech[t].append(r)

    def evidence(pool, terms, field_key, n=2):
        out = []
        for r in sorted(pool, key=lambda r: len(r[field_key])):
            low = r[field_key].lower()
            hit = [w for w in terms if w in low]
            if hit:
                out.append({"gid": r["g"], "sentence": r[field_key], "terms": hit})
            if len(out) >= n:
                break
        return out

    def cand(f, t):
        mt = world.matched_terms(f, t)
        return {
            "field": f, "technique": t,
            "papers_in_field": world.nF[f], "papers_using_technique": world.nT[t],
            "fields_already_using_technique": len(world.fields_of_T[t]),
            "matched_terms": mt,
            "field_states": evidence(by_dom[f], mt, "L"),
            "technique_claims": evidence(by_tech[t], mt, "KR"),
            "lim": round(world.s_lim(f, t), 3), "cf": round(world.s_cf(f, t), 3),
        }

    tiers = {}
    for name, sc in (("established", scorer), ("novel", "lim_spec")):
        tiers[name] = {
            "scorer": sc,
            "precision_at": validation["precision_at"][sc],
            "average_precision": validation["average_precision"][sc],
            "candidates": [cand(f, t) for f, t in ranks[sc][:top]],
        }
    dump_json(PROCESSED / "blue_ocean.json", {
        "note": "(field, technique) pairs no paper in the six editions has "
                "combined. fields = declared domains (card_terms dom, ~29% of "
                "papers declare one; ML-internal subfields excluded), "
                "techniques = builds-on/proposes names (model product names "
                "excluded). Evidence sentences are the papers' own; the ranking "
                "is an instrument, the judgment is the reader's. Validation = "
                "retrospective link prediction, editions <= 2025 predicting "
                "the two 2026 editions; each tier carries its own measured "
                "precision",
        "editions": TRAIN + TEST, "validation": validation, "tiers": tiers,
    })
    print("blue_ocean.json: " + ", ".join(
        f"{n} {len(t['candidates'])} by {t['scorer']} (P@50 {t['precision_at']['50']})"
        for n, t in tiers.items()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval", action="store_true")
    ap.add_argument("--emit", action="store_true")
    ap.add_argument("--scorer", default="lim_cf")
    ap.add_argument("--top", type=int, default=200)
    a = ap.parse_args()
    store = Store()
    rows = load_rows(store)
    print(f"{len(rows):,} papers with card terms; "
          f"{sum(1 for r in rows if r['dom']):,} declare a domain")
    res = None
    if a.eval or a.emit:
        res, *_ = evaluate(store, rows, verbose=a.eval)
    if a.emit:
        emit(store, rows, a.scorer, a.top, res)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
