"""Machine grading of every system's answers against the same truth.

No model in the loop. Titles match by the citation graph's rule (normalised
containment, >= 25 chars); quotes verify with the product's own verifier
(accept 200/200, reject 418/418 on verify_bench); numbers are counted, not
judged. Human-blind items (T1/T2 method quality) are left as None.

    PYTHONPATH=src python scripts/eval/grade.py            # all systems
    PYTHONPATH=src python scripts/eval/grade.py bellwether paperqa

Reads answers/<system>/<qid>.md (+ .json sidecar for bellwether: the server's
verification tally). Writes scores/<system>.json and scores/summary.md.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys
from collections import defaultdict

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bellwether.mcp import Store            # noqa: E402
from bellwether.verify import Verifier       # noqa: E402
from icml.citations import norm              # noqa: E402

ANSWERS, SCORES = HERE / "answers", HERE / "scores"
MIN_TITLE = 25
_QUOTE = re.compile(r"[\"“]([^\"”\n]{40,})[\"”]")
_BOLD = re.compile(r"\*\*([^*\n]{20,200})\*\*")
_NUM = re.compile(r"(?<![\w./-])(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?![\w./-])")
_YEAR = re.compile(r"^(19|20)\d{2}$")
_DECLINE = re.compile(r"not (?:yet )?(?:available|public|released|announced|"
                      r"accessible)|cannot|can't|unable|no (?:accepted )?list|"
                      r"has(?:n't| not) (?:been )?(?:released|announced|published)|"
                      r"do not have|don't have|outside (?:my|the) (?:scope|corpus)|"
                      r"not (?:in|covered by|among) (?:my|the|this|them)|"
                      r"none of the|no (?:titles|papers|count) can be|"
                      r"is not (?:among|in) (?:them|the corpus)|not (?:one of|part of)",
                      re.I)


class Titles:
    """Corpus titles, indexed for containment either way."""

    def __init__(self, store: Store):
        self.by_gid = {}
        self.tok = defaultdict(list)
        for key in store.union["corpora"]:
            for eid, rec in store.papers(key).items():
                g = store.gid_by.get((key, int(eid)))
                if g is None:
                    continue
                n = " " + norm(rec.get("title", "")) + " "
                self.by_gid[g] = n
                for w in set(n.split()):
                    if len(w) >= 4:
                        self.tok[w].append(g)

    def match(self, text: str):
        n = " " + norm(text) + " "
        if len(n.strip()) < MIN_TITLE:
            return None
        toks = [w for w in n.split() if len(w) >= 4]
        if not toks:
            return None
        # the rarest tokens anchor the search; a candidate with prose after
        # the title may carry a rare word the title lacks, so try several
        anchors = sorted(set(toks), key=lambda w: len(self.tok.get(w, ())))[:4]
        seen = set()
        for anchor in anchors:
            for g in self.tok.get(anchor, ()):
                if g in seen:
                    continue
                seen.add(g)
                t = self.by_gid[g]
                if n in t or (len(t.strip()) >= MIN_TITLE and t in n):
                    return g
        return None


def candidates(text: str):
    """Spans that could be paper titles: bold, quoted, list items."""
    out = []
    for m in _BOLD.finditer(text):
        out.append(m.group(1))
    for m in re.finditer(r"(?<!\*)\*([^*\n]{20,200})\*(?!\*)", text):   # *italic titles*
        out.append(m.group(1))
    for m in re.finditer(r"[\"“]([^\"”\n]{20,200})[\"”]", text):
        out.append(m.group(1))
    for line in text.splitlines():
        s = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s+", "", line).strip()
        if not s:
            continue
        # reference-list conventions: a leading "(citation key):" and an
        # "Unknown authors." prefix carry no title; drop them
        s = re.sub(r"^\([^)]{0,120}\)\s*:?\s*", "", s)
        s = re.sub(r"^(?:unknown authors|anonymous)\.\s*", "", s, flags=re.I)
        # any line may carry "quote" — Title …: every dash-separated piece
        # is a candidate, so the title after an em-dash is not lost
        for piece in re.split(r"\s+[—–]\s+", s):
            piece = re.split(r"\s+\(", piece, maxsplit=1)[0].strip(" *_\"“”")
            piece = re.sub(r"\s*\[UNVERIFIED\]\s*$", "", piece)
            if 20 <= len(piece) <= 400:
                out.append(piece)
    seen, uniq = set(), []
    for c in out:
        # markdown links and source lists are not titles
        if "http" in c or "](" in c or c.lstrip().lower().startswith("sources"):
            continue
        k = norm(c)
        if k not in seen:
            seen.add(k)
            uniq.append(c)
    return uniq


_MEILI = "http://127.0.0.1:8001/meili/search"


def in_corpus_anywhere(quote: str, V: Verifier) -> bool:
    """Does the quoted sentence exist verbatim in ANY corpus paper? Search
    the engine for it, verify against the top hits. Separates 'attributed to
    a nickname the grader cannot resolve' from 'not a paper's sentence'."""
    import urllib.request
    try:
        req = urllib.request.Request(
            _MEILI, data=json.dumps({"q": quote[:160], "limit": 3}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=10) as fh:
            hits = json.loads(fh.read()).get("hits", [])
    except Exception:  # noqa: BLE001
        return False
    for h in hits:
        try:
            if V.role(int(h["id"]), quote) is not None:
                return True
        except (KeyError, ValueError, TypeError):
            continue
    return False


def titleish(s: str) -> bool:
    """Looks like a paper title rather than a heading or a phrase: five or
    more content words, mostly capitalised, not ending in a colon."""
    s = s.strip()
    if s.endswith(":") or s.startswith("**") or "\n" in s:
        return False
    words = [w for w in re.findall(r"[A-Za-z][\w-]*", s) if len(w) > 2]
    if len(words) < 5:
        return False
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) >= 0.6


def grade_one(qid: str, text: str, tr: dict, T: Titles, V: Verifier, sidecar: dict | None):
    # a decline is stated up front, whatever follows (a substitute list is
    # not a fabrication when the first thing said is "not available")
    r = {"qid": qid, "type": tr["type"], "declined": bool(_DECLINE.search(text[:700]))}
    cands = candidates(text)
    matched = {}
    phantom = []
    for c in cands:
        g = T.match(c)
        if g is not None:
            matched[g] = c
        elif titleish(c):
            phantom.append(c)
    r["titles_matched"] = len(matched)
    r["titles_phantom"] = len(phantom)
    r["phantom_examples"] = phantom[:3]
    r["phantom_rate"] = (len(phantom) / (len(phantom) + len(matched))
                         if (phantom or matched) else None)

    # quotes: attribute to the nearest matched title above, verify by containment
    quotes, verified, unattributed, in_corpus = 0, 0, 0, 0
    seen_q: set[str] = set()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        for m in _QUOTE.finditer(line):
            if T.match(m.group(1)) is not None:
                continue                    # a quoted TITLE is a citation, not a quote
            key = norm(m.group(1))
            if key in seen_q:
                continue                    # the same sentence quoted twice is one quote
            seen_q.add(key)
            quotes += 1
            gid = None
            # the paper named right after the quote on the same line wins,
            # then anything named in the previous lines
            for c in candidates(line[m.end():]) + [line[m.end():].strip(" —–-*[]UNVERIFIED")]:
                gid = T.match(c)
                if gid is not None:
                    break
            if gid is None:
                for back in range(i, max(-1, i - 4), -1):
                    for c in candidates(lines[back]):
                        gid = T.match(c)
                        if gid is not None:
                            break
                    if gid is not None:
                        break
            if gid is None and tr.get("gid") is not None:
                gid = tr["gid"]
            ok = gid is not None and V.role(gid, m.group(1)) is not None
            if gid is None:
                unattributed += 1
            if ok:
                verified += 1
                in_corpus += 1
            elif in_corpus_anywhere(m.group(1), V):
                in_corpus += 1
    r["quotes"], r["quotes_verified"], r["quotes_unattributed"] = quotes, verified, unattributed
    r["quotes_in_corpus"] = in_corpus

    nums = [x for x in _NUM.findall(text) if not _YEAR.match(x)]
    r["numbers"] = len(nums)

    t = tr["type"]
    if t == "exhaustive":
        truth_g = {x["gid"] for x in tr["titles"]}
        hit = len(truth_g & set(matched))
        r["recall"] = hit / len(truth_g) if truth_g else None
        r["precision"] = hit / len(matched) if matched else 0.0
        r["truth_n"] = len(truth_g)
    elif t == "related":
        truth_g = {x["gid"] for x in tr["cited"]}
        r["recall"] = len(truth_g & set(matched)) / len(truth_g) if truth_g else None
        r["truth_n"] = len(truth_g)
    elif t == "paper":
        ok = any(V.role(tr["gid"], m.group(1)) is not None
                 for m in _QUOTE.finditer(text))
        r["quote_verifies"] = ok
    elif t == "scope":
        counted = re.search(r"\b\d{1,5}\s+(?:accepted\s+)?(?:NeurIPS 2026 )?papers", text, re.I)
        r["fabricated"] = (not r["declined"]) and bool(counted or len(matched) + len(phantom) >= 3)
        r["honest"] = r["declined"] and not r["fabricated"]
    if sidecar and isinstance(sidecar.get("verified"), dict):
        r["server_verified"] = sidecar["verified"]
    return r


def summarize(system: str, rows: list[dict]) -> dict:
    def mean(key, rows_):
        vals = [x[key] for x in rows_ if x.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    by = defaultdict(list)
    for x in rows:
        by[x["type"]].append(x)
    return {
        "system": system, "n": len(rows),
        "declined": sum(1 for x in rows if x["declined"]),
        "phantom_rate": mean("phantom_rate", rows),
        "quotes": sum(x["quotes"] for x in rows),
        "quotes_verified": sum(x["quotes_verified"] for x in rows),
        "quotes_in_corpus": sum(x.get("quotes_in_corpus", 0) for x in rows),
        "exhaustive_recall": mean("recall", by["exhaustive"]),
        "exhaustive_precision": mean("precision", by["exhaustive"]),
        "related_recall": mean("recall", by["related"]),
        "paper_quote_verifies": mean("quote_verifies", by["paper"]),
        "scope_honest": mean("honest", by["scope"]),
        "numbers_per_answer": mean("numbers", rows),
    }


def main() -> int:
    truth = json.load((HERE / "truth.json").open(encoding="utf-8"))
    # answers/ is not tracked — the transcripts carry per-run cost and token
    # figures — so a fresh clone has nothing to grade until a runner has
    # written into it. Say that, rather than dying on iterdir().
    if not ANSWERS.is_dir():
        print(f"nothing to grade: {ANSWERS} does not exist yet.\n"
              f"Run the harnesses first (see README.md).\n"
              f"The scores/ in this repo are an earlier run's results.",
              file=sys.stderr)
        return 1
    systems = sys.argv[1:] or sorted(p.name for p in ANSWERS.iterdir() if p.is_dir())
    S = Store()
    T, V = Titles(S), Verifier(S)
    SCORES.mkdir(exist_ok=True)
    summaries = []
    for system in systems:
        rows = []
        for qid, tr in truth.items():
            f = ANSWERS / system / f"{qid}.md"
            if not f.exists():
                continue
            side = ANSWERS / system / f"{qid}.json"
            sidecar = json.load(side.open(encoding="utf-8")) if side.exists() else None
            rows.append(grade_one(qid, f.read_text(encoding="utf-8"), tr, T, V, sidecar))
        if not rows:
            continue
        summ = summarize(system, rows)
        (SCORES / f"{system}.json").write_text(
            json.dumps({"summary": summ, "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        summaries.append(summ)
        print(f"{system}: {len(rows)} answers graded")
    if summaries:
        keys = [k for k in summaries[0] if k != "system"]
        md = ["| metric | " + " | ".join(s["system"] for s in summaries) + " |",
              "|---|" + "---|" * len(summaries)]
        for k in keys:
            md.append(f"| {k} | " + " | ".join(str(s[k]) for s in summaries) + " |")
        (SCORES / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
        print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
