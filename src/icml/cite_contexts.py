"""Verbatim citation-context sentences for in-corpus edges.

For every edge in citations.json (A cites B, both among our six editions),
find the sentences in A's parsed body where the citation actually happens,
and ship them verbatim — scite's idea, minus the classification: no
supporting/disputing labels, because a model judgment is not a fact.
The sentence itself is the fact, and it is checkable against the source.

How a citation is located: the HTML route's reference strings begin with
the in-text tag the paper itself uses ("Black et al. (2025b) Black, K., ...").
That head is extracted and searched for in the body as either narrative
"Black et al. (2025b)" or parenthetical "Black et al., 2025b" (with "and"/"&"
allowed to vary). References whose head is not name-(year) shaped — numeric
styles, mainly — yield no contexts rather than guessed ones.

    python3 -m icml.cite_contexts             # needs citations.json inputs

Output: data/processed/cite_contexts.json
    {"contexts": {cited_gid: [[citing_gid, bucket, sentence], ...]},
     "stats": ...}
Caps: 12 citing papers per cited paper, 2 sentences per citing paper,
sentences windowed to ~420 chars around the marker.
"""
from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from .common import INTERIM, PROCESSED, dump_json, load_json, read_jsonl
from .citations import MAX_HITS, MIN_TITLE, norm
from .corpus import available

MAX_CITERS = 12          # citing papers kept per cited paper
MAX_SENTS = 2            # sentences kept per (citing, cited) pair
MAX_LEN = 420            # sentence window, centred on the marker

# reference head = the in-text tag natbib prints: "Name et al. (2025b)"
_HEAD = re.compile(r"^\s*(.{2,70}?)[\s\u00a0]*\((\d{4}[a-z]?)\)")

# protect sentence-splitting hazards before splitting on ". Capital"
_PROT = [("et al.", "et al\u0000"), ("e.g.", "e\u0001g\u0001"),
         ("i.e.", "i\u0001e\u0001"), ("cf.", "cf\u0001"),
         ("vs.", "vs\u0001"), ("resp.", "resp\u0001"),
         ("Fig.", "Fig\u0001"), ("Eq.", "Eq\u0001"), ("Sec.", "Sec\u0001"),
         ("Tab.", "Tab\u0001"), ("No.", "No\u0001"), ("pp.", "pp\u0001")]
_SPLIT = re.compile(r"(?<=[.!?])[\s\u00a0]+(?=[A-Z(\u201c\"])")

# PDF linebreak hyphens ("Mag- nusson") are extractor artifacts, not the
# paper's words \u2014 joining them repairs the text and lets markers match
_DEHYPH = re.compile(r"(?<=[a-z])- (?=[a-z])")


def sentences(text: str):
    text = _DEHYPH.sub("", text)
    for a, b in _PROT:
        text = text.replace(a, b)
    for s in _SPLIT.split(text):
        for a, b in _PROT:
            s = s.replace(b, a)
        s = s.strip()
        if 30 <= len(s):
            yield s


def marker(ref: str):
    """In-text citation regex for one reference string, or None."""
    m = _HEAD.match(ref)
    if not m:
        return None
    name, year = m.group(1), m.group(2)
    if len(name) < 3 or not re.search(r"[A-Za-z]", name):
        return None
    nm = re.escape(name)
    # "Anderson and Rubin" in the bibliography may be "Anderson & Rubin" in text
    nm = re.sub(r"\\?\s+and\\?\s+", r"[\\s\\u00a0]+(?:and|&)[\\s\\u00a0]+", nm)
    # spaces flexible (NBSP shows up in HTML text); re.escape may or may not
    # have escaped them depending on Python version — handle both spellings
    nm = nm.replace("\\ ", "[\\s\\u00a0]+").replace(" ", "[\\s\\u00a0]+")
    return re.compile(nm + r"[\s\u00a0]*(?:\([\s\u00a0]*|,[\s\u00a0]*)"
                      + re.escape(year) + r"(?![0-9a-z])")


def window(sent: str, m: re.Match) -> str:
    if len(sent) <= MAX_LEN:
        return sent
    mid = (m.start() + m.end()) // 2
    a = max(0, mid - MAX_LEN // 2)
    b = min(len(sent), a + MAX_LEN)
    a = max(0, b - MAX_LEN)
    cut = sent[a:b]
    return ("… " if a else "") + cut + (" …" if b < len(sent) else "")


def main() -> int:
    u = load_json(PROCESSED / "union.json")
    gid = {(k, e): i for i, (k, e) in enumerate(zip(u["keys"], u["eids"]))}

    # ---- pass 0: distinctive titles, exactly as icml.citations
    titles: dict[int, str] = {}
    for c in available():
        for p in c.read_papers():
            g = gid.get((c.key, p["event_id"]))
            if g is None:
                continue
            t = norm(p["title"])
            if len(t) >= MIN_TITLE:
                titles[g] = " " + t + " "

    # ---- pass 1: references per citing gid (same walk as icml.citations)
    def bridge(f: Path):
        key = (f.stem.replace("fulltext_html_", "")
               .replace("fulltext_", "").replace("_", "-"))
        res = Path("data/raw/arxiv") / (f"resolved_{key}.jsonl" if key else
                                        "resolved.jsonl")
        if not res.exists():
            res, key = Path("data/raw/arxiv/resolved.jsonl"), "icml-2026"
        to_gid = {}
        for r in read_jsonl(res):
            if r.get("arxiv_base"):
                g = gid.get((key, r["event_id"]))
                if g is not None:
                    to_gid.setdefault(r["arxiv_base"], g)
        return to_gid

    src_files = sorted(INTERIM.glob("fulltext*.jsonl"))
    refs_of: dict[int, list[str]] = {}
    src_of: dict[int, str] = {}       # which file the refs came from — the
    for f in src_files:               # body searched must be the same one
        to_gid = bridge(f)
        for row in read_jsonl(f):
            if not row.get("ok") or not row.get("references"):
                continue
            g = to_gid.get(row.get("arxiv_base"))
            if g is not None and g not in refs_of:
                refs_of[g] = row["references"]
                src_of[g] = f.name

    # ---- pass 2: title -> (citing gid, ref index) hits, as icml.citations
    ref_norm: dict[tuple[int, int], str] = {}
    tok_ix: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for g, refs in refs_of.items():
        for j, r in enumerate(refs):
            n = " " + norm(r) + " "
            ref_norm[(g, j)] = n
            for t in set(n.split()):
                if len(t) >= 4:
                    tok_ix[t].append((g, j))

    # citing gid -> [(compiled marker, cited gid)]
    marks_of: dict[int, list] = defaultdict(list)
    n_edges = n_nohead = 0
    for tg, t in titles.items():
        toks = [w for w in t.split() if len(w) >= 4]
        if not toks:
            continue
        anchor = min(toks, key=lambda w: len(tok_ix.get(w, ())))
        hits = [(cg, j) for (cg, j) in tok_ix.get(anchor, ())
                if t in ref_norm[(cg, j)]]
        if len(hits) > MAX_HITS:
            continue
        for cg, j in hits:
            if cg == tg:
                continue
            n_edges += 1
            mk = marker(refs_of[cg][j])
            if mk is None:
                n_nohead += 1
            else:
                marks_of[cg].append((mk, tg))

    # ---- pass 3: re-read bodies, keep sentences where a marker fires
    ctx: dict[int, list] = defaultdict(list)      # cited gid -> rows
    pairs_hit: set[tuple[int, int]] = set()
    seen_g: set[int] = set()
    for f in src_files:
        to_gid = bridge(f)
        for row in read_jsonl(f):
            g = to_gid.get(row.get("arxiv_base"))
            # read the body of the SAME file the references came from — the
            # PDF twin of an HTML row lacks the natbib text the markers match
            if (g is None or src_of.get(g) != f.name or g in seen_g
                    or g not in marks_of or not row.get("ok")):
                continue
            seen_g.add(g)
            per_pair: dict[int, int] = defaultdict(int)
            for sec in row.get("sections") or ():
                bucket = sec.get("bucket") or ""
                for sent in sentences(sec.get("text") or ""):
                    for mk, tg in marks_of[g]:
                        if per_pair[tg] >= MAX_SENTS:
                            continue
                        m = mk.search(sent)
                        if m:
                            per_pair[tg] += 1
                            pairs_hit.add((g, tg))
                            ctx[tg].append([g, bucket, window(sent, m)])

    # ---- caps + dedup, stable by citing gid then document order
    out = {}
    for tg, rows in ctx.items():
        seen, kept, citers = set(), [], []
        for r in rows:                       # document order within citer
            if r[0] not in citers:
                if len(citers) >= MAX_CITERS:
                    continue
                citers.append(r[0])
            k = (r[0], r[2])
            if k not in seen:
                seen.add(k)
                kept.append(r)
        out[str(tg)] = [r for r in kept if r[0] in citers]

    n_sent = sum(len(v) for v in out.values())
    dump_json(PROCESSED / "cite_contexts.json", {
        "note": "verbatim sentences from the citing paper's parsed body where "
                "an in-corpus citation occurs; located by the reference head's "
                "own in-text tag (natbib), no classification, no paraphrase; "
                f"caps {MAX_CITERS} citers / {MAX_SENTS} sentences per pair",
        "contexts": out,
        "stats": {"edges": n_edges, "edges_with_context": len(pairs_hit),
                  "no_nameyear_head": n_nohead,
                  "cited_papers": len(out), "sentences": n_sent},
    }, indent=None)
    print(f"edges {n_edges:,} | with >=1 context sentence {len(pairs_hit):,} "
          f"({len(pairs_hit)*100//max(1,n_edges)}%) | numeric/odd ref heads "
          f"{n_nohead:,} | cited papers covered {len(out):,} | "
          f"sentences {n_sent:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
