"""The way out: a selection as BibTeX or CSV, a conversation as a bundle.

A tool that cannot hand its result to the next tool is a dead end. Everything
here is built from the same records the page shows — the papers' own metadata
and their verified sentences — and nothing is composed for the occasion.

BibTeX needs the full author list, which the browser payload does not carry
(one name and a count is enough for a card, not for a citation), so the export
runs server-side against papers_*.jsonl.
"""
from __future__ import annotations

import csv
import io
import json
import re
import time

from .mcp import Store, _VENUE

_BOOKTITLE = {
    "icml": "International Conference on Machine Learning",
    "neurips": "Advances in Neural Information Processing Systems",
    "iclr": "International Conference on Learning Representations",
}
_NONWORD = re.compile(r"[^a-z0-9]+")


def _key(rec: dict, venue: str, year: str, gid: int) -> str:
    """A citation key a human recognises: lastname + year + first title word."""
    au = (rec.get("authors") or [""])[0]
    last = _NONWORD.sub("", au.split()[-1].lower()) if au.strip() else ""
    word = ""
    for w in re.split(r"\s+", rec.get("title") or ""):
        w2 = _NONWORD.sub("", w.lower())
        if len(w2) > 3:
            word = w2
            break
    stem = "".join(x for x in (last, str(year), word) if x)
    return stem or f"{venue}{year}p{gid}"


def _brace(s: str) -> str:
    """Protect capitals (BibTeX lowercases titles) and escape the specials."""
    s = (s or "").replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    return s


def _fields(store: Store, gid: int) -> dict:
    key, eid = store.where(gid)
    venue, year = key.rsplit("-", 1)
    rec = store.rec(gid) or {}
    sp = store.span_entry(gid) or [None] * 4
    n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
    t = store.terms(key, eid)
    join = lambda v: ", ".join(v) if isinstance(v, list) else (v or "")
    return {
        "gid": gid, "key": key, "venue_key": venue, "year": year,
        "venue": _VENUE.get(venue, venue), "rec": rec,
        "title": rec.get("title") or "",
        "authors": rec.get("authors") or [],
        "url": rec.get("virtual_url") or rec.get("paper_url") or "",
        "limitation": L or "", "key_change": K or (n[0] if n else ""),
        "result_claim": R or "",
        "proposes": join(t.get("p")), "builds_on": join(t.get("b")),
        "tasks": join(t.get("t")), "data": join(t.get("d")),
    }


def bibtex(gids: list[int], store: Store | None = None) -> str:
    store = store or Store()
    out = []
    seen: set[str] = set()
    for gid in gids:
        try:
            f = _fields(store, gid)
        except (KeyError, IndexError, ValueError):
            continue
        if not f["title"]:
            continue
        k = _key(f["rec"], f["venue_key"], f["year"], gid)
        while k in seen:                      # two papers, one natural key
            k += "a"
        seen.add(k)
        rows = [f'  title = {{{_brace(f["title"])}}}',
                f'  author = {{{" and ".join(_brace(a) for a in f["authors"])}}}',
                f'  booktitle = {{{_BOOKTITLE.get(f["venue_key"], f["venue"])}}}',
                f'  year = {{{f["year"]}}}']
        if f["url"]:
            rows.append(f'  url = {{{f["url"]}}}')
        rows.append(f'  note = {{{f["venue"]} {f["year"]}}}')
        out.append("@inproceedings{" + k + ",\n" + ",\n".join(rows) + "\n}")
    return "\n\n".join(out) + ("\n" if out else "")


CSV_COLS = ["gid", "title", "venue", "year", "authors", "url",
            "proposes", "builds_on", "data", "tasks",
            "limitation", "key_change", "result_claim"]


def as_csv(gids: list[int], store: Store | None = None) -> str:
    store = store or Store()
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(CSV_COLS)
    for gid in gids:
        try:
            f = _fields(store, gid)
        except (KeyError, IndexError, ValueError):
            continue
        w.writerow([f["gid"], f["title"], f["venue"], f["year"],
                    "; ".join(f["authors"]), f["url"],
                    f["proposes"], f["builds_on"], f["data"], f["tasks"],
                    f["limitation"], f["key_change"], f["result_claim"]])
    return buf.getvalue()


# ---------------------------------------------------------------- conversation
# A published conversation is a research object: the question, what the agent
# wrote, and — the part that makes it checkable — every quote with the verdict
# the server gave it before the reader saw it. The bundle carries the corpus it
# was answered against, so the same question can be asked of the same shelf.

def _corpus_stamp(store: Store) -> dict:
    u = store.union
    return {"corpora": [c["k"] if isinstance(c, dict) else c
                        for c in (u.get("corpora") or [])],
            "papers_with_gid": len(u.get("keys") or [])}


def conversation(doc: dict, store: Store | None = None) -> dict:
    """The machine-readable bundle. Prose is the agent's; quotes are checked."""
    store = store or Store()
    cites, turns = [], []
    for t in doc.get("turns", []):
        text, tc = [], []
        for s in t.get("segs", []):
            if s.get("t") == "p":
                text.append(s.get("s", ""))
            else:
                c = {"gid": s.get("gid"), "title": s.get("title"),
                     "venue": s.get("venue"), "year": s.get("year"),
                     "quote": s.get("q"), "verified": bool(s.get("v")),
                     "matched_field": s.get("role")}
                tc.append(c)
                cites.append(c)
                text.append(f"[{c['venue']} {c['year']} · {c['title']}]")
        turns.append({"question": t.get("q"), "answer": "".join(text),
                      "citations": tc, "tools": t.get("trail") or [],
                      "verified": t.get("verified") or {}})
    return {
        "@context": "https://schema.org",
        "@type": "ScholarlyArticle",
        "name": doc.get("title"),
        "identifier": doc.get("id"),
        "dateCreated": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                     time.gmtime(doc.get("ts") or 0)),
        "generator": {"name": "Bellwether", "agent": doc.get("agent")},
        "corpus": _corpus_stamp(store),
        "turns": turns,
        "citations": cites,
        "verification": {
            "checked": sum(t.get("verified", {}).get("checked", 0) for t in turns),
            "passed": sum(t.get("verified", {}).get("passed", 0) for t in turns),
            "method": "each quote was matched against the paper's own text on "
                      "this machine before it was shown; unmatched quotes are "
                      "marked, never removed",
        },
    }


def conversation_md(doc: dict, store: Store | None = None) -> str:
    b = conversation(doc, store)
    out = [f"# {b['name']}", "",
           f"{b['dateCreated']} · Bellwether · agent: {b['generator']['agent']}",
           ""]
    for t in b["turns"]:
        out += [f"## {t['question']}", "", t["answer"], ""]
        if t["citations"]:
            out.append("| ✓ | paper | quote |")
            out.append("|---|---|---|")
            for c in t["citations"]:
                q = (c["quote"] or "").replace("|", "\\|")
                out.append(f"| {'✓' if c['verified'] else '✗'} | "
                           f"{c['venue']} {c['year']} · {c['title']} | {q} |")
            out.append("")
    v = b["verification"]
    out += ["---", "",
            f"{v['passed']}/{v['checked']} quotes verified against "
            f"{b['corpus']['papers_with_gid']:,} papers held locally "
            f"({', '.join(b['corpus']['corpora'])}).", ""]
    return "\n".join(out)
