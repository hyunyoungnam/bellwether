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
        "ids": store.ext_ids.get(str(gid)) or {},
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
        x = f["ids"]
        if x.get("arxiv"):
            rows.append(f'  eprint = {{{x["arxiv"]}}}')
            rows.append('  archiveprefix = {arXiv}')
        if x.get("doi"):
            rows.append(f'  doi = {{{x["doi"]}}}')
        rows.append(f'  note = {{{f["venue"]} {f["year"]}}}')
        out.append("@inproceedings{" + k + ",\n" + ",\n".join(rows) + "\n}")
    return "\n\n".join(out) + ("\n" if out else "")


CSV_COLS = ["gid", "title", "venue", "year", "authors", "url",
            "arxiv", "doi", "openalex", "s2",
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
        x = f["ids"]
        w.writerow([f["gid"], f["title"], f["venue"], f["year"],
                    "; ".join(f["authors"]), f["url"],
                    x.get("arxiv", ""), x.get("doi", ""),
                    f'https://openalex.org/{x["oa"]}' if x.get("oa") else "",
                    (f'https://www.semanticscholar.org/paper/{x["s2"]}'
                     if x.get("s2") else ""),
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
    cites, turns, figs = [], [], []
    for t in doc.get("turns", []):
        text, tc, tf = [], [], []
        for s in t.get("segs", []):
            if s.get("t") == "p":
                text.append(s.get("s", ""))
            elif s.get("t") == "n":
                # a figure carries its own provenance: the tool and argument
                # that produced it, and whether running them again agreed
                f = {"figures": s.get("s"), "tool": s.get("tool"),
                     "argument": s.get("arg"), "recomputed": s.get("v"),
                     "not_found": s.get("missing") or [],
                     # declared by the agent, or noticed by the server
                     "source": "server" if s.get("auto") else "anchor"}
                tf.append(f)
                figs.append(f)
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
                      "citations": tc, "figures": tf,
                      "tools": t.get("trail") or [],
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
        "figures": figs,
        "verification": {
            "checked": sum(t.get("verified", {}).get("checked", 0) for t in turns),
            "passed": sum(t.get("verified", {}).get("passed", 0) for t in turns),
            "figures_checked": sum(t.get("verified", {}).get("fchecked", 0)
                                   for t in turns),
            "figures_passed": sum(t.get("verified", {}).get("fpassed", 0)
                                  for t in turns),
            "method": "each quote was matched against the paper's own text on "
                      "this machine before it was shown, and each figure was "
                      "checked by running its tool again here; what failed is "
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
        if t["figures"]:
            out.append("| ✓ | figures | recomputed from |")
            out.append("|---|---|---|")
            for f in t["figures"]:
                if f["source"] == "server":
                    mark, src = "·", f"not in {f['tool']}"
                else:
                    mark = {"ok": "✓", "no": "✗"}.get(f["recomputed"], "·")
                    src = f"{f['tool']}({f['argument']})"
                out.append(f"| {mark} | {(f['figures'] or '').replace('|', chr(92)+'|')} "
                           f"| {src} |")
            out.append("")
    v = b["verification"]
    out += ["---", "",
            f"{v['passed']}/{v['checked']} quotes verified and "
            f"{v['figures_passed']}/{v['figures_checked']} figures recomputed "
            f"against {b['corpus']['papers_with_gid']:,} papers held locally "
            f"({', '.join(b['corpus']['corpora'])}).", ""]
    return "\n".join(out)
