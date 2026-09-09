"""Corresponding author, taken from the paper's own front matter.

The conference feed carries author names and institutions but no email — checked,
the field does not exist. The ICML LaTeX template does: every paper built from it
prints `Correspondence to: Name <addr@host>.` on page one, and 65% of the 4,770
extracted full texts contain that line.

That line is used and nothing else is. Guessing that the first author, or the
last, is the corresponding one would be inventing a fact the paper states plainly
when it states it at all — and staying silent for the other 35% is the honest
result, not a gap to paper over.

    .venv/bin/python -m icml.contacts
Output: data/processed/contacts.json
"""
from __future__ import annotations

import argparse
import re

from .common import INTERIM, PROCESSED, RAW, dump_json, read_jsonl
from .corpus import available

FULLTEXT = INTERIM / "fulltext.jsonl"
RESOLVED = RAW / "arxiv" / "resolved.jsonl"
OUT = PROCESSED / "contacts.json"

# "Correspondence to: Tae-Hoon Lee <th.lee@kaist.ac.kr>, Min-Soo Kim <...>."
_BLOCK = re.compile(r"correspondence\s+to\s*:?\s*(.{0,300})", re.I | re.S)
# The sentence that follows in the template, which must not be swallowed.
_STOP = re.compile(r"(proceedings of the|preprint|copyright|\d{4} by the author)", re.I)
_PAIR = re.compile(r"([A-Z][\w.'\-]*(?:\s+[A-Z][\w.'\-]*){0,3})\s*"
                   r"[<(\[]\s*([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\s*[>)\]]")
_EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_WS = re.compile(r"\s+")
# PDF line breaks split names AND addresses: "Li- jie Yang", "zhi- haoz3@cs.cmu.edu".
# Inside a correspondence line the hyphen is always the break, never a real one —
# genuinely hyphenated names are written without a space around the hyphen — so
# the join is unconditional here, unlike the suffix-aware repair used for prose.
_BREAK = re.compile(r"(\w)-\s+(\w)")


def parse(text: str) -> list[dict]:
    """Every Name <email> pair in the correspondence line, in order."""
    m = _BLOCK.search(text)
    if not m:
        return []
    seg = _BREAK.sub(r"\1\2", _WS.sub(" ", m.group(1)))
    stop = _STOP.search(seg)
    if stop:
        seg = seg[: stop.start()]
    out, seen = [], set()
    for name, email in _PAIR.findall(seg):
        # PDF extraction breaks addresses across lines: "minsoo.k@kaist" plus
        # "Min- Soo Kim". Repair only the space, never the characters.
        email = email.replace(" ", "").lower()
        name = _WS.sub(" ", name).strip(" ,.;")
        if email in seen or not name:
            continue
        seen.add(email)
        out.append({"name": name, "email": email})
    if out:
        return out
    # A bare address with no name attached still identifies who to write to.
    for email in _EMAIL.findall(seg)[:2]:
        e = email.replace(" ", "").lower()
        if e not in seen:
            seen.add(e)
            out.append({"name": None, "email": e})
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--head-chars", type=int, default=3000,
                    help="how much of each opening section to read")
    args = ap.parse_args()

    # One full-text source per corpus: arXiv preprints for the focus year
    # (bridged to event_id via resolved.jsonl), the PMLR camera-ready for
    # published years (rows carry event_id directly). Output is keyed by corpus,
    # because event_id is not a global identity.
    # The richest full-text file each corpus has, and the bridge from arXiv id
    # to event_id where its rows do not carry one. Same table as
    # bellwether.mcp._FT — a corpus added there needs adding here.
    SOURCES = {
        # NOT the HTML pass for ICML 2026: "Correspondence to:" is printed by
        # the ICML LaTeX template, so it survives in the PDF text and not in
        # the arXiv HTML. Measured — the HTML source yields 0%.
        "icml-2026": ("fulltext.jsonl", "resolved.jsonl"),
        "icml-2025": ("fulltext_pmlr_2025.jsonl", None),
        "neurips-2024": ("fulltext_neurips_2024.jsonl", "resolved_neurips-2024.jsonl"),
        "neurips-2025": ("fulltext_neurips_2025.jsonl", "resolved_neurips-2025.jsonl"),
        "iclr-2025": ("fulltext_iclr_2025.jsonl", "resolved_iclr-2025.jsonl"),
        "iclr-2026": ("fulltext_iclr_2026.jsonl", "resolved_iclr-2026.jsonl"),
    }

    def bridge(name: str | None) -> dict[str, int]:
        out: dict[str, int] = {}
        if not name:
            return out
        path = RAW / "arxiv" / name
        if not path.exists():
            return out
        for r in read_jsonl(path):
            if r.get("arxiv_base"):
                out.setdefault(r["arxiv_base"], r["event_id"])
        return out

    found: dict[str, dict] = {}
    totals = {}
    for c in available():
        spec = SOURCES.get(c.key)
        if spec is None:
            print(f"  {c.key}: no full-text source listed — skipped")
            continue
        src = INTERIM / spec[0]
        if not src.exists():
            src = FULLTEXT if c.is_focus else src
        if not src.exists():
            print(f"  {c.key}: {spec[0]} not on disk — skipped")
            continue
        to_event = bridge(spec[1])
        got, scanned = {}, 0
        for row in read_jsonl(src):
            if not row.get("ok"):
                continue
            scanned += 1
            eid = row.get("event_id") or to_event.get(row["arxiv_base"])
            if eid is None:
                continue
            head = " ".join((s.get("text") or "")[: args.head_chars]
                            for s in (row.get("sections") or [])[:2])
            people = parse(head)
            if people:
                got[eid] = people
        found[c.key] = got
        totals[c.key] = scanned
        print(f"  {c.key}: {len(got):,} of {scanned:,} full texts "
              f"({len(got)/max(scanned,1):.0%}) name a corresponding author")

    if not found:
        raise SystemExit("no full-text file found — run icml.pdf_extract or icml.pmlr")

    dump_json(OUT, {
        "method": ("parsed from the paper's own 'Correspondence to:' line; papers "
                   "that do not print one are absent rather than guessed"),
        "scanned_fulltexts": totals,
        "papers_with_contact": {k: len(v) for k, v in found.items()},
        "contacts": found,
    })
    print(f"wrote {OUT.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
