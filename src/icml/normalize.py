"""Stage 3 — join the feed snapshot with scraped abstracts into the canonical dataset.

Output: data/processed/papers.jsonl — one record per paper, the ONLY file
downstream stages (metrics, labeling, report) are allowed to read.

    python3 -m icml.normalize
"""
from __future__ import annotations

import argparse
import re

from .collect import latest_snapshot
from .common import (
    FOCUS_YEAR, PROCESSED, RAW, ROOT, dump_json, ensure_dirs, load_json, read_jsonl, write_jsonl,
)

ABSTRACTS = RAW / "abstracts" / "abstracts.jsonl"
OUT = PROCESSED / "papers.jsonl"

# Track identification, derived from the feed's sourceurl field.
TRACK_PATTERNS = [
    (re.compile(r"/\d{4}/Conference", re.I), "main"),
    (re.compile(r"Position_Paper_Track", re.I), "position"),
    (re.compile(r"^TMLR", re.I), "tmlr-journal"),
    (re.compile(r"JmlrOrg", re.I), "jmlr-journal"),
    (re.compile(r"ANN-STATS", re.I), "annals-stats"),
]

# Light-touch institution cleanup: the feed's free-text field is inconsistent.
_INST_TRIM = re.compile(r"\s*,?\s*(inc\.?|ltd\.?|llc)\s*$", re.I)


def classify_track(sourceurl: str | None) -> str:
    s = sourceurl or ""
    for pattern, name in TRACK_PATTERNS:
        if pattern.search(s):
            return name
    return "other"


_ALIAS_FILE = ROOT / "config" / "institution_aliases.json"
_ALIASES: dict[str, str] | None = None


def _aliases() -> dict[str, str]:
    global _ALIASES
    if _ALIASES is None:
        _ALIASES = load_json(_ALIAS_FILE)["aliases"] if _ALIAS_FILE.exists() else {}
    return _ALIASES


def normalize_institution(raw: str | None) -> str | None:
    """Free-text affiliation -> a canonical-ish name.

    Deliberately conservative: collapse whitespace, drop a repeated
    "X, X" self-reference, then apply the explicit alias table. Anything not in
    the table is left exactly as the author wrote it — inventing fuzzy matches
    here would silently merge distinct labs.
    """
    if not raw:
        return None
    s = re.sub(r"\s+", " ", raw).strip().strip(",")
    s = _INST_TRIM.sub("", s)
    if not s:
        return None

    # "Tsinghua University, Tsinghua University" -> "Tsinghua University"
    parts = [p.strip() for p in s.split(",") if p.strip()]
    if len(parts) > 1 and len(set(p.lower() for p in parts)) == 1:
        s = parts[0]

    return _aliases().get(s.lower(), s)


def split_topic(topic: str | None) -> tuple[str | None, str | None]:
    """'Deep Learning->Attention Mechanisms' -> ('Deep Learning', 'Attention Mechanisms')."""
    if not topic:
        return None, None
    parts = [p.strip() for p in topic.split("->") if p.strip()]
    if not parts:
        return None, None
    return parts[0], (parts[1] if len(parts) > 1 else None)


def openreview_id(paper_url: str | None) -> str | None:
    if not paper_url:
        return None
    m = re.search(r"[?&]id=([A-Za-z0-9_-]+)", paper_url)
    return m.group(1) if m else None


def dedupe_key(rec: dict) -> str:
    """Stable identity for a paper across its multiple schedule entries.

    The feed lists an oral paper TWICE — once as its Oral slot and once as its
    Poster slot — with different event ids. Counting rows would inflate the
    corpus by ~160 papers and double-count every oral.

    Identity is the OpenReview id, falling back to the normalised title. The
    feed's own `uid` is deliberately NOT used: it collides across tracks
    (verified — distinct main-track and position-track papers share a uid), so
    keying on it would silently merge unrelated papers.
    """
    oid = openreview_id(rec.get("paper_url"))
    if oid:
        return f"or:{oid}"
    return "title:" + re.sub(r"\W+", "", (rec.get("name") or "").lower())


def merge_duplicates(group: list[dict]) -> dict:
    """Collapse one paper's schedule entries into a single record.

    Oral status is the union (a paper with an Oral slot IS an oral), and we keep
    the entry that carries an abstract so nothing is lost.
    """
    best = sorted(group, key=lambda r: (r.get("abstract") is None, not r.get("is_oral")))[0]
    merged = dict(best)

    # Coalesce, don't pick a record wholesale. The Oral schedule entry usually
    # carries no `topic`, so choosing it discarded the Poster entry's
    # area/subarea — measured: 96 papers, and orals are exactly what people
    # browse first. Take each field from whichever sibling actually has it.
    for field in ("area", "subarea", "topic_raw", "decision", "abstract",
                  "paper_url", "openreview_id", "virtual_url"):
        if not merged.get(field):
            for r in group:
                if r.get(field):
                    merged[field] = r[field]
                    break

    merged["is_oral"] = any(r.get("is_oral") for r in group)
    merged["is_spotlight"] = any(r.get("is_spotlight") for r in group)
    if merged["is_oral"]:
        merged["presentation"] = "Oral"
    merged["duplicate_event_ids"] = sorted(
        r["event_id"] for r in group if r["event_id"] != best["event_id"])
    return merged


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the canonical papers dataset.")
    ap.add_argument("--allow-missing-abstracts", action="store_true",
                    help="build even if abstract coverage is below the 90%% threshold")
    ap.add_argument("--year", type=int, default=FOCUS_YEAR)
    args = ap.parse_args()

    # papers.jsonl is the 2026 canonical file the whole pipeline reads. Earlier
    # years are for trend comparison only and must never overwrite it.
    out = OUT if args.year == FOCUS_YEAR else PROCESSED / f"papers_{args.year}.jsonl"

    ensure_dirs()
    snap = latest_snapshot(args.year)
    if snap is None:
        raise SystemExit("no feed snapshot — run `python3 -m icml.collect` first")

    abstracts = {
        row["event_id"]: row["abstract"]
        for row in read_jsonl(ABSTRACTS)
        if row.get("abstract")
    }
    results = load_json(snap)["results"]

    rows = []
    for rec in results:
        if not rec.get("visible", True):
            continue
        area, subarea = split_topic(rec.get("topic"))
        authors = rec.get("authors") or []
        institutions = []
        for a in authors:
            inst = normalize_institution(a.get("institution"))
            if inst and inst not in institutions:
                institutions.append(inst)

        decision = rec.get("decision") or ""
        rows.append({
            "_key": dedupe_key(rec),
            "year": args.year,
            "event_id": rec["id"],
            "uid": rec.get("uid"),
            "title": (rec.get("name") or "").strip(),
            "abstract": abstracts.get(rec["id"]),
            "authors": [a.get("fullname") for a in authors if a.get("fullname")],
            "author_institutions": [
                {"name": a.get("fullname"), "institution": normalize_institution(a.get("institution"))}
                for a in authors
            ],
            "institutions": institutions,
            "n_authors": len(authors),
            "area": area,
            "subarea": subarea,
            "topic_raw": rec.get("topic") or None,
            "presentation": rec.get("eventtype"),
            "decision": decision or None,
            "is_spotlight": "spotlight" in decision.lower(),
            "is_oral": (rec.get("eventtype") or "").lower() == "oral",
            "track": classify_track(rec.get("sourceurl")),
            "openreview_id": openreview_id(rec.get("paper_url")),
            "paper_url": rec.get("paper_url"),
            "virtual_url": f"https://icml.cc{rec['virtualsite_url']}" if rec.get("virtualsite_url") else None,
        })

    # Collapse the oral/poster double-listing before anything is counted.
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["_key"], []).append(row)
    raw_rows = len(rows)
    rows = [merge_duplicates(g) for g in groups.values()]
    for row in rows:
        row.pop("_key", None)
    rows.sort(key=lambda r: r["event_id"])
    collapsed = raw_rows - len(rows)
    if collapsed:
        print(f"deduplicated {collapsed} duplicate schedule entries "
              f"({raw_rows} rows -> {len(rows)} papers)")

    with_abs = sum(1 for r in rows if r["abstract"])
    coverage = with_abs / len(rows) if rows else 0.0

    if coverage < 0.90 and not args.allow_missing_abstracts:
        raise SystemExit(
            f"abstract coverage {coverage:.1%} ({with_abs}/{len(rows)}) is below 90%.\n"
            "Finish `python3 -m icml.abstracts` first, or pass --allow-missing-abstracts "
            "to build a partial dataset (reports must then state the coverage)."
        )

    n = write_jsonl(out, rows)
    if args.year == FOCUS_YEAR:      # the manifest describes the canonical year only
        dump_json(PROCESSED / "dataset_manifest.json", {
            "source_snapshot": snap.name,
            "papers": n,
            "abstract_coverage": round(coverage, 4),
            "papers_with_abstract": with_abs,
        })
    print(f"wrote {out.relative_to(out.parents[2])}: {n} papers, abstract coverage {coverage:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
