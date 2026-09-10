"""OpenAlex / Semantic Scholar id mapping for the union.

The exports (.bib/.csv) end at our own site unless a row carries the ids the
rest of the world indexes by. This maps every paper we resolved to an arXiv id
(20,276 of 29,605 — icml-2025 is PMLR-only and stays unmapped rather than
title-guessed) to its Semantic Scholar and OpenAlex records, by exact id:

  - S2:      POST /graph/v1/paper/batch with ARXIV:<base>, 500 per call
  - OpenAlex: GET /works?filter=doi:10.48550/arXiv.<base>|..., 50 per call
              (every arXiv paper has that DataCite DOI)

No API keys; polite pacing; every fetched row is cached to
data/interim/ids_cache.jsonl so a rerun only fills gaps.

    python3 -m icml.ids

Output: data/processed/ids.json
    {"ids": {gid: {"arxiv", "doi", "s2", "oa"}}, "coverage": ...}
    doi = the DOI S2 lists (venue DOI when one exists, else the arXiv one).
"""
from __future__ import annotations

import json
import ssl
import time
import urllib.request
import urllib.error

# Python 3.13 verifies certificates strictly (VERIFY_X509_STRICT) and rejects
# S2's chain for a missing Authority Key Identifier. Verification stays on;
# only the strict extension checks are relaxed.
_SSL = ssl.create_default_context()
_SSL.verify_flags &= ~ssl.VERIFY_X509_STRICT

from .common import INTERIM, PROCESSED, dump_json, load_json, read_jsonl
from pathlib import Path

CACHE = INTERIM / "ids_cache.jsonl"
UA = {"User-Agent": "bellwether/1.0 (corpus id mapping; local research tool)"}

_RESOLVED = {
    "icml-2026": "resolved.jsonl",
    "neurips-2024": "resolved_neurips-2024.jsonl",
    "neurips-2025": "resolved_neurips-2025.jsonl",
    "iclr-2025": "resolved_iclr-2025.jsonl",
    "iclr-2026": "resolved_iclr-2026.jsonl",
}


def _http(url: str, data: bytes | None = None, tries: int = 8):
    wait = 5.0
    for _ in range(tries):
        req = urllib.request.Request(url, data=data, headers={
            **UA, **({"Content-Type": "application/json"} if data else {})})
        try:
            with urllib.request.urlopen(req, timeout=60, context=_SSL) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                time.sleep(wait)
                wait = min(wait * 1.7, 90)
                continue
            raise
        except (urllib.error.URLError, TimeoutError):
            time.sleep(wait)
            wait = min(wait * 1.7, 90)
    raise RuntimeError(f"gave up on {url[:80]}")


def main() -> int:
    u = load_json(PROCESSED / "union.json")
    gid = {(k, e): i for i, (k, e) in enumerate(zip(u["keys"], u["eids"]))}

    arx_of: dict[int, str] = {}
    for key, fn in _RESOLVED.items():
        p = Path("data/raw/arxiv") / fn
        if not p.exists():
            continue
        for r in read_jsonl(p):
            if r.get("arxiv_base"):
                g = gid.get((key, r["event_id"]))
                if g is not None and g not in arx_of:
                    arx_of[g] = r["arxiv_base"]
    print(f"arXiv-resolved papers: {len(arx_of):,} of {len(u['keys']):,}")

    cache: dict[str, dict] = {}
    if CACHE.exists():
        for row in read_jsonl(CACHE):
            cache[row["arxiv"]] = row
    print(f"cache: {len(cache):,} rows")
    cf = CACHE.open("a", encoding="utf-8")

    def note(base: str, kind: str, val):
        row = cache.setdefault(base, {"arxiv": base})
        if row.get(kind) != val:
            row[kind] = val
            cf.write(json.dumps(row, ensure_ascii=False) + "\n")
            cf.flush()

    bases = sorted(set(arx_of.values()))

    # ---- Semantic Scholar, 500 ids per POST
    todo = [b for b in bases if "s2" not in cache.get(b, {})]
    print(f"S2: fetching {len(todo):,}")
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        body = json.dumps({"ids": [f"ARXIV:{b}" for b in chunk]}).encode()
        try:
            res = _http("https://api.semanticscholar.org/graph/v1/paper/batch"
                        "?fields=paperId,externalIds", data=body)
        except (RuntimeError, urllib.error.HTTPError) as exc:
            # S2 throttles hard without a key — skip the batch, keep going;
            # a rerun refetches only what the cache is missing
            print(f"  S2 batch at {i} skipped: {str(exc)[:60]}")
            continue
        for b, r in zip(chunk, res):
            if r:
                ext = r.get("externalIds") or {}
                note(b, "s2", r.get("paperId"))
                if ext.get("DOI"):
                    note(b, "doi", ext["DOI"])
            else:
                note(b, "s2", None)
        print(f"  S2 {i + len(chunk):,}/{len(todo):,}")
        time.sleep(1.2)

    # ---- OpenAlex, 50 DOIs per GET (arXiv DataCite DOIs)
    todo = [b for b in bases if "oa" not in cache.get(b, {})]
    print(f"OpenAlex: fetching {len(todo):,}")
    for i in range(0, len(todo), 50):
        chunk = todo[i:i + 50]
        dois = "|".join(f"10.48550/arXiv.{b}" for b in chunk)
        res = _http("https://api.openalex.org/works?filter=doi:" + dois
                    + "&select=id,doi&per-page=50")
        found = {}
        for w in res.get("results", ()):
            d = (w.get("doi") or "").rsplit("10.48550/arxiv.", 1)
            if len(d) == 2:
                found[d[1]] = (w.get("id") or "").rsplit("/", 1)[-1]
        for b in chunk:
            note(b, "oa", found.get(b.lower()))
        if (i // 50) % 20 == 0:
            print(f"  OA {i + len(chunk):,}/{len(todo):,}")
        time.sleep(0.15)
    cf.close()

    ids = {}
    n_s2 = n_oa = n_doi = 0
    for g, b in arx_of.items():
        row = cache.get(b, {})
        doi = row.get("doi") or f"10.48550/arXiv.{b}"
        e = {"arxiv": b, "doi": doi}
        if row.get("s2"):
            e["s2"] = row["s2"]
            n_s2 += 1
        if row.get("oa"):
            e["oa"] = row["oa"]
            n_oa += 1
        if row.get("doi"):
            n_doi += 1
        ids[str(g)] = e

    dump_json(PROCESSED / "ids.json", {
        "note": "external ids by exact arXiv-id match only (icml-2025/PMLR "
                "has no arXiv resolution and is unmapped rather than "
                "title-guessed); doi = S2's listed DOI when present, else the "
                "arXiv DataCite DOI; s2 = Semantic Scholar paperId; oa = "
                "OpenAlex work id",
        "ids": ids,
        "coverage": {"papers": len(u["keys"]), "arxiv": len(arx_of),
                     "s2": n_s2, "oa": n_oa, "listed_doi": n_doi},
    }, indent=None)
    print(f"ids.json: {len(ids):,} papers | s2 {n_s2:,} | oa {n_oa:,} | "
          f"venue-or-listed doi {n_doi:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
