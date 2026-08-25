"""Push every paper into the Meilisearch index the page queries.

Search is bought, not built (decided 2026-08-25): Meilisearch supplies typo
tolerance, prefix search and relevance ranking that the home-grown term index
never had. This script is the whole integration on the data side — documents
are keyed by gid so the client can join hits straight onto its payload rows.

    python3 scripts/search_index.py          # (re)index all active corpora

Idempotent: documents are upserted; a corpus re-extraction just needs a rerun.
"""
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from icml.common import PROCESSED, load_json  # noqa: E402
from icml.corpus import available  # noqa: E402

KEY = (ROOT / "data/meili/master_key").read_text().strip()
BASE = "http://127.0.0.1:7700"


def req(method: str, path: str, body=None):
    r = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=120) as fh:
        return json.loads(fh.read() or b"{}")


def main() -> int:
    u = load_json(PROCESSED / "union.json")
    gid = {(k, e): i for i, (k, e) in enumerate(zip(u["keys"], u["eids"]))}
    nxt = len(gid)

    docs = []
    for c in available():
        for p in c.read_papers():
            key = (c.key, p["event_id"])
            if key not in gid:
                gid[key] = nxt
                nxt += 1
            docs.append({
                "id": gid[key],
                "title": p["title"],
                "abstract": p.get("abstract") or "",
                "venue": c.venue,
                "year": c.year,
            })

    req("PATCH", "/indexes/papers/settings", {
        "searchableAttributes": ["title", "abstract"],
        "filterableAttributes": ["venue", "year"],
        "displayedAttributes": ["id"],
        "pagination": {"maxTotalHits": 50000},
    }) if _ensure_index() else None
    t = req("POST", "/indexes/papers/documents?primaryKey=id", docs)
    print(f"queued {len(docs):,} documents (task {t.get('taskUid')})")
    return 0


def _ensure_index() -> bool:
    try:
        req("POST", "/indexes", {"uid": "papers", "primaryKey": "id"})
    except Exception:
        pass  # already exists
    return True


if __name__ == "__main__":
    raise SystemExit(main())
