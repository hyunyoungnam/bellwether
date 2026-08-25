"""Push every paper into the Meilisearch index the page queries.

Search is bought, not built (decided 2026-08-25): Meilisearch supplies typo
tolerance, prefix search and relevance ranking that the home-grown term index
never had. Similar papers ride the same engine (decided 2026-08-25): the BGE-M3
vectors already computed for the union go up as a userProvided embedder, and the
page asks /similar instead of being capped at the 20 precomputed neighbours.
This script is the whole integration on the data side — documents are keyed by
gid so the client can join hits straight onto its payload rows.

    .venv/bin/python scripts/search_index.py     # (re)index all active corpora

(.venv because the vectors live in .npy files; everything else is stdlib.)

Idempotent: documents are upserted; a corpus re-extraction just needs a rerun.
Vectors are part of the SAME document upload — a separate vector pass would be
silently dropped by the next full reindex, since POST /documents replaces.
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
EMB_MODEL = "BAAI_bge-m3"
CHUNK = 1000


def req(method: str, path: str, body=None):
    r = urllib.request.Request(
        BASE + path, method=method,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {KEY}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=300) as fh:
        return json.loads(fh.read() or b"{}")


def vectors_by_gid() -> dict[int, list[float]]:
    """gid -> BGE-M3 vector, assembled exactly as icml.embed defines gid:
    rows sorted by (corpus key, event_id), each corpus's cached .npy holding its
    rows in that same within-corpus order. Papers without an abstract have no
    vector and are simply absent."""
    import numpy as np

    u = load_json(PROCESSED / "union.json")
    keys = u["keys"]
    out: dict[int, list[float]] = {}
    g = 0
    while g < len(keys):
        k = keys[g]
        n = keys.count(k)
        mat = np.load(PROCESSED / f"emb_{k}_{n}_{EMB_MODEL}.npy").astype("float32")
        assert mat.shape[0] == n, (k, mat.shape, n)
        # 4 decimals ≈ int8 quantisation error; halves the upload size
        for r in range(n):
            out[g + r] = [round(float(x), 4) for x in mat[r]]
        g += n
    return out


def main() -> int:
    u = load_json(PROCESSED / "union.json")
    gid = {(k, e): i for i, (k, e) in enumerate(zip(u["keys"], u["eids"]))}
    nxt = len(gid)
    vec = vectors_by_gid()

    docs = []
    for c in available():
        for p in c.read_papers():
            key = (c.key, p["event_id"])
            if key not in gid:
                gid[key] = nxt
                nxt += 1
            d = {
                "id": gid[key],
                "title": p["title"],
                "abstract": p.get("abstract") or "",
                "venue": c.venue,
                "year": c.year,
            }
            # papers without an abstract have no vector; null opts them out of
            # the embedder instead of failing the whole settings task
            d["_vectors"] = {"bge": vec.get(gid[key])}
            docs.append(d)

    _ensure_index()
    req("PATCH", "/indexes/papers/settings", {
        "searchableAttributes": ["title", "abstract"],
        "filterableAttributes": ["venue", "year"],
        "displayedAttributes": ["id"],
        "pagination": {"maxTotalHits": 50000},
        "embedders": {"bge": {"source": "userProvided", "dimensions": 1024}},
    })
    last = None
    for s in range(0, len(docs), CHUNK):
        last = req("POST", "/indexes/papers/documents?primaryKey=id",
                   docs[s : s + CHUNK])
    n_vec = sum(1 for d in docs if "_vectors" in d)
    print(f"queued {len(docs):,} documents, {n_vec:,} with vectors "
          f"(last task {last.get('taskUid') if last else '-'})")
    return 0


def _ensure_index() -> bool:
    try:
        req("POST", "/indexes", {"uid": "papers", "primaryKey": "id"})
    except Exception:
        pass  # already exists
    return True


if __name__ == "__main__":
    raise SystemExit(main())
