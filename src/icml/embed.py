"""Embeddings and neighbours over the UNION of every active corpus.

`landscape.py` and `neighbors.py` both hardcode `papers.jsonl`, which silently
means "the focus year". That was fine with one conference and is wrong with six:
a reader asking "more like this" wants last year's paper and the other venue's
paper, and a per-corpus neighbour list can never return either. Neighbours are
therefore computed once, across everything.

**`event_id` is not a global identity.** It is the conference feed's own id, and
ICML 2025 and 2026 not overlapping is luck rather than a guarantee — nothing stops
NeurIPS from numbering into the same range. So this module defines the one global
key the rest of the project uses:

    gid = index into the union, ordered by (corpus key, event_id)

`union.json` is that ordering written down. Every other artifact here is indexed
by gid and by nothing else, and `icml.site` re-derives the same mapping rather
than assuming an order.

    .venv/bin/python -m icml.embed                 # all active corpora
    .venv/bin/python -m icml.embed --batch 8       # if the GPU is busy

Outputs: data/processed/union.json
         data/processed/neighbors_union.json
         data/processed/embed_union.json
"""
from __future__ import annotations

import argparse
import base64

from .common import PROCESSED, dump_json
from .corpus import Corpus, available

MODEL = "BAAI/bge-m3"


def union(venue: str | None = None) -> tuple[list[Corpus], list[tuple[str, int, str]]]:
    """Every abstract-bearing paper across the active corpora, in gid order.

    Papers without an abstract have no vector and are simply absent — they must
    be shown as uncounted downstream, never dropped silently (CLAUDE.md).
    """
    corpora = available(venue)
    rows: list[tuple[str, int, str]] = []
    for c in corpora:
        for p in c.read_papers(with_abstract=True):
            rows.append((c.key, p["event_id"], f"{p['title']}\n\n{p['abstract']}"))
    rows.sort(key=lambda r: (r[0], r[1]))
    return corpora, rows


def _embed(texts: list[str], model: str, batch: int, device: str):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    m = SentenceTransformer(model, device=device)
    m.max_seq_length = 1024        # abstracts run ~750 tokens; 8192 wastes memory
    return np.asarray(m.encode(texts, batch_size=batch, normalize_embeddings=True,
                               show_progress_bar=True), dtype="float32")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--venue", default=None)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--dims", type=int, default=128)
    # The GPU is routinely already holding a vLLM server from an extraction run
    # (CLAUDE.md > Traps). 16 fits in the ~8 GB that leaves; 64 does not.
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import numpy as np

    corpora, rows = union(args.venue)
    keys = [r[0] for r in rows]
    eids = [r[1] for r in rows]
    print(f"{len(rows):,} papers across {len(corpora)} corpora: "
          + " · ".join(f"{c.key} {keys.count(c.key):,}" for c in corpora))

    # Cached per corpus, not per union: adding a seventh conference must not
    # re-embed the six already done. The count is in the name so a corpus that
    # grew cannot silently reuse a stale matrix.
    blocks = []
    for c in corpora:
        idx = [i for i, k in enumerate(keys) if k == c.key]
        cache = PROCESSED / f"emb_{c.key}_{len(idx)}_{args.model.replace('/', '_')}.npy"
        if cache.exists():
            print(f"  {c.key}: cached")
            blocks.append(np.load(cache).astype("float32"))
            continue
        print(f"  {c.key}: embedding {len(idx):,}…")
        e = _embed([rows[i][2] for i in idx], args.model, args.batch, args.device)
        np.save(cache, e)
        blocks.append(e)
    emb = np.vstack(blocks)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    assert emb.shape[0] == len(rows)

    dump_json(PROCESSED / "union.json", {
        "note": "gid = index into these parallel arrays; the only global paper key",
        "order": "sorted by (corpus key, event_id)",
        "corpora": [c.key for c in corpora],
        "keys": keys, "eids": eids,
    }, indent=None)

    print(f"top-{args.k} neighbours across the union…")
    k = args.k
    flat_id: list[int] = []
    flat_sim: list[int] = []
    for s in range(0, len(rows), 512):
        block = emb[s : s + 512] @ emb.T
        for r, row in enumerate(block):
            i = s + r
            row[i] = -1.0
            top = np.argpartition(-row, k)[:k]
            top = top[np.argsort(-row[top])]
            flat_id.extend(int(j) for j in top)
            flat_sim.extend(int(round(max(float(row[j]), 0.0) * 255)) for j in top)
    dump_json(PROCESSED / "neighbors_union.json", {
        "method": f"cosine over {args.model} embeddings of title+abstract, across corpora",
        "format": "gid g has neighbours nbr[g*k:(g+1)*k] as gids, similarity sim/255",
        "k": k, "papers": len(rows), "nbr": flat_id, "sim": flat_sim,
    }, indent=None)

    from sklearn.decomposition import PCA
    p = PCA(n_components=min(args.dims, emb.shape[1]), random_state=0)
    red = p.fit_transform(emb).astype("float32")
    red /= np.linalg.norm(red, axis=1, keepdims=True).clip(1e-9)
    scale = float(np.abs(red).max())
    q = np.clip(np.round(red / scale * 127), -127, 127).astype("int8")
    dump_json(PROCESSED / "embed_union.json", {
        "method": f"PCA({red.shape[1]}) of {args.model}, int8, base64; row index = gid",
        "explained_variance": round(float(p.explained_variance_ratio_.sum()), 4),
        "scale": scale, "dims": int(red.shape[1]),
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
    }, indent=None)

    for name in ("union.json", "neighbors_union.json", "embed_union.json"):
        print(f"  wrote {name} ({(PROCESSED / name).stat().st_size/1024:,.0f} KB)")
    print(f"  PCA({red.shape[1]}) retains {p.explained_variance_ratio_.sum():.1%}")

    # Sanity: the point of the union is that neighbours cross corpora.
    cross = sum(1 for g in range(len(rows))
                if keys[flat_id[g * k]] != keys[g])
    print(f"\nnearest neighbour is in a different corpus for {cross:,}/{len(rows):,} "
          f"papers ({cross/len(rows):.0%}) — per-corpus lists could not return those")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
