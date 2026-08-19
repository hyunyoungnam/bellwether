"""Precompute semantic neighbours — the retrieval backbone.

Answers "which papers are in my field?" by example: give one or more seed papers,
get the papers nearest them in embedding space. A field is often cross-cutting
("efficient attention for long context") and matches no node in any taxonomy, so
retrieval-by-example is a first-class entry path, not a fallback.

Precomputed rather than computed live because the interface is a single static
HTML file with no server. k=20 neighbours for 6,592 papers is a few hundred KB —
small enough to ship inline, and it makes "more like this" instant.

Also exports a compact quantised embedding matrix so the page can score *multi-seed*
queries (average of several seed vectors) that no fixed neighbour list can answer.
int8 over 128 PCA dims keeps the whole corpus under ~1 MB.

    .venv/bin/python -m icml.neighbors
Output: data/processed/neighbors.json, data/processed/embed_compact.json
"""
from __future__ import annotations

import argparse
import json

from .common import PROCESSED, dump_json, read_jsonl


def main() -> int:
    ap = argparse.ArgumentParser(description="Precompute semantic neighbours.")
    ap.add_argument("--k", type=int, default=20, help="neighbours per paper")
    ap.add_argument("--dims", type=int, default=128, help="PCA dims for the compact matrix")
    ap.add_argument("--model", default="BAAI/bge-m3")
    args = ap.parse_args()

    import numpy as np

    papers = [p for p in read_jsonl(PROCESSED / "papers.jsonl") if p.get("abstract")]
    cache = PROCESSED / f"emb_{len(papers)}_{args.model.replace('/', '_')}.npy"
    if not cache.exists():
        raise SystemExit(f"no embedding cache {cache.name} — run `icml.landscape` first")

    emb = np.load(cache).astype(np.float32)
    if emb.shape[0] != len(papers):
        raise SystemExit(f"embedding rows {emb.shape[0]} != papers {len(papers)} — "
                         "the cache is stale; delete it and re-run icml.landscape")

    ids = [p["event_id"] for p in papers]
    # Embeddings are already L2-normalised, so a dot product IS cosine similarity.
    emb /= np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)

    print(f"{len(ids)} papers, {emb.shape[1]} dims — computing top-{args.k} neighbours…")
    k = args.k
    nbrs: dict[str, list] = {}
    # Chunked to keep the similarity matrix out of memory all at once.
    step = 512
    for s in range(0, len(ids), step):
        block = emb[s : s + step] @ emb.T
        for r, row in enumerate(block):
            i = s + r
            row[i] = -1.0  # never a neighbour of itself
            top = np.argpartition(-row, k)[:k]
            top = top[np.argsort(-row[top])]
            nbrs[str(ids[i])] = [[int(ids[j]), round(float(row[j]), 4)] for j in top]

    # Flat parallel arrays, not per-paper objects: the object form cost 2.2 MB of
    # repeated keys and quoting for data that is just two integer columns. The
    # page has to inline this, so the encoding matters.
    flat_id: list[int] = []
    flat_sim: list[int] = []
    for pid in ids:
        for nid, sim in nbrs[str(pid)]:
            flat_id.append(nid)
            flat_sim.append(int(round(sim * 255)))  # cosine in [0,1] -> uint8
    dump_json(PROCESSED / "neighbors.json", {
        "method": f"cosine similarity over {args.model} embeddings of title+abstract",
        "format": "ids[i] has neighbours nbr[i*k:(i+1)*k], similarity sim/255",
        "k": k, "papers": len(ids), "ids": ids, "nbr": flat_id, "sim": flat_sim,
    }, indent=None)

    # Compact matrix for multi-seed queries in the browser: PCA to `dims`, then
    # int8. Reconstruction is approximate, which is fine for ranking.
    from sklearn.decomposition import PCA
    p = PCA(n_components=min(args.dims, emb.shape[1]), random_state=0)
    red = p.fit_transform(emb).astype(np.float32)
    red /= np.linalg.norm(red, axis=1, keepdims=True).clip(1e-9)
    scale = float(np.abs(red).max())
    q = np.clip(np.round(red / scale * 127), -127, 127).astype(np.int8)

    # base64 of the raw int8 buffer: one byte per value instead of ~4 characters
    # of JSON text per value.
    import base64
    dump_json(PROCESSED / "embed_compact.json", {
        "method": f"PCA({red.shape[1]}) of {args.model} embeddings, int8-quantised, base64",
        "explained_variance": round(float(p.explained_variance_ratio_.sum()), 4),
        "scale": scale, "dims": int(red.shape[1]), "ids": ids,
        "b64": base64.b64encode(q.tobytes()).decode("ascii"),
    }, indent=None)

    kb = (PROCESSED / "neighbors.json").stat().st_size / 1024
    kb2 = (PROCESSED / "embed_compact.json").stat().st_size / 1024
    print(f"wrote neighbors.json ({kb:.0f} KB) and embed_compact.json ({kb2:.0f} KB)")
    print(f"  PCA({red.shape[1]}) retains {p.explained_variance_ratio_.sum():.1%} of variance")

    byid = {p["event_id"]: p for p in papers}
    seed = papers[0]
    print(f"\nsanity — nearest to: {seed['title'][:64]}")
    for pid, sim in nbrs[str(seed['event_id'])][:5]:
        print(f"   {sim:.3f}  {byid[pid]['title'][:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
