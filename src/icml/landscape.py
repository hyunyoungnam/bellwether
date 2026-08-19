"""Stage A — semantic landscape: embeddings -> 2D layout -> labelled clusters.

Input is **title + abstract**, never full text (see CLAUDE.md > Coverage caveat):
abstracts are a 99.3% census, full text is a ~70% subset biased by subfield, and
full text also adds related-work prose that blurs region boundaries.

Cluster labels come from **c-TF-IDF**, not from the embedding model and not from
an LLM: for each cluster, the terms that distinguish it from every other cluster.
That makes each region's name inspectable evidence rather than an invented phrase.

Reproducibility is enforced, not hoped for: model name, parameters and seed are
written into the output manifest. A landscape whose points move between runs
destroys the user's mental map, so never regenerate it casually.

    .venv/bin/python -m icml.landscape --limit 1000    # demo sample
    .venv/bin/python -m icml.landscape                 # full corpus

Output: data/processed/landscape.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
from collections import Counter, defaultdict

from .common import PROCESSED, ROOT, dump_json, load_json, read_jsonl

ROOT_CONFIG = ROOT / "config"

OUT = PROCESSED / "landscape.json"
FACTS = None  # set in main

DEFAULT_MODEL = "BAAI/bge-m3"
SEED = 20260730

_WORD = re.compile(r"[a-z][a-z0-9\-]{2,}")
STOP = set("""
the and for with without via from into onto our their its this that these those are was were been being
can could should would may might will not but all any both each few more most other some such only own same
too very just now new novel propose proposed proposes present presents paper study studies show shows shown
demonstrate approach approaches method methods model models framework based results result achieve achieves
improve improved improvement performance state art towards toward while however thus therefore furthermore
moreover recent recently work works learning learn learned learns training train trained data set sets large
small high low first second using use used uses which when where what how also than then them they have has
had between within across over under during through
""".split())


def tokens(text: str) -> list[str]:
    return [w for w in _WORD.findall(text.lower()) if w not in STOP]


def concept_labels(concepts: list[list[str]], labels: list[int],
                   top_k: int = 3) -> dict[int, list[str]]:
    """Label clusters from EXTRACTED concepts, not raw abstract words.

    c-TF-IDF over raw text produced labels like "adam · first-" and
    "collaboratively · old" — it happily picks verbs and fragments, because
    nothing constrains a token to be a research concept. The extraction pass
    already produced curated task/method names per paper, so scoring those
    instead yields labels that are concepts by construction.
    """
    per: dict[int, Counter] = defaultdict(Counter)
    for terms, lab in zip(concepts, labels):
        per[lab].update(set(terms))
    global_df: Counter = Counter()
    for c in per.values():
        global_df.update(c)

    out: dict[int, list[str]] = {}
    for lab, counts in per.items():
        n = max(sum(1 for x in labels if x == lab), 1)
        scored = []
        for term, c in counts.items():
            if c < 2:
                continue
            rest = max(global_df[term] - c, 1)
            scored.append(((c / n) * (c / rest), c, term))
        scored.sort(reverse=True)
        out[lab] = [t for _, _, t in scored[:top_k]]
    return out


def ctfidf_labels(texts: list[str], labels: list[int], top_k: int = 4) -> dict[int, list[str]]:
    """Class-based TF-IDF: terms that distinguish each cluster from the rest.

    Plain TF would just return the corpus's most common words for every cluster
    ("model", "data"). Weighting each cluster's term frequency against the term's
    frequency across all other clusters is what makes the label discriminative.
    """
    per_cluster: dict[int, Counter] = defaultdict(Counter)
    for text, lab in zip(texts, labels):
        per_cluster[lab].update(set(tokens(text)))  # doc-frequency within cluster

    global_df: Counter = Counter()
    for c in per_cluster.values():
        global_df.update(c)

    sizes = {lab: sum(1 for x in labels if x == lab) for lab in per_cluster}
    out: dict[int, list[str]] = {}
    for lab, counts in per_cluster.items():
        n = max(sizes[lab], 1)
        scored = []
        for term, c in counts.items():
            if c < max(3, 0.05 * n):
                continue
            tf = c / n
            # How much more common here than everywhere else.
            rest = max(global_df[term] - c, 1)
            scored.append((tf * (c / rest), term))
        scored.sort(reverse=True)
        out[lab] = [t for _, t in scored[:top_k]]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the semantic landscape.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--limit", type=int, default=0, help="sample N papers (0 = all)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--neighbors", type=int, default=15, help="UMAP n_neighbors")
    ap.add_argument("--min-dist", type=float, default=0.08)
    ap.add_argument("--min-cluster", type=int, default=0,
                    help="HDBSCAN min_cluster_size (0 = scale with corpus)")
    ap.add_argument("--cluster", choices=["kmeans","hdbscan"], default="kmeans",
                    help="kmeans tiles the map evenly; hdbscan finds denser cores but leaves gaps")
    ap.add_argument("--k", type=int, default=0, help="kmeans regions (0 = scale with corpus)")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    papers = [p for p in read_jsonl(PROCESSED / "papers.jsonl") if p.get("abstract")]
    if args.limit and args.limit < len(papers):
        # Stratify by area so a sample keeps the corpus's structure instead of
        # over-representing the biggest area.
        by_area: dict[str, list] = defaultdict(list)
        for p in papers:
            by_area[p.get("area") or "Unspecified"].append(p)
        rng = random.Random(args.seed)
        share = args.limit / len(papers)
        sample = []
        for area, group in by_area.items():
            k = max(1, round(len(group) * share))
            sample.extend(rng.sample(group, min(k, len(group))))
        papers = sample[: args.limit]

    print(f"{len(papers)} papers | model={args.model} | seed={args.seed}")

    texts = [f"{p['title']}\n\n{p['abstract']}" for p in papers]

    import numpy as np

    # Cache embeddings: they are the slow step (~2 min for the full corpus) and
    # are unaffected by layout/clustering parameters, which is what you actually
    # iterate on.
    cache = PROCESSED / f"emb_{len(papers)}_{args.model.replace('/', '_')}.npy"
    if cache.exists():
        emb = np.load(cache)
        print(f"embeddings from cache {cache.name} {emb.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        t0 = time.time()
        model = SentenceTransformer(args.model, device="cuda")
        emb = model.encode(texts, batch_size=args.batch, normalize_embeddings=True,
                           show_progress_bar=False, convert_to_numpy=True)
        np.save(cache, emb)
        print(f"embedded {emb.shape} in {time.time()-t0:.0f}s -> {cache.name}")
    import umap
    from sklearn.cluster import HDBSCAN

    t0 = time.time()
    reducer = umap.UMAP(n_neighbors=args.neighbors, min_dist=args.min_dist,
                        n_components=2, metric="cosine", random_state=args.seed)
    xy = reducer.fit_transform(emb)
    print(f"umap in {time.time()-t0:.0f}s")

    # k-means by default. HDBSCAN is better at finding "real" clusters, but for a
    # MAP we want even territories covering the whole surface: on this corpus it
    # produced one blob of 5,851 papers plus four fragments, which names nothing.
    # k-means tiles the space and assigns every paper, so every region gets a name.
    if args.cluster == "hdbscan":
        min_cluster = args.min_cluster or max(8, len(papers) // 90)
        labels = HDBSCAN(min_cluster_size=min_cluster, min_samples=5).fit_predict(xy)
        noise = int((labels < 0).sum())
    else:
        from sklearn.cluster import KMeans
        k = args.k or max(10, min(40, len(papers) // 180))
        labels = KMeans(n_clusters=k, random_state=args.seed, n_init=10).fit_predict(xy)
        noise = 0
    n_clusters = len({int(l) for l in labels if l >= 0})
    sizes = sorted(Counter(int(l) for l in labels).values(), reverse=True)
    print(f"{args.cluster}: {n_clusters} regions, {noise} unclustered, "
          f"sizes {sizes[:3]}…{sizes[-2:] if len(sizes) > 3 else ''}")

    # Prefer extracted concepts for labels; fall back to c-TF-IDF over raw text
    # only where a cluster has too few extracted terms to name itself.
    from .common import INTERIM
    from .normalize_terms import basic
    alias_path = ROOT_CONFIG / "term_aliases.json"
    alias = load_json(alias_path)["methods"]["mapping"] if alias_path.exists() else {}
    facts_by_id: dict[int, list[str]] = {}
    facts_path = INTERIM / "facts_abstract.jsonl"
    if facts_path.exists():
        for r in read_jsonl(facts_path):
            if not r.get("ok"):
                continue
            terms = [alias.get(m["name"], basic(m["name"]))
                     for m in (r["facts"].get("methods") or [])]
            terms += [basic(t["name"]) for t in (r["facts"].get("tasks") or [])]
            facts_by_id[r["event_id"]] = [t for t in terms if t]

    concepts = [facts_by_id.get(p["event_id"], []) for p in papers]
    label_terms = concept_labels(concepts, [int(l) for l in labels])
    fallback = ctfidf_labels(texts, [int(l) for l in labels])
    for lab, terms in label_terms.items():
        if len(terms) < 2:
            label_terms[lab] = (terms + fallback.get(lab, []))[:3]

    # Normalise coordinates to [0,1] so the frontend never rescales.
    xmin, ymin = xy[:, 0].min(), xy[:, 1].min()
    xmax, ymax = xy[:, 0].max(), xy[:, 1].max()
    nx = (xy[:, 0] - xmin) / max(xmax - xmin, 1e-9)
    ny = (xy[:, 1] - ymin) / max(ymax - ymin, 1e-9)

    points = []
    for i, p in enumerate(papers):
        points.append({
            "id": p["event_id"],
            "x": round(float(nx[i]), 5),
            "y": round(float(ny[i]), 5),
            "c": int(labels[i]),
            "area": p.get("area") or "Unspecified",
            "title": p["title"],
            "oral": bool(p.get("is_oral")),
            "spot": bool(p.get("is_spotlight")),
        })

    clusters = []
    for c in sorted({int(l) for l in labels if l >= 0}):
        members = [i for i, l in enumerate(labels) if l == c]
        clusters.append({
            "id": c,
            "size": len(members),
            "label": " · ".join(label_terms.get(c, [])) or f"cluster {c}",
            "terms": label_terms.get(c, []),
            "cx": round(float(nx[members].mean()), 5),
            "cy": round(float(ny[members].mean()), 5),
            "areas": Counter(papers[i].get("area") or "Unspecified"
                             for i in members).most_common(3),
        })

    dump_json(OUT, {
        "manifest": {
            "model": args.model,
            "seed": args.seed,
            "umap": {"n_neighbors": args.neighbors, "min_dist": args.min_dist,
                     "metric": "cosine"},
            "clustering": ({"method": "hdbscan", "min_cluster_size": min_cluster,
                            "min_samples": 5} if args.cluster == "hdbscan"
                           else {"method": "kmeans", "k": n_clusters}),
            "input": "title + abstract (census, never full text)",
            "labels": "c-TF-IDF over cluster members",
            "sampled": bool(args.limit),
            "papers": len(papers),
        },
        "stats": {"clusters": n_clusters, "unclustered": noise},
        "points": points,
        "clusters": clusters,
    })
    print(f"wrote {OUT.name}: {len(points)} points, {n_clusters} clusters")
    for c in sorted(clusters, key=lambda c: -c["size"])[:8]:
        print(f"   {c['size']:4d}  {c['label'][:62]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
