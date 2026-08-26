"""Full-text phases 3+4 — PDF -> sectioned text, with the noise dropped.

GROBID would be the textbook choice, but Docker needs sudo here. arXiv PDFs are
born-digital with regular formatting, so heading detection on font weight/size
plus a section-name regex recovers the structure at ~50x the speed of any
model-based parser. Validated on samples before adoption.

The point of this stage is NOT faithful reconstruction — it is throwing away
what the extraction model must not waste tokens on:

  * everything from References onward (an ICML paper is ~8 body pages and
    ~20 appendix pages; this single cut removes most of the file)
  * Related Work, which describes OTHER papers and would pollute per-paper
    method extraction

Requires the project venv: .venv/bin/python -m icml.pdf_extract

Output: data/interim/fulltext.jsonl  (one row per PDF)
"""
from __future__ import annotations

import argparse
import json
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover
    raise SystemExit("PyMuPDF missing — run: .venv/bin/pip install pymupdf")

from .common import INTERIM, RAW

PDF_DIR = RAW / "pdf"
OUT = INTERIM / "fulltext.jsonl"

BOLD = 1 << 4

_SEC = re.compile(
    r"^\s*(?P<num>\d+(?:\.\d+)*)?[\.\)]?\s*"
    r"(?P<name>abstract|introduction|related work|prior work|background|preliminaries"
    r"|method(?:s|ology)?|approach|framework|experiment(?:s|al setup)?|evaluation"
    r"|result(?:s)?|analysis|ablation|discussion|limitations?|conclusion(?:s)?"
    r"|references|bibliography|appendi(?:x|ces)|acknowledg\w*)\b",
    re.I)

# Section name -> bucket. Buckets decide what survives.
BUCKET = {
    "abstract": "abstract",
    "introduction": "intro",
    "related work": "related", "prior work": "related",
    "background": "background", "preliminaries": "background",
    "method": "method", "methods": "method", "methodology": "method",
    "approach": "method", "framework": "method",
    "experiment": "experiments", "experiments": "experiments",
    "experimental setup": "experiments", "evaluation": "experiments",
    "result": "experiments", "results": "experiments",
    "analysis": "experiments", "ablation": "experiments",
    "discussion": "conclusion", "limitation": "conclusion",
    "limitations": "conclusion", "conclusion": "conclusion", "conclusions": "conclusion",
    "references": "back", "bibliography": "back",
    "appendix": "back", "appendices": "back",
}

KEEP = {"abstract", "intro", "background", "method", "experiments", "conclusion"}


def bucket_for(name: str) -> str:
    n = name.lower().strip()
    if n.startswith("acknowledg"):
        return "back"
    return BUCKET.get(n, "other")


def extract(path: Path) -> dict:
    """PDF -> {sections: [...], kept_text, stats}. Never raises; reports errors."""
    row: dict = {"arxiv_base": path.stem}
    try:
        doc = fitz.open(path)
    except Exception as exc:  # noqa: BLE001
        return row | {"ok": False, "error": f"open: {type(exc).__name__}"}

    try:
        lines: list[tuple[int, str, bool, float]] = []
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    spans = line.get("spans") or []
                    if not spans:
                        continue
                    text = "".join(s["text"] for s in spans).strip()
                    if not text:
                        continue
                    s0 = spans[0]
                    lines.append((page.number, text, bool(s0["flags"] & BOLD), round(s0["size"], 1)))
        n_pages = len(doc)
    except Exception as exc:  # noqa: BLE001
        return row | {"ok": False, "error": f"parse: {type(exc).__name__}"}
    finally:
        doc.close()

    if not lines:
        return row | {"ok": False, "error": "no text (scanned or image-only PDF?)"}

    body_size = max(
        {sz: sum(len(t) for _, t, _, s in lines if s == sz) for _, _, _, sz in lines}.items(),
        key=lambda kv: kv[1])[0]

    # Heading = looks like a section name AND is visually distinguished.
    # A numbered heading is high confidence; an unnumbered one must be one of the
    # known standalone headings, or bold text mid-paragraph produces false splits.
    heads: list[tuple[int, str, str]] = []  # (line index, raw, bucket)
    for i, (_pg, text, bold, size) in enumerate(lines):
        if len(text) > 80:
            continue
        m = _SEC.match(text)
        if not m or not (bold or size > body_size + 1.0):
            continue
        b = bucket_for(m.group("name"))
        if b == "other":
            continue
        if not m.group("num") and b not in {"abstract", "back"}:
            continue  # unnumbered mid-body match -> almost always a false positive
        heads.append((i, text, b))

    total_chars = sum(len(t) for _, t, _, _ in lines)

    sections = []
    if heads:
        for j, (start, raw, b) in enumerate(heads):
            end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
            body = " ".join(t for _, t, _, _ in lines[start + 1 : end])
            body = re.sub(r"\s+", " ", body).strip()
            if body:
                sections.append({"heading": re.sub(r"\s+", " ", raw)[:80], "bucket": b,
                                 "chars": len(body), "text": body})
    else:
        # No headings recovered: keep everything, flagged, rather than guessing.
        body = re.sub(r"\s+", " ", " ".join(t for _, t, _, _ in lines))
        sections = [{"heading": "(unsegmented)", "bucket": "unsegmented",
                     "chars": len(body), "text": body}]

    kept = [s for s in sections if s["bucket"] in KEEP or s["bucket"] == "unsegmented"]
    kept_chars = sum(s["chars"] for s in kept)

    return row | {
        "ok": True,
        "pages": n_pages,
        "segmented": bool(heads),
        "total_chars": total_chars,
        "kept_chars": kept_chars,
        "reduction": round(1 - kept_chars / total_chars, 3) if total_chars else 0.0,
        "buckets": sorted({s["bucket"] for s in sections}),
        "sections": [{k: s[k] for k in ("heading", "bucket", "chars", "text")} for s in kept],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract and filter full text from arXiv PDFs.")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--rebuild", action="store_true", help="re-extract everything")
    ap.add_argument("--out", default=None,
                    help="alternate output (per-corpus fulltext jsonl)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else OUT
    out.parent.mkdir(parents=True, exist_ok=True)
    done: set[str] = set()
    if out.exists() and not args.rebuild:
        done = {r["arxiv_base"] for r in (json.loads(l) for l in out.open() if l.strip())}

    pdfs = [p for p in sorted(PDF_DIR.glob("*.pdf")) if p.stem not in done]
    if args.limit:
        pdfs = pdfs[: args.limit]
    if not pdfs:
        print(f"nothing to do ({len(done)} already extracted)")
        return 0

    print(f"extracting {len(pdfs)} PDFs with {args.workers} workers…")
    start = time.time()
    ok = err = 0
    tot_before = tot_after = 0

    mode = "w" if args.rebuild else "a"
    with out.open(mode, encoding="utf-8") as fh, \
         ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, row in enumerate(pool.map(extract, pdfs, chunksize=8), 1):
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            if row.get("ok"):
                ok += 1
                tot_before += row["total_chars"]
                tot_after += row["kept_chars"]
            else:
                err += 1
            if i % 200 == 0 or i == len(pdfs):
                rate = i / max(time.time() - start, 1e-6)
                print(f"  {i}/{len(pdfs)}  ok={ok} err={err}  {rate:.0f}/s", flush=True)

    print(f"\ndone in {time.time()-start:.0f}s — {ok} ok, {err} failed")
    if tot_before:
        print(f"  text volume: {tot_before/1e6:.0f}M -> {tot_after/1e6:.0f}M chars "
              f"({1-tot_after/tot_before:.0%} dropped)")
        print(f"  est. tokens for extraction: ~{tot_after/4/1e6:.1f}M "
              f"(vs ~{tot_before/4/1e6:.1f}M unfiltered)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
