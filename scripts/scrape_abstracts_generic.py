"""Abstracts for a non-ICML venue, from its virtual site — same pattern as
icml.abstracts (server-rendered div.abstract-content, ~5 req/s), parameterised
by venue until the module itself is generalised.

    python3 scripts/scrape_abstracts_generic.py --venue iclr --year 2026

Output: data/raw/abstracts_<venue>_<year>.jsonl   (resume-safe; a failed fetch
is NOT written, so re-running retries it — a 429 recorded as a result would be
baked in forever, see CLAUDE.md > Traps)
"""
import argparse
import glob
import json
import re
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_ABS = re.compile(r'<div class="abstract-content">(.*?)</div>\s*</div>', re.S)
_TAG = re.compile(r"<[^>]+>")
UA = {"User-Agent": "conference-research-tool/1.0"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venue", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--delay", type=float, default=0.2)
    args = ap.parse_args()

    feed = sorted(glob.glob(str(ROOT / f"data/raw/virtual_feed_{args.venue}_{args.year}_*.json")))[-1]
    rows = json.load(open(feed)).get("results", [])
    out = ROOT / f"data/raw/abstracts_{args.venue}_{args.year}.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in open(out)}
    todo = [r for r in rows if r.get("id") not in done]
    print(f"{len(rows)} rows · {len(done)} done · {len(todo)} to fetch", flush=True)

    ok = fail = 0
    with out.open("a", encoding="utf-8") as fh:
        for i, r in enumerate(todo, 1):
            url = f"https://{args.venue}.cc/virtual/{args.year}/poster/{r['id']}"
            try:
                req = urllib.request.Request(url, headers=UA)
                with urllib.request.urlopen(req, timeout=60) as resp:
                    html = resp.read().decode("utf-8", "replace")
                m = _ABS.search(html)
                if not m:
                    raise ValueError("no abstract-content div")
                text = _TAG.sub("", m.group(1)).strip()
                fh.write(json.dumps({"id": r["id"], "abstract": text},
                                    ensure_ascii=False) + "\n")
                ok += 1
            except Exception as exc:  # noqa: BLE001
                fail += 1
                if fail <= 20:
                    print(f"  fail {r['id']}: {type(exc).__name__}", flush=True)
            if i % 200 == 0:
                fh.flush()
                print(f"  {i}/{len(todo)}  ok={ok} fail={fail}", flush=True)
            time.sleep(args.delay)
    print(f"done: ok={ok} fail={fail} -> {out.name}")


if __name__ == "__main__":
    main()
