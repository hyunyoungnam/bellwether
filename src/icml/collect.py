"""Stage 1 — download ICML virtual-site feeds into the immutable raw cache.

The feed is the single source of truth for *which* papers exist. Every
downstream number must trace back to a snapshot written here.

    python3 -m icml.collect                    # focus year only
    python3 -m icml.collect --years 2023 2024 2025 2026
    python3 -m icml.collect --force            # re-download today's snapshot
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json

from .common import (
    FOCUS_YEAR, RAW, TREND_YEARS, dump_json, ensure_dirs, feed_url, fetch, load_json,
)


def snapshot_path(year: int, stamp: str):
    return RAW / f"virtual_feed_{year}_{stamp}.json"


def latest_snapshot(year: int = FOCUS_YEAR):
    """Most recent snapshot for a year, or None."""
    snaps = sorted(RAW.glob(f"virtual_feed_{year}_*.json"))
    return snaps[-1] if snaps else None


def collect_year(year: int, force: bool) -> dict | None:
    out = snapshot_path(year, dt.date.today().isoformat())
    if out.exists() and not force:
        data = load_json(out)
        print(f"[{year}] snapshot present: {out.name} ({data.get('count')} records) — use --force to refresh")
        return None

    url = feed_url(year)
    print(f"[{year}] GET {url}")
    body = fetch(url, timeout=240)
    digest = hashlib.sha256(body).hexdigest()[:16]
    data = json.loads(body)

    results = data.get("results") or []
    if not results:
        print(f"[{year}] feed returned no results — skipping (cache left untouched)")
        return None

    dump_json(out, data, indent=None)

    prev = None
    others = [p for p in sorted(RAW.glob(f"virtual_feed_{year}_*.json")) if p != out]
    if others:
        prev = len(load_json(others[-1]).get("results") or [])

    print(f"[{year}] wrote {out.name}: {len(results)} records, {len(body)/1e6:.1f} MB, sha={digest}")
    if prev is not None and prev != len(results):
        print(f"[{year}] NOTE: record count changed {prev} -> {len(results)} since last snapshot")

    return {
        "year": year,
        "url": url,
        "fetched_at": dt.datetime.now().astimezone().isoformat(),
        "snapshot": out.name,
        "sha256_16": digest,
        "bytes": len(body),
        "record_count": len(results),
        "previous_count": prev,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Download ICML virtual-site paper feeds.")
    ap.add_argument("--years", type=int, nargs="+", default=[FOCUS_YEAR],
                    help=f"years to fetch (known good: {TREND_YEARS})")
    ap.add_argument("--all-years", action="store_true", help=f"shorthand for --years {' '.join(map(str, TREND_YEARS))}")
    ap.add_argument("--force", action="store_true", help="re-download even if today's snapshot exists")
    args = ap.parse_args()

    ensure_dirs()
    years = TREND_YEARS if args.all_years else args.years

    manifest_path = RAW / "feed_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {}
    if not isinstance(manifest, dict) or "years" not in manifest:
        manifest = {"years": {}}

    for year in years:
        entry = collect_year(year, args.force)
        if entry:
            manifest["years"][str(year)] = entry

    if manifest["years"]:
        dump_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
