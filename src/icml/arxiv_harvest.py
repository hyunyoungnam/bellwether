"""Full-text phase 1a — harvest an arXiv metadata index via OAI-PMH.

Why this exists: resolving papers one query at a time against the arXiv search
API drew HTTP 429s even at 4.5s spacing, projecting 8+ hours. OAI-PMH returns
~700 records per request with no throttling, so a few hundred requests build a
COMPLETE local index that resolution can then match against offline, instantly
and repeatably.

Harvests the `cs` and `stat` sets from --since. ICML 2026 preprints are
overwhelmingly 2024 onward; widen the window if match rate disappoints.

    python3 -m icml.arxiv_harvest                    # resume until complete
    python3 -m icml.arxiv_harvest --since 2023-01-01

Output: data/raw/arxiv/oai_index.jsonl  +  oai_state.json (resumption tokens)
"""
from __future__ import annotations

import argparse
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

from .common import RAW, dump_json, ensure_dirs, load_json

OUT = RAW / "arxiv" / "oai_index.jsonl"
STATE = RAW / "arxiv" / "oai_state.json"
BASE = "http://export.arxiv.org/oai2"

OAI = "{http://www.openarchives.org/OAI/2.0/}"
DC = "{http://purl.org/dc/elements/1.1/}"

UA = "icml-atlas/0.1 (academic research; building a conference research map)"


def request(params: dict, timeout: int = 120) -> ET.Element:
    """OAI request honouring 503 + Retry-After, which is arXiv's flow control."""
    url = f"{BASE}?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return ET.fromstring(resp.read())
        except urllib.error.HTTPError as exc:
            # 503 + Retry-After is the documented way OAI servers say "slow down".
            if exc.code in (503, 429):
                wait = int(exc.headers.get("Retry-After", 20) or 20)
                print(f"    {exc.code} — waiting {wait}s", flush=True)
                time.sleep(wait + 1)
                continue
            raise
        except Exception as exc:  # noqa: BLE001
            print(f"    {type(exc).__name__}: {exc} — retrying", flush=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("OAI request failed after retries")


def parse_records(root: ET.Element) -> tuple[list[dict], str | None]:
    rows = []
    for rec in root.iter(f"{OAI}record"):
        header = rec.find(f"{OAI}header")
        if header is not None and header.get("status") == "deleted":
            continue
        ident = rec.findtext(f"{OAI}header/{OAI}identifier") or ""
        aid = ident.rsplit(":", 1)[-1]
        title = rec.findtext(f".//{DC}title") or ""
        title = re.sub(r"\s+", " ", title).strip()
        if not aid or not title:
            continue
        rows.append({
            "arxiv_base": aid,
            "title": title,
            "date": rec.findtext(f".//{DC}date"),
        })
    tok_el = root.find(f".//{OAI}resumptionToken")
    tok = (tok_el.text or "").strip() if tok_el is not None else ""
    return rows, (tok or None)


def harvest_set(name: str, since: str, state: dict, delay: float) -> int:
    token = state.get(name, {}).get("token")
    done = state.get(name, {}).get("complete", False)
    if done:
        print(f"[{name}] already complete")
        return 0

    n = 0
    start = time.time()
    with OUT.open("a", encoding="utf-8") as fh:
        while True:
            params = ({"verb": "ListRecords", "resumptionToken": token} if token
                      else {"verb": "ListRecords", "metadataPrefix": "oai_dc",
                            "set": name, "from": since})
            root = request(params)
            rows, token = parse_records(root)
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
            fh.flush()
            n += len(rows)

            state[name] = {"token": token, "complete": token is None}
            dump_json(STATE, state)

            rate = n / max(time.time() - start, 1e-6)
            print(f"[{name}] +{len(rows):4d}  total={n:6d}  {rate:.0f} rec/s"
                  f"{'' if token else '  COMPLETE'}", flush=True)
            if not token:
                break
            time.sleep(delay)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="Harvest arXiv metadata via OAI-PMH.")
    ap.add_argument("--since", default="2024-01-01", help="harvest records from this date")
    ap.add_argument("--sets", nargs="+", default=["cs", "stat"])
    ap.add_argument("--delay", type=float, default=3.0)
    args = ap.parse_args()

    ensure_dirs()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    state = load_json(STATE) if STATE.exists() else {}

    total = 0
    for s in args.sets:
        total += harvest_set(s, args.since, state, args.delay)

    have = sum(1 for _ in OUT.open()) if OUT.exists() else 0
    print(f"\nharvested +{total} records this run; index now {have:,} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
