"""Verify a research brief: every cited quote must exist in its paper.

A brief is agent-written prose over agent-chosen citations, and the prose is
allowed to be wrong — the citations are not. Each cite carries the verbatim
quote the claim rests on; this script checks that quote against everything we
hold for that paper (title, abstract, the verified card sentences), stamps
each cite `v: true/false`, stamps the brief's verified counts, and updates
reports/briefs/index.json. The page renders unverified cites with a warning —
they are never silently dropped, and never silently trusted.

    python3 scripts/verify_brief.py reports/briefs/<id>.json

Stdlib only; run with PYTHONPATH=src from the repo root.
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from wnai.mcp import Store  # noqa: E402  (read-only corpus access)

_NORM = re.compile(r"[^a-z0-9]+")


def norm(s: str) -> str:
    return _NORM.sub(" ", (s or "").lower()).strip()


def source_text(store: Store, gid: int) -> str:
    parts = []
    r = store.rec(gid)
    if r:
        parts += [r.get("title") or "", r.get("abstract") or ""]
    sp = store.span_entry(gid)
    if sp:
        n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
        parts += list(n) + [L or "", K or "", R or ""]
        if isinstance(sp[5], list):          # deep-pass sentences, when present
            for arr in sp[5]:
                if isinstance(arr, list):
                    parts += [x for x in arr if isinstance(x, str)]
    return " " + norm(" ".join(parts)) + " "


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.split("\n")[0], file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    brief = json.loads(path.read_text())
    store = Store()

    checked = passed = 0
    src_cache: dict[int, str] = {}
    for block in brief.get("answer", []):
        for cite in block.get("cites", []):
            checked += 1
            gid = int(cite["gid"])
            q = norm(cite.get("quote") or "")
            if gid not in src_cache:
                try:
                    src_cache[gid] = source_text(store, gid)
                except Exception:  # noqa: BLE001 — bad gid is just unverified
                    src_cache[gid] = ""
            ok = bool(q) and len(q) >= 20 and (" " + q + " ") in src_cache[gid]
            cite["v"] = ok
            passed += ok

    brief["verified"] = {"checked": checked, "passed": passed,
                         "date": time.strftime("%Y-%m-%d")}
    path.write_text(json.dumps(brief, ensure_ascii=False, indent=1))

    idx_path = path.parent / "index.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else []
    idx = [e for e in idx if e.get("id") != brief["id"]]
    idx.append({"id": brief["id"], "question": brief["question"],
                "date": brief.get("date") or time.strftime("%Y-%m-%d"),
                "cites": checked, "verified": passed})
    idx_path.write_text(json.dumps(idx, ensure_ascii=False, indent=1))

    print(f"{brief['id']}: {passed}/{checked} quotes verified")
    if passed < checked:
        print("unverified quotes are MARKED on the page — fix them by quoting "
              "the paper's own sentences from get_paper, or remove the cite")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
