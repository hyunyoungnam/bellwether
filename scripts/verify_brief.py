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
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bellwether.verify import Verifier  # noqa: E402  (the one shared implementation)


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__.split("\n")[0], file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    brief = json.loads(path.read_text())
    ver = Verifier()

    checked = passed = 0
    for block in brief.get("answer", []):
        for cite in block.get("cites", []):
            checked += 1
            ok = ver.check(int(cite["gid"]), cite.get("quote") or "")
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
