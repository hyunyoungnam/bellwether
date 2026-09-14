"""Bellwether's answers to the frozen questions, via the running server.

Each question is a fresh conversation (no carry-over), answered by the
user's signed-in claude CLI through POST /chat — no API key, no budget entry.
The server's segments are rendered to markdown the grader can read (quotes
in quotation marks, the cited title right after), and the raw response is
kept as a .json sidecar carrying the verification tally.

    PYTHONPATH=src python scripts/eval/run_bellwether.py [--only T5-01 ...] [--base URL]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = HERE / "answers" / "bellwether"


def render(segs, brief) -> str:
    out = []
    for s in segs:
        t = s.get("t")
        if t == "p":
            out.append(s.get("s", ""))
        elif t == "c":
            title = brief(s["gid"]).get("title", "") if s.get("gid") is not None else ""
            mark = "" if s.get("v") else " [UNVERIFIED]"
            out.append(f'"{s.get("q", "")}" — {title}{mark}')
        elif t == "n":
            out.append(s.get("s", ""))
    return "".join(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    a = ap.parse_args()
    from bellwether.mcp import Store
    S = Store()
    qs = json.load((HERE / "questions_resolved.json").open(encoding="utf-8"))["questions"]
    if a.only:
        qs = [q for q in qs if q["id"] in set(a.only)]
    OUT.mkdir(parents=True, exist_ok=True)
    for q in qs:
        f = OUT / f"{q['id']}.md"
        if f.exists():
            continue
        t0 = time.time()
        req = urllib.request.Request(
            a.base + "/chat", data=json.dumps({"q": q["q"]}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=a.timeout) as fh:
            r = json.loads(fh.read())
        if "error" in r:
            print(f"{q['id']}: ERROR {r['error']}")
            continue
        text = render(r.get("segs", []), S.brief)
        f.write_text(f"<!-- system: bellwether | {time.strftime('%Y-%m-%d %H:%M')} | "
                     f"{time.time() - t0:.0f}s | verified: {json.dumps(r.get('verified'))} -->\n"
                     f"{text}\n", encoding="utf-8")
        (OUT / f"{q['id']}.json").write_text(
            json.dumps(r, ensure_ascii=False), encoding="utf-8")
        print(f"{q['id']}: {len(text)} chars, {time.time() - t0:.0f}s, "
              f"verified {r.get('verified')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
