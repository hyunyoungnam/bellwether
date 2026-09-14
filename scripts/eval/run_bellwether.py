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
import re
import sys
import time
import urllib.request

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = HERE / "answers" / "bellwether"


def _norm(s: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", (s or "").lower()))


def render(segs, brief) -> str:
    """Segments -> markdown the grader reads. A chip whose quote IS the paper's
    title is a citation, rendered once in bold; a sentence chip is rendered
    as "quote" — Title so the grader can attribute and verify it."""
    out = []
    for s in segs:
        t = s.get("t")
        if t == "p":
            out.append(s.get("s", ""))
        elif t == "c":
            title = brief(s["gid"]).get("title", "") if s.get("gid") is not None else ""
            q = s.get("q", "")
            mark = "" if s.get("v") else " [UNVERIFIED]"
            if title and _norm(q) == _norm(title):
                out.append(f"**{title}**{mark}")
            else:
                out.append(f'"{q}" — {title}{mark}')
        elif t == "n":
            out.append(s.get("s", ""))
    return "".join(out)


def rerender(brief) -> int:
    """Rebuild every .md from its .json sidecar (rendering changed, answers did not)."""
    n = 0
    for js in sorted(OUT.glob("*.json")):
        r = json.loads(js.read_text(encoding="utf-8"))
        md = OUT / (js.stem + ".md")
        head = md.read_text(encoding="utf-8").splitlines()[0] if md.exists() else "<!-- system: bellwether -->"
        md.write_text(head + "\n" + render(r.get("segs", []), brief) + "\n", encoding="utf-8")
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8001")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--rerender", action="store_true", help="rebuild .md from .json sidecars")
    a = ap.parse_args()
    from bellwether.mcp import Store
    S = Store()
    if a.rerender:
        print(f"re-rendered {rerender(S.brief)} answers")
        return 0
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
