"""The same Claude CLI Bellwether runs on, with the platform taken away.

Two profiles, both the user's signed-in `claude` (subscription, no key), the
same default model and the same 12-turn limit as Bellwether's agent:

  web   Claude Code's built-in WebSearch/WebFetch only — "the frontier model
        with a browser": answers from weights + the open web
  orx   the OpenResearch skill (alphaXiv / OpenAlex retrieval via `orx
        discover`, `orx paper`) — a gateway harness on the same model

Neither sees Bellwether's MCP tools or verifier. Answers land in
answers/claude-<profile>/<qid>.md with the raw JSON as a sidecar.

    PYTHONPATH=src python scripts/eval/run_claude.py --profile web [--only T5-01 ...]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]

PROFILES = {
    "web": {"allowed": ["WebSearch", "WebFetch"],
            "system": "You are a research assistant answering questions about "
                      "recent ML conference papers (ICML, NeurIPS, ICLR 2024-2026). "
                      "Use web search when it helps. Name papers by exact title. "
                      "When you quote a paper, quote it verbatim in quotation marks. "
                      "If something cannot be determined, say so plainly."},
    "orx": {"allowed": ["Bash(orx *)", "WebFetch"],
            "system": "You are a research assistant answering questions about "
                      "recent ML conference papers (ICML, NeurIPS, ICLR 2024-2026). "
                      "Use the OpenResearch `orx` CLI for paper retrieval "
                      "(`orx discover keyword|embedding|openalex ...`, `orx paper ...`). "
                      "Name papers by exact title. When you quote a paper, quote it "
                      "verbatim in quotation marks. If something cannot be "
                      "determined, say so plainly."},
}


# Run OUTSIDE the repository: inside it, Claude Code loads CLAUDE.md and the
# project memory and the "baseline" starts answering from Bellwether's own
# notes (measured: "from this project's corpora …"). An empty directory
# under $HOME carries no project context.
NEUTRAL = pathlib.Path.home() / ".eval-baseline"


def ask(q: str, profile: dict, timeout: int) -> dict:
    NEUTRAL.mkdir(exist_ok=True)
    cmd = ["claude", "-p", q, "--output-format", "json", "--max-turns", "12",
           "--no-session-persistence", "--strict-mcp-config",
           "--allowedTools", *profile["allowed"],
           "--append-system-prompt", profile["system"]]
    out = subprocess.run(cmd, cwd=str(NEUTRAL), capture_output=True,
                         text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout or "agent failed")[-400:]}
    try:
        d = json.loads(out.stdout)
    except ValueError:
        return {"error": out.stdout[-400:]}
    return {"text": d.get("result") or "", "turns": d.get("num_turns"),
            "raw": d}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", choices=list(PROFILES), required=True)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--timeout", type=int, default=900)
    a = ap.parse_args()
    prof = PROFILES[a.profile]
    out_dir = HERE / "answers" / f"claude-{a.profile}"
    out_dir.mkdir(parents=True, exist_ok=True)
    qs = json.load((HERE / "questions_resolved.json").open(encoding="utf-8"))["questions"]
    if a.only:
        qs = [q for q in qs if q["id"] in set(a.only)]
    for q in qs:
        f = out_dir / f"{q['id']}.md"
        if f.exists():
            continue
        t0 = time.time()
        r = ask(q["q"], prof, a.timeout)
        if "error" in r:
            print(f"{q['id']}: ERROR {r['error'][:120]}")
            continue
        model = (r["raw"].get("modelUsage") or {})
        f.write_text(f"<!-- system: claude-{a.profile} | tools: {','.join(prof['allowed'])} | "
                     f"models: {','.join(model) if isinstance(model, dict) else model} | "
                     f"{time.strftime('%Y-%m-%d %H:%M')} | {time.time() - t0:.0f}s | "
                     f"turns {r['turns']} -->\n{r['text']}\n", encoding="utf-8")
        (out_dir / f"{q['id']}.json").write_text(json.dumps(r["raw"], ensure_ascii=False),
                                                 encoding="utf-8")
        print(f"{q['id']}: {len(r['text'])} chars, {time.time() - t0:.0f}s, turns {r['turns']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
