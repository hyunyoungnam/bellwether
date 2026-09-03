"""The conversational loop: browser chat -> the user's own coding agent.

This is the OpenResearch-shaped frame with our difference inside it: the
agent (Claude Code today; Codex later) runs headless under the USER'S OWN
subscription login — no API key — with only the wnai MCP tools, and every
factual sentence it writes must carry an anchor `⟦gid|exact quote⟧`. The
server verifies each anchor against the locally held corpus BEFORE the
browser shows it, so the reader sees, per citation, whether the quote really
exists in the paper. Prose can be wrong; a green check cannot.
"""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from .mcp import Store, _VENUE
from .verify import Verifier

ROOT = Path(__file__).resolve().parents[2]

SYSTEM = (
    "You are the research assistant of a local literature engine holding six "
    "conference editions in full: ICML 2025/26, NeurIPS 2024/25, ICLR "
    "2025/26 — 29,605 papers with verified sentences, embeddings, topics and "
    "citations. Evidence comes ONLY from the wnai MCP tools; never answer "
    "about papers from memory. PROTOCOL: after each factual claim, append an "
    "anchor of the exact form ⟦gid|quote⟧ where quote is copied verbatim "
    "from a tool response (a verified_sentences field or an abstract "
    "sentence, >=20 chars, never edited). Claims without an anchor will be "
    "shown to the reader as unbacked. If the corpus cannot answer, say so "
    "plainly. Answer in the user's language; keep quotes in their original "
    "language. Never rank papers by importance. Keep answers compact — a few "
    "sentences with anchors beat an essay."
)

_ANCHOR = re.compile(r"⟦\s*(\d+)\s*\|([^⟧]+)⟧")


def ask(message: str, sid: str | None = None, timeout: int = 300) -> dict:
    cmd = ["claude", "-p", message, "--output-format", "json",
           "--max-turns", "12",
           "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
           "--allowedTools", "mcp__wnai",
           "--append-system-prompt", SYSTEM]
    if sid:
        cmd += ["--resume", sid]
    out = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                         timeout=timeout, stdin=subprocess.DEVNULL)
    if out.returncode != 0:
        return {"error": (out.stderr or out.stdout or "agent failed")[-400:]}
    d = json.loads(out.stdout)
    return {"text": d.get("result") or "", "sid": d.get("session_id"),
            "turns": d.get("num_turns")}


def segment(text: str, store: Store, ver: Verifier) -> tuple[list, int, int]:
    """Split the agent's text into prose and verified/unverified cite chips."""
    segs: list[dict] = []
    checked = passed = 0
    pos = 0
    for m in _ANCHOR.finditer(text):
        if m.start() > pos:
            segs.append({"t": "p", "s": text[pos:m.start()]})
        gid, quote = int(m.group(1)), m.group(2).strip()
        ok = ver.check(gid, quote)
        checked += 1
        passed += ok
        r = store.rec(gid)
        key = store.where(gid)[0] if r else None
        venue, year = (key.rsplit("-", 1) if key else (None, None))
        segs.append({"t": "c", "gid": gid, "q": quote, "v": ok,
                     "title": r["title"] if r else f"gid {gid}",
                     "venue": _VENUE.get(venue, venue), "year": year})
        pos = m.end()
    if pos < len(text):
        segs.append({"t": "p", "s": text[pos:]})
    return segs, checked, passed


_STORE: Store | None = None
_VER: Verifier | None = None


def handle(body: dict) -> dict:
    """POST /chat entry point. {q, sid?} -> {segs, sid, verified}."""
    global _STORE, _VER
    if _STORE is None:
        _STORE = Store()
        _VER = Verifier(_STORE)
    q = (body.get("q") or "").strip()
    if not q:
        return {"error": "empty question"}
    r = ask(q, body.get("sid") or None)
    if "error" in r:
        return r
    segs, checked, passed = segment(r["text"], _STORE, _VER)
    return {"segs": segs, "sid": r["sid"],
            "verified": {"checked": checked, "passed": passed}}
