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
import secrets
import subprocess
import time
from pathlib import Path

from .mcp import Store, _VENUE
from .verify import Verifier

ROOT = Path(__file__).resolve().parents[2]
CHAT_DIR = ROOT / "data" / "chats"

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
        role = ver.role(gid, quote)
        ok = role is not None
        checked += 1
        passed += ok
        r = store.rec(gid)
        key = store.where(gid)[0] if r else None
        venue, year = (key.rsplit("-", 1) if key else (None, None))
        segs.append({"t": "c", "gid": gid, "q": quote, "v": ok, "role": role,
                     "title": r["title"] if r else f"gid {gid}",
                     "venue": _VENUE.get(venue, venue), "year": year})
        pos = m.end()
    if pos < len(text):
        segs.append({"t": "p", "s": text[pos:]})
    return segs, checked, passed


_STORE: Store | None = None
_VER: Verifier | None = None


def _env():
    global _STORE, _VER
    if _STORE is None:
        _STORE = Store()
        _VER = Verifier(_STORE)
    return _STORE, _VER


def handle(body: dict) -> dict:
    """POST /chat entry point. {q, sid?} -> {segs, sid, verified}."""
    store, ver = _env()
    q = (body.get("q") or "").strip()
    if not q:
        return {"error": "empty question"}
    r = ask(q, body.get("sid") or None)
    if "error" in r:
        return r
    segs, checked, passed = segment(r["text"], store, ver)
    return {"segs": segs, "sid": r["sid"],
            "verified": {"checked": checked, "passed": passed}}


def card(gid: int) -> dict:
    """GET /paper/<gid>: the highlight card's material, for inline display.

    The same three-colour passage the browse cards carry — the paper's own
    sentences (pink: why it was needed / yellow: what is new / blue: what it
    achieved), never rewritten here or anywhere."""
    store, _ = _env()
    r = store.rec(gid)
    if r is None:
        return {"error": f"no record for gid {gid}"}
    key, _eid = store.where(gid)
    venue, year = key.rsplit("-", 1)
    sp = store.span_entry(gid) or [None] * 8
    n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
    return {"gid": gid, "title": r["title"],
            "venue": _VENUE.get(venue, venue), "year": year,
            "authors": (r.get("authors") or [])[:3],
            "n_authors": r.get("n_authors"),
            "L": L or None,
            "K": K or (n[0] if n else None),
            "R": R or None}


# ------------------------------------------------------------- conversations
# A conversation is OURS, keyed by a stable chat id; the agent CLI issues a
# NEW session id on every resume, so the latest one is stored inside the doc
# and never shown to the client.

def _doc_path(cid: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{12}", cid):
        raise ValueError("bad chat id")
    return CHAT_DIR / f"{cid}.json"


def list_chats() -> list[dict]:
    out = []
    if CHAT_DIR.exists():
        for f in CHAT_DIR.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                out.append({"id": d["id"], "title": d["title"], "ts": d["ts"]})
            except (OSError, KeyError, json.JSONDecodeError):
                continue
    out.sort(key=lambda e: e["ts"], reverse=True)
    return out


def get_chat(cid: str) -> dict:
    d = json.loads(_doc_path(cid).read_text())
    d.pop("sid", None)                      # internal
    return d


def _summ(tool_input: dict) -> str:
    for k in ("query", "topic", "gid"):
        if k in (tool_input or {}):
            return str(tool_input[k])[:60]
    return ""


def stream(body: dict, emit) -> None:
    """POST /chat/stream: emit({'t':'tool'|'done'|'error', ...}) as SSE.

    Tool calls surface live — the reader watches the agent walk the corpus —
    and the final text arrives verified, exactly like handle()."""
    store, ver = _env()
    q = (body.get("q") or "").strip()
    if not q:
        emit({"t": "error", "error": "empty question"})
        return
    cid = body.get("chat") or secrets.token_hex(6)
    doc = None
    if _doc_path(cid).exists():
        doc = json.loads(_doc_path(cid).read_text())
    cmd = ["claude", "-p", q, "--output-format", "stream-json", "--verbose",
           "--max-turns", "12",
           "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
           "--allowedTools", "mcp__wnai",
           "--append-system-prompt", SYSTEM]
    if doc and doc.get("sid"):
        cmd += ["--resume", doc["sid"]]
    result = None
    try:
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              stdin=subprocess.DEVNULL) as p:
            for line in p.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if ev.get("type") == "assistant":
                    for c in (ev.get("message") or {}).get("content", []):
                        # only corpus tools make the visible trail — harness
                        # plumbing (ToolSearch etc.) is noise to the reader
                        if c.get("type") == "tool_use" \
                                and c["name"].startswith("mcp__wnai__"):
                            emit({"t": "tool",
                                  "name": c["name"].split("__")[-1],
                                  "arg": _summ(c.get("input"))})
                elif ev.get("type") == "result":
                    result = ev
    except Exception as exc:  # noqa: BLE001
        emit({"t": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return
    if not result or result.get("is_error"):
        emit({"t": "error", "error": (result or {}).get("result", "agent failed")[:300]})
        return
    segs, checked, passed = segment(result.get("result") or "", store, ver)
    turn = {"q": q, "segs": segs,
            "verified": {"checked": checked, "passed": passed}}
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if doc is None:
        doc = {"id": cid, "title": q[:80], "ts": int(time.time()), "turns": []}
    doc["sid"] = result.get("session_id")
    doc["ts"] = int(time.time())
    doc["turns"].append(turn)
    _doc_path(cid).write_text(json.dumps(doc, ensure_ascii=False))
    emit({"t": "done", "chat": cid, "segs": segs,
          "verified": turn["verified"]})
