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
    "about papers from memory. METHOD for field-level questions (what is "
    "rising, what is new this year, where are the gaps): read the field in "
    "bulk with field_cards, get computed shares from field_trend, then derive "
    "your conclusion FROM the cards — recurring key_change ideas are a "
    "technique rising, recurring limitations nobody's key_change answers are "
    "a gap. Distribution figures must be restated exactly as the tools "
    "computed them (per-1k shares, z), never invented. PROTOCOL: after each "
    "claim about a specific paper, append an anchor of the exact form "
    "⟦gid|quote⟧ where quote is copied verbatim from a tool response (a card "
    "field or abstract sentence, >=20 chars, never edited). Claims without "
    "an anchor will be shown to the reader as unbacked. If the corpus cannot "
    "answer, say so plainly. Answer in the user's language; keep quotes in "
    "their original language. Never rank papers by importance. Keep answers "
    "compact — a few sentences with anchors beat an essay."
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


# ------------------------------------------------------------- agents
# Like orx: the machine's installed, signed-in agent CLIs are what "connect".

def _claude_account() -> str | None:
    try:
        d = json.loads((Path.home() / ".claude.json").read_text())
        return (d.get("oauthAccount") or {}).get("emailAddress")
    except Exception:  # noqa: BLE001
        return None


def _codex_account() -> str | None:
    """The email inside the id_token JWT — best-effort, display only."""
    try:
        import base64
        d = json.loads((Path.home() / ".codex" / "auth.json").read_text())
        tok = (d.get("tokens") or {}).get("id_token") or ""
        pay = tok.split(".")[1]
        pay += "=" * (-len(pay) % 4)
        return json.loads(base64.urlsafe_b64decode(pay)).get("email")
    except Exception:  # noqa: BLE001
        return None


def agents() -> dict:
    import shutil
    out = {}
    for name, binname, acct in (("claude", "claude", _claude_account),
                                ("codex", "codex", _codex_account)):
        path = shutil.which(binname) or (
            str(Path.home() / ".local/bin" / binname)
            if (Path.home() / ".local/bin" / binname).exists() else None)
        out[name] = {"installed": bool(path),
                     "account": acct() if path else None}
    return out


_CODEX_MCP = """
[mcp_servers.wnai]
command = "python3"
args = ["-m", "wnai", "mcp"]
env = { PYTHONPATH = "%s" }
"""


def _ensure_codex_mcp() -> None:
    cfg = Path.home() / ".codex" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    text = cfg.read_text() if cfg.exists() else ""
    if "mcp_servers.wnai" not in text:
        cfg.write_text(text + _CODEX_MCP % (ROOT / "src"))


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
    topics = []
    try:
        for t in store.topics(key)["topics"]:
            if _eid in t.get("explicit", ()) or _eid in t.get("via_child", ()):
                topics.append(t["label"])
    except FileNotFoundError:
        pass
    return {"gid": gid, "title": r["title"],
            "venue": _VENUE.get(venue, venue), "year": year,
            "authors": (r.get("authors") or [])[:3],
            "n_authors": r.get("n_authors"),
            "area": r.get("area"), "subarea": r.get("subarea"),
            "L": L or None,
            "K": K or (n[0] if n else None),
            "R": R or None,
            "novelty": n, "topics": topics,
            "abstract": r.get("abstract")}


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
    agent = body.get("agent") or "claude"
    doc = None
    if _doc_path(cid).exists():
        doc = json.loads(_doc_path(cid).read_text())
    sid = doc.get("sid") if doc else None
    # a conversation stays on the agent it started with — sessions don't port
    if doc and doc.get("agent") and doc["agent"] != agent:
        agent = doc["agent"]

    if agent == "codex":
        _ensure_codex_mcp()
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if sid:
            cmd = ["codex", "exec", "resume", sid, "--json",
                   "--skip-git-repo-check"]
        cmd.append(SYSTEM + "\n\nUSER QUESTION:\n" + q)
    else:
        cmd = ["claude", "-p", q, "--output-format", "stream-json", "--verbose",
               "--max-turns", "12",
               "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
               "--allowedTools", "mcp__wnai",
               "--append-system-prompt", SYSTEM]
        if sid:
            cmd += ["--resume", sid]

    result_text, new_sid, failed = None, None, None
    try:
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              stdin=subprocess.DEVNULL) as p:
            for line in p.stdout:
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if agent == "codex":
                    # measured event shape (codex 0.153): thread.started
                    # carries thread_id; item.completed carries typed items.
                    # An 'error' item is a WARNING (the run continues) — only
                    # turn.failed is fatal.
                    ty = ev.get("type") or ""
                    if ty == "thread.started":
                        new_sid = ev.get("thread_id") or new_sid
                    elif ty == "turn.failed":
                        failed = str(ev.get("error") or "codex turn failed")[:300]
                    elif ty in ("item.started", "item.completed"):
                        it = ev.get("item") or {}
                        ity = it.get("type") or ""
                        if "mcp" in ity and ty == "item.started":
                            emit({"t": "tool",
                                  "name": (it.get("tool") or it.get("name")
                                           or "tool").split("__")[-1],
                                  "arg": _summ(it.get("arguments")
                                               if isinstance(it.get("arguments"), dict)
                                               else {})})
                        elif ity == "agent_message" and ty == "item.completed":
                            result_text = it.get("text") or result_text
                else:
                    if ev.get("type") == "assistant":
                        for c in (ev.get("message") or {}).get("content", []):
                            # only corpus tools make the visible trail —
                            # harness plumbing is noise to the reader
                            if c.get("type") == "tool_use" \
                                    and c["name"].startswith("mcp__wnai__"):
                                emit({"t": "tool",
                                      "name": c["name"].split("__")[-1],
                                      "arg": _summ(c.get("input"))})
                    elif ev.get("type") == "result":
                        result_text = ev.get("result") or ""
                        new_sid = ev.get("session_id")
                        if ev.get("is_error"):
                            failed = (result_text or "agent failed")[:300]
    except Exception as exc:  # noqa: BLE001
        emit({"t": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return
    if failed or result_text is None:
        emit({"t": "error", "error": failed or "the agent returned nothing"})
        return
    segs, checked, passed = segment(result_text, store, ver)
    turn = {"q": q, "segs": segs,
            "verified": {"checked": checked, "passed": passed}}
    CHAT_DIR.mkdir(parents=True, exist_ok=True)
    if doc is None:
        doc = {"id": cid, "title": q[:80], "ts": int(time.time()),
               "agent": agent, "turns": []}
    doc["sid"] = new_sid or sid
    doc["ts"] = int(time.time())
    doc["turns"].append(turn)
    _doc_path(cid).write_text(json.dumps(doc, ensure_ascii=False))
    emit({"t": "done", "chat": cid, "segs": segs,
          "verified": turn["verified"]})
