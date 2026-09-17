"""The conversational loop: browser chat -> the user's own coding agent.

This is the OpenResearch-shaped frame with our difference inside it: the
agent (Claude Code today; Codex later) runs headless under the USER'S OWN
subscription login — no API key — with only the bellwether MCP tools, and every
factual sentence it writes must carry an anchor `⟦gid|exact quote⟧`. The
server verifies each anchor against the locally held corpus BEFORE the
browser shows it, so the reader sees, per citation, whether the quote really
exists in the paper. Prose can be wrong; a green check cannot.
"""
from __future__ import annotations

import json
import os
import re
import secrets
import signal
import subprocess
import time
from pathlib import Path

from . import figures
from .mcp import Store, _VENUE
from .verify import Verifier

ROOT = Path(__file__).resolve().parents[2]
CHAT_DIR = ROOT / "data" / "chats"

SYSTEM = (
    "You are the research assistant of a local literature engine holding six "
    "conference editions in full: ICML 2025/26, NeurIPS 2024/25, ICLR "
    "2025/26 — 29,605 papers with verified sentences, embeddings, topics and "
    "citations. Evidence comes ONLY from the bellwether MCP tools; never answer "
    "about papers from memory. METHOD for field-level questions (what is "
    "rising, what is new this year, where are the gaps): read the field in "
    "bulk with field_cards, get computed shares from field_trend, then derive "
    "your conclusion FROM the cards — recurring key_change ideas are a "
    "technique rising, recurring limitations nobody's key_change answers are "
    "a gap. Distribution figures must be restated exactly as the tools "
    "computed them (per-1k shares, z), never invented. For GAP / blue-ocean "
    "questions call gap_scan(topic): the reader sees its derivation tree "
    "rendered directly, so do NOT restate every branch — interpret it: which "
    "gap candidates look real, which are lexical artifacts, anchored to "
    "specific papers. For DEPTH questions "
    "about one paper (how it works, its numbers), read its actual text with "
    "paper_text: list sections, then read the relevant one. PROTOCOL: after each "
    "claim about a specific paper, append an anchor of the exact form "
    "⟦gid|quote⟧ where quote is copied verbatim from a tool response (a card "
    "field or abstract sentence, >=20 chars, never edited). Claims without "
    "an anchor will be shown to the reader as unbacked. FIGURES: every number "
    "you print is checked against what the tools returned this turn — restate "
    "them exactly as computed and never round a share differently. When a "
    "figure matters enough to bind to one call, anchor it as "
    "⟦tool:argument|the figures⟧, e.g. ⟦gap_scan:healthcare|31 name it, 3 "
    "attack it⟧, using exactly the argument you called. If the corpus cannot "
    "answer, say so plainly. For WHERE-TO-GET questions (a paper's code, "
    "released data or weights; where a benchmark lives) call paper_resources "
    "and benchmark_info — their links are either printed in the paper or our "
    "API-checked mapping, and stars/downloads are facts to report, never a "
    "reason to rank. ALWAYS write your answer in English, whatever "
    "language the question is in — the reader's Korean is rendered from this "
    "English by a separate step, so one text is written, verified and stored. "
    "Never rank papers by importance. Keep answers "
    "compact — a few sentences with anchors beat an essay."
)

# The interface language no longer changes what the agent writes — it always
# writes English, and Korean is rendered from it. Kept as a seam because the
# callers still pass a language and a second renderer may want it.
def system_for(lang: str | None) -> str:
    return SYSTEM

# Two kinds of anchor, one scan so the segments come out in reading order:
#   ⟦gid|quote⟧              a sentence, matched against that paper
#   ⟦tool:arg|figures⟧       a number, recomputed by running the tool again
_ANCHOR = re.compile(
    r"⟦\s*(?:(\d+)\s*\|([^⟧]+)|([a-z_]{3,20})\s*:\s*([^|⟧]{1,90})\|([^⟧]{1,200}))⟧")


def ask(message: str, sid: str | None = None, timeout: int = 300) -> dict:
    cmd = ["claude", "-p", message, "--output-format", "json",
           "--max-turns", "12",
           "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
           "--allowedTools", "mcp__bellwether",
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


def segment(text: str, store: Store, ver: Verifier,
            trail: list | None = None) -> tuple[list, dict]:
    """Prose, checked quotes, recomputed figures — and every OTHER number too.

    The anchors are the agent's declaration; the automatic pass is ours. Both
    end in the same place: a number is either in what the tools returned this
    turn, derivable from two of those, or in neither."""
    segs: list[dict] = []
    v = {"checked": 0, "passed": 0, "fchecked": 0, "fpassed": 0}
    fcache: dict = {}
    pos = 0
    for m in _ANCHOR.finditer(text):
        if m.start() > pos:
            segs.append({"t": "p", "s": text[pos:m.start()]})
        pos = m.end()
        if m.group(1):                                   # ⟦gid|quote⟧
            gid, quote = int(m.group(1)), m.group(2).strip()
            role = ver.role(gid, quote)
            ok = role is not None
            v["checked"] += 1
            v["passed"] += ok
            r = store.rec(gid)
            key = store.where(gid)[0] if r else None
            venue, year = (key.rsplit("-", 1) if key else (None, None))
            segs.append({"t": "c", "gid": gid, "q": quote, "v": ok, "role": role,
                         "title": r["title"] if r else f"gid {gid}",
                         "venue": _VENUE.get(venue, venue), "year": year})
            continue
        tool, arg, claim = (m.group(3), m.group(4).strip(), m.group(5).strip())
        fig = figures.check(tool, arg, claim, fcache)
        if fig["state"] != "na":                         # 'na' claims nothing
            v["fchecked"] += 1
            v["fpassed"] += fig["state"] == "ok"
        segs.append({"t": "n", "s": claim, "v": fig["state"], "tool": tool,
                     "arg": arg, "missing": fig.get("missing") or [],
                     "why": fig.get("why")})
    if pos < len(text):
        segs.append({"t": "p", "s": text[pos:]})
    return _auto_figures(segs, v, trail, fcache), v


def _auto_figures(segs: list, v: dict, trail, fcache: dict) -> list:
    """Check every printed number against this turn's own tool results.

    Numbers that hold up stay plain prose and are only counted; a number that
    is in no tool result and is not arithmetic over two of them is wrapped so
    the reader can see WHICH one it is. The mark states a fact — this number is
    not in what the tools returned — and accuses the sentence of nothing."""
    vals, used = figures.turn_pool(trail, fcache)
    if not vals:
        return segs
    deriv = figures.derived_set(vals)
    skip = {float(s["gid"]) for s in segs if s.get("t") == "c"}
    out: list = []
    for seg in segs:
        if seg.get("t") != "p":
            out.append(seg)
            continue
        text, pos = seg["s"], 0
        for a, b, raw, verdict in figures.scan(text, vals, deriv, skip):
            v["fchecked"] += 1
            v["fpassed"] += verdict != "no"
            if verdict != "no":
                continue
            if a > pos:
                out.append({"t": "p", "s": text[pos:a]})
            out.append({"t": "n", "s": raw, "v": "no", "auto": True,
                        "tool": ", ".join(used[:2]) or "this turn's tools",
                        "arg": "", "missing": [raw]})
            pos = b
        out.append({"t": "p", "s": text[pos:]})
    return out


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
    segs, v = segment(r["text"], store, ver)
    return {"segs": segs, "sid": r["sid"], "verified": v}


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
[mcp_servers.bellwether]
command = "python3"
args = ["-m", "wnai", "mcp"]
env = { PYTHONPATH = "%s" }
"""


def _ensure_codex_mcp() -> None:
    cfg = Path.home() / ".codex" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    text = cfg.read_text() if cfg.exists() else ""
    if "mcp_servers.bellwether" not in text:
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


def delete_chat(cid: str) -> dict:
    _doc_path(cid).unlink()
    return {"ok": True}


def update_chat(cid: str, body: dict) -> dict:
    """Rename and/or publish. A published conversation is the brief's heir:
    the exploration that earned a place on the front screen."""
    p2 = _doc_path(cid)
    d = json.loads(p2.read_text())
    if body.get("title"):
        d["title"] = body["title"].strip()[:80] or d["title"]
    if "pub" in body:
        d["pub"] = bool(body["pub"])
    p2.write_text(json.dumps(d, ensure_ascii=False))
    return {"ok": True, "title": d["title"], "pub": d.get("pub", False)}


def list_chats() -> list[dict]:
    out = []
    if CHAT_DIR.exists():
        for f in CHAT_DIR.glob("*.json"):
            try:
                d = json.loads(f.read_text())
                out.append({"id": d["id"], "title": d["title"], "ts": d["ts"],
                            "pub": d.get("pub", False)})
            except (OSError, KeyError, json.JSONDecodeError):
                continue
    out.sort(key=lambda e: e["ts"], reverse=True)
    return out


def translate_chat(cid: str, lang: str = "ko") -> dict:
    """Render this conversation's prose into `lang`, once, and keep it.

    Quotes and figures are copied untouched — the verdicts on this page were
    reached against the English and cannot move. A segment whose figures do
    not survive translation stays English."""
    from . import translate
    if lang != "ko":
        return {"error": f"no renderer for {lang}"}
    if not translate.up():
        return {"error": "no translation engine", "engine": translate.ENDPOINT}
    p2 = _doc_path(cid)
    d = json.loads(p2.read_text())
    done = 0
    for t in d.get("turns", []):
        if t.get("ko"):
            continue
        ko = translate.turn(t.get("segs", []))
        if ko:
            t["ko"] = ko
            done += 1
    if done:
        p2.write_text(json.dumps(d, ensure_ascii=False))
    d.pop("sid", None)
    return d


def get_chat(cid: str) -> dict:
    d = json.loads(_doc_path(cid).read_text())
    d.pop("sid", None)                      # internal
    return d


# A generation the reader can call off. The agent CLI spawns children (the MCP
# server among them), so the run gets its own process group and the whole group
# is signalled — terminating the parent alone leaves the tools running.
_RUNS: dict[str, dict] = {}


def stop_run(run: str) -> dict:
    e = _RUNS.get(run)
    if e is None:
        return {"ok": False}
    e["stopped"] = True
    p = e.get("proc")
    if p is not None and p.poll() is None:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            p.terminate()
    return {"ok": True}


def _corpus_id(store) -> dict:
    u = store.union
    return {"papers": len(u.get("keys") or []),
            "corpora": list(u.get("corpora") or [])}


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

    system = system_for(body.get("lang"))
    if agent == "codex":
        _ensure_codex_mcp()
        cmd = ["codex", "exec", "--json", "--skip-git-repo-check"]
        if sid:
            cmd = ["codex", "exec", "resume", sid, "--json",
                   "--skip-git-repo-check"]
        cmd.append(system + "\n\nUSER QUESTION:\n" + q)
    else:
        cmd = ["claude", "-p", q, "--output-format", "stream-json", "--verbose",
               "--include-partial-messages",
               "--max-turns", "12",
               "--mcp-config", str(ROOT / ".mcp.json"), "--strict-mcp-config",
               "--allowedTools", "mcp__bellwether",
               "--append-system-prompt", system]
        if sid:
            cmd += ["--resume", sid]

    result_text, new_sid, failed = None, None, None
    trees: list[dict] = []
    trail: list[dict] = []
    run = str(body.get("run") or secrets.token_hex(6))
    entry = _RUNS[run] = {"stopped": False, "proc": None}

    def _tool_event(name: str, arg_map: dict) -> None:
        emit({"t": "tool", "name": name, "arg": _summ(arg_map)})
        # the trail is evidence about HOW the answer was reached, so it is
        # stored with the turn — until now it lived only in the open tab and
        # vanished when the conversation was reopened
        trail.append({"name": name, "arg": _summ(arg_map)})
        # the derivation tree renders as a component, not prose: recompute the
        # same deterministic scan server-side and hand it to the page directly
        if name == "gap_scan" and (arg_map or {}).get("topic"):
            try:
                from .mcp import t_gap_scan
                tree = t_gap_scan({"topic": arg_map["topic"]})
                if "error" not in tree:
                    trees.append(tree)
                    emit({"t": "tree", "tree": tree})
            except Exception:  # noqa: BLE001 — the prose answer still lands
                pass

    try:
        with subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True,
                              stdin=subprocess.DEVNULL,
                              start_new_session=True) as p:
            entry["proc"] = p
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
                            _tool_event((it.get("tool") or it.get("name")
                                         or "tool").split("__")[-1],
                                        it.get("arguments")
                                        if isinstance(it.get("arguments"), dict)
                                        else {})
                        elif ity == "agent_message" and ty == "item.completed":
                            result_text = it.get("text") or result_text
                else:
                    if ev.get("type") == "stream_event":
                        e2 = ev.get("event") or {}
                        if e2.get("type") == "content_block_delta":
                            d2 = e2.get("delta") or {}
                            if d2.get("type") == "text_delta" and d2.get("text"):
                                emit({"t": "d", "s": d2["text"]})
                    elif ev.get("type") == "assistant":
                        for c in (ev.get("message") or {}).get("content", []):
                            # only corpus tools make the visible trail —
                            # harness plumbing is noise to the reader
                            if c.get("type") == "tool_use" \
                                    and c["name"].startswith("mcp__bellwether__"):
                                _tool_event(c["name"].split("__")[-1],
                                            c.get("input") or {})
                    elif ev.get("type") == "result":
                        result_text = ev.get("result") or ""
                        new_sid = ev.get("session_id")
                        if ev.get("is_error"):
                            failed = (result_text or "agent failed")[:300]
    except Exception as exc:  # noqa: BLE001
        emit({"t": "error", "error": f"{type(exc).__name__}: {exc}"[:300]})
        return
    finally:
        _RUNS.pop(run, None)
    if entry["stopped"]:
        # a half-finished answer has unverified anchors — it is not saved and
        # not shown as the paper's words
        emit({"t": "stopped"})
        return
    if failed or result_text is None:
        emit({"t": "error", "error": failed or "the agent returned nothing"})
        return
    segs, v = segment(result_text, store, ver, trail)
    turn = {"q": q, "segs": segs, "verified": v,
            # what it was answered against, so the same question can be put to
            # the same shelf later — the corpus is fixed, that is the point
            "on": _corpus_id(store)}
    if trail:
        turn["trail"] = trail
    if trees:
        turn["trees"] = trees
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
