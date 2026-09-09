"""Every HTTP route and every MCP tool, called once, checked once.

Run against a live server:  python3 scripts/audit/api_audit.py [base-url]
Prints one line per check; exits non-zero if any FAIL.
"""
from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
rows: list[tuple[str, str, str]] = []


def req(method: str, path: str, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(BASE + path, data=data, method=method,
                               headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=timeout) as fh:
            return fh.status, fh.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except Exception as exc:  # noqa: BLE001
        return 0, str(exc).encode()


def check(name: str, ok: bool, note: str = "") -> None:
    rows.append(("PASS" if ok else "FAIL", name, note))
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  — {note}" if note else ""))


def jget(path: str):
    code, raw = req("GET", path)
    try:
        return code, json.loads(raw)
    except Exception:  # noqa: BLE001
        return code, None


# ---------------------------------------------------------------- static
code, raw = req("GET", "/")
check("GET / serves the chat shell", code == 200 and b"Bellwether" in raw)
check("GET / has a favicon", b'rel="icon"' in raw)
code, raw = req("GET", "/browse")
check("GET /browse serves the corpus view", code == 200 and b"<h1 id=\"ttl\"" in raw)
check("GET /browse carries the home link", b'id="homelink"' in raw)
code, _ = req("GET", "/nope.html")
check("missing file 404s", code == 404)
code, raw = req("GET", "/../src/bellwether/chat.py")
check("path traversal refused", code != 200 or b"SYSTEM" not in raw)

# ---------------------------------------------------------------- json api
code, d = jget("/agents")
check("GET /agents", code == 200 and isinstance(d, dict) and "claude" in d,
      f"claude installed={bool((d or {}).get('claude', {}).get('installed'))}")
code, chats = jget("/chats")
check("GET /chats", code == 200 and isinstance(chats, list), f"{len(chats or [])} chats")
pubs = [c for c in (chats or []) if c.get("pub")]
check("published conversations exist", len(pubs) > 0, f"{len(pubs)} published")

if chats:
    cid = chats[0]["id"]
    code, d = jget("/chats/" + cid)
    check("GET /chats/<id>", code == 200 and "turns" in (d or {}))
    check("chat doc hides the agent session id", "sid" not in (d or {}))
    seg_ok = all("gid" in s for t in (d or {}).get("turns", [])
                 for s in t.get("segs", []) if s.get("t") == "c")
    check("cite segments carry a gid", seg_ok)
code, d = jget("/chats/zzzz")
check("GET /chats/<bad id> is an error, not a 500 page", code in (400, 404))

# a scratch conversation for the write routes
code, d = jget("/chats")
before = len(d or [])
import pathlib
import secrets
import time
scratch = secrets.token_hex(6)
p = pathlib.Path(__file__).resolve().parents[2] / "data/chats" / f"{scratch}.json"
p.write_text(json.dumps({"id": scratch, "title": "audit scratch",
                         "ts": int(time.time()), "agent": "claude", "turns": []}))
code, d = req("POST", "/chats/" + scratch, {"title": "audit renamed"})
check("POST /chats/<id> renames", code == 200 and json.loads(d)["title"] == "audit renamed")
code, d = req("POST", "/chats/" + scratch, {"pub": True})
check("POST /chats/<id> publishes", code == 200 and json.loads(d)["pub"] is True)
code, d = req("POST", "/chats/" + scratch, {"pub": False})
check("publish is reversible", code == 200 and json.loads(d)["pub"] is False)
code, _ = req("DELETE", "/chats/" + scratch)
check("DELETE /chats/<id>", code == 200 and not p.exists())
code, _ = req("DELETE", "/chats/" + scratch)
check("DELETE of a gone chat 404s", code == 404)

code, d = jget("/paper/19662")
check("GET /paper/<gid>", code == 200 and (d or {}).get("title"),
      (d or {}).get("title", "")[:40])
check("paper card carries the three roles",
      all(k in (d or {}) for k in ("L", "K", "R")))
code, d = jget("/paper/99999999")
check("GET /paper/<absent gid> answers, not crashes",
      code in (200, 400) and (d or {}).get("error"))
code, d = jget("/paper/abc")
check("GET /paper/<non-numeric> is a 400", code == 400)

code, d = req("POST", "/chat/stop", {"run": "no-such-run"})
check("POST /chat/stop on an unknown run", code == 200 and json.loads(d) == {"ok": False})

code, raw = req("POST", "/meili/search", {"q": "diffusion", "limit": 2})
try:
    hits = json.loads(raw).get("hits", [])
except Exception:  # noqa: BLE001
    hits = []
check("POST /meili/search proxy", code == 200 and len(hits) > 0, f"{len(hits)} hits")
code, raw = req("POST", "/meili/similar",
                {"id": 19662, "embedder": "bge", "limit": 3})
try:
    sim = json.loads(raw).get("hits", [])
except Exception:  # noqa: BLE001
    sim = []
check("POST /meili/similar proxy", code == 200 and len(sim) > 0, f"{len(sim)} neighbours")
code, _ = req("POST", "/meili/nope", {})
check("no other Meilisearch path is exposed", code == 404)

# ---------------------------------------------------------------- mcp tools
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from bellwether import mcp  # noqa: E402

ARGS = {"search_papers": {"query": "kv cache compression", "limit": 3},
        "similar_papers": {"gid": 19662, "limit": 3},
        "get_paper": {"gid": 19662},
        "list_topics": {},
        "topic_papers": {"topic": "llm inference", "limit": 5},
        "field_cards": {"topic": "llm inference", "limit": 5},
        "field_trend": {"topic": "llm inference"},
        "gap_scan": {"topic": "llm inference"},
        "paper_text": {"gid": 19662},
        "get_citations": {"gid": 19662}}
check("every declared tool is audited",
      {t["name"] for t in mcp.TOOLS} == set(ARGS),
      str({t["name"] for t in mcp.TOOLS} ^ set(ARGS)))
for spec in mcp.TOOLS:
    name = spec["name"]
    try:
        out = spec["fn"](ARGS[name])
        bad = isinstance(out, dict) and out.get("error")
        check(f"mcp {name}", not bad, str(bad or "")[:60])
    except Exception as exc:  # noqa: BLE001
        check(f"mcp {name}", False, f"{type(exc).__name__}: {exc}"[:80])

# the tool the tree renders must carry a reader-facing caption
try:
    tree = mcp.t_gap_scan({"topic": "llm inference"})
    check("gap_scan carries a reader caption", bool(tree.get("caption")))
    check("gap_scan branches have evidence",
          all(b.get("rep", {}).get("sentence") for b in tree.get("branches", [])))
except Exception as exc:  # noqa: BLE001
    check("gap_scan shape", False, str(exc)[:60])

# verification, the invariant everything else rests on
try:
    from bellwether.verify import Verifier
    from bellwether.mcp import Store
    v = Verifier(Store())
    card = mcp.t_paper({"gid": 19662})
    vs = card.get("verified_sentences") or {}
    quote = (vs.get("limitation_of_prior_work") or vs.get("key_change") or "")[:80]
    check("verifier accepts a real quote", bool(quote) and v.role(19662, quote) is not None)
    check("verifier rejects an invented one",
          v.role(19662, "this sentence was never in any paper at all") is None)
except Exception as exc:  # noqa: BLE001
    check("verifier", False, str(exc)[:60])

# the interface language reaches the agent as a tie-breaker only
from bellwether.chat import SYSTEM, system_for  # noqa: E402
check("system prompt takes the reader's language",
      system_for("ko") != SYSTEM and system_for("en") != SYSTEM
      and system_for(None) == SYSTEM and "Korean" in system_for("ko"))

bad = [r for r in rows if r[0] == "FAIL"]
print(f"\n{len(rows) - len(bad)}/{len(rows)} passed")
sys.exit(1 if bad else 0)
