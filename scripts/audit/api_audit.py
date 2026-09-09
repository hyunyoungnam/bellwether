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

code, raw = req("POST", "/export", {"gids": [19662, 100], "fmt": "bib"})
bib = raw.decode("utf-8", "replace")
check("POST /export bibtex", code == 200 and bib.count("@inproceedings") == 2,
      f"{bib.count('@inproceedings')} entries")
check("bibtex carries the full author list", " and " in bib and "author = {" in bib)
code, raw = req("POST", "/export", {"gids": [19662], "fmt": "csv"})
rowsv = raw.decode("utf-8", "replace").strip().split("\n")
check("POST /export csv", code == 200 and len(rowsv) == 2, f"{len(rowsv)} lines")
check("csv carries the verified sentences",
      "limitation" in rowsv[0] and "result_claim" in rowsv[0])
code, raw = req("POST", "/export", {"gids": [], "fmt": "bib"})
check("export of nothing is empty, not an error", code == 200 and not raw.strip())

if pubs:
    pid = pubs[0]["id"]
    code, raw = req("GET", f"/chats/{pid}/export?fmt=md")
    md = raw.decode("utf-8", "replace")
    check("GET /chats/<id>/export markdown", code == 200 and md.startswith("#"))
    check("the export tables every quote with its verdict",
          "| ✓ |" in md or "| ✗ |" in md)
    code, raw = req("GET", f"/chats/{pid}/export?fmt=json")
    try:
        bundle = json.loads(raw)
    except Exception:  # noqa: BLE001
        bundle = {}
    check("GET /chats/<id>/export json", code == 200 and bundle.get("turns"))
    check("the bundle stamps the corpus it was answered against",
          bool((bundle.get("corpus") or {}).get("papers_with_gid")),
          str((bundle.get("corpus") or {}).get("papers_with_gid")))
    check("and carries the verification totals",
          (bundle.get("verification") or {}).get("checked", 0) > 0)
code, _ = req("GET", "/chats/zzzz/export?fmt=md")
check("export of a missing chat 404s", code in (400, 404))

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

ARGS = {"verify_quote": {"gid": 19662, "quote": "the KV cache"},
        "search_papers": {"query": "kv cache compression", "limit": 3},
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

# ---- figures: a number is checked by running its tool again
from bellwether import figures        # noqa: E402
from bellwether.mcp import Store      # noqa: E402
from bellwether.chat import segment   # noqa: E402
from bellwether.verify import Verifier as _V   # noqa: E402

_st = Store()
_cache: dict = {}
FIG_CASES = [
    ("gap_scan", "healthcare", "31 name it, 3 attack it", "ok"),
    ("gap_scan", "healthcare", "51 name it, 3 attack it", "no"),
    ("gap_scan", "healthcare", "by_year 2025:13, 2026:11", "ok"),
    ("field_trend", "code generation", "4.4->6.2 per 1k, z=1.15", "ok"),
    ("field_trend", "code generation", "z=2.9", "no"),
    ("get_citations", "19662", "28 papers cite it", "ok"),
    ("search_papers", "kv cache", "315 papers", "na"),
    ("gap_scan", "no such topic at all", "12 papers", "na"),
]
for tool, arg, claim, want in FIG_CASES:
    got = figures.check(tool, arg, claim, _cache)["state"]
    check(f"figure {want}: {tool}({arg}) “{claim[:28]}”", got == want, f"got {got}")
check("search figures are never marked verified",
      figures.RECOMPUTABLE.get("search_papers") is None)
check("paper ids are not figures",
      2 > len([x for x in figures.pool({"attackers": [11, 12], "named_by": 7})
               if x not in (2.0, 7.0)]),
      str(figures.pool({"attackers": [11, 12], "named_by": 7})))

# the whole path: an answer with both anchor kinds, through segment()
_txt = ("A ⟦gap_scan:healthcare|31 name it, 3 attack it⟧ and "
        "B ⟦19662|The size of the KV cache grows linearly with sequence length⟧ "
        "and C ⟦gap_scan:healthcare|99 name it⟧.")
_segs, _v = segment(_txt, _st, _V(_st))
check("segment() reads both anchor kinds",
      [x["t"] for x in _segs].count("n") == 2 and
      [x["t"] for x in _segs].count("c") == 1,
      str([x["t"] for x in _segs]))
check("and counts quotes and figures apart",
      _v == {"checked": 1, "passed": 1, "fchecked": 2, "fpassed": 1}, str(_v))
check("a wrong figure names the number that failed",
      [x for x in _segs if x["t"] == "n" and x["v"] == "no"][0]["missing"] == ["99"])

# ---- the guarantee, offered as a tool
_vq_card = mcp.t_paper({"gid": 19662})["verified_sentences"]
_vq = _vq_card.get("limitation_of_prior_work") or _vq_card.get("key_change")
check("verify_quote accepts the paper's own sentence",
      mcp.t_verify_quote({"gid": 19662, "quote": _vq})["verified"])
check("verify_quote names the field that matched",
      mcp.t_verify_quote({"gid": 19662, "quote": _vq})["matched_field"] is not None)
check("verify_quote rejects an invented one",
      not mcp.t_verify_quote({"gid": 19662,
                              "quote": "this sentence is in no paper at all"})["verified"])
check("and rejects a quote cut mid-word (the containment is space-bounded)",
      not mcp.t_verify_quote({"gid": 19662, "quote": _vq[:70]})["verified"]
      if len(_vq) > 75 and not _vq[70].isspace() else True)

# ---- the automatic pass: every printed number, anchor or not
_trail = [{"name": "field_trend", "arg": "code generation"},
          {"name": "gap_scan", "arg": "healthcare"}]
_auto = ("ICLR 4.4->6.2 per 1k, z=1.15, under the |z| >= 2.576 (99%) bar. "
         "The corpus went 3,324 -> 6,592. Benchmarks: 31 against 3, and "
         "705 papers of which 686 state a limitation. Invented: 4,242.")
_segs2, _v2 = segment(_auto, _st, _V(_st), _trail)
_marked = [x["s"] for x in _segs2 if x.get("auto")]
check("the auto pass checks numbers with no anchor at all",
      _v2["fchecked"] >= 12, f"{_v2['fchecked']} figures checked")
check("a thousands separator is one number, not two",
      "324" not in _marked and "592" not in _marked, str(_marked))
check("the rule's own threshold verifies (the tool states it)",
      "2.576" not in _marked and "99" not in _marked, str(_marked))
check("only the invented figure is marked", _marked == ["4,242"], str(_marked))
check("and it is marked as the server's notice, not as a failed claim",
      all(x.get("auto") for x in _segs2 if x.get("t") == "n"))
check("a turn with no recomputable tool marks nothing",
      not [x for x in segment(_auto, _st, _V(_st), [])[0] if x.get("t") == "n"])
check("derived arithmetic counts as explained",
      "derived" == figures.scan("4.4", [31.0, 705.0], figures.derived_set([31.0, 705.0]),
                                set())[0][3])

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
