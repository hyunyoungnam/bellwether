"""MCP server on stdio: a coding agent's hands on the literature engine.

Connect Claude Code / Codex to this (no API key — the agent brings its own
model) and it can search all six editions, walk similarity and citations, and
read each paper's VERIFIED sentences. Every sentence a tool returns is either
a paper's own verbatim text (machine-verified against the source) or computed
from data/processed/ — the same guarantee the site gives, now for agents.

Protocol: JSON-RPC 2.0, one message per line (MCP stdio transport). Stdlib
only, read-only, no network beyond the local search engine.

    wnai mcp                      # normally launched by the agent, via .mcp.json
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RDATA = ROOT / "reports" / "data"
MEILI_ADDR = os.environ.get("WNAI_MEILI_ADDR", "127.0.0.1:7700")

_VENUE = {"icml": "ICML", "neurips": "NeurIPS", "iclr": "ICLR"}


def _papers_file(key: str) -> Path:
    # historical naming: the first corpus kept its original file names
    if key == "icml-2026":
        return PROCESSED / "papers.jsonl"
    if key == "icml-2025":
        return PROCESSED / "papers_2025.jsonl"
    venue, year = key.rsplit("-", 1)
    return PROCESSED / f"papers_{venue}_{year}.jsonl"


class Store:
    """Lazy, cached access to the processed corpus. Read-only."""

    def __init__(self) -> None:
        self._c: dict = {}

    def _json(self, path: Path):
        k = str(path)
        if k not in self._c:
            with open(path, encoding="utf-8") as fh:
                self._c[k] = json.load(fh)
        return self._c[k]

    @property
    def union(self) -> dict:
        return self._json(PROCESSED / "union.json")

    @property
    def gid_by(self) -> dict:
        if "gid_by" not in self._c:
            u = self.union
            self._c["gid_by"] = {(k, e): i for i, (k, e)
                                 in enumerate(zip(u["keys"], u["eids"]))}
        return self._c["gid_by"]

    def where(self, gid: int) -> tuple[str, int]:
        u = self.union
        if not (0 <= gid < len(u["keys"])):
            raise ValueError(f"gid {gid} out of range 0..{len(u['keys'])-1}")
        return u["keys"][gid], u["eids"][gid]

    def papers(self, key: str) -> dict:
        ck = f"papers:{key}"
        if ck not in self._c:
            recs = {}
            with open(_papers_file(key), encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    recs[r["event_id"]] = r
            self._c[ck] = recs
        return self._c[ck]

    def rec(self, gid: int) -> dict | None:
        key, eid = self.where(gid)
        return self.papers(key).get(eid)

    def span_entry(self, gid: int) -> list | None:
        key, _ = self.where(gid)
        return self._json(RDATA / f"spans_{key}.json").get(str(gid))

    def topics(self, key: str) -> dict:
        return self._json(PROCESSED / f"topics_{key}.json")

    @property
    def cites_out(self) -> dict:
        if "cout" not in self._c:
            c = self._json(PROCESSED / "citations.json")
            out: dict[int, list] = {}
            inn: dict[int, list] = {}
            for a, b in c["edges"]:
                out.setdefault(a, []).append(b)
                inn.setdefault(b, []).append(a)
            self._c["cout"], self._c["cin"] = out, inn
            self._c["cnote"] = (f"references parsed for {c['papers_with_refs']:,} "
                                f"of {len(self.union['keys']):,} papers — a paper "
                                "with no edges may simply lack parsed references")
        return self._c["cout"]

    @property
    def cites_in(self) -> dict:
        _ = self.cites_out
        return self._c["cin"]

    @property
    def cite_note(self) -> str:
        _ = self.cites_out
        return self._c["cnote"]

    @property
    def search_key(self) -> str | None:
        try:
            return (ROOT / "data/meili/search_key").read_text().strip()
        except OSError:
            return None

    def meili(self, path: str, body: dict) -> dict | None:
        key = self.search_key
        if key is None:
            return None
        req = urllib.request.Request(
            f"http://{MEILI_ADDR}/indexes/papers/{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {key}",
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as fh:
                return json.loads(fh.read())
        except (urllib.error.URLError, OSError):
            return None

    def brief(self, gid: int) -> dict:
        r = self.rec(gid)
        key, _ = self.where(gid)
        venue, year = key.rsplit("-", 1)
        if r is None:
            return {"gid": gid, "venue": _VENUE.get(venue, venue), "year": int(year)}
        return {"gid": gid, "title": r["title"],
                "venue": _VENUE.get(venue, venue), "year": int(year),
                "authors": r.get("authors") or [],
                "area": r.get("area"), "subarea": r.get("subarea"),
                "decision": r.get("decision")}


S = Store()


# ---------------------------------------------------------------- tools

def t_search(a: dict) -> dict:
    limit = min(int(a.get("limit", 10)), 50)
    res = S.meili("search", {"q": a["query"], "limit": limit})
    if res is not None:
        return {"total_estimate": res.get("estimatedTotalHits"),
                "results": [S.brief(h["id"]) for h in res.get("hits", [])],
                "engine": "meilisearch (typo-tolerant, relevance-ranked)"}
    # engine down: word-AND over titles+abstracts, exact words only
    words = [w.lower() for w in a["query"].split() if w]
    out = []
    for key in S.union["corpora"]:
        for eid, r in S.papers(key).items():
            hay = (r["title"] + " " + (r.get("abstract") or "")).lower()
            if all(w in hay for w in words):
                out.append(S.gid_by[(key, eid)])
                if len(out) >= limit * 3:
                    break
    return {"total_estimate": len(out), "results": [S.brief(g) for g in out[:limit]],
            "engine": "fallback word-AND (search engine is down — no typo "
                      "tolerance, no relevance ranking)"}


def t_similar(a: dict) -> dict:
    gid = int(a["gid"])
    limit = min(int(a.get("limit", 10)), 30)
    res = S.meili("similar", {"id": gid, "limit": limit, "embedder": "bge"})
    if res is not None:
        return {"anchor": S.brief(gid),
                "results": [S.brief(h["id"]) for h in res.get("hits", [])]}
    nb = S._json(PROCESSED / "neighbors_union.json")
    k = nb["k"]
    sl = nb["nbr"][gid * k:(gid + 1) * k][:limit]
    return {"anchor": S.brief(gid), "results": [S.brief(g) for g in sl],
            "note": "engine down — precomputed 20-nearest fallback"}


def t_paper(a: dict) -> dict:
    gid = int(a["gid"])
    r = S.rec(gid)
    if r is None:
        return {"error": f"no record for gid {gid}"}
    out = S.brief(gid)
    out["abstract"] = r.get("abstract")
    out["links"] = {"virtual_site": r.get("virtual_url") or r.get("paper_url")}
    sp = S.span_entry(gid)
    if sp:
        n, L, K, R = sp[0], sp[1], sp[2], sp[3]
        out["verified_sentences"] = {
            "note": "verbatim sentences from the paper itself, machine-verified "
                    "against the source; missing fields mean the paper's text "
                    "did not yield a verified sentence, never invent one",
            "limitation_of_prior_work": L or None,
            "key_change": K or None,
            "result_claim": R or None,
            "novelty_spans": n or [],
        }
    key, eid = S.where(gid)
    labels = []
    try:
        for t in S.topics(key)["topics"]:
            if eid in t.get("explicit", ()) or eid in t.get("via_child", ()):
                labels.append(t["label"])
    except FileNotFoundError:
        pass
    out["topics"] = labels
    out["citations"] = {"cites_in_corpus": len(S.cites_out.get(gid, ())),
                        "cited_by_in_corpus": len(S.cites_in.get(gid, ())),
                        "note": S.cite_note}
    return out


def t_topics(a: dict) -> dict:
    agg: dict[str, dict] = {}
    for key in S.union["corpora"]:
        try:
            tj = S.topics(key)
        except FileNotFoundError:
            continue
        for t in tj["topics"]:
            if t.get("junk"):
                continue
            e = agg.setdefault(t["label"], {"label": t["label"],
                                            "family": t.get("family"),
                                            "papers": 0, "by_corpus": {}})
            n = t.get("n_explicit", 0) + t.get("n_via_child", 0)
            e["papers"] += n
            e["by_corpus"][key] = n
    rows = sorted(agg.values(), key=lambda e: -e["papers"])
    return {"topics": rows,
            "note": "multi-label tags — papers carry several, counts do not sum "
                    "to corpus size; roughly half of papers carry no tag "
                    "(they declare no domain and only generic tasks), so a tag "
                    "list is never exhaustive"}


def t_topic_papers(a: dict) -> dict:
    want = a["topic"].strip().lower()
    limit = min(int(a.get("limit", 50)), 200)
    hits, per = [], {}
    for key in S.union["corpora"]:
        try:
            tj = S.topics(key)
        except FileNotFoundError:
            continue
        for t in tj["topics"]:
            if t["label"].lower() != want:
                continue
            eids = list(t.get("explicit", ())) + list(t.get("via_child", ()))
            per[key] = len(eids)
            hits += [S.gid_by[(key, e)] for e in eids if (key, e) in S.gid_by]
    if not hits:
        return {"error": f"no topic labelled '{a['topic']}' — see list_topics",
                "hint": "topics are a fixed shared vocabulary; free-text goes "
                        "to search_papers instead"}
    return {"topic": want, "total": len(hits), "by_corpus": per,
            "results": [S.brief(g) for g in hits[:limit]],
            "truncated": len(hits) > limit}


def t_citations(a: dict) -> dict:
    gid = int(a["gid"])
    return {"paper": S.brief(gid),
            "cites": [S.brief(g) for g in S.cites_out.get(gid, ())[:40]],
            "cited_by": [S.brief(g) for g in S.cites_in.get(gid, ())[:40]],
            "note": "in-corpus edges only (between our six editions); "
                    + S.cite_note}


_GID = {"type": "integer", "description":
        "global paper id (gid) as returned by the other tools"}
TOOLS = [
    {"name": "search_papers",
     "description": "Full-text search over title+abstract of all six conference "
                    "editions (ICML/NeurIPS/ICLR, 29,605 papers). Typo-tolerant, "
                    "relevance-ranked. Returns gids for use with other tools.",
     "fn": t_search,
     "inputSchema": {"type": "object", "required": ["query"], "properties": {
         "query": {"type": "string"},
         "limit": {"type": "integer", "default": 10, "maximum": 50}}}},
    {"name": "similar_papers",
     "description": "Papers nearest to a given paper in embedding space "
                    "(BGE-M3 over abstracts), across all editions.",
     "fn": t_similar,
     "inputSchema": {"type": "object", "required": ["gid"], "properties": {
         "gid": _GID, "limit": {"type": "integer", "default": 10}}}},
    {"name": "get_paper",
     "description": "One paper in full: record, abstract, topics, citation "
                    "counts, and its VERIFIED sentences — the paper's own "
                    "verbatim words for what prior work failed at, what this "
                    "paper changes, and what it achieved. Quote these; never "
                    "paraphrase them as if quoted.",
     "fn": t_paper,
     "inputSchema": {"type": "object", "required": ["gid"], "properties": {
         "gid": _GID}}},
    {"name": "list_topics",
     "description": "The shared level-3 topic vocabulary (multi-label field "
                    "tags like 'autonomous driving'), with paper counts per "
                    "corpus. Coverage ~50% — absence of a tag is not evidence.",
     "fn": t_topics,
     "inputSchema": {"type": "object", "properties": {}}},
    {"name": "topic_papers",
     "description": "All papers carrying a topic tag, across editions.",
     "fn": t_topic_papers,
     "inputSchema": {"type": "object", "required": ["topic"], "properties": {
         "topic": {"type": "string", "description": "a label from list_topics"},
         "limit": {"type": "integer", "default": 50}}}},
    {"name": "get_citations",
     "description": "Which of our corpus papers this paper cites, and which "
                    "cite it (edges within the six editions only).",
     "fn": t_citations,
     "inputSchema": {"type": "object", "required": ["gid"], "properties": {
         "gid": _GID}}},
]


# ---------------------------------------------------------------- protocol

def _reply(msg_id, result=None, error=None) -> None:
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def serve_stdio() -> int:
    tools_by = {t["name"]: t for t in TOOLS}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        method = msg.get("method", "")
        msg_id = msg.get("id")
        if method == "initialize":
            _reply(msg_id, {
                "protocolVersion": msg.get("params", {}).get(
                    "protocolVersion", "2024-11-05"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "wnai", "version": "0.1.0"}})
        elif method == "tools/list":
            _reply(msg_id, {"tools": [
                {k: t[k] for k in ("name", "description", "inputSchema")}
                for t in TOOLS]})
        elif method == "tools/call":
            p = msg.get("params", {})
            t = tools_by.get(p.get("name"))
            if t is None:
                _reply(msg_id, error={"code": -32602,
                                      "message": f"unknown tool {p.get('name')}"})
                continue
            try:
                res = t["fn"](p.get("arguments") or {})
                _reply(msg_id, {"content": [{"type": "text",
                                             "text": json.dumps(res, ensure_ascii=False)}],
                                "isError": "error" in res})
            except Exception as exc:  # noqa: BLE001
                _reply(msg_id, {"content": [{"type": "text",
                                             "text": f"{type(exc).__name__}: {exc}"}],
                                "isError": True})
        elif method == "ping":
            _reply(msg_id, {})
        elif msg_id is not None:
            _reply(msg_id, error={"code": -32601,
                                  "message": f"method not found: {method}"})
        # notifications (no id) are consumed silently
    return 0
