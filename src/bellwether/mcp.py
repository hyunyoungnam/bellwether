"""MCP server on stdio: a coding agent's hands on the literature engine.

Connect Claude Code / Codex to this (no API key — the agent brings its own
model) and it can search all six editions, walk similarity and citations, and
read each paper's VERIFIED sentences. Every sentence a tool returns is either
a paper's own verbatim text (machine-verified against the source) or computed
from data/processed/ — the same guarantee the site gives, now for agents.

Protocol: JSON-RPC 2.0, one message per line (MCP stdio transport). Stdlib
only, read-only, no network beyond the local search engine.

    bellwether mcp                      # normally launched by the agent, via .mcp.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
RDATA = ROOT / "reports" / "data"
INTERIM = ROOT / "data" / "interim"
ARXRAW = ROOT / "data" / "raw" / "arxiv"

# parsed full text per corpus (richest source each), plus the arxiv-id bridge
_FT = {
    "icml-2026": ("fulltext_html_icml_2026.jsonl", "resolved.jsonl"),
    "icml-2025": ("fulltext_pmlr_2025.jsonl", None),   # rows carry event_id
    "neurips-2024": ("fulltext_neurips_2024.jsonl", "resolved_neurips-2024.jsonl"),
    "neurips-2025": ("fulltext_neurips_2025.jsonl", "resolved_neurips-2025.jsonl"),
    "iclr-2025": ("fulltext_iclr_2025.jsonl", "resolved_iclr-2025.jsonl"),
    "iclr-2026": ("fulltext_iclr_2026.jsonl", "resolved_iclr-2026.jsonl"),
}
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

    def terms(self, key: str, eid: int) -> dict:
        """Aggregation names from the abstract pass: p(roposes) / b(uilds on)
        / t(asks) / d(ata) / dom(ain). Empty when the file is absent."""
        try:
            return self._json(PROCESSED / "card_terms.json").get(key, {}) \
                .get(str(eid), {})
        except FileNotFoundError:
            return {}

    def _ft_index(self, key: str) -> dict:
        """eid -> byte offset into the corpus's full-text jsonl. One linear
        scan per corpus per process; rows are then read by seek."""
        ck = f"ftix:{key}"
        if ck not in self._c:
            ix: dict[int, int] = {}
            spec = _FT.get(key)
            if spec and (INTERIM / spec[0]).exists():
                to_eid = None
                if spec[1]:
                    to_eid = {}
                    for line in open(ARXRAW / spec[1], encoding="utf-8"):
                        r = json.loads(line)
                        if r.get("arxiv_base"):
                            to_eid.setdefault(r["arxiv_base"], r["event_id"])
                with open(INTERIM / spec[0], "rb") as fh:
                    off = 0
                    for raw in fh:
                        head = raw[:200].decode("utf-8", "ignore")
                        eid = None
                        if to_eid is None:
                            m = re.search(r'"event_id":\s*(\d+)', head)
                            eid = int(m.group(1)) if m else None
                        else:
                            m = re.search(r'"arxiv_base":\s*"([^"]+)"', head)
                            eid = to_eid.get(m.group(1)) if m else None
                        if eid is not None and eid not in ix:
                            ix[eid] = off
                        off += len(raw)
            self._c[ck] = ix
        return self._c[ck]

    def fulltext(self, gid: int) -> dict | None:
        key, eid = self.where(gid)
        ix = self._ft_index(key)
        if eid not in ix:
            return None
        spec = _FT[key]
        with open(INTERIM / spec[0], "rb") as fh:
            fh.seek(ix[eid])
            row = json.loads(fh.readline())
        return row if row.get("ok", True) and row.get("sections") else None

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
    def cite_ctx(self) -> dict:
        """cited gid (str) -> [[citing gid, bucket, sentence], ...]."""
        if "cctx" not in self._c:
            p = PROCESSED / "cite_contexts.json"
            self._c["cctx"] = self._json(p)["contexts"] if p.exists() else {}
        return self._c["cctx"]

    @property
    def blue_ocean(self) -> dict:
        """icml.blue_ocean output: validated (field, technique) candidates."""
        if "bo" not in self._c:
            p = PROCESSED / "blue_ocean.json"
            self._c["bo"] = self._json(p) if p.exists() else {}
        return self._c["bo"]

    @property
    def ext_ids(self) -> dict:
        """gid (str) -> {arxiv, doi, s2, oa} — icml.ids, exact matches only."""
        if "xids" not in self._c:
            p = PROCESSED / "ids.json"
            self._c["xids"] = self._json(p)["ids"] if p.exists() else {}
        return self._c["xids"]

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


def _agent_card(gid: int) -> dict:
    """The compact agent-readable card: extracted verbatim fields, no prose."""
    r = S.rec(gid)
    key, _ = S.where(gid)
    venue, year = key.rsplit("-", 1)
    sp = S.span_entry(gid) or [None] * 4
    n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
    kc = K or (n[0] if n else None)
    _, eid = S.where(gid)
    t = S.terms(key, eid)
    return {"gid": gid, "title": r["title"] if r else None,
            "venue": _VENUE.get(venue, venue), "year": int(year),
            "limitation": L[:280] if L else None,
            "key_change": kc[:280] if kc else None,
            "result_claim": R[:280] if R else None,
            # names, for counting across cards — the aggregation axis
            "proposes": t.get("p"), "builds_on": t.get("b"),
            "tasks": t.get("t"), "data": t.get("d"), "domain": t.get("dom")}


def t_field_cards(a: dict) -> dict:
    """Bulk agent cards for a whole field — the raw material for derivation."""
    limit = min(int(a.get("limit", 30)), 60)
    gids: list[int] = []
    if a.get("topic"):
        r = t_topic_papers({"topic": a["topic"], "limit": 400})
        if "error" in r:
            return r
        gids = [x["gid"] for x in r["results"]]
        total, per = r["total"], r["by_corpus"]
    elif a.get("query"):
        r = S.meili("search", {"q": a["query"], "limit": limit})
        if r is None:
            return {"error": "search engine down — use topic instead"}
        gids = [h["id"] for h in r.get("hits", [])]
        total, per = r.get("estimatedTotalHits"), None
    else:
        return {"error": "pass topic or query"}
    # newest editions first, so 'what is rising' reads the right material
    gids.sort(key=lambda g: -int(S.where(g)[0].rsplit("-", 1)[1]))
    return {"total_in_field": total, "by_corpus": per,
            "cards": [_agent_card(g) for g in gids[:limit]],
            "note": "card fields are the papers' own verbatim sentences; a "
                    "missing field means the paper's text yielded no verified "
                    "sentence, never invent one. Cards shown newest-edition "
                    "first and capped — by_corpus has the full counts."}


def _ztest(k1: int, n1: int, k2: int, n2: int) -> float:
    import math
    if not n1 or not n2:
        return 0.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    d = math.sqrt(max(p * (1 - p) * (1 / n1 + 1 / n2), 1e-12))
    return (p1 - p2) / d


def t_field_trend(a: dict) -> dict:
    """A topic's share per edition — computed, guardrail-encoded numbers.

    Shares are per 1,000 papers (editions differ ~2x in size; raw counts
    reverse conclusions) and each venue's newest-vs-previous change carries a
    two-proportion z — |z| >= 2.576 is the product's standing bar for 'real'."""
    want = a["topic"].strip().lower()
    rows = []
    for key in S.union["corpora"]:
        try:
            tj = S.topics(key)
        except FileNotFoundError:
            continue
        for t in tj["topics"]:
            if t["label"].lower() == want:
                n = t.get("n_explicit", 0) + t.get("n_via_child", 0)
                size = tj.get("papers") or 1
                venue, year = key.rsplit("-", 1)
                rows.append({"venue": _VENUE.get(venue, venue),
                             "year": int(year), "papers": n,
                             "corpus_size": size,
                             "per_1k": round(n / size * 1000, 1)})
    if not rows:
        return {"error": f"no topic labelled '{a['topic']}' — see list_topics"}
    rows.sort(key=lambda r: (r["venue"], r["year"]))
    changes = []
    for i in range(1, len(rows)):
        a2, b2 = rows[i - 1], rows[i]
        if a2["venue"] == b2["venue"]:
            z = _ztest(b2["papers"], b2["corpus_size"],
                       a2["papers"], a2["corpus_size"])
            changes.append({"venue": b2["venue"],
                            "years": f"{a2['year']}->{b2['year']}",
                            "per_1k": f"{a2['per_1k']}->{b2['per_1k']}",
                            "z": round(z, 2),
                            "significant_99": abs(z) >= 2.576})
    return {"topic": want, "editions": rows, "changes": changes,
            # the rule states its own constants, so a sentence that quotes the
            # bar ("|z| >= 2.576, the 99% level") is checkable like any figure
            "rule": {"z_threshold": 2.576, "confidence_pct": 99,
                     "shares_per": 1000},
            "note": "tag coverage is ~50% of papers, so shares understate "
                    "fields that avoid the expected vocabulary; state figures "
                    "as computed shares, never as paper quotes"}


# discourse words that name no failure — the register, not the content
_GAP_STOP = {
    "existing", "current", "prior", "previous", "conventional", "traditional",
    "methods", "method", "approaches", "approach", "models", "model", "works",
    "work", "often", "typically", "however", "remain", "remains", "limited",
    "limitation", "limitations", "challenge", "challenges", "challenging",
    "difficult", "difficulty", "problem", "problems", "issue", "issues",
    "significant", "significantly", "large", "high", "low", "widely", "many",
    "various", "based", "rely", "relies", "require", "requires", "requiring",
    "expensive", "costly", "cost", "costs", "lack", "lacks", "fail", "fails",
    "failure", "unable", "cannot", "still", "these", "those", "such", "them",
    "their", "this", "that", "which", "when", "while", "with", "without",
    "performance", "tasks", "task", "data", "training", "learning", "either",
    "struggle", "struggles", "suffer", "suffers", "make", "makes", "making",
    # function words the tokenizer admits
    "the", "and", "for", "are", "but", "not", "can", "due", "may", "has",
    "have", "been", "was", "were", "will", "would", "could", "should", "its",
    "into", "from", "they", "there", "where", "what", "who", "how", "why",
    "than", "then", "also", "only", "both", "each", "more", "most", "less",
    "other", "others", "same", "well", "even", "much", "very", "between",
    "across", "over", "under", "within", "about", "because", "since",
    "although", "though", "thus", "hence", "therefore", "moreover",
    "furthermore", "despite", "unlike", "among", "along", "against", "does",
    "usually", "generally", "highly", "particularly", "especially", "several",
    "including", "include", "includes", "leading", "leads", "lead", "given",
    "using", "used", "uses", "use", "via", "per", "yet", "own", "new", "one",
    "two", "way", "ways", "key", "main", "real", "world", "recent", "recently",
    "different", "single", "multiple", "specific", "important", "common",
    "poor", "poorly", "hard", "hinder", "hinders", "hindering", "prevent",
    "prevents", "prevented", "ignore", "ignores", "ignoring", "overlook",
    "overlooks", "neglect", "neglects", "capture", "captures", "capturing",
    "achieve", "achieves", "achieving", "obtain", "handle", "handling",
    "address", "addresses", "addressing", "remain", "result", "results",
    "focus", "focuses", "focusing", "focused", "consider", "considers",
    "resulting", "propose", "proposed", "approaches", "solutions", "solution",
}
_TOK = re.compile(r"[a-z][a-z\-]{2,}")


def t_gap_scan(a: dict) -> dict:
    """The blue-ocean derivation, computed — the agent interprets, never
    invents. Within a field: cluster the failures its papers NAME (verbatim
    limitation sentences), then split each cluster into papers that merely
    cite the failure as motivation vs papers whose key_change/proposals
    ATTACK it. A failure widely named but barely attacked is a gap
    candidate — a structural fact, not a score."""
    want = a["topic"].strip().lower()
    gids = []
    for key in S.union["corpora"]:
        try:
            tj = S.topics(key)
        except FileNotFoundError:
            continue
        for t0 in tj["topics"]:
            if t0["label"].lower() == want:
                eids = list(t0.get("explicit", ())) + list(t0.get("via_child", ()))
                gids += [S.gid_by[(key, e)] for e in eids if (key, e) in S.gid_by]
    if not gids:
        return {"error": f"no topic labelled '{a['topic']}' — see list_topics"}
    label_words = set(_TOK.findall(a["topic"].lower()))
    # gather each member's L text, K text, names, edition
    rows = []
    for g in gids:
        sp = S.span_entry(g) or [None] * 4
        L, K = sp[1] or "", sp[2] or ""
        n = sp[0] or []
        key, eid = S.where(g)
        t2 = S.terms(key, eid)
        names = " ".join((t2.get("p") or []) + (t2.get("t") or []))
        rows.append({"g": g, "L": L.lower(), "Lraw": L,
                     "atk": (K + " " + " ".join(n) + " " + names).lower(),
                     "year": int(key.rsplit("-", 1)[1])})
    withL = [x for x in rows if x["L"]]
    # candidate failure terms: unigrams+bigrams by document frequency in L
    from collections import Counter
    df: Counter = Counter()
    for x in withL:
        toks = [w for w in _TOK.findall(x["L"]) if w not in _GAP_STOP
                and w not in label_words]
        toks_u = [w for w in toks if len(w) >= 5]
        big = [f"{u} {v}" for u, v in zip(toks, toks[1:])]
        for term in set(toks_u) | set(big):
            df[term] += 1
    n_mem = len(withL)
    cands = [(t3, c) for t3, c in df.items()
             if c >= max(3, n_mem // 60) and c <= n_mem * 0.5]
    # prefer bigrams over their own words when nearly as frequent
    keep = []
    bigs = {t3 for t3, _ in cands if " " in t3}
    for t3, c in sorted(cands, key=lambda p: -p[1]):
        if " " not in t3 and any(t3 in b for b in bigs
                                 if df[b] >= c * 0.6):
            continue
        keep.append((t3, c))
    branches = []
    used: set = set()
    for term, _c in keep:
        if any(term in u or u in term for u in used):
            continue
        namers = [x for x in withL if term in x["L"]]
        if len(namers) < 3:
            continue
        attackers = [x for x in namers if term in x["atk"]]
        motiv = [x for x in namers if term not in x["atk"]]
        rep = min(namers, key=lambda x: len(x["Lraw"]))
        yrs = Counter(x["year"] for x in namers)
        branches.append({
            "term": term,
            "named_by": len(namers),
            "attacked_by": len(attackers),
            "motivation_only": len(motiv),
            "by_year": dict(sorted(yrs.items())),
            "rep": {"gid": rep["g"], "sentence": rep["Lraw"]},
            "attackers": [x["g"] for x in attackers][:12],
            "namers": [x["g"] for x in motiv][:12],
        })
        used.add(term)
        if len(branches) >= 10:
            break
    # structural order: unattacked first, then how widely named
    branches.sort(key=lambda b: (b["attacked_by"] > 0, -b["named_by"]))
    return {"field": a["topic"], "papers_in_field": len(gids),
            "with_limitation": n_mem, "branches": branches,
            # what the READER sees under the tree: a coverage caveat, no
            # reading instructions — the note below is written for the agent
            "caption": "failure terms taken from the papers' own limitation "
                       "sentences; matching is lexical, so a field that "
                       "phrases its failures differently is undercounted",
            "note": "branches are failure terms from the papers' own "
                    "limitation sentences (register words and the field's own "
                    "name filtered); attacked_by counts papers whose "
                    "key_change or proposal names the term. A widely named, "
                    "barely attacked branch is a gap CANDIDATE — judge it, "
                    "don't rank it. Term matching is lexical; coverage "
                    "understates fields that phrase failures differently."}


def t_paper_text(a: dict) -> dict:
    """Read a paper's parsed full text, section by section — the on-demand
    depth path (no pre-extraction): list sections first, then read one."""
    gid = int(a["gid"])
    row = S.fulltext(gid)
    if row is None:
        return {"error": "no parsed full text for this paper — coverage is "
                         "~77% and not random (theory/statistics preprint "
                         "less); the card fields and abstract still apply",
                "paper": S.brief(gid)}
    secs = row["sections"]
    sec = a.get("section")
    if not sec:
        return {"paper": S.brief(gid),
                "sections": [{"bucket": x["bucket"], "title": x.get("title"),
                              "chars": len(x["text"])} for x in secs],
                "note": "call again with section=<bucket> to read one"}
    txt = " ".join(x["text"] for x in secs if x["bucket"] == sec)
    if not txt:
        return {"error": f"no section bucket '{sec}'",
                "available": sorted({x["bucket"] for x in secs})}
    return {"paper": S.brief(gid), "section": sec, "chars": len(txt),
            "text": txt[:12000], "truncated": len(txt) > 12000}


def t_verify_quote(a: dict) -> dict:
    """The product's own guarantee, offered as a tool.

    Every literature MCP server surveyed is a remote search gateway; this one
    holds the papers, so it can answer the question none of them can: does this
    sentence actually appear in that paper? Any agent — ours or someone else's
    — can put a quote to it before publishing the quote.
    """
    from .verify import Verifier
    gid = int(a["gid"])
    quote = (a.get("quote") or "").strip()
    ver = _verifier()
    role = ver.role(gid, quote) if quote else None
    r = S.rec(gid)
    key, _ = S.where(gid)
    venue, year = key.rsplit("-", 1)
    return {"gid": gid, "verified": role is not None,
            "matched_field": role,
            "title": r["title"] if r else None,
            "venue": _VENUE.get(venue, venue), "year": int(year),
            "note": "matched against this paper's own text on this machine: "
                    "the extracted sentences first, then the abstract and "
                    "title, then the parsed full text where we hold it. "
                    "verified=false means the sentence is not in what we hold "
                    "— it does not mean the paper says otherwise."}


_VER_CACHE: list = []


def _verifier():
    if not _VER_CACHE:
        from .verify import Verifier
        _VER_CACHE.append(Verifier(S))
    return _VER_CACHE[0]


def t_blue_ocean(a: dict) -> dict:
    """(field, technique) pairs no paper in the corpus has combined, ranked
    by how the field's own stated failures match what the technique claims to
    fix elsewhere — computed by icml.blue_ocean and validated by time (editions
    <= 2025 predicting the 2026 editions). Two tiers with their own measured
    precision; the agent relays the evidence sentences, never a verdict."""
    bo = S.blue_ocean
    if not bo:
        return {"error": "blue_ocean.json not built — run icml.blue_ocean --emit"}
    want = (a.get("field") or "").strip().lower()
    k = max(1, min(int(a.get("k") or 8), 25))
    out = {"field": want or None, "tiers": {}}
    for name, tier in bo["tiers"].items():
        rows = [c for c in tier["candidates"]
                if not want or want in c["field"] or c["field"] in want]
        out["tiers"][name] = {
            "scorer": tier["scorer"],
            "validated_precision_at_50": tier["precision_at"]["50"],
            "candidates": rows[:k],
        }
    v = bo["validation"]
    out["rule"] = {
        "validation": "retrospective link prediction, editions <= 2025 -> 2026",
        "base_rate": v["base_rate"],
        "precision_at_50": {s: v["precision_at"][s]["50"] for s in v["precision_at"]},
        "fields": v["fields"], "techniques": v["techniques"],
    }
    out["note"] = ("a candidate is a pair NO paper in the six editions has "
                   "combined; 'established' = most likely to be filled within a "
                   "year (measured P@50 above), 'novel' = techniques few fields "
                   "use yet, about half the hit rate. field_states are the "
                   "field's own limitation sentences, technique_claims the "
                   "technique's own result/key-change sentences elsewhere — "
                   "quote them, do not paraphrase them into a recommendation. "
                   "Fields are declared domains (~29% of papers declare one)")
    return out


def t_citations(a: dict) -> dict:
    gid = int(a["gid"])

    def _cx(citing: int, cited: int):
        # the citing paper's own sentence at the citation, where the reference
        # style let icml.cite_contexts locate it (name-year heads only)
        for cg, bucket, sent in S.cite_ctx.get(str(cited), ()):
            if cg == citing:
                return {"section": bucket, "sentence": sent}
        return None

    def _side(gids, direction):
        out = []
        for g in gids[:40]:
            b = S.brief(g)
            c = _cx(gid, g) if direction == "out" else _cx(g, gid)
            if c:
                b["context"] = c
            out.append(b)
        return out

    return {"paper": S.brief(gid),
            "cites": _side(S.cites_out.get(gid, ()), "out"),
            "cited_by": _side(S.cites_in.get(gid, ()), "in"),
            "note": "in-corpus edges only (between our six editions); context "
                    "= the citing paper's verbatim sentence at the citation, "
                    "present where the reference style allowed locating it; "
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
    {"name": "field_cards",
     "description": "Bulk agent-readable cards for a whole field (by topic "
                    "label or free query): each paper's verbatim limitation / "
                    "key_change / result_claim in one call. THE tool for "
                    "deriving what is rising, what recurs, and what is "
                    "missing across a field — read many cards, then conclude.",
     "fn": t_field_cards,
     "inputSchema": {"type": "object", "properties": {
         "topic": {"type": "string", "description": "a label from list_topics"},
         "query": {"type": "string", "description": "free text when no label fits"},
         "limit": {"type": "integer", "default": 30, "maximum": 60}}}},
    {"name": "field_trend",
     "description": "A topic's share per conference edition, computed with "
                    "the product's statistical guardrails (per-1,000 shares, "
                    "two-proportion z per venue pair). Use for 'is this field "
                    "growing' — quote the returned figures as computed shares.",
     "fn": t_field_trend,
     "inputSchema": {"type": "object", "required": ["topic"], "properties": {
         "topic": {"type": "string"}}}},
    {"name": "gap_scan",
     "description": "The blue-ocean derivation for a field: clusters the "
                    "failures its papers name (verbatim limitation "
                    "sentences) and splits each into namers vs attackers — "
                    "a widely named, barely attacked failure is a gap "
                    "candidate. Call this for 'where are the gaps / blue "
                    "ocean / unsolved problems' questions, then interpret "
                    "the branches; the reader sees the tree itself.",
     "fn": t_gap_scan,
     "inputSchema": {"type": "object", "required": ["topic"], "properties": {
         "topic": {"type": "string", "description": "a label from list_topics"}}}},
    {"name": "paper_text",
     "description": "Read one paper's parsed FULL TEXT on demand — first call "
                    "lists its sections (abstract/intro/method/experiments/"
                    "conclusion/appendix/...), a second call with section= "
                    "returns that section's own words. THE tool for depth "
                    "questions about how a specific paper works.",
     "fn": t_paper_text,
     "inputSchema": {"type": "object", "required": ["gid"], "properties": {
         "gid": _GID, "section": {"type": "string"}}}},
    {"name": "verify_quote",
     "description": "Check a sentence against the paper it is attributed to, "
                    "on this machine. Returns whether the quote appears in "
                    "that paper's own text and which field matched. Use it "
                    "before publishing any sentence as a paper's words — "
                    "including sentences you got from somewhere else.",
     "fn": t_verify_quote,
     "inputSchema": {"type": "object", "required": ["gid", "quote"],
                     "properties": {"gid": _GID,
                                    "quote": {"type": "string"}}}},
    {"name": "blue_ocean",
     "description": "Blue-ocean candidates: (application field, technique) "
                    "pairs no paper in the six editions has combined, ranked by "
                    "how the field's stated failures match the technique's "
                    "claims elsewhere; time-validated (editions <= 2025 "
                    "predicting 2026, precision stated in the result). Optional "
                    "field filter. Returns the papers' own evidence sentences.",
     "fn": t_blue_ocean,
     "inputSchema": {"type": "object", "properties": {
         "field": {"type": "string", "description":
                   "application domain, e.g. healthcare, robotics, drug discovery"},
         "k": {"type": "integer", "description": "candidates per tier (<= 25)"}}}},
    {"name": "get_citations",
     "description": "Which of our corpus papers this paper cites, and which "
                    "cite it (edges within the six editions only). Where the "
                    "reference style allows it, each entry carries the citing "
                    "paper's verbatim sentence at the citation (context).",
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
                "serverInfo": {"name": "bellwether", "version": "0.1.0"}})
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
