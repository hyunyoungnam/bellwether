"""Where a paper's own artifacts live — code, released data, models — and
where the benchmarks it tested on can be fetched.

Two different facts, from two different sources, and the boundary matters:

1. **The paper's own links.** "Code is available at https://github.com/…"
   is printed in the paper; the URL is a verbatim string and the sentence
   around it is the evidence. Extracted here by rules over the parsed full
   text (no model, no GPU: a URL is a regex, and a sentence is cut, not
   written). Coverage is full-text coverage — about 70% of each corpus — and
   is stated wherever the links are shown.

2. **Where a benchmark lives.** A paper says "we evaluate on GSM8K"; it does
   not say GSM8K is `openai/gsm8k` on the Hub. That mapping is OURS —
   `config/benchmarks.json`, authored like DOMAIN_FAMILIES, marked as such in
   the interface, and every id in it is checked against the Hub / GitHub API
   before it may carry a link. A name with no confirmed location stays a
   name.

Both are enrichment of a paper's own view. Nothing sorts or scores on them
(guardrail 5); GitHub stars and Hub downloads are shown as facts and are
never a sort axis.

    python3 -m icml.resources                 # extract  -> data/processed/resources.json
    python3 -m icml.resources --resolve       # GitHub / Hub metadata (gh token or GITHUB_TOKEN)
    python3 -m icml.resources --registry      # check config/benchmarks.json ids against the APIs
"""
from __future__ import annotations

import argparse
import json
import os
import re
import ssl
import subprocess
import sys
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

from .common import INTERIM, PROCESSED, RAW, ROOT, dump_json, load_json, read_jsonl
from .corpus import available
from .taxonomy import dataset_key

OUT = PROCESSED / "resources.json"
CACHE = INTERIM / "resolve_cache.jsonl"
REGISTRY = ROOT / "config" / "benchmarks.json"

# The richest parsed full text per corpus (the same table the MCP store
# reads), plus the arxiv-id -> event_id bridge where rows are keyed by arXiv.
FULLTEXT = {
    "icml-2026": ("fulltext_html_icml_2026.jsonl", "resolved.jsonl"),
    "icml-2025": ("fulltext_pmlr_2025.jsonl", None),   # rows carry event_id
    "neurips-2024": ("fulltext_neurips_2024.jsonl", "resolved_neurips-2024.jsonl"),
    "neurips-2025": ("fulltext_neurips_2025.jsonl", "resolved_neurips-2025.jsonl"),
    "iclr-2025": ("fulltext_iclr_2025.jsonl", "resolved_iclr-2025.jsonl"),
    "iclr-2026": ("fulltext_iclr_2026.jsonl", "resolved_iclr-2026.jsonl"),
}

# ------------------------------------------------------------------ extraction

_URL = re.compile(
    r"https?://(?:www\.)?"
    r"(github\.com|github\.io|huggingface\.co|hf\.co|gitlab\.com|zenodo\.org|osf\.io|"
    r"[\w-]+\.github\.io)"
    r"(/[!-~]*)?", re.I)      # ASCII only: a curly quote after the URL is prose
_TRAIL = ".,;:!?)]}'\"›»>*"
# PDF line wrapping splits a URL after the dot or the slash; the HTML route
# does not, so the repairs are no-ops there.
_REPAIRS = [(re.compile(r"(github|huggingface|gitlab|zenodo)\.\s*\n?\s*(com|co|org)\b", re.I), r"\1.\2"),
            (re.compile(r"(https?://)\s+"), r"\1"),
            (re.compile(r"(github\.com|huggingface\.co|gitlab\.com)/\s+(?=[\w.-]+/)", re.I), r"\1/"),
            (re.compile(r"(github\.com|gitlab\.com)/([\w.-]+)/\s+(?=[\w.-]{3,})", re.I), r"\1/\2/"),
            (re.compile(r"(huggingface\.co)/(datasets/|spaces/)?([\w.-]+)/\s+(?=[\w.-]{3,})", re.I), r"\1/\2\3/")]
_GH_NOT_REPO = {"features", "topics", "orgs", "search", "sponsors", "settings",
                "marketplace", "apps", "login", "about", "site", "explore",
                "pulls", "issues", "notifications", "blog", "collections"}
_HF_NOT_REPO = {"papers", "docs", "blog", "spaces", "collections", "settings",
                "join", "login", "pricing", "learn", "tasks", "models", "datasets",
                "posts", "chat", "new", "organizations", "enterprise"}

# "Code / models / data … available / released / open-sourced", either order.
_AVAIL = re.compile(
    r"\b(code|codes|codebase|implementation|source|scripts?|models?|checkpoints?|weights|"
    r"data|datasets?|benchmark|software|package|library|toolkit|repository|repo|"
    r"project page|website|demo|artifacts?|materials?)\b.{0,90}?"
    r"\b(available|released?|releases|open-?sourced?|public(?:ly)?|accessible|hosted|"
    r"found at|shared at|hf\.co|hub)\b"
    r"|\b(available|released|hosted|open-?sourced)\b.{0,40}?"
    r"\b(code|codes|implementation|models?|checkpoints?|weights|data|datasets?|"
    r"benchmark|repository|repo)\b", re.I | re.S)
# "our" counts only when it owns the artifact ("our code", "we release"):
# "we train our model on the data from <url>" is someone else's data
_OURS = re.compile(
    r"\b(our (?:source )?(?:code|codes|codebase|implementation|data|datasets?|benchmark|"
    r"models?|checkpoints?|weights|repository|repo|project page|website|demo|package|library)|"
    r"we release|we open-?source|we provide|we publish|we make .{0,40}?\bavailable)\b", re.I)
# "we adopt the code from …", "the official implementation of X at …": a
# repository the paper USES, however the sentence phrases availability
_THEIRS = re.compile(
    r"\b(adopt|adapt|borrow|follow|reuse|re-use|use|used|using|utili[sz]e|leverage|"
    r"taken|obtained|downloaded|based on|built? on|build upon|provided by the authors|"
    r"official (?:implementation|code|repository)|original (?:implementation|code|repository)|"
    r"refer to|see also|details in|in accordance with|following the|provided in|as in|"
    r"setup of|implementation of|baseline|baselines|we considered|we used|we use|"
    r"we train on|we evaluate on|taken from|we list)\b", re.I)
# a footnote label: "Code: URL", "Project page: URL", or the URL on its own
_FOOTNOTE = re.compile(
    r"^(?:\d\s*)?(?:(?:source\s+)?code|project\s*page|website|homepage|demo|repository|repo|"
    r"github|hugging\s*face|models?|data(?:set)?|weights|checkpoints?|page)\s*[:：]?\s*https?://", re.I)
_DATAWORD = re.compile(r"\b(dataset|datasets|benchmark|data|corpus|annotations)\b", re.I)
_CODEWORD = re.compile(r"\b(code|codes|implementation|codebase|scripts?|library|package|toolkit)\b", re.I)
_MODELWORD = re.compile(r"\b(models?|checkpoints?|weights)\b", re.I)
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\d(\[])")
_LATER = re.compile(r"\b(will be|to be)\s+(made\s+)?(available|released|open-?sourced|published)", re.I)

# a repo that exists only in a name: "Method-Name" vs "methodname"
def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def _clean_url(u: str) -> str:
    u = u.strip()
    while u and u[-1] in _TRAIL:
        u = u[:-1]
    # a closing paren that has no opening one inside the url is punctuation
    return u


def _classify_url(host: str, path: str) -> tuple[str, str] | None:
    """-> (host_kind, repo) or None. host_kind: gh / hfd / hfm / hfs / page /
    gl / zenodo / osf. repo is the canonical id (owner/name, lowercase for
    GitHub which is case-insensitive; Hub ids keep case)."""
    h = host.lower()
    segs = [s for s in (path or "").split("/") if s]
    if h == "github.com":
        if len(segs) < 2 or segs[0] in _GH_NOT_REPO:
            return None
        name = re.sub(r"\.git$", "", segs[1])
        if not name or not re.match(r"^[\w.\-]+$", segs[0]):
            return None
        return "gh", f"{segs[0]}/{name}".lower()
    if h.endswith("github.io"):
        owner = h.split(".")[0] if h != "github.io" else (segs[0] if segs else "")
        if not owner:
            return None
        return "page", f"{owner}.github.io" + ("/" + segs[0] if h != "github.io" and segs else "")
    if h in ("huggingface.co", "hf.co"):
        if not segs or segs[0] in {"papers", "docs", "blog", "settings", "join",
                                   "login", "pricing", "learn", "tasks", "posts",
                                   "chat", "new", "organizations", "enterprise"}:
            return None
        if segs[0] == "datasets":
            return ("hfd", f"{segs[1]}/{segs[2]}") if len(segs) >= 3 else None
        if segs[0] == "spaces":
            return ("hfs", f"{segs[1]}/{segs[2]}") if len(segs) >= 3 else None
        if segs[0] in ("collections",):
            return ("hfc", "/".join(segs[1:3])) if len(segs) >= 3 else None
        if segs[0] in ("models",):
            segs = segs[1:]
        if len(segs) >= 2 and segs[0] not in _HF_NOT_REPO:
            return "hfm", f"{segs[0]}/{segs[1]}"
        return None
    if h == "gitlab.com":
        return ("gl", f"{segs[0]}/{segs[1]}".lower()) if len(segs) >= 2 else None
    if h == "zenodo.org":
        m = re.search(r"(\d{5,})", path or "")
        return ("zenodo", m.group(1)) if m else None
    if h == "osf.io":
        return ("osf", segs[0]) if segs else None
    return None


def _sentence_at(text: str, start: int, end: int) -> str:
    lo = max(0, start - 400)
    hi = min(len(text), end + 400)
    win = text[lo:hi]
    rel = start - lo
    # cut at sentence boundaries that do not fall inside the URL itself
    left = 0
    for m in _SENT_SPLIT.finditer(win):
        if m.end() <= rel:
            left = m.end()
        else:
            break
    right = len(win)
    for m in _SENT_SPLIT.finditer(win):
        if m.start() >= rel + (end - start):
            right = m.start()
            break
    s = win[left:right].strip()
    s = re.sub(r"\s+", " ", s)
    return s[:320]


def extract_paper(sections: list[dict], proposed: list[str]) -> dict:
    """Own links (code / data / model / page) with evidence sentences, and
    the repos merely mentioned (a baseline's code, a tool)."""
    props = {_slug(p) for p in proposed if len(_slug(p)) >= 4}
    own: dict[tuple[str, str], dict] = {}
    mention: dict[tuple[str, str], dict] = {}
    for sec in sections:
        bucket = sec.get("bucket") or "other"
        if bucket == "references":
            continue
        text = sec.get("text") or ""
        for rx, rep in _REPAIRS:
            text = rx.sub(rep, text)
        # footnotes run together on the PDF route ("…here: URL 4URL 5URL"):
        # a footnote number glued to a URL starts a new sentence, so one
        # availability phrase cannot vouch for three repositories
        text = re.sub(r"\s(\d{1,2})(?=https?://)", r". \1 ", text)
        for m in _URL.finditer(text):
            url = _clean_url(m.group(0))
            host = m.group(1)
            path = url[len(m.group(0)) - len(m.group(2) or ""):] if m.group(2) else ""
            # recompute path from the cleaned url
            path = re.sub(r"^https?://(?:www\.)?[^/]+", "", url)
            cl = _classify_url(host, path)
            if cl is None:
                continue
            kind_host, repo = cl
            sent = _sentence_at(text, m.start(), m.end())
            first_page = bucket == "abstract"
            avail = bool(_AVAIL.search(sent))
            ours = bool(_OURS.search(sent))
            named = _slug(repo.split("/")[-1]) in props or any(
                p in _slug(repo) for p in props if len(p) >= 6)
            theirs = bool(_THEIRS.search(sent)) and not ours
            words = len(re.sub(r"https?://\S+", "", sent).split())
            footnote = bool(_FOOTNOTE.search(sent)) or words <= 3
            # first page alone is a weak signal on the PDF route (page-6
            # footnotes land in the same bucket): it needs a footnote shape
            is_own = (avail and not theirs) or ours or (named and not theirs) or (first_page and footnote)
            if kind_host == "hfd" or kind_host in ("zenodo", "osf"):
                kind = "data"
            elif kind_host == "hfm":
                kind = "model"
            elif kind_host in ("hfs", "page"):
                kind = "page"
            elif kind_host == "hfc":
                kind = "model"
            else:  # github / gitlab: code unless the sentence talks only of data
                if _DATAWORD.search(sent) and not _CODEWORD.search(sent) and not _MODELWORD.search(sent):
                    kind = "data"
                else:
                    kind = "code"
            k = (kind_host, repo)
            item = {"u": url, "r": repo, "h": kind_host, "k": kind, "s": sent, "b": bucket,
                    "w": ("p" if first_page else "") + ("a" if avail else "") +
                         ("o" if ours else "") + ("n" if named else "") +
                         ("l" if _LATER.search(sent) else "")}
            if is_own:
                cur = own.get(k)
                # prefer the first-page / availability sentence as evidence
                if cur is None or (len(item["w"]) > len(cur["w"])):
                    own[k] = item
                mention.pop(k, None)
            elif k not in own and k not in mention:
                mention[k] = item
    out: dict[str, list] = {"code": [], "data": [], "model": [], "page": []}
    for it in own.values():
        out[it["k"]].append(it)
    for kind in out:
        # first-page links first, then by evidence strength; stable otherwise
        out[kind].sort(key=lambda it: (-len(it["w"]), it["b"] != "abstract"))
    caps = {"code": 4, "data": 5, "model": 4, "page": 2}
    out = {k: v[:caps[k]] for k, v in out.items()}
    out["mentions"] = [[it["h"], it["r"]] for it in list(mention.values())[:25]]
    return out


def _bridge(spec: tuple[str, str | None]) -> dict[str, int] | None:
    if spec[1] is None:
        return None
    to_eid: dict[str, int] = {}
    for r in read_jsonl(RAW / "arxiv" / spec[1]):
        if r.get("arxiv_base"):
            to_eid.setdefault(r["arxiv_base"], r["event_id"])
    return to_eid


def build() -> dict:
    from .site import gid_map   # the one legal gid bridge
    corpora = available()
    GID, _ = gid_map(corpora)
    papers: dict[int, dict] = {}
    coverage: dict[str, dict] = {}
    ft_all: list[int] = []      # every gid with full text on disk: the filter's denominator
    for c in corpora:
        spec = FULLTEXT.get(c.key)
        n_papers = sum(1 for _ in c.read_papers())
        cov = {"papers": n_papers, "fulltext": 0, "code": 0, "data": 0, "model": 0, "page": 0}
        coverage[c.key] = cov
        if not spec or not (INTERIM / spec[0]).exists():
            print(f"  {c.key}: no full text on disk", file=sys.stderr)
            continue
        # proposed names: full-text facts lead, abstract facts fill
        proposed: dict[int, list[str]] = defaultdict(list)
        for path in (c.facts_fulltext, c.facts):
            if path.exists():
                for r in read_jsonl(path):
                    if not r.get("ok"):
                        continue
                    for m in (r["facts"].get("methods") or []):
                        if m.get("role") == "proposed" and m.get("name"):
                            proposed[r["event_id"]].append(m["name"])
        to_eid = _bridge(spec)
        seen: set[int] = set()
        for row in read_jsonl(INTERIM / spec[0]):
            if not row.get("ok") or not row.get("sections"):
                continue
            eid = row.get("event_id") if to_eid is None else to_eid.get(row.get("arxiv_base"))
            if eid is None or eid in seen:
                continue
            gid = GID.get((c.key, eid))
            if gid is None:
                continue
            seen.add(eid)
            cov["fulltext"] += 1
            ft_all.append(gid)
            res = extract_paper(row["sections"], proposed.get(eid, []))
            for kind in ("code", "data", "model", "page"):
                if res[kind]:
                    cov[kind] += 1
            if any(res[k] for k in ("code", "data", "model", "page")) or res["mentions"]:
                papers[gid] = res
        print(f"  {c.key}: full text {cov['fulltext']}/{n_papers}  code {cov['code']}  "
              f"data {cov['data']}  model {cov['model']}  page {cov['page']}")
    return {
        "note": "Links printed in the papers themselves, cut with the sentence that "
                "states them (w: p=first page, a=availability phrase, o='our', "
                "n=repo named after the proposed method, l=promised for later). "
                "Coverage is full-text coverage per corpus; a paper without a "
                "preprint shows no link, which is not evidence it has none. "
                "Metadata (m) is fetched from GitHub / the Hub and dated; it is "
                "shown as a fact and never used to sort.",
        "built": date.today().isoformat(),
        "coverage": coverage,
        "fulltext_gids": sorted(ft_all),
        "papers": {str(g): v for g, v in sorted(papers.items())},
    }


# ------------------------------------------------------------------ resolution

def _gh_token() -> str | None:
    tok = os.environ.get("GITHUB_TOKEN")
    if tok:
        return tok
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        tok = out.stdout.strip()
        return tok or None
    except (OSError, subprocess.SubprocessError):
        return None


class RateLimited(Exception):
    pass


# Python 3.13+ turns on VERIFY_X509_STRICT, which rejects api.github.com's
# chain for a missing Authority Key Identifier; the chain is still verified
# against the system store without the strict flag (curl accepts it).
_SSL = ssl.create_default_context()
_SSL.verify_flags &= ~getattr(ssl, "VERIFY_X509_STRICT", 0)


def _get(url: str, headers: dict) -> tuple[int, dict | None]:
    req = urllib.request.Request(url, headers={"User-Agent": "bellwether/0.1", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30, context=_SSL) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        if e.code in (403, 429):
            # GitHub reports its limit in the body/headers; a limit is not a fact about the repo
            remaining = e.headers.get("X-RateLimit-Remaining")
            if e.code == 429 or remaining == "0" or "rate limit" in (e.read()[:300].decode("utf-8", "ignore")).lower():
                raise RateLimited(url)
            return e.code, None
        return e.code, None
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise RateLimited(f"network: {e}")   # transient: never recorded
    except (UnicodeError, ValueError):
        return 0, None                        # malformed id: recorded as an error, not retried


def fetch_meta(kind: str, repo: str, gh_headers: dict) -> dict | None:
    """One repo's public facts. None = transient failure (not recorded)."""
    today = date.today().isoformat()
    if kind == "gh":
        st, d = _get(f"https://api.github.com/repos/{repo}", gh_headers)
        if st == 404 or (st == 451):
            return {"gone": True, "at": today}
        if d is None:
            return {"err": st, "at": today} if st < 500 else None
        return {"stars": d.get("stargazers_count"), "pushed": (d.get("pushed_at") or "")[:10],
                "license": (d.get("license") or {}).get("spdx_id"),
                "archived": bool(d.get("archived")), "fork": bool(d.get("fork")),
                "name": d.get("full_name"), "at": today}
    if kind in ("hfd", "hfm", "hfs"):
        path = {"hfd": "datasets", "hfm": "models", "hfs": "spaces"}[kind]
        st, d = _get(f"https://huggingface.co/api/{path}/{repo}", {})
        if st in (404, 401):     # 401 = gated/private to anonymous callers
            return {"gone": True, "at": today} if st == 404 else {"gated": True, "at": today}
        if d is None:
            return {"err": st, "at": today} if st < 500 else None
        return {"downloads": d.get("downloads"), "likes": d.get("likes"),
                "modified": (d.get("lastModified") or "")[:10], "gated": bool(d.get("gated")),
                "name": d.get("id"), "at": today}
    return {"skip": True, "at": today}


def _load_cache() -> dict[str, dict]:
    cache: dict[str, dict] = {}
    if CACHE.exists():
        for r in read_jsonl(CACHE):
            cache[r["k"]] = r["m"]
    return cache


def resolve(keys: list[tuple[str, str]], workers: int = 6, max_age_days: int = 60) -> dict[str, dict]:
    cache = _load_cache()
    today = date.today()
    todo = []
    for kind, repo in keys:
        k = f"{kind}:{repo}"
        m = cache.get(k)
        if m and (today - date.fromisoformat(m.get("at", "2000-01-01"))).days <= max_age_days:
            continue
        todo.append((k, kind, repo))
    tok = _gh_token()
    gh_headers = {"Authorization": f"Bearer {tok}", "Accept": "application/vnd.github+json"} if tok else {}
    if not tok:
        print("  no GitHub token (gh auth login, or GITHUB_TOKEN): 60 requests/hour", file=sys.stderr)
    print(f"  {len(keys)} repos, {len(todo)} to fetch, {len(cache)} cached")
    done = 0
    # The Hub has no hourly cap and GitHub does (5,000 with a token), so the
    # hosts run as separate batches: a GitHub limit must not strand the Hub
    # lookups, and once a host is limited its remaining lookups are skipped
    # rather than each failing in turn.
    import threading
    batches = [[t for t in todo if t[1] != "gh"], [t for t in todo if t[1] == "gh"]]
    with open(CACHE, "a", encoding="utf-8") as fh:
        for batch in batches:
            if not batch:
                continue
            limited = threading.Event()

            def _one(kind, repo):
                if limited.is_set():
                    return None
                return fetch_meta(kind, repo, gh_headers)

            with ThreadPoolExecutor(workers) as ex:
                futs = {ex.submit(_one, kind, repo): (k, kind, repo) for k, kind, repo in batch}
                for fut in as_completed(futs):
                    k, kind, repo = futs[fut]
                    try:
                        m = fut.result()
                    except RateLimited as e:
                        if not limited.is_set():
                            print(f"  {kind}: stopped at {e} — rerun later, the cache resumes",
                                  file=sys.stderr)
                        limited.set()
                        continue
                    except Exception as e:      # noqa: BLE001 — one repo, not the run
                        print(f"  skip {k}: {type(e).__name__}: {e}", file=sys.stderr)
                        continue
                    if m is None:
                        continue
                    cache[k] = m
                    fh.write(json.dumps({"k": k, "m": m}, ensure_ascii=False) + "\n")
                    fh.flush()
                    done += 1
                    if done % 500 == 0:
                        print(f"  {done}/{len(todo)}")
    print(f"  fetched {done}, {len(todo) - done} left for a later run")
    return cache


def attach_meta(doc: dict, cache: dict[str, dict]) -> int:
    n = 0
    for res in doc["papers"].values():
        for kind in ("code", "data", "model", "page"):
            for it in res.get(kind, []):
                m = cache.get(f"{it['h']}:{it['r']}")
                if m:
                    it["m"] = {k: v for k, v in m.items() if k != "at"} | {"at": m.get("at")}
                    n += 1
    return n


# ------------------------------------------------------------------ registry

def load_registry() -> dict:
    """config/benchmarks.json -> {fold key: entry}. Entries keep their hand-
    written fields; `ok` is what the last API check said about each id."""
    reg = load_json(REGISTRY)
    by_key: dict[str, dict] = {}
    for e in reg["benchmarks"]:
        for surface in [e["name"], *e.get("aliases", [])]:
            k = dataset_key(surface)
            if k:
                by_key.setdefault(k, e)
    return {"entries": reg["benchmarks"], "by_key": by_key,
            "not_a_dataset": {dataset_key(s) for s in reg.get("not_a_dataset", [])}}


def check_registry(workers: int = 6) -> None:
    reg = load_json(REGISTRY)
    keys = []
    for e in reg["benchmarks"]:
        if e.get("hf"):
            keys.append(("hfd", e["hf"]))
        if e.get("gh"):
            keys.append(("gh", e["gh"]))
    cache = resolve(keys, workers=workers, max_age_days=30)
    bad = 0
    for e in reg["benchmarks"]:
        ok = {}
        for field, kind in (("hf", "hfd"), ("gh", "gh")):
            if e.get(field):
                m = cache.get(f"{kind}:{e[field]}")
                ok[field] = bool(m) and not m.get("gone") and not m.get("err")
                if m and m.get("name") and m["name"].lower() != e[field].lower() and not m.get("gone"):
                    print(f"  {e['name']}: {field} {e[field]} -> renamed {m['name']}")
                if not ok[field]:
                    bad += 1
                    print(f"  {e['name']}: {field} {e[field]} NOT FOUND ({m})")
        e["ok"] = ok
    reg["checked"] = date.today().isoformat()
    dump_json(REGISTRY, reg)
    print(f"  registry: {len(reg['benchmarks'])} entries, {bad} ids failed the check "
          f"(kept as names, no link)")


# ------------------------------------------------------------------ cli

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--resolve", action="store_true", help="fetch GitHub / Hub metadata for own links")
    ap.add_argument("--registry", action="store_true", help="check config/benchmarks.json ids")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.registry:
        check_registry(a.workers)
        return 0
    if a.resolve:
        doc = load_json(OUT)
        keys = sorted({(it["h"], it["r"]) for res in doc["papers"].values()
                       for kind in ("code", "data", "model") for it in res.get(kind, [])
                       if it["h"] in ("gh", "hfd", "hfm")})
        cache = resolve(keys, workers=a.workers)
        n = attach_meta(doc, cache)
        doc["resolved"] = date.today().isoformat()
        dump_json(OUT, doc)
        print(f"  attached metadata to {n} links -> {OUT}")
        return 0
    doc = build()
    if OUT.exists():        # keep metadata already fetched
        old = load_json(OUT)
        cache = _load_cache()
        attach_meta(doc, cache)
        doc["resolved"] = old.get("resolved")
    dump_json(OUT, doc)
    tot = doc["coverage"]
    print(f"  -> {OUT}  ({len(doc['papers'])} papers with links; "
          f"code {sum(c['code'] for c in tot.values())} across "
          f"{sum(c['fulltext'] for c in tot.values())} full texts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
