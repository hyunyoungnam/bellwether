"""Build the search-first interface — a single self-contained HTML file.

Serves the three questions this project exists for:

  1. "Which papers are in my field?"  -> abstract-wide search, the shared topic
     vocabulary, benchmarks, and "papers like this one".
  2. "What is in the set I picked?"   -> counts and a share of the corpus.
  3. "What is new in this paper?"     -> the paper's OWN sentence, verified
     verbatim upstream, marked by role with a highlighter.

Design rules that come from CLAUDE.md and from what failed before:

* **Nothing is listed until the reader asks.** 6,637 rows on arrival is the
  problem this exists to remove.
* **Said-so and looks-like are never merged.** A topic chip separates papers that
  named it from those that named something narrower; search separates literal
  matches from papers that are only close in meaning.
* **Only verified spans are shown as the paper's words.** Rendering is allowed
  (TeX -> glyphs, opening discourse frames cut) but nothing is ever written.

Vocabulary is interned — method and dataset names repeat heavily across 6,592
papers, and inlining them as strings roughly doubled the payload.

    .venv/bin/python -m icml.site --spans abstract --dist
Output: reports/index.html (+ dist/ when --dist)
"""
from __future__ import annotations

import argparse
import json
import re as _re
from collections import Counter, defaultdict
from pathlib import Path

from .common import INTERIM, PROCESSED, REPORTS, ROOT, load_json, read_jsonl

UNION = PROCESSED / "union.json"
EMBED = PROCESSED / "embed_union.json"
NEIGHBORS = PROCESSED / "neighbors_union.json"
CONTACTS = PROCESSED / "contacts.json"

VENUE = "ICML"          # override with --venue when another conference is added
MAX_SPANS = 2
SPAN_CHARS = 320

# Search used to see only titles and extracted names — 20% of the corpus text, so
# "autonomous driving" returned 57 papers when the topic held 102. The abstract is
# shipped as a salient-term index rather than raw prose: same word-AND semantics,
# 100% coverage, and 1.0 MB gzipped instead of 2.8 MB.
_WORD = _re.compile(r"[a-z][a-z0-9\-]{2,}")
TERM_DF_MIN = 2        # a term in one paper cannot connect it to anything
TERM_DF_MAX = 0.15     # a term in a sixth of the corpus does not discriminate

SEARCH_STOP = set("""the of and to in a for we is are that with on this by as an be our it
can from which at or not but have has been more most their its using use used show shows
shown propose proposed present presents new novel results result method methods approach
model models learning data paper work also than such these those when where while however
thus each other both between into over under only very much many any all some one two
three does do done make makes made may might will would could should because since if then
so no nor own same too here there about after before during through against you your""".split())

# ---------------------------------------------------------------- TeX rendering
# Authors write raw TeX in the submission form, and it survives into abstracts,
# extracted names and novelty spans alike — 2.2% of spans carry some. Rendering
# `ME$^2$` as `ME²` is display, not rewriting: the characters the author typed are
# shown as the glyphs they typed them for. Verification against the source already
# happened upstream, on the raw text, so this cannot weaken the extractive
# guarantee. Nothing here paraphrases.
_SUP = str.maketrans("0123456789+-=()/naibcdefghjklmoprstuvwxyz",
                     "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼⁽⁾⁄ⁿᵃⁱᵇᶜᵈᵉᶠᵍʰʲᵏˡᵐᵒᵖʳˢᵗᵘᵛʷˣʸᶻ")
_SUB = str.maketrans("0123456789+-=()aehijklmnoprstuvx",
                     "₀₁₂₃₄₅₆₇₈₉₊₋₌₍₎ₐₑₕᵢⱼₖₗₘₙₒₚᵣₛₜᵤᵥₓ")
_GREEK = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ", "epsilon": "ε",
    "zeta": "ζ", "eta": "η", "theta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π", "rho": "ρ",
    "sigma": "σ", "tau": "τ", "upsilon": "υ", "phi": "φ", "chi": "χ",
    "psi": "ψ", "omega": "ω", "Gamma": "Γ", "Delta": "Δ", "Theta": "Θ",
    "Lambda": "Λ", "Xi": "Ξ", "Pi": "Π", "Sigma": "Σ", "Phi": "Φ",
    "Psi": "Ψ", "Omega": "Ω", "times": "×", "cdot": "·", "leq": "≤",
    "geq": "≥", "neq": "≠", "approx": "≈", "sim": "~", "pm": "±",
    "infty": "∞", "rightarrow": "→", "to": "→", "ell": "ℓ",
}
_UNWRAP = _re.compile(
    r"\\(?:texttt|textit|textbf|textsc|textrm|emph|text|mathrm|mathbf|mathit|"
    r"mathcal|mathbb|mathsf|mathtt|operatorname|boldsymbol|bm)\s*\{([^{}]*)\}")
_DROP = _re.compile(r"\\(?:cite[tp]?|ref|label|footnote)\s*\{[^{}]*\}")


def _script(body: str, table: dict, fallback: str) -> str:
    """Unicode super/subscript when every character maps, else a readable ^(...)."""
    out = body.translate(table)
    if any(ch in body for ch in "\\{}"):
        return f"{fallback}({body})"
    if out != body:
        return out
    return f"{fallback}({body})" if body else body


# PDF extraction breaks words across lines and leaves the hyphen behind. Two
# different repairs are needed and telling them apart matters:
#   "condition- ing"  -> conditioning   (a word was split; the hyphen is an artifact)
#   "token- level"    -> token-level    (a real compound; only the space is wrong)
# The split is decided by whether the right-hand fragment is a word ending rather
# than a word. Getting it wrong is visible either way — "tokenlevel" or
# "condition-ing" — so the suffix list is deliberately short and common.
_SUFFIXES = ("ing", "tion", "sion", "ment", "ness", "ity", "ally", "ized", "ised",
             "ance", "ence", "able", "ible", "ers", "ies", "ted", "ual", "ary",
             "ive", "ous", "ful", "less", "ward", "wise", "ation", "ical",
             "ization", "isation", "ational", "ingly", "ability")
_PDF_HYPHEN = _re.compile(r"(\w)-\s+([a-z]\w*)")


def _dehyphen(m):
    left, right = m.group(1), m.group(2)
    joined = right.startswith(_SUFFIXES) and len(right) <= 8
    return left + ("" if joined else "-") + right


def detex(t: str) -> str:
    """Render author TeX as text. Display-only; never used for matching or counts."""
    s = _PDF_HYPHEN.sub(_dehyphen, t or "")
    s = _DROP.sub("", s)
    for _ in range(3):                       # nested \textbf{\texttt{x}}
        s2 = _UNWRAP.sub(r"\1", s)
        if s2 == s:
            break
        s = s2
    # word-boundary so "\alpha + x" keeps its spacing and "\alphabet" is untouched
    s = _re.sub(r"\\(" + "|".join(sorted(_GREEK, key=len, reverse=True)) + r")(?![a-zA-Z])",
                lambda m: _GREEK[m.group(1)], s)
    s = _re.sub(r"\^\{([^{}]*)\}", lambda m: _script(m.group(1), _SUP, "^"), s)
    s = _re.sub(r"_\{([^{}]*)\}", lambda m: _script(m.group(1), _SUB, "_"), s)
    s = _re.sub(r"\^(\w)", lambda m: _script(m.group(1), _SUP, "^"), s)
    s = _re.sub(r"_(\w)", lambda m: _script(m.group(1), _SUB, "_"), s)
    s = s.replace("\\$", "\x00")             # an escaped dollar is a real dollar sign
    s = s.replace("$", "")                   # math delimiters carry no meaning here
    s = s.replace("\x00", "$")
    s = _re.sub(r"\\([%&#_])", r"\1", s)     # escaped punctuation
    s = _re.sub(r"\\[a-zA-Z]+", "", s)       # any macro we do not know
    s = s.replace("{", "").replace("}", "").replace("\\", "")
    return _re.sub(r"\s+", " ", s).strip()


# Deletion-only editing: characters may be removed, never added or changed. What
# is left is still the paper's sentence, so the extractive guarantee survives —
# checked below as an ordered subsequence rather than a substring.
#
# Only the opening discourse frame is cut. Trailing "which/where/thereby" clauses
# were tried and rejected: measured over 15,256 spans, that cut usually removed
# the MECHANISM ("...which learns isometric embeddings by recovering the intrinsic
# surface geodesic metric"), which is the one thing the card exists to show.
_LEAD = _re.compile(
    r"^\s*(?:in this (?:paper|work|study|article)|in particular|specifically|"
    r"to this end|to that end|moreover|furthermore|additionally|in addition|"
    r"notably|importantly|as a result|consequently|therefore|thus|hence|"
    r"building on (?:this|these)|motivated by (?:this|these)|based on (?:this|these)|"
    r"inspired by (?:this|these)|to address (?:this|these)(?: \w+){0,2}|"
    r"to overcome (?:this|these)(?: \w+){0,2}|to (?:fill|bridge) (?:this|the) gap|"
    r"in response|finally|first|second|third|to do so|as such|in contrast|however)"
    r"\s*,\s*", _re.I)


# "We introduce a framework that ..." reads as a scraped sentence. Cutting the
# announcement leaves "a framework that ...", which reads as an edited one. Still
# pure deletion: the remaining characters occur in the source, in order.
_ANNOUNCE = _re.compile(
    r"^\s*(?:we|this (?:paper|work|study)|the (?:authors|paper)|here\s+we)\s+"
    r"(?:therefore\s+|thus\s+|further\s+|also\s+|first\s+|then\s+|hereby\s+)?"
    r"(?:propose|present|introduce|develop|design|construct|build|offer|provide|"
    r"describe|formulate|derive|establish|demonstrate|show|prove|argue|"
    r"stud(?:y|ies)|investigate|explore|examine|analys(?:e|es)|analyz(?:e|es)|"
    r"revisit|extend|generalis(?:e|es)|generaliz(?:e|es))e?s?\b"
    r"\s*(?:that\s+|a\s+new\s+|an\s+new\s+)?", _re.I)


def elide(t: str) -> str:
    """Cut the opening discourse frame and the announcement. Never writes."""
    s = (t or "").strip()
    m = _LEAD.match(s)
    if m:
        s = s[m.end():].strip()
    m = _ANNOUNCE.match(s)
    if m and len(s) - m.end() > 30:      # never leave a stub
        s = s[m.end():].strip()
    return s


def is_subsequence(short: str, source: str) -> bool:
    """Every retained character occurs in the source, in order."""
    a = _re.sub(r"\s+", " ", short).strip().lower()
    b = _re.sub(r"\s+", " ", source).strip().lower()
    i = 0
    for ch in a:
        i = b.find(ch, i)
        if i < 0:
            return False
        i += 1
    return True


def edit_span(t: str) -> str:
    """detex + elide, with the guarantee checked rather than assumed."""
    rendered = detex(t)
    short = elide(rendered)
    return short if is_subsequence(short, rendered) else rendered


def clean_title(t: str) -> str:
    """Kept for callers that ask for a title specifically."""
    return detex(t)


class Vocab:
    """Intern strings to integer ids so repeated names cost 2-3 bytes, not 20."""

    def __init__(self):
        self.items: list[str] = []
        self._ix: dict[str, int] = {}

    def id(self, s: str) -> int:
        i = self._ix.get(s)
        if i is None:
            i = len(self.items)
            self._ix[s] = i
            self.items.append(s)
        return i


def build_term_index(abstracts: list) -> dict:
    """Salient abstract vocabulary per paper, interned.

    Terms too rare to link two papers, or too common to separate them, are cut —
    that is what makes the index a third of the size of the prose it replaces.
    """
    docs, df = [], Counter()
    for text in abstracts:
        t = {w for w in _WORD.findall((text or "").lower()) if w not in SEARCH_STOP}
        docs.append(t)
        df.update(t)
    n = max(len(docs), 1)
    keep = {w for w, c in df.items() if TERM_DF_MIN <= c < TERM_DF_MAX * n}
    vocab = sorted(keep)
    ix = {w: i for i, w in enumerate(vocab)}
    return {"v": vocab, "p": [sorted(ix[w] for w in d if w in keep) for d in docs]}


def gid_map(corpora) -> tuple[dict[tuple[str, int], int], int]:
    """(corpus key, event_id) -> gid, the one global paper key.

    `event_id` is the conference feed's own id and collides across tracks and
    venues; ICML 2025 and 2026 not overlapping is luck. gids 0..n_emb-1 come from
    `union.json` and index the embedding and neighbour matrices directly. Papers
    with no abstract are not in the union — they get gids after it, so they still
    appear and are simply shown as having no neighbours, never dropped.
    """
    g: dict[tuple[str, int], int] = {}
    if UNION.exists():
        u = load_json(UNION)
        for i, (k, e) in enumerate(zip(u["keys"], u["eids"])):
            g[(k, e)] = i
    n_emb = len(g)
    nxt = n_emb
    for c in corpora:
        for p in c.read_papers():
            if (c.key, p["event_id"]) not in g:
                g[(c.key, p["event_id"])] = nxt
                nxt += 1
    return g, n_emb


def build_payload(span_source: str) -> dict:
    from .corpus import available
    corpora = available(VENUE)
    if not corpora:
        raise SystemExit("no corpus has both papers and an extraction")
    ci = {c.key: i for i, c in enumerate(corpora)}
    GID, N_EMB = gid_map(corpora)

    # Every paper across every active corpus, keyed by gid. The corpus a paper
    # came from travels with it — nothing downstream may assume one conference.
    papers: dict[int, dict] = {}
    for c in corpora:
        for p in c.read_papers():
            g = GID[(c.key, p["event_id"])]
            papers[g] = {**p, "_ck": c.key, "_ci": ci[c.key]}

    # Prefer full-text facts where available: intro sections state contributions
    # more specifically than abstracts. Fall back to the census pass.
    facts: dict[int, dict] = {}
    src_of: dict[int, str] = {}
    for c in corpora:
        for r in read_jsonl(c.facts):
            if r.get("ok"):
                g = GID.get((c.key, r["event_id"]))
                if g is None:
                    continue
                facts[g] = r["facts"]
                src_of[g] = "abstract"
    # Field-by-field, not record-by-record. Measured on the 2026 re-run: the
    # abstract pass fills `limitation` for 90% of papers and the full-text pass
    # for 59% — PDF text is dirty enough that 40.7% of its "verbatim" quotes fail
    # verification. So the abstract leads, and full text only fills the gaps it
    # leaves; that union reaches 94%.
    # Full-text source differs by corpus — arXiv preprints for the focus year
    # (71.9%, biased by subfield), the PMLR camera-ready for published years
    # (99.3%). Coverage per corpus is disclosed in `corpora[].full`, and nothing
    # sorts, scores or filters on it.
    n_full = 0
    focus = next((c for c in corpora if c.is_focus), corpora[-1])
    if span_source == "fulltext":
        FILLABLE = ("limitation", "key_change", "result_claim", "novelty_spans",
                    "datasets", "methods", "tasks", "domain")
        for c in corpora:
            if not c.facts_fulltext.exists():
                continue
            for r in read_jsonl(c.facts_fulltext):
                if not r.get("ok"):
                    continue
                g = GID.get((c.key, r["event_id"]))
                if g is None:
                    continue
                base = facts.get(g)
                if base is None:
                    facts[g] = r["facts"]
                    src_of[g] = "fulltext"
                    n_full += 1
                    continue
                used = False
                for k in FILLABLE:
                    if not base.get(k) and r["facts"].get(k):
                        base[k] = r["facts"][k]
                        used = True
                if used:
                    src_of[g] = "fulltext"
                    n_full += 1

    # The shared vocabulary from icml.taxonomy, not a per-run derivation. Its two
    # membership kinds are kept apart all the way to the screen: the paper said
    # this, or it said something narrower that rolls up to this.
    # One label list for every corpus, because the vocabulary is frozen in
    # config/taxonomy.json and applied per corpus. Verified at build time: if the
    # lists ever diverge, a topic index would mean two different things and every
    # cross-corpus count would be silently wrong.
    tdocs = {c.key: load_json(c.topics) for c in corpora}
    labels = [[t["label"] for t in d["topics"]] for d in tdocs.values()]
    if any(l != labels[0] for l in labels):
        raise SystemExit("topic vocabularies differ between corpora — re-run "
                         "`taxonomy label --all` so every corpus shares one list")
    base_topics = tdocs[corpora[0].key]["topics"]
    by_label = {t["label"]: i for i, t in enumerate(base_topics)}
    topic_of: dict[int, list[int]] = defaultdict(list)
    topic_kind: dict[tuple[int, int], int] = {}
    topics_out = []
    for ti in range(len(base_topics)):
        n = e = 0
        for c in corpora:
            t = tdocs[c.key]["topics"][ti]
            n += t["n_explicit"]
            e += t["n_via_child"]
            for pid in t["explicit"]:
                g = GID.get((c.key, pid))
                if g is not None:
                    topic_of[g].append(ti)
                    topic_kind[(g, ti)] = 1
            for pid in t["via_child"]:
                g = GID.get((c.key, pid))
                if g is not None:
                    topic_of[g].append(ti)
                    topic_kind.setdefault((g, ti), 2)
        t0 = base_topics[ti]
        topics_out.append({
            "l": t0["label"],
            "n": n,                        # the paper named this topic
            "e": e,                        # it named something narrower
            "u": by_label.get(t0.get("nested_under")) if t0.get("nested_under") else None,
            "f": t0.get("facet", 0),       # 1 = a domain the work is applied to
            "F": t0.get("family"),         # family: curated for domains, rules for tasks
            "j": 1 if t0.get("junk") else 0,
        })

    # Keyed by corpus, then event_id (see icml.contacts). The old flat shape
    # (focus year only) is still read so a stale contacts.json degrades to
    # focus-year coverage instead of crashing the build.
    craw = (load_json(CONTACTS)["contacts"] if CONTACTS.exists() else {})
    contacts: dict[int, list] = {}
    if craw and all(isinstance(v, dict) for v in craw.values()):
        for ck, per in craw.items():
            for k, v in per.items():
                g = GID.get((ck, int(k)))
                if g is not None:
                    contacts[g] = v
    else:
        contacts = {GID[(focus.key, int(k))]: v for k, v in craw.items()
                    if (focus.key, int(k)) in GID}

    meth, data, task = Vocab(), Vocab(), Vocab()
    rows = []
    order = sorted(papers.values(), key=lambda p: (not p.get("is_oral"),
                                                   not p.get("is_spotlight"),
                                                   p["title"]))
    for p in order:
        eid = GID[(p["_ck"], p["event_id"])]
        f = facts.get(eid) or {}
        spans = [edit_span(s)[:SPAN_CHARS] for s in (f.get("novelty_spans") or [])[:MAX_SPANS]]
        methods = f.get("methods") or []
        tl = sorted(set(topic_of.get(eid, ())))
        rows.append({
            "i": eid,                      # gid: the global key, see gid_map()
            "cy": p["_ci"],                # which corpus this paper came from
            "t": clean_title(p["title"]),
            "a": p.get("area") or "",
            "b": p.get("subarea") or "",
            # Three tiers, mutually exclusive, and this is a normalisation, not
            # the feed's own words: the two years use different decision
            # vocabularies. 2025 writes poster / spotlight poster / oral as
            # parallel labels; 2026 writes regular / spotlight and picks its 168
            # orals FROM the spotlights, so its feed marks every oral a
            # spotlight. Summing the raw flags across years therefore adds two
            # different quantities. Read as tiers — regular < spotlight < oral —
            # both years mean the same thing, so `o` is the tier and the raw
            # overlap is disclosed in the Selection hint rather than summed.
            "o": 1 if p.get("is_oral") else (2 if p.get("is_spotlight") else 0),
            "c": f.get("contribution_type") or "",
            "d": f.get("domain") or "",
            "p": [meth.id(detex(m["name"])) for m in methods if m.get("role") == "proposed"][:4],
            "u": [meth.id(detex(m["name"])) for m in methods if m.get("role") == "building-block"][:5],
            "v": [meth.id(detex(m["name"])) for m in methods if m.get("role") == "baseline"][:4],
            "k": [data.id(detex(x["name"])) for x in (f.get("datasets") or [])][:5],
            "s": [task.id(detex(x["name"])) for x in (f.get("tasks") or [])][:3],
            "n": spans,
            "L": edit_span(f.get("limitation") or "")[:SPAN_CHARS],
            "K": edit_span(f.get("key_change") or "")[:SPAN_CHARS],
            "R": edit_span(f.get("result_claim") or "")[:SPAN_CHARS],
            "g": [[ti, topic_kind.get((eid, ti), 0)] for ti in tl],
            "w": p.get("virtual_url") or p.get("paper_url") or "",
            # From the paper's own "Correspondence to:" line, or absent.
            "c1": contacts.get(str(eid)) or contacts.get(eid) or None,
            "au": (p.get("authors") or [])[:1],
            "na": p.get("n_authors") or len(p.get("authors") or []),
            "f": 1 if src_of.get(eid) == "fulltext" else 0,
        })

    nb = load_json(NEIGHBORS) if NEIGHBORS.exists() else None
    with_span = sum(1 for r in rows if r["n"])
    tagged = sum(1 for r in rows if any(k in (1, 2) for _, k in r["g"]))

    # Parallel to `rows` by construction — keyed lookups here died once already,
    # when the key changed from event_id to gid and every lookup silently missed.
    terms = build_term_index([p.get("abstract") for p in order])

    # Datasets a reader can enter by. Placeholders ("three datasets") are counts
    # wearing a name and outrank every real dataset if left in.
    from .taxonomy import (is_placeholder, building_blocks, method_key,
                           method_family, METHOD_FAMILIES)
    ds_count: dict[int, int] = defaultdict(int)
    for r in rows:
        for di in set(r["k"]):
            ds_count[di] += 1
    ds_entry = sorted(((c, di) for di, c in ds_count.items()
                       if c >= 3 and not is_placeholder(data.items[di])), reverse=True)

    # Building blocks, normalised: "reinforcement learning" and "reinforcement
    # learning (RL)" were 148 and 97 until the acronym gloss was folded.
    mrows, mids, mdisp, mparent = building_blocks(facts, min_papers=3)
    mvocab = Vocab()
    mid_of = {k: mvocab.id(detex(mdisp.get(k, k))) for _, k in mrows}
    for r in rows:
        f = facts.get(r["i"]) or {}
        got = set()
        for m in f.get("methods") or []:
            if m.get("role") != "building-block":
                continue
            key, _ = method_key(m["name"])
            if key in mid_of:
                got.add(mid_of[key])
        r["mu"] = sorted(got)
    reach_method = len({e for _, k in mrows for e in mids[k]})
    # The same word can be a subject and a tool. Measured: "reinforcement
    # learning" is 299 papers as a topic and 245 as a method, and only 17 are in
    # both — one set studies it, the other uses it. Neither row is redundant, but
    # a reader who sees two different numbers under one word deserves to be told
    # why, so the pairs are shipped and surfaced when one is picked.
    from .taxonomy import canon as _canon
    mby = {_canon(mdisp.get(k, k)): (mid_of[k], n) for n, k in mrows}
    both = []
    for ti, t in enumerate(topics_out):
        hit = mby.get(_canon(t["l"]))
        if hit:
            both.append([ti, hit[0], t["n"] + t["e"], hit[1]])
    # "Other" last: it is the ungrouped tail, and sorting it by size would put the
    # least informative chip first.
    mfam_names = [n for n, _ in METHOD_FAMILIES] + ["Other"]
    mfam_ix = {n: i for i, n in enumerate(mfam_names)}

    return {
        "papers": rows,
        "vocab": {"m": meth.items, "d": data.items, "t": task.items},
        "terms": terms,
        # PCA(128) int8 of the BGE-M3 vectors, 1.1 MB. Search is lexical and
        # always will be — a browser cannot embed a query — but these let a
        # lexical hit set reach the papers that mean the same thing without
        # sharing the words. Kept as a SEPARATE, marked group, never blended in.
        "emb": load_json(EMBED) if EMBED.exists() else None,
        "datasets": [[di, c] for c, di in ds_entry[:400]],
        # id, papers, family index, parent id (null when top level)
        "methods": [[mid_of[k], n,
                     mfam_ix[method_family(mdisp.get(k, k), k)],
                     mid_of.get(mparent.get(k))] for n, k in mrows],
        "mvocab": mvocab.items,
        "mfams": mfam_names,
        "both": both,
        "n_datasets_real": sum(1 for di in ds_count if not is_placeholder(data.items[di])),
        "topics": topics_out,
        # Indexed by gid, and computed over the UNION: "more like this" has to be
        # able to return last year's paper and the other venue's paper.
        "neighbors": ({"k": nb["k"], "nbr": nb["nbr"]} if nb else None),
        "nemb": N_EMB,   # gids below this have a vector; the rest have no abstract
        # Ordered as the payload's `cy` indexes them. Sizes differ by 2x, so any
        # cross-corpus number must be a share of these, never a raw count.
        "corpora": [{"k": c.key, "v": c.venue, "y": c.year,
                     "os": sum(1 for p in c.read_papers()
                               if p.get("is_oral") and p.get("is_spotlight")),
                     "n": sum(1 for r in rows if r["cy"] == ci[c.key]),
                     "full": sum(1 for r in rows if r["cy"] == ci[c.key] and r["f"]),
                     "emb": sum(1 for r in rows if r["cy"] == ci[c.key] and r["i"] < N_EMB)}
                    for c in corpora],
        "coverage": {
            "total": len(rows),
            "with_span": with_span,
            "tagged": tagged,
            "from_fulltext": n_full,
            "span_source": span_source,
            # How far each facet actually reaches. Two thirds of ML papers are
            # method work with no application domain — the extractor is told to
            # write null there — so a domain menu can never cover the corpus, and
            # the screen has to say so rather than let the gap read as a bug.
            "named_domain": sum(1 for f in facts.values() if f.get("domain")),
            "reach_domain": len({g for g, tl in topic_of.items()
                                 if any(topics_out[ti]["f"] == 1 for ti in tl)}),
            "reach_topic": len({g for g, tl in topic_of.items()
                                if any(topics_out[ti]["f"] != 1 for ti in tl)}),
            "reach_method": reach_method,
            # The venue belongs in the data. Other conferences are coming, and a
            # hard-coded title would have to be edited for each of them.
            "venue": VENUE,
        },
    }


HTML = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<!-- Unlisted while this is shown to a first group: a URL people can click,
     but not one search engines index. Remove when it should be public. -->
<meta name="robots" content="noindex,nofollow">
<title>What's new in AI research</title>
<style>
:root{--bg:#eceef0;--card:#fff;--ink:#141414;--ink2:#454545;--mut:#7c7c78;
--line:#e2e4e6;--ring:#d2d5d8;--acc:#2a6fd0;--warm:#d2551f;--dim:#c6c8ca;--hi:#fff3c4;
/* one hue per venue, in VENUES order, never cycled; the pale step is the same
   hue knocked back — year is carried by depth, identity by hue */
--v0:#2a6fd0;--v1:#0c9a85;--v2:#8a5cd6}
*{box-sizing:border-box}
[hidden]{display:none!important}   /* display:flex on .facets/.pane outranks it otherwise */
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
-webkit-font-smoothing:antialiased}
/* One column for the whole page. With the sheet at page width, a 1380px band of
   controls above it left the content looking pushed to the right. */
.wrap{max-width:980px;margin:0 auto;padding:18px 20px 60px}
header{margin:14px 0 22px;text-align:center}
h1{font-size:40px;margin:0;font-weight:700;letter-spacing:-.025em;line-height:1.15;cursor:pointer}
@media(max-width:600px){h1{font-size:27px}}
.search{display:flex;gap:8px;align-items:center;margin-bottom:10px}
/* With no placeholder the box has to say "search" by itself: a magnifier and a
   border in the accent colour. Paler tints were measured and rejected — only the
   full accent clears 3:1 against BOTH the page (#eceef0) and the white field, so
   idle vs active is carried by weight and a ring, not by a lighter blue. */
.search{position:relative}
.search .mag{position:absolute;left:15px;top:50%;transform:translateY(-50%);
width:18px;height:18px;fill:none;stroke:var(--acc);stroke-width:2;
stroke-linecap:round;pointer-events:none;opacity:.8;transition:stroke .12s,opacity .12s}
input[type=search]{flex:1;font:inherit;font-size:15px;padding:12px 15px;
border-radius:10px;border:2px solid #1d5fbd;background:#fff;color:var(--ink);
transition:border-color .12s,box-shadow .12s,background .12s}
input[type=search]:hover{box-shadow:0 0 0 3px rgba(42,111,208,.10)}
input[type=search]:focus{outline:none;border-color:#0f3f8c;
background:#fafcff;box-shadow:0 0 0 4px rgba(29,95,189,.24)}
.search:focus-within .mag{opacity:1;stroke:#1a56a8}
select,button{font:inherit;font-size:12.5px;padding:7px 10px;border-radius:8px;
border:1px solid var(--ring);background:var(--card);color:var(--ink2);cursor:pointer;max-width:100%}
button:hover,select:hover{color:var(--ink)}
button.on{background:var(--acc);color:#fff;border-color:transparent}
.facets{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:10px}
/* A subarea name sets the select's intrinsic width and overflowed the viewport
   on a phone. min-width:0 lets it shrink; below 560px each control gets a row. */
.facets select{min-width:0;flex:0 1 auto;text-overflow:ellipsis}
@media(max-width:560px){.facets select{flex:1 1 100%}
  .facets button{flex:1 1 calc(50% - 4px)}}
/* Topics collapse to whole rows — a half-cut row read as a rendering bug, and a
   scroll area inside the page went unnoticed. Exact height is measured in JS. */
.sug{background:var(--card);border:1px solid var(--ring);border-radius:10px;
padding:12px 14px 10px;margin-bottom:14px}
.sughd{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);
margin:0 0 7px;display:flex;gap:8px;align-items:baseline}
.sughd span{text-transform:none;letter-spacing:0;font-size:11px;color:var(--dim)}
.sughd .reach{color:var(--mut)}
.sughd em{font-style:normal;text-transform:none;letter-spacing:0;font-size:11.5px;
color:var(--ink2);font-weight:400}
.xref{max-width:920px;margin:0 auto 10px;padding:9px 14px;border-radius:8px;
background:var(--card);border:1px dashed var(--ring);color:var(--ink2);font-size:12px;
line-height:1.5}
.xref b{font-weight:600}
.xref button{font:inherit;font-size:11.5px;padding:2px 9px;margin-left:8px;
border-radius:6px;border:1px solid var(--ring);background:#fff;color:var(--acc);cursor:pointer}
.sughd{flex-wrap:wrap}
.sughd:not(:first-child){margin-top:14px;padding-top:12px;border-top:1px solid var(--line)}
.fams{display:flex;gap:6px;flex-wrap:wrap}
.famopen{margin-top:2px}
.famopenrow{display:grid;grid-template-columns:150px 1fr;gap:6px 12px;align-items:start;
margin:8px 0 4px}
.famname{font-size:11px;color:var(--mut);padding-top:5px;line-height:1.35}
@media(max-width:640px){.famopenrow{grid-template-columns:1fr;gap:2px}}
.famchip{font:inherit;font-size:12.5px;padding:5px 12px;border-radius:7px;
border:1px solid var(--ring);background:#fff;color:#2b2b2b;cursor:pointer;
display:inline-flex;align-items:baseline;gap:7px;box-shadow:0 1px 1px rgba(0,0,0,.03)}
.famchip:hover{border-color:var(--acc);color:var(--acc)}
.famchip.open{border-color:var(--acc);color:var(--acc)}
.famchip.sel{background:var(--acc);color:#fff;border-color:transparent}
.famchip.sel .car,.famchip.sel .n,.famchip.sel .pk{color:#fff;opacity:.8}
.famchip.haspick{border-color:var(--acc);box-shadow:0 0 0 2px rgba(42,111,208,.14)}
.famchip .car{color:var(--dim);font-size:10px}
.famchip.open .car{color:var(--acc)}
.famchip .n{color:var(--mut);font-size:11px}
.famchip .pk{color:var(--acc);font-size:10.5px;font-weight:600}
.famchips{display:flex;gap:6px;flex-wrap:wrap}
.chipwrap{margin-bottom:0}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chips.clip{overflow:hidden}
.chiptog{margin-top:7px;font-size:11.5px;padding:3px 9px}
/* On the old near-white page these read as ghosts. White pill, defined border,
   real ink — they are the primary way in, so they must look pressable. */
.chip{font-size:12.5px;padding:5px 12px;border-radius:7px;border:1px solid var(--ring);
background:#fff;color:#2b2b2b;cursor:pointer;white-space:nowrap;
box-shadow:0 1px 1px rgba(0,0,0,.03)}
.chip:hover{border-color:var(--acc);color:var(--acc)}
.chip.on{background:var(--acc);color:#fff;border-color:transparent}
.chip .n{color:var(--mut);margin-left:5px;font-size:11px}
.chip .kid{color:var(--dim);margin-right:4px}
.chip .kn{color:var(--acc);margin-left:5px;font-size:10px;font-weight:600}
/* How much of the corpus is on screen. One value, one hue — a bar, not a pie:
   the topics overlap, so parts never sum to a whole and nothing may imply they do.
   The numbers are direct labels, so the bar carries no information alone. */
.meter{max-width:920px;margin:0 auto 12px}
.mrow{display:flex;align-items:baseline;gap:10px;font-size:12.5px;color:var(--ink2);
flex-wrap:wrap;margin-bottom:5px}
.mrow b{font-size:17px;color:var(--ink);font-weight:660;font-variant-numeric:tabular-nums}
.mrow .pct{color:var(--acc);font-weight:600;font-variant-numeric:tabular-nums}
.mrow .of{color:var(--mut)}
.mtrack{height:8px;background:#dfe2e5;border-radius:4px;overflow:hidden}
.mtrack i{display:block;height:100%;background:var(--acc);border-radius:0 4px 4px 0;
min-width:3px;transition:width .18s ease}
.mtrack.full i{border-radius:4px}
/* The sheet is a PAGE, so it has a page's width. A 1340px sheet holding a
   700px measure reads as a wide box with the text pushed off-centre. */
.res{display:flex;flex-direction:column;gap:16px;max-width:920px;margin:0 auto}

/* A card is a SHEET. Opening it should feel like opening the paper: white
   ground, black type, generous margins — so it reads as the paper re-edited
   rather than as a database row. The sheet stays white in dark mode for the
   same reason a PDF viewer does. */
.p{background:#fff;color:#111;border:1px solid rgba(0,0,0,.11);border-radius:5px;
box-shadow:0 1px 2px rgba(0,0,0,.06),0 10px 26px rgba(0,0,0,.045);
padding:24px 30px 18px;cursor:default}
.p.sel{box-shadow:0 0 0 2px var(--acc),0 10px 26px rgba(0,0,0,.06)}
.p .body{max-width:72ch;margin:0 auto}      /* a measure you can actually read */
.p .ti{font-size:15px;font-weight:660;line-height:1.36;color:#111;letter-spacing:-.005em}
.p .meta{color:#78776f;font-size:11.5px;margin-top:4px}
.rule{height:1px;background:rgba(0,0,0,.09);margin:15px 0}

/* The three sentences ARE the card. Everything else is smaller than they are. */
/* Three highlighter colours, one per question. Text stays near-black on a
   translucent wash, so contrast is ~14:1 whichever colour lands under it. */
.passage{font-size:16px;line-height:1.72;color:#141414;margin:14px 0 4px}
.passage.miss{font-size:12.5px;color:#a8a79f;font-style:italic}
.hl{padding:.16em .32em .2em;margin:0 .04em;border-radius:.2em;
-webkit-box-decoration-break:clone;box-decoration-break:clone}
.hl.why{background-image:linear-gradient(101deg,rgba(255,158,186,0) .5%,
rgba(255,158,186,.62) 1.8%,rgba(255,145,176,.5) 96.5%,rgba(255,158,186,0) 99.4%)}
.hl.new{background-image:linear-gradient(99deg,rgba(255,216,48,0) .5%,
rgba(255,216,48,.95) 1.8%,rgba(255,206,26,.82) 96.5%,rgba(255,216,48,0) 99.4%)}
.hl.eff{background-image:linear-gradient(103deg,rgba(126,206,255,0) .5%,
rgba(126,206,255,.72) 1.8%,rgba(108,196,252,.58) 96.5%,rgba(126,206,255,0) 99.4%)}
.terms{margin:12px 0 2px;display:flex;flex-wrap:wrap;gap:5px 14px;font-size:12px;
color:#4a4943;line-height:1.6}
.tm b{font-weight:600;color:#8a8981;font-size:10px;text-transform:uppercase;
letter-spacing:.07em;margin-right:5px}
.tm.new b{color:#9a7a00}
.tm.eff b{color:#1d6ea8}
.meta i{font-style:italic;color:#8a8981}
/* The address is copied, not opened: a mailto: hands the reader off to whatever
   client the OS picked, which is rarely what they wanted. */
.au{font:inherit;font-size:inherit;color:#3a3a3a;background:none;border:0;padding:0;
border-bottom:1px solid rgba(0,0,0,.22);cursor:pointer}
.au:hover{color:var(--acc);border-bottom-color:var(--acc)}
.au.done{color:#1a7f45;border-bottom-color:#1a7f45}
.tm.src{color:var(--mut);font-size:11px}
.tm.src::before{content:'·';margin-right:8px;color:var(--dim)}
.blk{display:grid;grid-template-columns:66px 1fr;gap:4px 18px;margin:13px 0}
.blk .k{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:#9a9a92;
padding-top:6px;white-space:nowrap}
.blk .s{font-size:15.5px;line-height:1.6;color:#141414}
.blk.chg .s{font-size:17px;line-height:1.55}
.blk.miss .s{font-size:12px;color:#b3b2aa;font-style:italic;padding-top:3px}

.foot{display:flex;align-items:flex-end;gap:14px;margin-top:14px;
padding-top:11px;border-top:1px solid rgba(0,0,0,.07)}
.foot .fx{font-size:11.5px;color:#5c5b55;line-height:1.5;flex:1}
.foot .fx b{font-weight:600;color:#8a8981;font-size:10px;text-transform:uppercase;
letter-spacing:.07em;margin-right:5px}
.simbtn{font-size:11px;padding:4px 10px;border-radius:7px;border:1px solid rgba(0,0,0,.16);
background:#fff;color:#333;cursor:pointer;white-space:nowrap;text-decoration:none;
display:inline-block;line-height:1.5;font-family:inherit}
.simbtn:hover{border-color:var(--acc);color:var(--acc)}
.badge{display:inline-block;font-size:9px;padding:1px 5px;border-radius:4px;
background:var(--acc);color:#fff;margin-left:6px;vertical-align:1.5px}
.badge.sp{background:var(--warm)}
.near{display:inline-block;margin-left:6px;padding:0 5px;border-radius:4px;
border:1px dashed #c9c8c0;color:#8a8981;font-size:9.5px;vertical-align:1px}

/* Highlighter, not typography. A marker swipe leaves an uneven, translucent
   edge and sits UNDER black text, so contrast stays ~15:1 — the coloured-text
   version failed at 4.3:1. box-decoration-break keeps the swipe continuous
   across a line wrap instead of chopping it into rectangles. */
.hm{padding:.1em .3em .16em;margin:0 -.14em;border-radius:.16em;
-webkit-box-decoration-break:clone;box-decoration-break:clone}
.hm-p{background-image:linear-gradient(101deg,rgba(255,222,64,0) .6%,
rgba(255,222,64,.92) 2.4%,rgba(255,216,50,.78) 95.5%,rgba(255,222,64,0) 99.2%)}
.hm-k{background-image:linear-gradient(99deg,rgba(120,201,255,0) .6%,
rgba(120,201,255,.82) 2.4%,rgba(101,190,252,.66) 95.5%,rgba(120,201,255,0) 99.2%)}
.hm-v{background-image:linear-gradient(103deg,rgba(255,151,178,0) .6%,
rgba(255,151,178,.78) 2.4%,rgba(255,138,168,.62) 95.5%,rgba(255,151,178,0) 99.2%)}
.hm-u{background-image:linear-gradient(100deg,rgba(0,0,0,0) .6%,
rgba(0,0,0,.10) 2.4%,rgba(0,0,0,.075) 95.5%,rgba(0,0,0,0) 99.2%)}
mark{background:none;background-image:linear-gradient(102deg,rgba(180,255,170,0) .6%,
rgba(180,255,170,.85) 2.4%,rgba(165,250,155,.7) 95.5%,rgba(180,255,170,0) 99.2%);
color:inherit;padding:.1em .3em .16em;margin:0 -.14em;border-radius:.16em;
-webkit-box-decoration-break:clone;box-decoration-break:clone}
.legend{display:flex;gap:12px;flex-wrap:wrap;color:var(--mut);font-size:11px;
margin:2px auto 14px;align-items:center;max-width:920px;line-height:2}
.legend .sw{color:#111;background:#fff;padding:2px 3px;border-radius:3px;white-space:nowrap}
.det .sp{border-left:2px solid rgba(0,0,0,.14);padding-left:11px;margin:9px 0;
font-size:13px;line-height:1.55;color:#333}
.nb{font-size:12.5px;padding:5px 0;border-bottom:1px solid rgba(0,0,0,.07);
cursor:pointer;line-height:1.35;color:#333}
.nb:hover{color:var(--acc)}
.expand{margin-top:13px;padding-top:12px;border-top:1px solid rgba(0,0,0,.07)}
.expand h4{font-size:10px;letter-spacing:.09em;text-transform:uppercase;
color:#9a9a92;margin:0 0 7px;font-weight:600}
.empty{color:var(--mut);padding:24px;text-align:center}
/* The first screen. An empty result area is the point, so it has to read as a
   deliberate state rather than a failure to load. */
/* Similar papers open beside the list, not in place of it — losing your results
   to see a neighbour was the wrong trade. */
.panel{position:fixed;top:0;right:0;width:340px;height:100vh;background:var(--card);
border-left:1px solid var(--ring);box-shadow:-8px 0 28px rgba(0,0,0,.07);
display:flex;flex-direction:column;z-index:20}
.phd{display:flex;align-items:center;justify-content:space-between;gap:10px;
padding:14px 16px;border-bottom:1px solid var(--line)}
.phd span{font-size:10.5px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut)}
.phd button{font:inherit;font-size:11.5px;padding:4px 10px;border-radius:7px;
border:1px solid var(--ring);background:#fff;color:var(--ink2);cursor:pointer}
.pbody{overflow-y:auto;padding:12px 16px 24px}
.pseed{font-size:12.5px;line-height:1.4;color:var(--ink);font-weight:600;margin-bottom:4px}
.pseedm{font-size:11px;color:var(--mut);margin-bottom:12px;
padding-bottom:12px;border-bottom:1px solid var(--line)}
.nb{font-size:12.5px;padding:9px 0;border-bottom:1px solid var(--line);
cursor:pointer;line-height:1.4;color:#2b2b2b}
.nb:hover{color:var(--acc)}
.nb .nbm{display:block;font-size:10.5px;color:var(--mut);margin-top:3px}
@media(max-width:1500px){body.haspanel .wrap{margin-right:360px}}
@media(max-width:900px){.panel{width:100%;height:70vh;top:auto;bottom:0;
border-left:0;border-top:1px solid var(--ring)}
  body.haspanel .wrap{margin-right:0;padding-bottom:70vh}}
.start{max-width:920px;margin:26px auto;padding:26px 28px;background:var(--card);
border:1px dashed var(--ring);border-radius:10px;color:var(--ink2);font-size:14px;
line-height:1.6}
.start span{display:block;margin-top:6px;color:var(--mut);font-size:12px}
.nearhd{max-width:920px;margin:22px auto 2px;padding:11px 16px;border-radius:9px;
background:var(--card);border:1px dashed var(--ring);color:var(--ink2);font-size:12.5px}
.nearhd span{display:block;color:var(--mut);font-size:11px;margin-top:3px}
.grp{max-width:920px;margin:0 auto 12px;background:var(--card);border:1px solid var(--ring);
border-radius:10px;padding:13px 16px}
.grphd{display:flex;align-items:baseline;gap:10px;font-size:14px;font-weight:640;
color:var(--ink);margin-bottom:8px}
.grphd .sep{color:var(--dim);font-weight:400}
.grphd .n{margin-left:auto;font-size:11.5px;color:var(--mut);font-weight:400;white-space:nowrap}
.grow{font-size:12.5px;line-height:1.5;padding:4px 0;color:#2b2b2b;cursor:pointer;
border-bottom:1px solid var(--line)}
.grow:last-of-type{border-bottom:0}
.grow:hover{color:var(--acc)}
.gmore{font-size:11.5px;color:var(--mut);padding-top:6px}
.capped{max-width:920px;margin:14px auto 0;padding:12px 16px;background:var(--card);
border:1px solid var(--ring);border-radius:9px;color:var(--mut);font-size:12px;
line-height:1.55}
/* Three ways in, named. Chips reach 29% of papers, so they cannot be the only
   door — search covers everything, datasets are the highest-trust anchor, and a
   dataset is the highest-trust anchor a paper commits to. */
.modes{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap}
.modes button{font-size:12.5px;padding:7px 13px}
.modes button.on{background:var(--acc);color:#fff;border-color:transparent}
.pane{margin-bottom:10px}

.dslist{display:flex;gap:6px;flex-wrap:wrap}
.dslist.clip{overflow:hidden}
.ds{font-size:12.5px;padding:5px 12px;border-radius:7px;border:1px solid var(--ring);
background:#fff;color:#2b2b2b;cursor:pointer;white-space:nowrap;
box-shadow:0 1px 1px rgba(0,0,0,.03)}
.ds:hover{border-color:var(--acc);color:var(--acc)}
.ds.on{background:var(--acc);color:#fff;border-color:transparent}
.ds .n{color:var(--mut);margin-left:5px;font-size:11px}
/* Filters were behind a toggle and nobody found them. They are a permanent row
   now, labelled, with an active-count so it is obvious when one is on. */
.facets{background:var(--card);border:1px solid var(--ring);border-radius:9px;
padding:8px 10px;margin-bottom:11px}
.flabel{font-size:10px;letter-spacing:.09em;text-transform:uppercase;color:var(--mut);
margin-right:3px}
.facets select,.facets button{font-size:12.5px}
#fclear{margin-left:auto;color:var(--warm);border-color:transparent;background:transparent}
.near{display:inline-block;margin-left:6px;padding:0 5px;border-radius:4px;
border:1px dashed var(--dim);color:var(--mut);font-size:9.5px;vertical-align:1px}
.simbtn{font-size:10.5px;padding:2px 7px;border-radius:6px;margin-left:auto}
/* Which proceedings this row is from. Muted and first: it identifies the row,
   it is not a thing to compare rows by. */
.cyst{color:var(--mut);font-size:10.5px;font-weight:600;letter-spacing:.02em;
margin-right:8px;white-space:nowrap}
.per{margin-left:5px;color:var(--acc);font-size:10px;font-weight:600}
.simbtn.on{background:var(--ink);color:#fff;border-color:var(--ink)}
/* landing */
.venues{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:4px}
@media(max-width:760px){.venues{grid-template-columns:1fr}}
.vcard{background:var(--card);border-radius:12px;padding:16px 18px}
.vcard.dim{opacity:.55}
.vname{font-size:18px;font-weight:700;letter-spacing:-.01em}
.vfull{font-size:11px;color:var(--mut);margin:2px 0 12px;min-height:28px}
.vcard{border:1px solid transparent;text-align:left;font:inherit;cursor:pointer}
.vcard:not(.dim):hover{border-color:var(--acc)}
.vcard.dim{cursor:default}
.vyr{font-size:12px;font-weight:600;color:var(--mut);margin-left:8px}
.vn{font-size:11.5px;color:var(--ink2)}
.vsoon{font-size:11.5px;color:var(--mut)}
#rail{display:flex;gap:8px;margin:0 0 12px}
.rv{font:inherit;font-size:13px;font-weight:650;padding:7px 14px;border-radius:9px;cursor:pointer;
border:1px solid var(--ring);background:var(--card);color:var(--ink2);display:flex;gap:6px;align-items:baseline}
.rv span{font-size:10.5px;font-weight:500;color:var(--mut)}
.rv.on{background:var(--ink);border-color:var(--ink);color:#fff}
.rv.on span{color:#c9c9c9}
.rv.dim{opacity:.45;cursor:default;padding:7px 14px;border:1px dashed var(--ring);border-radius:9px}
.rtr{display:none}
/* on wide screens the rail is a real grid column and sticks while the list
   scrolls — it holds "what to press next", which is exactly what a reader
   forgets mid-scroll */
@media(min-width:1300px){
  .wrap.withrail{max-width:1266px;display:grid;grid-template-columns:250px minmax(0,980px);
    gap:0 26px;align-items:start}
  .wrap.withrail>header{grid-column:1/-1}
  #rail{grid-column:1;position:sticky;top:12px;flex-direction:column;margin:0;
    max-height:calc(100vh - 24px);overflow-y:auto;scrollbar-width:none}
  #main{grid-column:2;min-width:0}
  .rv{justify-content:space-between}
  .rtr{display:block;margin-top:14px;border-top:1px solid var(--ring);padding-top:10px}
}
.rh{font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-bottom:7px}
.rh em{font-style:normal;font-weight:500;letter-spacing:0;text-transform:none;margin-left:4px}
/* rail mini chart: one venue pair per row, log x like the landing chart */
.mrr{display:block;width:100%;text-align:left;font:inherit;background:none;border:0;
  padding:2.5px 0;cursor:pointer;border-radius:4px}
.mrr:hover .mrl{color:var(--acc)}
.mrl{font-size:11.5px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  display:block;margin-bottom:1px}
.mrl .tag{margin-left:5px}
.mrt{position:relative;height:6px;display:block}
.mrt .lane{position:absolute;inset:0;height:auto;margin:0}
/* the set card: composition of the current selection, in reach while scrolling */
.setcard{margin-top:14px;border-top:1px solid var(--ring);padding-top:10px}
.scn{font-size:20px;font-weight:700;letter-spacing:-.02em}
.scn small{font-size:11px;font-weight:500;color:var(--mut);margin-left:5px}
.scyr{margin:8px 0 2px}
.scyrow{display:grid;grid-template-columns:34px 1fr 70px;gap:8px;align-items:center;
  font-size:11px;color:var(--ink2);padding:1.5px 0}
.scyrow .yb{position:relative;height:7px}
.scyrow .yb i{position:absolute;left:0;top:0;height:100%;border-radius:2px}
.scyrow b{font-weight:600;font-size:10px;text-align:right;color:var(--ink2);white-space:nowrap}
.scsub{font-size:10.5px;color:var(--mut);margin:2px 0 0}
.scchips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.scchip{font:inherit;font-size:10.5px;padding:2px 8px;border-radius:20px;cursor:pointer;
  border:1px solid var(--ring);background:var(--card);color:var(--ink2)}
.scchip b{font-weight:600;color:var(--mut);margin-left:3px}
.scchip:hover{border-color:var(--acc);color:var(--acc)}
/* what-moved entry view */
.chgbox{background:var(--card);border-radius:12px;padding:16px 18px;margin-top:14px}
.chghd{font-size:15px;font-weight:660;display:flex;align-items:center;gap:10px}
.chgtabs{margin-left:auto;display:flex;gap:4px}
.chgtab{font-size:11px;padding:2px 9px;border-radius:6px;border:1px solid var(--ring);
background:none;cursor:pointer;color:var(--ink2)}
.chgtab.on{background:var(--ink);color:#fff;border-color:var(--ink)}
.chgsub{font-size:11px;color:var(--mut);margin:3px 0 10px}
.chgleg{display:flex;gap:14px;margin:6px 0 0;font-size:10.5px;color:var(--ink2)}
.chgleg i{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;vertical-align:-1px}
.chgplot{position:relative}
/* gridlines and axis share one geometry: 172px label column + 12px gap */
.chgg{position:absolute;top:0;bottom:0;left:184px;right:0;pointer-events:none}
.chgg i{position:absolute;top:0;bottom:0;width:1px;background:var(--line)}
.chgax{position:relative;height:13px;margin:10px 0 0 184px}
.chgax b{position:absolute;transform:translateX(-50%);font-size:9px;font-weight:500;color:var(--mut);white-space:nowrap}
.chgsec{font-size:9.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);
  margin:11px 0 3px;position:relative}
.cr{display:grid;grid-template-columns:172px 1fr;gap:12px;align-items:center;
  padding:2px 0;position:relative;font-size:12.5px;color:var(--ink2);border:0;background:none;
  width:100%;text-align:left;font-family:inherit;cursor:pointer;border-radius:4px}
.cr:hover .crl{color:var(--acc)}
.crl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.tag{font-style:normal;font-size:9px;font-weight:700;letter-spacing:.04em;color:var(--acc);
  margin-left:6px;vertical-align:1px}
.tag.gone{color:var(--mut)}
.trk{position:relative}
.lane{display:block;position:relative;height:8px;margin:1.5px 0}
.lane i{position:absolute;left:0;top:0;height:100%;border-radius:0 2px 2px 0}

.famchip.sel .per{color:#fff}
.pfoot{display:flex;align-items:center;gap:8px;margin-top:7px}
</style></head><body><div class="wrap">
<header><h1 id="home">What's new in AI research</h1></header>

<div id="landing" hidden></div>
<nav id="rail" hidden></nav>

<div id="main">

<div class="search">
  <input type="search" id="q" aria-label="Search every abstract">
</div>

<div class="sug">
  <div class="sughd">Topic <em>what the paper is about</em> <span id="tophint"></span></div>
  <div class="fams" id="topfams"></div>
  <div class="famopen" id="topopen"></div>
  <div class="sughd">Method <em>what it builds on</em> <span id="mhint"></span></div>
  <div class="fams" id="mfams"></div>
  <div class="famopen" id="mopen"></div>
  <div class="sughd">Benchmarks <em>where it was tested</em> <span id="dshint"></span></div>
  <div class="chipwrap">
    <div class="dslist clip" id="dslist"></div>
    <button class="chiptog" id="dstog" hidden></button>
  </div>
</div>

<div class="xref" id="xref" hidden></div>
<div class="meter" id="meter">
  <div class="mrow"><span id="count"></span><span id="ctx"></span>
    <button id="grptog" class="simbtn" style="margin-left:auto">Split into subgroups</button></div>
  <div class="mtrack"><i id="mfill"></i></div>
</div>
<div class="legend" id="legend">
  <span class="sw"><span class="hl why">why it was needed</span></span>
  <span class="sw"><span class="hl new">what is new</span></span>
  <span class="sw"><span class="hl eff">what it achieved</span></span>
  <span>— the paper's own sentences, cut but never rewritten. The colours and the order are ours.</span></div>

<div class="res" id="results"></div>

<aside class="panel" id="panel" hidden>
  <div class="phd"><span>Closest papers</span><button id="pclose">Close</button></div>
  <div class="pbody" id="pbody"></div>
</aside>
</div>
</div>
<script>const D=__DATA__;</script>
<script>
const $=s=>document.querySelector(s);
const esc=s=>String(s??'').replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const V=D.vocab, P=D.papers, T=D.topics;
const mname=i=>V.m[i], dname=i=>V.d[i], tname=i=>V.t[i];

// Names only — titles and extracted concepts. Kept because dataset and method
// names do not always appear in the abstract prose.
const HAY=P.map(p=>[p.t,p.a,p.b,p.d,
  ...p.p.map(mname),...p.u.map(mname),...p.v.map(mname),
  ...p.k.map(dname),...p.s.map(tname)].join(' ').toLowerCase());
const BYID={}; P.forEach((p,i)=>BYID[p.i]=i);

// Neighbours and vectors are indexed by gid, so there is no lookup table: a
// paper's row IS p.i. Papers with no abstract sit above D.nemb and have neither.
const NBK=D.neighbors?D.neighbors.k:0, NBR=D.neighbors?D.neighbors.nbr:null;
const nbrsOf=g=>NBR&&g<D.nemb?NBR.slice(g*NBK,(g+1)*NBK):[];

// ---- PCA(128) int8 vectors, dequantised and row-normalised once ----
// A browser cannot embed a query, so search stays lexical. These are used the
// other way round: take what the words DID find, and reach the papers that mean
// the same thing without sharing the vocabulary. Those arrive as a separate,
// labelled group — never mixed into the literal matches.
const EMB=(()=>{
  if(!D.emb)return null;
  const {dims,b64,scale}=D.emb;
  const bin=atob(b64), n=D.nemb;
  const v=new Float32Array(n*dims);
  for(let i=0;i<n;i++){
    let s=0;
    for(let d=0;d<dims;d++){
      const q=(bin.charCodeAt(i*dims+d)<<24)>>24;   // byte -> signed int8
      const x=q*scale; v[i*dims+d]=x; s+=x*x;
    }
    s=Math.sqrt(s)||1;
    for(let d=0;d<dims;d++) v[i*dims+d]/=s;
  }
  return {v,dims,n};
})();
const hasVec=g=>EMB&&g<EMB.n;

// ---- abstract term index: search now sees all 6,637 abstracts, not 20% ----
const TV=D.terms.v, TP=D.terms.p;
const POST=new Map();                       // term id -> paper indices
TP.forEach((ids,i)=>{for(const t of ids){let a=POST.get(t);if(!a)POST.set(t,a=[]);a.push(i);}});

// TV is sorted, so every term sharing a prefix is one contiguous run.
function prefixTerms(w){
  let lo=0,hi=TV.length;
  while(lo<hi){const m=(lo+hi)>>1; if(TV[m]<w)lo=m+1; else hi=m;}
  const out=[];
  for(let i=lo;i<TV.length&&TV[i].startsWith(w);i++)out.push(i);
  return out;
}
function papersWithWord(w){
  const s=new Set();
  for(const t of prefixTerms(w))for(const i of (POST.get(t)||[]))s.add(i);
  for(let i=0;i<P.length;i++)if(!s.has(i)&&HAY[i].includes(w))s.add(i);
  return s;
}

// Papers are shown only once the reader has asked for some. Listing all 6,637 on
// arrival is the problem this product exists to remove, not a neutral default.
const MAX_SHOWN=80, NEIGHBOURS_SHOWN=5;
// Declared here, not beside clipRow: drawDatasets() runs during setup, before
// that part of the script is reached, and a const in its temporal dead zone
// throws rather than reading as undefined.
const rowCount={}, openRow={}, openFam=new Set();
const st={corp:null,q:[],topics:new Set(),fams:new Set(),sel:null,panel:null,ds:null,meth:null,mfam:null,grouped:false};
// A conference pick alone is NOT a selection. Picking "ICML 2026" leaves 6,637
// papers, which is the problem this product exists to remove — the reader still
// has to say what their field is.
const chosen=()=>st.q.length||st.topics.size||st.fams.size||st.ds!==null||st.meth!==null||st.mfam!==null;

// How strongly each literal hit matches — used to pick seeds, so the centroid is
// built from the papers the query is actually about, not the weakest 700.
function hitScore(i){
  let sc=0;
  for(const w of st.q){
    for(const t of prefixTerms(w)){
      const post=POST.get(t);
      if(post&&post.length&&TP[i].includes(t)) sc+=Math.log(1+P.length/post.length);
    }
  }
  return sc;
}

// Papers close in meaning to what the words found, excluding what the words found.
function semanticExtras(hits,cap=60){
  if(!EMB||!hits||!hits.size||!st.q.length)return [];
  const seeds=[...hits].map(i=>[hitScore(i),i]).sort((a,b)=>b[0]-a[0])
                       .slice(0,40).map(([,i])=>P[i].i).filter(hasVec);
  if(seeds.length<3)return [];
  const D_=EMB.dims, c=new Float32Array(D_);
  for(const r of seeds) for(let d=0;d<D_;d++) c[d]+=EMB.v[r*D_+d];
  let s=0; for(let d=0;d<D_;d++)s+=c[d]*c[d];
  s=Math.sqrt(s)||1; for(let d=0;d<D_;d++)c[d]/=s;
  const dot=r=>{let x=0;for(let d=0;d<D_;d++)x+=c[d]*EMB.v[r*D_+d];return x;};
  // Self-calibrating bar, as in topics.py: a candidate must sit at least as close
  // to the centre as a typical paper the words already found. A fixed threshold
  // cannot work — a tight query and a diffuse one have different geometry.
  const own=seeds.map(dot).sort((a,b)=>b-a);
  const bar=own[Math.floor(own.length*0.6)]||0.4;
  const out=[];
  for(let i=0;i<P.length;i++){
    if(hits.has(i))continue;
    const r=P[i].i; if(!hasVec(r))continue;
    const v=dot(r);
    if(v>=bar) out.push([v,i]);
  }
  out.sort((a,b)=>b[0]-a[0]);
  return out.slice(0,cap).map(([,i])=>i);
}

function queryHits(){
  if(!st.q.length)return null;
  let acc=null;
  for(const w of st.q){
    const s=papersWithWord(w);
    acc=acc===null?s:new Set([...acc].filter(x=>s.has(x)));
    if(!acc.size)break;
  }
  return acc;
}

// A chip's count is its size WITHIN what is already chosen. Measured: 139
// benchmarks drop to 7-15 once a topic is picked. No hierarchy exists in this
// data to lean on — 0 topic pairs show 80% containment — so narrowing by context
// is what replaces it. Nothing is invented; a chip vanishes only when no paper in
// the current set uses it.
function baseSet(skip){
  const out=new Set(), hits=queryHits();
  for(let i=0;i<P.length;i++){
    const p=P[i];
    if(st.corp!==null&&p.cy!==st.corp)continue;
    if(skip!=='ds'&&st.ds!==null&&!p.k.includes(st.ds))continue;
    if(skip!=='meth'&&!methPass(p))continue;
    if(skip!=='topics'&&!facetPass(p,false))continue;
    if(skip!=='domains'&&!facetPass(p,true))continue;
    if(hits&&!hits.has(i))continue;
    out.add(i);
  }
  return out;
}

// a method pick matches the method itself or any of its narrower forms; a family
// pick matches anything in that family
let MCHILD=null, MFAMOF=null;
function methIndex(){
  if(MCHILD)return;
  MCHILD=new Map(); MFAMOF=new Map();
  for(const [id,n,f,par] of MS){ MFAMOF.set(id,f);
    if(par!=null){ if(!MCHILD.has(par))MCHILD.set(par,[]); MCHILD.get(par).push(id);} }
}
function methPass(p){
  if(st.meth===null&&st.mfam===null)return true;
  methIndex();
  const mu=p.mu||[];
  if(st.meth!==null){
    if(mu.includes(st.meth))return true;
    for(const k of (MCHILD.get(st.meth)||[])) if(mu.includes(k))return true;
    return false;
  }
  for(const m of mu) if(MFAMOF.get(m)===st.mfam)return true;
  return false;
}

function facetPass(p,wantDomain){
  let picked=false, hit=false;
  for(let ti=0;ti<T.length;ti++){
    if((T[ti].f===1)!==wantDomain)continue;
    if(st.topics.has(ti)||st.fams.has(FAMKEY[ti])){picked=true;break;}
  }
  if(!picked)return true;                       // nothing chosen on this side
  for(const [ti] of p.g){
    if((T[ti].f===1)!==wantDomain)continue;
    if(st.topics.has(ti)||st.fams.has(FAMKEY[ti])){hit=true;break;}
  }
  return hit;
}

function match(i,hits,corp){
  const p=P[i];
  const cc=corp===undefined?st.corp:corp;
  if(cc!==null&&p.cy!==cc)return false;
  if(st.ds!==null&&!p.k.includes(st.ds))return false;
  if(!methPass(p))return false;
  // Within a facet the picks are OR — two topics widen the set. ACROSS facets
  // they are AND: "reasoning" plus "healthcare" means reasoning applied to
  // healthcare, not everything that is either. Anything else makes the two rows
  // fight each other.
  if(!facetPass(p,false)||!facetPass(p,true))return false;
  if(hits&&!hits.has(i))return false;
  return true;
}

function results(){
  const hits=queryHits();
  const out=[];
  for(let i=0;i<P.length;i++) if(match(i,hits)) out.push(i);
  return out;
}

// The near-miss group: close in meaning, no shared wording. Kept apart from the
// literal matches for the same reason "looks similar" is kept apart from "said
// so" — they are different claims about why a paper is on screen.
function nearby(){
  if(!st.q.length)return [];
  const hits=queryHits();
  if(!hits)return [];
  const lit=new Set(); for(const i of hits) if(match(i,hits)) lit.add(i);
  return semanticExtras(lit).filter(i=>match(i,null));
}

// topics.json keeps "the paper said so" and "it sits near the centre" apart, and
// the page must too — otherwise a chip labelled 51 quietly returns 102.
// The panel is a view onto one paper's neighbours; it never changes the result set.
const RX_ESC=s=>s.replace(/[.*+?^${}()|[\]\\]/g,'\\$&');
function roleRanges(text,p){
  const groups=[[p.p,'hm-p',4],[p.k,'hm-k',3],[p.v,'hm-v',2],[p.u,'hm-u',1]];
  const out=[];
  for(const [ids,cls,prio] of groups){
    for(const id of ids){
      const name=(cls==='hm-k'?dname(id):mname(id))||'';
      if(name.length<3)continue;                   // "3D", "RL" match everywhere
      let re;
      try{re=new RegExp('(^|[^A-Za-z0-9])('+RX_ESC(name)+')(?![A-Za-z0-9])','gi');}
      catch(e){continue;}
      let m;
      while((m=re.exec(text))!==null){
        const s=m.index+m[1].length;
        out.push({s,e:s+m[2].length,cls,prio});
        if(re.lastIndex<=m.index)re.lastIndex=m.index+1;
      }
    }
  }
  return out;
}
function annotate(text,p,plain){
  const rs=plain?[]:roleRanges(text,p);
  for(const w of st.q){                            // search terms share the layer
    if(w.length<2)continue;
    let re; try{re=new RegExp(RX_ESC(w),'gi');}catch(e){continue;}
    let m; while((m=re.exec(text))!==null){
      rs.push({s:m.index,e:m.index+m[0].length,cls:'q',prio:0});
      if(re.lastIndex<=m.index)re.lastIndex=m.index+1;}
  }
  // longest wins, then highest priority; keep only non-overlapping ranges
  rs.sort((a,b)=>(b.e-b.s)-(a.e-a.s)||b.prio-a.prio||a.s-b.s);
  const kept=[];
  for(const r of rs) if(!kept.some(k=>r.s<k.e&&k.s<r.e)) kept.push(r);
  kept.sort((a,b)=>a.s-b.s);
  let outp='',at=0;
  for(const r of kept){
    outp+=esc(text.slice(at,r.s));
    const inner=esc(text.slice(r.s,r.e));
    outp+= r.cls==='q' ? `<mark>${inner}</mark>` : `<span class="hm ${r.cls}">${inner}</span>`;
    at=r.e;
  }
  return outp+esc(text.slice(at));
}

const hl=(s)=>{ if(!st.q.length)return esc(s);
  let out=esc(s);
  for(const w of st.q){ if(w.length<2)continue;
    out=out.replace(new RegExp('('+w.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','ig'),'<mark>$1</mark>');}
  return out;};

function card(i){
  const p=P[i];
  const names=(arr,fn)=>arr.map(fn).join(', ');
  // One passage, not three labelled rows. The three sentences are the paper's,
  // in the order a reader needs them, and the colour says which question each
  // answers: why it was needed, what is new, what it achieved.
  const parts=[];
  if(p.L)parts.push(`<span class="hl why">${annotate(p.L,p,true)}</span>`);
  const change=p.K||p.n[0]||'';
  if(change)parts.push(`<span class="hl new">${annotate(change,p,true)}</span>`);
  if(p.R)parts.push(`<span class="hl eff">${annotate(p.R,p,true)}</span>`);

  // The extracted terms are part of the summary, not a footnote under it.
  const term=(lab,v,cls)=>v?`<span class="tm ${cls}"><b>${lab}</b>${hl(v)}</span>`:'';
  const src=p.f?'<span class="tm src">read from the full paper</span>':'';
  const terms=[term('proposes',names(p.p,mname),'new'),
               term('builds on',names(p.u,mname),''),
               term('compared with',names(p.v,mname),''),
               term('data',names(p.k,dname),'eff'),
               term('tasks',names(p.s,tname),'')].filter(Boolean).join('')+src;

  // One name: the corresponding author when the paper printed one, else the
  // first. "et al." stands for the rest rather than listing eight names.
  const c1=(p.c1||[])[0];
  const lead=c1?(c1.name||c1.email):((p.au||[])[0]||'');
  const etal=(p.na||0)>1?' <i>et al.</i>':'';
  const who=lead
    ? (c1&&c1.email
        ? `<button class="au" data-mail="${esc(c1.email)}" title="${esc(c1.email)} — click to copy">${esc(lead)}</button>${etal}`
        : `${esc(lead)}${etal}`)
    : '';

  return `<div class="p ${st.sel===i?'sel':''}" data-i="${i}"><div class="body">
    <div class="ti">${hl(p.t)}${p.o===1?'<span class="badge">Oral</span>':p.o===2?'<span class="badge sp">Spotlight</span>':''}</div>
    <div class="meta"><span class="cyst">${esc(CY[p.cy].v)} ${CY[p.cy].y}</span>${who}${nearBadge(p)}</div>
    <div class="rule"></div>
    ${parts.length?`<div class="passage">${parts.join(' ')}</div>`
      :`<div class="passage miss">no sentence in this paper states what is new</div>`}
    ${terms?`<div class="terms">${terms}</div>`:''}
    <div class="foot">
      <div class="fx"></div>
      <div style="display:flex;gap:7px">
        ${p.w?`<a class="simbtn" href="${esc(p.w)}" target="_blank" rel="noopener">See paper ↗</a>`:''}
        <button class="simbtn" data-sim="${i}">Similar</button>
      </div>
    </div>
  </div></div>`;
}

// topics.json keeps "the paper said so" and "it sits near the centre" apart, and
// the page must too — otherwise a chip labelled 51 quietly returns 102.
// The panel is a view onto one paper's neighbours; it never changes the result set.
function openPanel(i){
  st.panel=i;
  const p=P[i];
  // Five, not twelve. The panel is a nudge sideways, not a second result list —
  // past the first handful the neighbours stop being obviously related anyway.
  const rows=nbrsOf(p.i).slice(0,NEIGHBOURS_SHOWN).map(id=>{
    const j=BYID[id]; if(j===undefined)return '';
    const q=P[j];
    return `<div class="nb" data-go="${j}">${esc(q.t)}`+
           `<span class="nbm">${st.corp!==null&&q.cy!==st.corp?CY[q.cy].y+' · ':''}`+
           `${esc(q.b||q.a||'')}${q.o===1?' · Oral':q.o===2?' · Spotlight':''}</span></div>`;
  }).join('');
  $('#pbody').innerHTML=`<div class="pseed">${esc(p.t)}</div>`+
    `<div class="pseedm">nearest by meaning — this year and last, every subarea</div>`+
    (rows||'<div style="color:var(--mut);font-size:12px">No neighbours for this paper.</div>');
  $('#panel').hidden=false; document.body.classList.add('haspanel');
  $('#pbody').querySelectorAll('[data-go]').forEach(el=>el.onclick=()=>{
    const j=+el.dataset.go;
    if($(`.p[data-i="${j}"]`)){
      st.sel=j; render(); openPanel(st.panel);
      $(`.p[data-i="${j}"]`)?.scrollIntoView({block:'center',behavior:'smooth'});
    } else openPanel(j);          // not in the current set — walk on to its neighbours
  });
}
function closePanel(){ st.panel=null; $('#panel').hidden=true;
  document.body.classList.remove('haspanel'); }

function nearBadge(p){
  if(!st.topics.size)return '';
  // kind 1 = the paper named this topic. kind 2 = it named something narrower
  // that rolls up to it. Different claims, so they are never shown as the same.
  for(const [ti,kind] of p.g) if(st.topics.has(ti)&&kind===1) return '';
  for(const [ti,kind] of p.g) if(st.topics.has(ti)&&kind===2)
    return '<span class="near">said something narrower</span>';
  return '';
}

// When a word exists on both rows, say so where it is noticed — at the moment
// the reader picks one and sees a number that does not match the other row.
function drawXref(){
  const B=D.both||[]; const el=$('#xref'); if(!el)return;
  let msg='';
  for(const [ti,mid,nt,nm] of B){
    if(st.topics.has(ti)&&st.meth!==mid){
      msg=`<b>${esc(T[ti].l)}</b> is also a <b>method</b>: ${nm.toLocaleString()} papers `+
        `build on it rather than study it.<button data-x="m:${mid}">Show those instead</button>`; break;
    }
    if(st.meth===mid&&!st.topics.has(ti)){
      msg=`<b>${esc(MV[mid])}</b> is also a <b>topic</b>: ${nt.toLocaleString()} papers `+
        `are about it rather than built on it.<button data-x="t:${ti}">Show those instead</button>`; break;
    }
  }
  el.hidden=!msg; el.innerHTML=msg;
  el.querySelectorAll('[data-x]').forEach(b=>b.onclick=()=>{
    const [kind,v]=b.dataset.x.split(':');
    if(kind==='m'){ st.topics.clear(); st.fams.clear(); st.meth=+v; st.mfam=null; }
    else { st.meth=null; st.mfam=null; st.topics.clear(); st.topics.add(+v); }
    render();});
}

// ---- Problem 3: show the SET, not just the rows ----------------------------
// Clusters whatever the reader selected, using the PCA vectors already shipped
// for semantic search, and labels each group with the words its papers use and
// the rest of the selection does not. Ported from icml.setview, with the two
// findings that draft produced kept intact: a group under 3 papers or over 60%
// of the set is not a split, and some fields genuinely do not split at all.
const MIN_GROUP=3, MAX_GROUP_SHARE=0.6, MAX_K=6;
// Abstract terms alone label a group "evaluating · deployed · rates". Titles
// carry the vocabulary a reader recognises, at 100% coverage, and they are the
// paper's own words — the setview draft found the same thing.
const TITLE_STOP=new Set(('a an the of for with via using toward towards on in to and or is are be as at by '+
 'from into over under between across their there here than then them they learning model models method '+
 'methods framework approach network networks new novel efficient effective robust improved improving better '+
 'fast faster scalable simple unified general generalized deep neural training train based aware guided driven '+
 'free aided we our this that it its can does do how what why when where which who all any data analysis study '+
 'beyond rethinking revisiting understanding exploring enhancing enhanced boosting bridging leveraging '+
 'exploiting position make makes made more most less least high low large small first second one two three '+
 'multi single self semi cross pre post non anti sub super inter intra without within through against about '+
 'after before during while whether if not no you your').split(' '));
// Words that survive the search index but say nothing as a group heading.
const LABEL_STOP=new Set(('how why what when which who whom whose does did done doing '+
 'whether either neither thus hence therefore however moreover furthermore also still yet '+
 'given since because although though while whereas hence able unable often rarely '+
 'typically usually generally particularly especially significantly substantially '+
 'various several multiple different similar same certain specific general common '+
 'recent recently prior previous existing current novel simple effective efficient '+
 'important significant key main major minor overall total').split(' '));
const TITLE_W=P.map(p=>{
  const w=(p.t||'').toLowerCase().replace(/[^a-z0-9 -]/g,' ').split(/\s+/)
          .filter(x=>x.length>3&&!TITLE_STOP.has(x));
  const s=new Set(w);
  for(let i=0;i+1<w.length;i++)s.add(w[i]+' '+w[i+1]);
  return [...s];
});
function kmeans(rows,k,iters=40){
  const D_=EMB.dims, n=rows.length;
  const C=new Float32Array(k*D_);
  for(let j=0;j<k;j++){                       // deterministic spread, no RNG
    const r=rows[Math.floor(j*n/k)];
    for(let d=0;d<D_;d++)C[j*D_+d]=EMB.v[r*D_+d];
  }
  let a=new Int32Array(n);
  for(let it=0;it<iters;it++){
    for(let i=0;i<n;i++){
      let best=-2,bj=0;
      for(let j=0;j<k;j++){
        let dot=0; for(let d=0;d<D_;d++)dot+=EMB.v[rows[i]*D_+d]*C[j*D_+d];
        if(dot>best){best=dot;bj=j;}
      }
      a[i]=bj;
    }
    C.fill(0);
    const cnt=new Int32Array(k);
    for(let i=0;i<n;i++){ cnt[a[i]]++;
      for(let d=0;d<D_;d++)C[a[i]*D_+d]+=EMB.v[rows[i]*D_+d]; }
    for(let j=0;j<k;j++){
      let s=0; for(let d=0;d<D_;d++)s+=C[j*D_+d]*C[j*D_+d];
      s=Math.sqrt(s)||1; for(let d=0;d<D_;d++)C[j*D_+d]/=s;
    }
  }
  return a;
}
function groupsFor(idx){
  if(!EMB||idx.length<12)return null;
  const rows=[],keep=[];
  for(const i of idx){ const r=P[i].i; if(hasVec(r)){rows.push(r);keep.push(i);} }
  if(rows.length<12)return null;
  let best=null;
  for(let k=2;k<=Math.min(MAX_K,Math.floor(rows.length/MIN_GROUP));k++){
    const a=kmeans(rows,k);
    const cnt=new Array(k).fill(0); for(const x of a)cnt[x]++;
    if(Math.min(...cnt)<MIN_GROUP)continue;
    const share=Math.max(...cnt)/rows.length;
    if(share>MAX_GROUP_SHARE)continue;
    if(!best||share<best.share)best={k,a,share};
  }
  if(!best)return {split:false};
  // label each group by the terms distinctive to it INSIDE this selection
  const out=[];
  for(let j=0;j<best.k;j++){
    const mem=keep.filter((_,i)=>best.a[i]===j);
    const rest=keep.filter((_,i)=>best.a[i]!==j);
    const A=new Map(),B=new Map();
    const add=(map,i)=>{ for(const t of TP[i]){ const w=TV[t];
                           if(!LABEL_STOP.has(w))map.set(w,(map.get(w)||0)+1); }
                         for(const w of TITLE_W[i])map.set(w,(map.get(w)||0)+1); };
    for(const i of mem) add(A,i);
    for(const i of rest) add(B,i);
    const scored=[];
    for(const [t,c] of A){
      if(c<2)continue;
      const pa=c/mem.length, pb=(B.get(t)||0)/Math.max(rest.length,1);
      scored.push([(pa+.02)/(pb+.02)*Math.log1p(c),t]);
    }
    scored.sort((x,y)=>y[0]-x[0]);
    const lab=[];
    for(const [,t] of scored){                       // drop "lidar" after "lidar point"
      if(lab.some(k=>k.includes(t)||t.includes(k)))continue;
      lab.push(t); if(lab.length===3)break;
    }
    out.push({n:mem.length, ids:mem, label:lab});
  }
  out.sort((a,b)=>b.n-a.n);
  return {split:true, groups:out};
}

const CY=D.corpora, CYN=CY.map(c=>c.n);
// ---- The landing: which conference, and what moved since last year ---------
// The reader arrives around a conference. The venue blurbs are interface copy —
// one factual line each, no adjectives; every number beside them is counted from
// the payload. Venues not yet collected say so instead of pretending.
const VENUES=[
  {v:'ICML', full:'International Conference on Machine Learning'},
  {v:'NeurIPS', full:'Conference on Neural Information Processing Systems'},
  {v:'ICLR', full:'International Conference on Learning Representations'},
];
function landingHTML(){
  // One door per venue, always the newest year. Last year is not a place anyone
  // browses — it exists to say what changed, and it does that in the chart below
  // and in each paper's neighbours, not as a category of its own.
  const cards=VENUES.map(ven=>{
    const yrs=CY.map((c,j)=>({...c,j})).filter(c=>c.v===ven.v);
    const now=yrs[yrs.length-1];
    return `<button class="vcard ${now?'':'dim'}" ${now?`data-corp="${now.j}"`:'disabled'}>
      <div class="vname">${esc(ven.v)}${now?`<span class="vyr">${now.y}</span>`:''}</div>
      <div class="vfull">${esc(ven.full)}</div>
      ${now?`<div class="vn">${now.n.toLocaleString()} papers</div>`:`<div class="vsoon">not collected yet</div>`}
    </button>`;
  }).join('');
  return `<div class="venues">${cards}</div>`+changedHTML();
}
// The chosen conference lives in the URL hash, so the browser's back button
// returns to the landing and a reload keeps the reader where they were.
const corpFromHash=()=>{const j=CY.findIndex(c=>c.k===location.hash.slice(1));return j<0?null:j;};
function go(j){
  st.corp=j; st.sel=null; st.grouped=false;
  history.pushState(null,'',j===null?location.pathname+location.search:'#'+CY[j].k);
  render(); window.scrollTo({top:0});
}
window.addEventListener('popstate',()=>{ st.corp=corpFromHash(); st.sel=null; render(); });
$('#home').onclick=()=>{ if(st.corp!==null)go(null); };
function wireLanding(){
  document.querySelectorAll('[data-corp]').forEach(el=>el.onclick=()=>go(+el.dataset.corp));
  wireChanged();
}

function render(){
  const landing=st.corp===null;
  $('#landing').hidden=!landing;
  $('#rail').hidden=landing;
  document.querySelector('.wrap').classList.toggle('withrail',!landing);
  document.querySelector('.search').hidden=landing;
  document.querySelector('.sug').hidden=landing;
  if(landing){
    $('#meter').hidden=true; $('#legend').hidden=true; $('#results').innerHTML='';
    $('#landing').innerHTML=landingHTML();
    wireLanding();
    return;
  }
  // The landing is used once; after that the rail moves between venues freely.
  // Filters survive a switch — the vocabulary is shared, so "robotics" means the
  // same thing at the next conference.
  $('#rail').innerHTML=VENUES.map(ven=>{
    const yrs=CY.map((c,j)=>({...c,j})).filter(c=>c.v===ven.v);
    const now=yrs[yrs.length-1];
    if(!now)return `<span class="rv dim">${esc(ven.v)}</span>`;
    return `<button class="rv ${st.corp===now.j?'on':''}" data-rv="${now.j}">`+
      `${esc(ven.v)}<span>${now.y}</span></button>`;
  }).join('');
  // The rail carries whatever the reader should press NEXT: before a pick, this
  // venue's own movers (each row applies itself as the selection); after one,
  // the composition of the set they picked (see railSetCard).
  $('#rail').insertAdjacentHTML('beforeend','<div class="rtr" id="rtr"></div>');
  $('#rail').querySelectorAll('[data-rv]').forEach(el=>el.onclick=()=>{
    if(+el.dataset.rv!==st.corp)go(+el.dataset.rv); });
  const on=chosen();
  drawXref();
  $('#meter').hidden=!on;
  const gt=$('#grptog'); if(gt){ gt.textContent=st.grouped?'Show papers':'Split into subgroups';
    gt.classList.toggle('on',st.grouped); }
  $('#legend').hidden=!on;
  if(!on){
    $('#results').innerHTML=`<div class="start">Pick a topic or a benchmark above, or search.`+
      `<span>Nothing is listed until you do — ${CY[st.corp].n.toLocaleString()} papers is the problem, not the answer.</span></div>`;
    const rt=$('#rtr'); if(rt){ rt.innerHTML=railTrend();
      rt.querySelectorAll('[data-tid]').forEach(el=>el.onclick=()=>{
        applyChgRow('t',+el.dataset.tid); render();}); }
    drawTopics(); drawMethods(); drawDatasets(); fitChips();
    return;
  }
  const res=results();
  { const rt=$('#rtr'); if(rt){ rt.innerHTML=railSetCard(res);
      rt.querySelectorAll('[data-ck]').forEach(el=>el.onclick=()=>{
        const id=+el.dataset.cid;
        if(el.dataset.ck==='d')st.ds=st.ds===id?null:id; else st.meth=st.meth===id?null:id;
        render();}); } }

  const extra=nearby();
  const show=res.slice(0,MAX_SHOWN);
  // No "show more": nobody reads to the end of 6,637. Past the cap the answer is
  // to narrow, and the count above says how much is not on screen.
  const extraShown=extra.slice(0, res.length>=MAX_SHOWN?0:Math.min(extra.length,MAX_SHOWN-res.length));
  $('#results').innerHTML=(show.map(card).join('')
    +(res.length>show.length
      ? `<div class="capped">Showing the first ${MAX_SHOWN} of ${res.length.toLocaleString()}. `+
        `Add a topic, a benchmark, or a search word to narrow this.</div>` : '')
    +(extraShown.length
      ? `<div class="nearhd">${extraShown.length}${extra.length>extraShown.length?' of '+extra.length:''} more `+
        `that are close in meaning but never use these words`+
        `<span>found through the paper embeddings, not the text — read them as suggestions</span></div>`
        +extraShown.map(card).join('') : ''))
    ||'<div class="empty">No papers match all of these. Remove one.</div>';
  // Denominator is the chosen conference, never the whole payload: the reader
  // entered one proceedings, and "of 9,976" answers a question nobody asked.
  const tot=st.corp!==null?CY[st.corp].n:P.length;
  const pct=tot?100*res.length/tot:0;
  const all=res.length===tot;
  const VL=st.corp!==null?`${CY[st.corp].v} ${CY[st.corp].y}`:'all';
  // Orals are not a filter — the tier shows as a badge, the set is already
  // ordered orals first, and this line says how many the selection holds.
  let nOral=0,nSpot=0;
  for(const i of res){ if(P[i].o===1)nOral++; else if(P[i].o===2)nSpot++; }
  const tiers=(nOral||nSpot)?` <span class="of">·${nOral?` ${nOral} oral${nOral>1?'s':''}`:''}`+
    `${nSpot?` ${nOral?'and ':''}${nSpot} spotlight${nSpot>1?'s':''}`:''}, listed first</span>`:'';
  $('#count').innerHTML=all
    ? `<b>${tot.toLocaleString()}</b> <span class="of">papers — all of ${esc(VL)}</span>`
    : `<b>${res.length.toLocaleString()}</b> <span class="of">of ${tot.toLocaleString()} ${esc(VL)} papers</span>`
      +` <span class="pct">${pct<0.1&&res.length?'<0.1':pct.toFixed(1)}%</span>`+tiers;
  $('#mfill').style.width=(res.length?Math.max(pct,0.35):0)+'%';
  $('#meter').querySelector('.mtrack').classList.toggle('full',all);
  if(st.grouped){ renderGrouped(res); return; }
  // Say what is currently narrowing the set — with two kinds of chip active at
  // once, "11 papers" on its own does not explain itself.
  {
    const parts=[];
    if(st.fams.size)parts.push([...st.fams].map(k=>k.split('|')[1]).join(' or '));
    if(st.topics.size)parts.push([...st.topics].map(t=>T[t].l).join(' or '));
    if(st.meth!==null)parts.push('built on '+MV[st.meth]);
    else if(st.mfam!==null)parts.push('built on '+MF[st.mfam]);
    if(st.ds!==null)parts.push('using '+dname(st.ds));
    if(st.q.length)parts.push('matching “'+st.q.join(' ')+'”');
    $('#ctx').textContent=parts.join(' · ');
  }
  $('#results').querySelectorAll('[data-mail]').forEach(el=>el.onclick=async ev=>{
    ev.stopPropagation();
    const addr=el.dataset.mail, was=el.textContent;
    try{ await navigator.clipboard.writeText(addr); }
    catch(e){ const t=document.createElement('textarea'); t.value=addr;
      document.body.appendChild(t); t.select(); document.execCommand('copy'); t.remove(); }
    el.textContent='copied'; el.classList.add('done');
    setTimeout(()=>{ el.textContent=was; el.classList.remove('done'); },1200);});
  $('#results').querySelectorAll('[data-sim]').forEach(el=>el.onclick=ev=>{
    ev.stopPropagation();
    openPanel(+el.dataset.sim);});
  drawTopics(); drawMethods(); drawDatasets(); fitChips();
}

// The count shown is the count returned, within whatever else is chosen.
const TOPIC_OF=P.map(p=>new Set(p.g.map(x=>x[0])));
// facet-qualified family key, so "Trust & safety" under Topic and a future
// same-named domain family never collide.
const TOPIC_PAPERS=T.map(()=>[]);
P.forEach((p,i)=>{ for(const [ti] of p.g) TOPIC_PAPERS[ti].push(i); });
const FAMKEY=T.map(t=>(t.f===1?'#domfams|':'#topfams|')+(t.F||'Other'));
// Clip a chip strip at a whole row. Chips wrap, so the cut point depends on
// viewport width and font metrics — measure it rather than guess a pixel height.
const CHIP_ROWS=2;
function clipRow(boxSel,togSel,noun){
  const box=$(boxSel), tog=$(togSel);
  if(!box||!tog)return;
  const total=rowCount[boxSel]??0;
  box.classList.remove('clip'); box.style.maxHeight='';
  const items=[...box.children];
  if(!items.length){tog.hidden=true;return;}
  const tops=[...new Set(items.map(el=>el.offsetTop))].sort((a,b)=>a-b);
  if(tops.length<=CHIP_ROWS){tog.hidden=true;return;}
  tog.hidden=false;
  tog.textContent=openRow[boxSel]?`Show fewer ${noun}`:`Show all ${total} ${noun}`;
  if(openRow[boxSel])return;
  box.style.maxHeight=(tops[CHIP_ROWS-1]-tops[0]+items[0].offsetHeight)+'px';
  box.classList.add('clip');
}
function fitChips(){
  clipRow('#dslist','#dstog','benchmarks');
}

function countInto(base){
  const n=new Array(T.length).fill(0);
  for(const i of base) for(const t of TOPIC_OF[i]) n[t]++;
  return n;
}

function drawTopics(){
  // Two facets, because that is the only first level this data supports —
  // see declared_split() in taxonomy.py for the three groupings that failed.
  drawFacet('#chips','#chiptog','#tophint',i=>T[i].f!==1,n,'topics');
  drawDomains(n);
}

// Domains are shown by family. The families are curated, not derived — the topic
// centroids all sit within 0.85-0.92 cosine of each other, so the embeddings have
// no resolving power at this level and pretending otherwise would be a fiction.
// Both facets render the same way: a single flowing row of family chips, and the
// members of whichever families are open underneath. Collapsing them into a
// vertical stack saved nothing — five stacked rows occupy the same screen as five
// rows of chips. Flowing them horizontally is what actually shrinks the picker.
function drawFamilies(famSel,openSel,hintSel,pick,n,noun,base){
  const rows=T.map((t,i)=>[i,n[i]])
              .filter(([i,c])=>pick(i)&&!T[i].j&&(c>0||st.topics.has(i)));
  const fams=new Map();
  for(const [i,c] of rows){ const f=T[i].F||'Other';
    if(!fams.has(f))fams.set(f,[]); fams.get(f).push([i,c]); }
  const order=[...fams.entries()].sort((a,b)=>
    b[1].reduce((s,x)=>s+x[1],0)-a[1].reduce((s,x)=>s+x[1],0));

  $(famSel).innerHTML=order.map(([f,list])=>{
    const key=famSel+'|'+f, on=openFam.has(key), sel=st.fams.has(key);
    const picked=list.filter(([i])=>st.topics.has(i)).length;
    // PAPERS, not labels — and only those still in play, or the family chip
    // would claim more than the share bar under it can show.
    const seen=new Set();
    for(const [i] of list) for(const j of TOPIC_PAPERS[i]) if(base.has(j)) seen.add(j);
    return `<button class="famchip ${on?'open':''} ${sel?'sel':''} ${picked?'haspick':''}" `+
      `data-fam="${esc(key)}" title="${list.length} topics">`+
      `<span class="car">${on?'▾':'▸'}</span>${esc(f)}`+
      `<span class="n">${seen.size.toLocaleString()}</span>`+
      `${picked?`<span class="pk">${picked} picked</span>`:''}</button>`;
  }).join('')||`<span style="color:var(--mut);font-size:12px">none apply to these papers</span>`;

  $(openSel).innerHTML=order.filter(([f])=>openFam.has(famSel+'|'+f)).map(([f,list])=>{
    list.sort((a,b)=>b[1]-a[1]);
    return `<div class="famopenrow"><div class="famname">${esc(f)}</div><div class="famchips">`+
      list.map(([i,c])=>`<span class="chip ${st.topics.has(i)?'on':''}" data-t="${i}">`+
        `${T[i].u!=null?'<span class="kid">↳</span>':''}${esc(T[i].l)}`+
        `<span class="n">${c}</span></span>`).join('')+`</div></div>`;
  }).join('');

  const total=T.filter((t,i)=>pick(i)&&!t.j).length;
  const cv=D.coverage, isDom=noun==='domains';
  let reach=0,tot=0;
  for(let i=0;i<P.length;i++){ if(st.corp!==null&&P[i].cy!==st.corp)continue; tot++;
    for(const [ti] of P[i].g) if((T[ti].f===1)===isDom){reach++;break;} }
  $(hintSel).innerHTML=(rows.length<total
      ? `${rows.length} of ${total} ${noun} apply here`
      : `${total} ${noun} in ${order.length} groups`)
    +` · <span class="reach">covers ${reach.toLocaleString()} of ${tot.toLocaleString()} papers`
    +(isDom?`, because only ${cv.named_domain.toLocaleString()} name a domain at all — `
           +`the rest is method work with no application area`:'')+`</span>`;

  for(const sel of [famSel,openSel]){
    $(sel).querySelectorAll('[data-fam]').forEach(el=>el.onclick=()=>{
      // one click does both: pick the family AND open it, so the share bar moves
      // and the way to narrow further is already on screen.
      const k=el.dataset.fam;
      if(st.fams.has(k)){
        st.fams.delete(k); openFam.delete(k);
        // and everything picked under it — a family that is off must not keep
        // filtering (and colouring) through the topics it opened
        for(let i=0;i<T.length;i++) if(FAMKEY[i]===k) st.topics.delete(i);
      }
      else { st.fams.add(k); openFam.add(k); }
      render();});
    $(sel).querySelectorAll('.chip').forEach(el=>el.onclick=()=>{
      const t=+el.dataset.t; st.topics.has(t)?st.topics.delete(t):st.topics.add(t); render();});
  }
}

function countInto(base){
  const n=new Array(T.length).fill(0);
  for(const i of base) for(const t of TOPIC_OF[i]) n[t]++;
  return n;
}

function drawTopics(){
  // Domains were dropped on purpose: the core idea matters more than where it is
  // applied, and only a third of papers name an application area at all. The
  // facet is still computed in taxonomy.py, so bringing the row back is markup,
  // not a re-derivation — and every domain word still works in the search box.
  const base=baseSet('topics');
  drawFamilies('#topfams','#topopen','#tophint',i=>T[i].f!==1,
               countInto(base),'topics',base);
}

// ---- suggestion list: building blocks — what a paper stands on ----
const MV=D.mvocab||[], MS=D.methods||[], MF=D.mfams||[];
// ---- Since last year — the entry answer to "what is rising?" ---------------
// Three measurable readings of that question, and nothing beyond them: which
// fields take a larger share of the conference than last year, which barely
// existed last year (marked "new" — a small delta can be the bigger signal),
// and which take less. Everything is a counted share per 1,000 papers, from the
// same abstract-pass extraction for both years (Guardrail 5); sizes differ 2×,
// so only shares compare. The glyph is a dumbbell — last year's dot, this
// year's dot, one scale — because a delta bar hides the base: +6 from 1.8 is a
// 4× rise, +6 from 22 is a fifth. Two editions are a difference, not a trend
// line, and no row is called hot or important (Guardrail 1) — the benchmark tab
// is the closest thing to that claim we allow ourselves: where the field tests
// is a commitment, not a word choice.
// Each venue's latest edition vs its previous one. The pair is chosen per
// venue — never across venues — and every number below is a share per 1,000
// of its own edition, because edition sizes differ 2x.
function venuePairs(){
  const out=[];
  VENUES.forEach((ven,hue)=>{
    const ys=CY.map((c,j)=>({...c,j})).filter(c=>c.v===ven.v);
    if(ys.length>=2)out.push({v:ven.v,hue,c0:ys[ys.length-2].j,c1:ys[ys.length-1].j,
                              y0:ys[ys.length-2].y,y1:ys[ys.length-1].y});
  });
  return out;
}
const CHG={};
function changedRows(c0,c1){
  const key=c0+':'+c1;
  if(CHG[key])return CHG[key];
  const out={t:[],m:[],d:[]};
  const push=(arr,l,id,a,b,min)=>{
    if(a+b<min)return;
    const s0=a/CYN[c0]*1000, s1=b/CYN[c1]*1000;
    arr.push({l,id,a,b,s0,s1});
  };
  T.forEach((t,ti)=>{
    if(t.j)return;
    let a=0,b=0;
    for(const i of TOPIC_PAPERS[ti]){ if(P[i].cy===c0)a++; else if(P[i].cy===c1)b++; }
    push(out.t,t.l,ti,a,b,12);
  });
  // methods: children roll up to their top-level parent, as the menu does
  const par=new Map(); for(const [id,,,pa] of MS) par.set(id,pa==null?id:pa);
  const mc=new Map();
  for(const p of P){ if(p.cy!==c0&&p.cy!==c1)continue; const seen=new Set();
    for(const m of (p.mu||[])){ const top=par.get(m)??m;
      if(seen.has(top))continue; seen.add(top);
      let c=mc.get(top); if(!c)mc.set(top,c=new Map()); c.set(p.cy,(c.get(p.cy)||0)+1); } }
  for(const [id,,,pa] of MS){
    if(pa!=null)continue;
    const c=mc.get(id);
    push(out.m,MV[id],id,c?.get(c0)||0,c?.get(c1)||0,12);
  }
  // benchmarks: the ids DS already filtered (placeholders like "three datasets"
  // are counts wearing a name and would top any list)
  const ok=new Set(DS.map(([di])=>di));
  // "MATH-500" and "MATH500" are one benchmark spelled two ways; fold rows whose
  // letters and digits agree, keep the commoner spelling. Display-time only —
  // the underlying alias problem (LIBERO vs "LIBERO benchmark") is still open.
  const dc=new Map();
  for(const p of P){ if(p.cy!==c0&&p.cy!==c1)continue; const seen=new Set();
    for(const di of new Set(p.k)){ if(!ok.has(di))continue;
      const key2=dname(di).toLowerCase().replace(/[^a-z0-9]/g,'');
      if(seen.has(key2))continue; seen.add(key2);
      let c=dc.get(key2); if(!c)dc.set(key2,c={n:new Map(),a:0,b:0});
      c.n.set(di,(c.n.get(di)||0)+1); if(p.cy===c0)c.a++; else c.b++; } }
  for(const c of dc.values()){
    const di=[...c.n.entries()].sort((x,y)=>y[1]-x[1])[0][0];
    push(out.d,dname(di),di,c.a,c.b,8);
  }
  return CHG[key]=out;
}
let chgTab='t';
// Which rows earn ink — a rule, not a top-N. Per venue pair, a row must pass
// TWO tests at once:
//   real      |z| >= 2.576 — a two-proportion test at 99%, so the edition sizes
//             decide what is sampling noise, not a hand-picked cutoff;
//   material  the share moved by >= 2 per 1,000 OR by >= 1.5x.
// With several venues loaded, agreement is the tiebreak: a topic that clears the
// test at three venues outranks one that clears at one, and a row where venues
// pull in OPPOSITE directions is not drawn at all — a disagreement stated as a
// trend would be a lie. Sections replace arrows and numbers: LARGEST holds the
// two biggest current shares whatever their change (context — "rose" means
// nothing without what is big), RISING/NEW/FALLING carry the direction.
const Z_SHOW=2.576, LABEL_CAP=5, D_MIN=2, FOLD_MIN=1.5;
function sections(tab){
  const pairs=venuePairs();
  if(!pairs.length)return null;
  const minS=tab==='d'?2:3;
  let tot1=0; for(const pr of pairs)tot1+=CYN[pr.c1];
  const by=new Map();
  for(const pr of pairs){
    for(const r of changedRows(pr.c0,pr.c1)[tab]){
      const n0=CYN[pr.c0], n1=CYN[pr.c1];
      const pp=(r.a+r.b)/(n0+n1), se=Math.sqrt(pp*(1-pp)*(1/n0+1/n1));
      const z=se?(r.b/n1-r.a/n0)/se:0;
      const lo=Math.min(r.s0,r.s1), hi=Math.max(r.s0,r.s1);
      const mat=Math.abs(r.s1-r.s0)>=D_MIN||(lo>0?hi/lo:1e9)>=FOLD_MIN;
      let e=by.get(r.l); if(!e)by.set(r.l,e={l:r.l,id:r.id,lanes:[],b:0});
      e.b+=r.b;
      e.lanes.push({hue:pr.hue,s0:r.s0,s1:r.s1,z,mat,nw:r.a<=2&&r.b>2,gn:r.b<=2&&r.a>2});
    }
  }
  const rows=[...by.values()].filter(e=>e.lanes.some(x=>Math.max(x.s0,x.s1)>=minS));
  for(const e of rows){
    e.up=e.lanes.filter(x=>x.z>=Z_SHOW&&x.mat).length;
    e.dn=e.lanes.filter(x=>x.z<=-Z_SHOW&&x.mat).length;
    e.nw=e.lanes.some(x=>x.nw); e.gn=e.lanes.some(x=>x.gn);
    e.maxz=Math.max(...e.lanes.map(x=>Math.abs(x.z)));
    e.s1u=e.b/tot1*1000;
  }
  const rising=rows.filter(e=>e.up&&!e.dn&&!e.nw)
    .sort((x,y)=>y.up-x.up||y.maxz-x.maxz).slice(0,LABEL_CAP);
  const fresh=rows.filter(e=>e.up&&!e.dn&&e.nw)
    .sort((x,y)=>y.s1u-x.s1u).slice(0,LABEL_CAP);
  const falling=rows.filter(e=>e.dn&&!e.up)
    .sort((x,y)=>y.maxz-x.maxz).slice(0,LABEL_CAP);
  const inSec=new Set([...rising,...fresh,...falling]);
  const largest=rows.filter(e=>!inSec.has(e)).sort((x,y)=>y.s1u-x.s1u).slice(0,2);
  return {pairs,largest,rising,fresh,falling};
}
// Log x. The linear form could not show both size and growth: length is an
// absolute encoding, so 0.2->0.9% (a 4.5x rise) was invisible next to a big
// flat bar. On a log axis equal tails are equal FOLD changes, which is the
// comparison the sections are making — and the axis says so with plain ticks,
// not a caption. Bars start at the 0.1% floor; nothing below it is drawn.
const CHG_F=1;   // floor, per-1,000
function logX(M){ return v=>v<=CHG_F?0:Math.log(v/CHG_F)/Math.log(M/CHG_F)*100; }
function laneHTML(x,grey){
  const dark=grey?'#9a9c9e':`var(--v${x.hue})`;
  const pale=grey?'#dcdee0':`color-mix(in srgb, var(--v${x.hue}) 26%, var(--card))`;
  const t=`title="${(x.s0/10).toFixed(1)}% → ${(x.s1/10).toFixed(1)}%"`;
  const w0=x.w0.toFixed(1), w1=Math.max(x.w1,.6).toFixed(1);
  return `<span class="lane" ${t}>`+
    (x.w0>x.w1
      ?`<i style="width:${w0}%;background:${pale}"></i><i style="width:${w1}%;background:${dark}"></i>`
      :`<i style="width:${w1}%;background:${dark}"></i><i style="width:${w0}%;background:${pale}"></i>`)+
    `</span>`;
}
function changedHTML(){
  const S=sections(chgTab);
  if(!S)return '';
  const secs=[['largest',S.largest],['rising',S.rising],['new',S.fresh],['falling',S.falling]]
    .filter(x=>x[1].length);
  if(!secs.length)return '';
  const all=secs.flatMap(x=>x[1]);
  const M=Math.max(...all.flatMap(e=>e.lanes.flatMap(x=>[x.s0,x.s1])),CHG_F*2);
  const X=logX(M);
  const ticks=[1,3,10,30,100].filter(t=>t<=M*1.04);
  const grid='<div class="chgg">'+ticks.map(t=>`<i style="left:${X(t).toFixed(2)}%"></i>`).join('')+'</div>';
  const axis='<div class="chgax">'+ticks.map(t=>`<b style="left:${X(t).toFixed(2)}%">${t/10}%</b>`).join('')+'</div>';
  const body=secs.map(([name,rows])=>`<div class="chgsec">${name}</div>`+rows.map(e=>{
    const lanes=e.lanes.map(x=>laneHTML({...x,w0:X(x.s0),w1:X(x.s1)},name==='largest')).join('');
    const tag=e.gn?'<em class="tag gone">gone</em>':'';
    return `<button class="cr" data-k="${chgTab}" data-id="${e.id}">`+
      `<span class="crl" title="${esc(e.l)}">${esc(e.l)}${tag}</span><span class="trk">${lanes}</span></button>`;
  }).join('')).join('');
  const multi=S.pairs.length>1;
  const leg=multi
    ?`<div class="chgleg">${S.pairs.map(p=>`<span><i style="background:var(--v${p.hue})"></i>${esc(p.v)}</span>`).join('')}</div>`
    :`<div class="chgleg"><span><i style="background:color-mix(in srgb, var(--v${S.pairs[0].hue}) 26%, var(--card))"></i>${S.pairs[0].y0}</span>`+
     `<span><i style="background:var(--v${S.pairs[0].hue})"></i>${S.pairs[0].y1}</span></div>`;
  const tab=(k,l)=>`<button class="chgtab ${chgTab===k?'on':''}" data-tab="${k}">${l}</button>`;
  return `<div class="chgbox"><div class="chghd">Since last year`+
    `<span class="chgtabs">${tab('t','topics')}${tab('m','methods')}${tab('d','benchmarks')}</span></div>`+
    `<div class="chgplot">${grid}${body}</div>${axis}${leg}</div>`;
}
function applyChgRow(k,id){
  st.q=[]; const q=$('#q'); if(q)q.value='';
  st.sel=null; st.grouped=false;
  st.topics.clear(); st.fams.clear(); st.meth=null; st.mfam=null; st.ds=null;
  if(k==='t')st.topics.add(id); else if(k==='m')st.meth=id; else st.ds=id;
}
// The rail's before-a-pick view: this venue's own significant movers, the same
// two-test rule as the landing, one pair-lane each. A row IS a filter — clicking
// it applies the topic, because a trend a reader cannot act on is trivia.
function railTrend(){
  const pr=venuePairs().find(p=>p.c1===st.corp||p.c0===st.corp);
  if(!pr)return '';
  const rows=[];
  for(const r of changedRows(pr.c0,pr.c1).t){
    const n0=CYN[pr.c0], n1=CYN[pr.c1];
    const pp=(r.a+r.b)/(n0+n1), se=Math.sqrt(pp*(1-pp)*(1/n0+1/n1));
    const z=se?(r.b/n1-r.a/n0)/se:0;
    const lo=Math.min(r.s0,r.s1), hi=Math.max(r.s0,r.s1);
    const mat=Math.abs(r.s1-r.s0)>=D_MIN||(lo>0?hi/lo:1e9)>=FOLD_MIN;
    if(Math.max(r.s0,r.s1)>=3&&Math.abs(z)>=Z_SHOW&&mat)rows.push({...r,z});
  }
  rows.sort((x,y)=>Math.abs(y.z)-Math.abs(x.z));
  const top=rows.slice(0,8);
  if(!top.length)return '';
  const M=Math.max(...top.flatMap(r=>[r.s0,r.s1]),CHG_F*2);
  const X=logX(M);
  return `<div class="rh">Since last year <em>${pr.y0} → ${pr.y1}</em></div>`+
    top.map(r=>{
      const tag=r.a<=2&&r.b>2?'<em class="tag">new</em>'
               :r.b<=2&&r.a>2?'<em class="tag gone">gone</em>':'';
      return `<button class="mrr" data-tid="${r.id}"><span class="mrl" title="${esc(r.l)}">${esc(r.l)}${tag}</span>`+
        `<span class="mrt">${laneHTML({hue:pr.hue,s0:r.s0,s1:r.s1,w0:X(r.s0),w1:X(r.s1)},false)}</span></button>`;
    }).join('');
}
// After a pick: what the chosen set is MADE OF, held sticky while the list
// scrolls. Same selection counted in each edition of this venue (share of that
// year — sizes differ 2x), then the vocabulary that recurs inside the set,
// scoped tighter than the corpus-wide menus above the results.
function railSetCard(res){
  const hits=queryHits();
  const cur=CY[st.corp];
  const hue=Math.max(VENUES.findIndex(v=>v.v===cur.v),0);
  const sib=CY.map((c,j)=>({c,j})).filter(x=>x.c.v===cur.v);
  const ys=sib.map(({c,j})=>{
    let n=0; for(let i=0;i<P.length;i++) if(match(i,hits,j))n++;
    return {y:c.y,n,sh:n/CYN[j]*1000};
  });
  const mx=Math.max(...ys.map(r=>r.sh),1e-9);
  const yrows=ys.map((r,ix)=>{
    const col=ix===ys.length-1?`var(--v${hue})`
              :`color-mix(in srgb, var(--v${hue}) 26%, var(--card))`;
    return `<div class="scyrow"><span>${r.y}</span>`+
      `<span class="yb"><i style="width:${Math.max(r.sh/mx*100,1.5).toFixed(1)}%;background:${col}"></i></span>`+
      `<b>${r.n} · ${(r.sh/10).toFixed(1)}%</b></div>`;
  }).join('');
  const dk=new Map(), mk=new Map();
  const par=new Map(); for(const [id,,,pa] of MS)par.set(id,pa==null?id:pa);
  for(const i of res){
    for(const d of new Set(P[i].k)) if(DSSET.has(d)&&d!==st.ds)dk.set(d,(dk.get(d)||0)+1);
    const seen=new Set();
    for(const m of (P[i].mu||[])){ const t2=par.get(m)??m;
      if(seen.has(t2))continue; seen.add(t2);
      if(t2!==st.meth)mk.set(t2,(mk.get(t2)||0)+1); }
  }
  const chips=(map,kind,name)=>[...map.entries()].filter(([,c])=>c>=2)
    .sort((a,b)=>b[1]-a[1]).slice(0,4)
    .map(([id,c])=>`<button class="scchip" data-ck="${kind}" data-cid="${id}">${esc(name(id))}<b>${c}</b></button>`).join('');
  const dch=chips(dk,'d',dname), mch=chips(mk,'m',i=>MV[i]);
  const nf=res.filter(i=>P[i].f).length;
  return `<div class="setcard"><div class="rh">This set</div>`+
    `<div class="scn">${res.length.toLocaleString()}<small>papers</small></div>`+
    `<div class="scyr">${yrows}</div>`+
    (sib.length>1?`<div class="scsub">the same pick, in each edition — share of that year</div>`:'')+
    (dch?`<div class="rh" style="margin-top:11px">Tested on <em>in this set</em></div><div class="scchips">${dch}</div>`:'')+
    (mch?`<div class="rh" style="margin-top:11px">Builds on <em>in this set</em></div><div class="scchips">${mch}</div>`:'')+
    (nf?`<div class="scsub" style="margin-top:9px">${nf} of ${res.length} cards use full text</div>`:'')+
    `</div>`;
}
function wireChanged(){
  document.querySelectorAll('[data-tab]').forEach(el=>el.onclick=()=>{
    chgTab=el.dataset.tab; render();});
  document.querySelectorAll('.cr[data-id]').forEach(el=>el.onclick=()=>{
    applyChgRow(el.dataset.k,+el.dataset.id);
    const j=CY.length-1;
    if(st.corp!==j)go(j); else render();
  });
}

function renderGrouped(res){
  const g=groupsFor(res.slice(0,400));
  let html='';
  if(!g) html='<div class="capped">Too few papers to group. Pick a wider set.</div>';
  else if(!g.split) html='<div class="capped">This set does not split — no grouping holds '+
    'together without leaving a group of one or two, or putting nearly all of them in one pile. '+
    'These papers are variations on the same thing.</div>';
  else html=g.groups.map(gr=>
    `<div class="grp"><div class="grphd">${gr.label.map(esc).join('<span class="sep"> · </span>')||
      '<i>no vocabulary sets these apart</i>'}<span class="n">${gr.n} papers</span></div>`+
    gr.ids.slice(0,6).map(i=>`<div class="grow" data-i="${i}">${esc(P[i].t)}</div>`).join('')+
    (gr.ids.length>6?`<div class="gmore">and ${gr.ids.length-6} more</div>`:'')+`</div>`).join('');
  $('#results').innerHTML=html;
  $('#results').querySelectorAll('[data-i]').forEach(el=>el.onclick=()=>{
    st.grouped=false; st.q=[]; $('#q').value=''; st.sel=+el.dataset.i; render();
    $(`.p[data-i="${st.sel}"]`)?.scrollIntoView({block:'center'});});
}

function drawMethods(){
  const base=baseSet('meth');
  const cnt=new Map();
  for(const i of base) for(const m of new Set(P[i].mu||[])) cnt.set(m,(cnt.get(m)||0)+1);
  // a child rolls its papers up to its parent, so picking "attention" does not
  // silently exclude the six kinds of attention that were named separately
  const kidsOf=new Map(), meta=new Map();
  for(const [id,n,f,par] of MS){ meta.set(id,{n,f,par});
    if(par!=null){ if(!kidsOf.has(par))kidsOf.set(par,[]); kidsOf.get(par).push(id);} }
  const papersOf=id=>{
    const s=new Set(); const add=x=>{ for(let i=0;i<P.length;i++) if((P[i].mu||[]).includes(x)&&base.has(i)) s.add(i); };
    add(id); for(const k of (kidsOf.get(id)||[])) add(k); return s;
  };
  const fams=new Map();
  for(const [id,n,f,par] of MS){
    if(par!=null)continue;
    const c=cnt.get(id)||0;
    let tot=c; for(const k of (kidsOf.get(id)||[])) tot+=cnt.get(k)||0;
    if(!tot&&st.meth!==id)continue;
    if(!fams.has(f))fams.set(f,[]); fams.get(f).push([id,tot]);
  }
  const order=[...fams.entries()].sort((a,b)=>{
    const oa=MF[a[0]]==='Other', ob=MF[b[0]]==='Other';
    if(oa!==ob)return oa?1:-1;
    return b[1].reduce((s,x)=>s+x[1],0)-a[1].reduce((s,x)=>s+x[1],0);});

  $('#mfams').innerHTML=order.map(([f,list])=>{
    const key='#mfams|'+MF[f], on=openFam.has(key), sel=st.mfam===f;
    const seen=new Set(); for(const [id] of list) for(const i of papersOf(id)) seen.add(i);
    return `<button class="famchip ${on?'open':''} ${sel?'sel':''}" data-mfam="${f}" `+
      `title="${list.length} methods">`+
      `<span class="car">${on?'▾':'▸'}</span>${esc(MF[f])}`+
      `<span class="n">${seen.size.toLocaleString()}</span></button>`;
  }).join('')||'<span style="color:var(--mut);font-size:12px">no method is shared by these papers</span>';

  $('#mopen').innerHTML=order.filter(([f])=>openFam.has('#mfams|'+MF[f])).map(([f,list])=>{
    list.sort((a,b)=>b[1]-a[1]);
    return `<div class="famopenrow"><div class="famname">${esc(MF[f])}</div><div class="famchips">`+
      list.slice(0,60).map(([id,c])=>{
        const k=kidsOf.get(id)||[];
        return `<span class="chip ${st.meth===id?'on':''}" data-m="${id}">${esc(MV[id])}`+
          `${k.length?`<span class="kn">+${k.length}</span>`:''}<span class="n">${c}</span></span>`;
      }).join('')+`</div></div>`;
  }).join('');

  $('#mhint').innerHTML=`${MS.length} used by 3+ papers in ${order.length} groups`
    +(()=>{let r=0,t=0;for(let i=0;i<P.length;i++){if(st.corp!==null&&P[i].cy!==st.corp)continue;t++;
      if((P[i].mu||[]).length)r++;}
      return ` · <span class="reach">covers ${r.toLocaleString()} of ${t.toLocaleString()} papers</span>`;})();
  for(const sel of ['#mfams','#mopen']){
    $(sel).querySelectorAll('[data-mfam]').forEach(el=>el.onclick=()=>{
      const f=+el.dataset.mfam, key='#mfams|'+MF[f];
      if(st.mfam===f){ st.mfam=null; openFam.delete(key); }
      else { st.mfam=f; openFam.add(key); st.meth=null; }
      render();});
    $(sel).querySelectorAll('.chip').forEach(el=>el.onclick=()=>{
      const m=+el.dataset.m; st.meth=st.meth===m?null:m; st.mfam=null; render();});
  }
}

// ---- suggestion list: benchmarks, the highest-trust anchor in the corpus ----
const DS=D.datasets, DSSET=new Set(DS.map(([d])=>d));
function drawDatasets(){
  const base=baseSet('ds');
  const narrowed=base.size<P.length;
  const cnt=new Map();
  if(narrowed) for(const i of base) for(const d of new Set(P[i].k)) cnt.set(d,(cnt.get(d)||0)+1);
  const rows=(narrowed
    ? [...cnt].filter(([d,c])=>c>=2&&DSSET.has(d)).sort((a,b)=>b[1]-a[1])
    : DS).slice(0,160);
  if(narrowed&&st.ds!==null&&!rows.some(([d])=>d===st.ds))
    rows.unshift([st.ds,cnt.get(st.ds)||0]);
  rowCount['#dslist']=rows.length;
  $('#dslist').innerHTML=rows.map(([di,c])=>
    `<span class="ds ${st.ds===di?'on':''}" data-d="${di}">${esc(dname(di))}<span class="n">${c}</span></span>`).join('')
    ||'<span style="color:var(--mut);font-size:12px">No benchmark is shared by two or more of these papers.</span>';
  $('#dslist').querySelectorAll('.ds').forEach(el=>el.onclick=()=>{
    const d=+el.dataset.d; st.ds=st.ds===d?null:d; render();});
  $('#dshint').textContent=narrowed
    ? `${rows.length} used by 2+ of the papers now selected, of ${DS.length}`
    : `${DS.length} used by 3+ papers · ${D.n_datasets_real.toLocaleString()} named in all`;
  fitChips();
}
drawDatasets();
drawMethods();

$('#q').addEventListener('input',e=>{
  st.q=e.target.value.toLowerCase().split(/\s+/).filter(Boolean);
  render();});
$('#pclose').onclick=closePanel;
$('#grptog').onclick=()=>{st.grouped=!st.grouped;render();};
$('#dstog').onclick=()=>{openRow['#dslist']=!openRow['#dslist'];render();};
addEventListener('resize',fitChips);

const c=D.coverage;
// The coverage caveats are no longer a paragraph at the foot of the page. Each
// one now sits where it actually bites: the tag share next to the topic list, the
// "looks similar" mark on the papers it applies to, and "no sentence states what
// is new" on the cards that have none. Stating them in place beats a block that
// is read once and then ignored.

st.corp=corpFromHash();
render();
fitChips();
</script></body></html>"""


ROBOTS = "User-agent: *\nDisallow: /\n"

# Cloudflare Pages / Netlify read this file. The meta tag already asks crawlers
# not to index; the header says the same thing to fetchers that ignore HTML.
HEADERS = "/*\n  X-Robots-Tag: noindex, nofollow\n"


def write_dist(html: str):
    """Emit a folder that can be dragged onto a static host as-is."""
    d = ROOT / "dist"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text(html, encoding="utf-8")
    (d / "robots.txt").write_text(ROBOTS, encoding="utf-8")
    (d / "_headers").write_text(HEADERS, encoding="utf-8")
    return d


def main() -> int:
    global VENUE
    ap = argparse.ArgumentParser(description="Build the search interface.")
    ap.add_argument("--spans", choices=["abstract", "fulltext"], default="fulltext",
                    help="prefer full-text spans when available")
    ap.add_argument("--venue", default=VENUE,
                    help="conference name shown beside the counts; the title itself "
                         "stays venue-neutral so other conferences can be added")
    ap.add_argument("--dist", action="store_true",
                    help="also write dist/ — index.html + robots.txt + _headers")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    VENUE = args.venue
    payload = build_payload(args.spans)
    html = HTML.replace("__DATA__", json.dumps(payload, ensure_ascii=False,
                                               separators=(",", ":")))
    out = REPORTS / "index.html" if args.out is None else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    c = payload["coverage"]
    print(f"wrote {out} ({len(html)/1024/1024:.1f} MB)")
    if args.dist:
        print(f"  dist/ ready to upload: {write_dist(html)}")
    print(f"  {c['total']:,} papers · {c['with_span']:,} with a verified span "
          f"({c['with_span']/c['total']:.0%}) · {c['tagged']:,} carrying a topic "
          f"({c['tagged']/c['total']:.0%}) · {len(payload['topics'])} topics")
    if c["from_fulltext"]:
        print(f"  {c['from_fulltext']:,} papers using full-text spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
