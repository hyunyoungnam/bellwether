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


def build_term_index(abstracts: list, bigrams: bool = False) -> dict:
    """Salient abstract vocabulary per paper, interned.

    Terms too rare to link two papers, or too common to separate them, are cut —
    that is what makes the index a third of the size of the prose it replaces.
    """
    docs, df = [], Counter()
    for text in abstracts:
        ws = [w for w in _WORD.findall((text or "").lower())]
        t = {w for w in ws if w not in SEARCH_STOP}
        if bigrams:
            t |= {f"{a} {b}" for a, b in zip(ws, ws[1:])
                  if a not in SEARCH_STOP and b not in SEARCH_STOP}
        docs.append(t)
        df.update(t)
    n = max(len(docs), 1)
    # a bigram must be commoner to earn a slot — df>=2 lets in one-off phrasing
    keep = {w for w, c in df.items()
            if (TERM_DF_MIN if " " not in w else 5) <= c < TERM_DF_MAX * n}
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
    corpora = available()          # every venue far enough through the pipeline
    # mid-pipeline grace: a corpus whose extraction exists but whose labels do
    # not yet (the chain runs label after extract) must wait, not kill the build
    skipped = [c.key for c in corpora if not c.topics.exists()]
    corpora = [c for c in corpora if c.topics.exists()]
    if skipped:
        print(f"  (skipping mid-pipeline corpora: {', '.join(skipped)})")
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
    # Everything counted, compared or filtered on runs on the abstract pass
    # (Guardrail 5): full-text coverage is 99% for the PMLR year and 72% for the
    # arXiv year, so merged fields would let coverage masquerade as trend. The
    # snapshot is taken here, before the merge below mutates `facts` in place.
    abs_of = {g: (f.get("methods") or [], f.get("datasets") or [],
                  f.get("tasks") or [], f.get("limitation") or "")
              for g, f in facts.items()}

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

    # One vocabulary id per benchmark, however it is spelled: surfaces that share
    # a dataset_key intern to the commonest spelling. Fixes the menu/search/V5
    # fragmentation (LIBERO vs "LIBERO benchmark") at the source, not at draw
    # time. is_placeholder still filters at the menu, unchanged.
    from .taxonomy import dataset_key
    ds_surface = Counter()
    for f in facts.values():
        for x in f.get("datasets") or []:
            n = detex(x["name"]).strip()
            if n:
                ds_surface[n] += 1
    ds_disp: dict[str, str] = {}
    for n, _c in ds_surface.most_common():
        k = dataset_key(n)
        if k and k not in ds_disp:
            ds_disp[k] = n

    meth, data, task = Vocab(), Vocab(), Vocab()

    def ds_id(name: str) -> int:
        n = detex(name).strip()
        return data.id(ds_disp.get(dataset_key(n), n))

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
            "k": [ds_id(x["name"]) for x in (f.get("datasets") or [])][:5],
            "s": [task.id(detex(x["name"])) for x in (f.get("tasks") or [])][:3],
            # abstract-pass-only twins of k/s — the ONLY fields countable
            # across papers or years; k/s above are display, marked "full text"
            "k0": [ds_id(x["name"]) for x in abs_of.get(eid, ((),(),()))[1]][:5],
            "s0": [task.id(detex(x["name"])) for x in abs_of.get(eid, ((),(),()))[2]][:3],
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

    # "Who fights my problem" (V5): the same interning trick, but over the
    # papers' own limitation sentences — so a reader can enter by the failure
    # they care about, not the area it sits in. Unigrams plus the bigrams
    # frequent enough to be a name ("catastrophic forgetting").
    lim_texts = []
    for r in rows:
        lim_texts.append(r["L"] or "")
    lims = build_term_index(lim_texts, bigrams=True)
    # …and its abstract-pass twin, the only one countable across years
    lim0_texts = [edit_span((abs_of.get(r["i"]) or ((), (), (), ""))[3])[:SPAN_CHARS]
                  for r in rows]
    lims0 = build_term_index(lim0_texts, bigrams=True)

    # Datasets a reader can enter by. Placeholders ("three datasets") are counts
    # wearing a name and outrank every real dataset if left in.
    from .taxonomy import (is_placeholder, building_blocks, method_key,
                           method_family, METHOD_FAMILIES)
    ds_count: dict[int, int] = defaultdict(int)
    for r in rows:
        for di in set(r["k0"]):
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
        got0 = set()
        for m in abs_of.get(r["i"], ((),))[0]:
            if m.get("role") != "building-block":
                continue
            key, _ = method_key(m["name"])
            if key in mid_of:
                got0.add(mid_of[key])
        r["mu0"] = sorted(got0)
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


    payload_methods = [[mid_of[k], n,
                        mfam_ix[method_family(mdisp.get(k, k), k)],
                        mid_of.get(mparent.get(k))] for n, k in mrows]

    # ------------------------------------------------------------- the digest
    # Analysis -> selection: the landing leads with what MOVED, computed here at
    # build time so first paint needs no scan and no lazy files. Every row is a
    # door — the client applies it as a selection. Scanning 159 topics x four
    # axes is thousands of significance tests, so the two-test rule alone would
    # seed the digest with ~50 chance "trends"; Benjamini-Hochberg at q=0.05,
    # per family, is what makes every printed row real.
    import math

    def _z(a, b, n0, n1):
        pp = (a + b) / (n0 + n1)
        se = math.sqrt(pp * (1 - pp) * (1 / n0 + 1 / n1)) or 1e-12
        return (b / n1 - a / n0) / se

    def _p_of(z):
        return math.erfc(abs(z) / math.sqrt(2))

    # Limitation sentences share a writing register, and register drifts between
    # editions — "rely", "significant", "typically" all shift with huge z while
    # naming no failure. Two guards keep the failure axis about failures: a
    # discourse stoplist, and a df ceiling (a term in >5% of either edition's
    # papers is the genre's phrasing, not a specific failure).
    FIGHT_STOP = set("""rely relies relying significant significantly challenge
        challenges challenging typically usually generally often frequently yet
        still previous prior existing current recent recently explicit implicit
        remains remain remained limited limits limitation limitations struggle
        struggles struggled fail fails failed failure failures difficult
        difficulty hard unable lack lacks lacking without despite however
        moreover furthermore approaches methods models works studies designed
        based large small high low due suffer suffers text several various
        substantial extensive additional certain specific particular real
        overall poor strong weak key main major crucial essential important
        require requires requiring costly expensive prohibitive they treat
        treats evidence standard standards signals signal training benchmarks
        benchmark agents costs cost capture captures potential techniques
        technique algorithms algorithm text llms robot actions largely rather
        principled global merely inherently fundamentally solely primarily
        increasingly especially highly directly effectively naturally internal
        errors error pipelines pipeline outputs output settings setting
        implicitly explicitly jointly separately independently""".split())

    def fight_ok(term: str, a: int, b: int) -> bool:
        words = term.split()
        if all(w in FIGHT_STOP for w in words):
            return False
        if len(words) == 1 and words[0] in FIGHT_STOP:
            return False
        return a / n0 <= 0.05 and b / n1 <= 0.05

    digest = {"pair": None, "mix": [], "fights": [], "fresh": []}
    by_venue: dict[str, list[int]] = defaultdict(list)
    for i2, c in enumerate(corpora):
        by_venue[c.venue].append(i2)
    pair = next((v[-2:] for v in by_venue.values() if len(v) >= 2), None)
    if pair:
        pc0, pc1 = pair
        n0 = sum(1 for r in rows if r["cy"] == pc0)
        n1 = sum(1 for r in rows if r["cy"] == pc1)
        digest["pair"] = {"v": corpora[pc0].venue, "y0": corpora[pc0].year,
                          "y1": corpora[pc1].year, "c0": pc0, "c1": pc1,
                          "n0": n0, "n1": n1}
        row_of = {r["i"]: ix for ix, r in enumerate(rows)}
        # per-paper limitation terms, abstract pass, aligned with rows
        l0_terms = lims0["p"]
        mtop = {}
        for mid, _n, _f, pa in payload_methods:
            mtop[mid] = pa if pa is not None else mid

        def items_of(r, ix, ax):
            if ax == "u":
                return {mtop.get(m, m) for m in r["mu0"]}
            if ax == "s":
                return set(r["s0"])
            if ax == "d":
                return {ti for ti, _k in r["g"] if topics_out[ti]["f"] == 1
                        and not topics_out[ti]["j"]}
            return set(l0_terms[ix])

        name_of = {"u": lambda i2: mvocab.items[i2],
                   "s": lambda i2: task.items[i2],
                   "d": lambda i2: topics_out[i2]["l"],
                   "l": lambda i2: lims0["v"][i2]}

        # ---- mix: inside each topic, which building blocks / tasks / domains /
        # attacked failures took a different share of the set ----
        tests = []
        for ti, t0 in enumerate(topics_out):
            if t0["j"]:
                continue
            members = [g for g, tl in topic_of.items() if ti in tl]
            side = {g: rows[row_of[g]]["cy"] for g in members if g in row_of}
            s0g = [g for g, cyv in side.items() if cyv == pc0]
            s1g = [g for g, cyv in side.items() if cyv == pc1]
            if len(s0g) < 20 or len(s1g) < 20:
                continue
            # u/s/d only: at topic granularity the limitation axis is register
            # noise even after the stoplist; failures get their own corpus-wide
            # section below, where the sample is big enough to hold them.
            for ax in ("u", "s", "d"):
                cnt: dict[int, list[int]] = defaultdict(lambda: [0, 0])
                for sidei, gl in ((0, s0g), (1, s1g)):
                    for g in gl:
                        ix = row_of[g]
                        for item in items_of(rows[ix], ix, ax):
                            if ax == "d" and item == ti:
                                continue
                            cnt[item][sidei] += 1
                for item, (a, b) in cnt.items():
                    if a + b < 6:
                        continue
                    if ax == "l" and not fight_ok(lims0["v"][item], 0, 0):
                        continue
                    sh0, sh1 = a / len(s0g) * 1000, b / len(s1g) * 1000
                    lo, hi = min(sh0, sh1), max(sh0, sh1)
                    z = _z(a, b, len(s0g), len(s1g))
                    moved = (abs(z) >= 2.576
                             and (abs(sh1 - sh0) >= 30 or (lo > 0 and hi / lo >= 1.5)))
                    # appearing is its own evidence: 0 -> 11 papers needs no
                    # z-test to be a fact worth a row (GRPO, RLVR)
                    fresh2 = a <= 1 and b >= 8
                    gone2 = b <= 1 and a >= 8
                    if not (moved or fresh2 or gone2):
                        continue
                    tests.append({"ti": ti, "ax": ax, "item": item,
                                  "a": a, "b": b, "s0": round(sh0), "s1": round(sh1),
                                  "n0": len(s0g), "n1": len(s1g),
                                  "z": z if moved else (3.0 if fresh2 else -3.0)})
        survivors = tests
        digest["mix_tested"] = len(tests)
        per_topic: dict[int, list[dict]] = defaultdict(list)
        for t2 in survivors:
            per_topic[t2["ti"]].append(t2)
        mix_rows = []
        for ti, sh in per_topic.items():
            sh.sort(key=lambda t2: -abs(t2["z"]))
            mix_rows.append({
                "ti": ti, "l": topics_out[ti]["l"],
                "n0": sh[0]["n0"], "n1": sh[0]["n1"],
                "maxz": round(abs(sh[0]["z"]), 2),
                "shifts": [{"ax": t2["ax"], "id": t2["item"],
                            "l": name_of[t2["ax"]](t2["item"]),
                            "s0": t2["s0"], "s1": t2["s1"],
                            "up": 1 if t2["z"] > 0 else 0,
                            "nw": 1 if t2["a"] <= 1 else 0}
                           for t2 in sh[:4]],
            })
        mix_rows.sort(key=lambda r2: -r2["maxz"])
        digest["mix"] = mix_rows[:10]

        # Evidence papers make each claim inspectable in place: for a row's top
        # shift, the latest-edition papers that CARRY it, orals first. Titles
        # are already inline in the payload, so only gids ship.
        gid_rows = {r["i"]: r for r in rows}
        for r2 in digest["mix"]:
            top = r2["shifts"][0]
            members = [g for g, tl in topic_of.items()
                       if r2["ti"] in tl and g in row_of
                       and rows[row_of[g]]["cy"] == pc1]
            hits = []
            for g in members:
                ix = row_of[g]
                if top["id"] in items_of(rows[ix], ix, top["ax"]):
                    hits.append(rows[ix])
            hits.sort(key=lambda p2: (p2["o"] != 1, p2["o"] != 2, p2["t"]))
            r2["ev"] = [p2["i"] for p2 in hits[:4]]


        # ---- fights: which stated failures grew, conference-wide ----
        ftests = []
        for t2, post in enumerate(lims0["p"] and range(0) or []):
            pass  # (placeholder guard, replaced by the loop below)
        post_by_term: dict[int, list[int]] = defaultdict(list)
        for ix, ids in enumerate(lims0["p"]):
            for t2 in ids:
                post_by_term[t2].append(ix)
        for t2, ixs in post_by_term.items():
            a = sum(1 for ix in ixs if rows[ix]["cy"] == pc0)
            b = sum(1 for ix in ixs if rows[ix]["cy"] == pc1)
            if a + b < 12 or not fight_ok(lims0["v"][t2], a, b):
                continue
            sh0, sh1 = a / n0 * 1000, b / n1 * 1000
            lo, hi = min(sh0, sh1), max(sh0, sh1)
            if abs(sh1 - sh0) < 1 and (lo == 0 or hi / lo < 1.5):
                continue
            z = _z(a, b, n0, n1)
            if z < 2.576:      # rising only: a failure fading tracks its field
                continue       # fading, and section one already tells that story
            ftests.append({"t": t2, "a": a, "b": b,
                           "s0": round(sh0, 1), "s1": round(sh1, 1),
                           "z": z, "p": _p_of(z)})
        # a term that is a topic label's word is a FIELD echo, not a failure
        topic_words = {w for t3 in topics_out for w in t3["l"].lower().split()}
        fs = [t2 for t2 in ftests
              if not all(w in topic_words for w in lims0["v"][t2["t"]].split())]
        fs.sort(key=lambda t2: -abs(t2["z"]))
        # one row per failure family: "hallucination" and "hallucinations" both
        # survive the test; the shorter term that contains-or-is-contained wins
        seen_sub: list[str] = []
        frows = []
        for t2 in fs:
            term = lims0["v"][t2["t"]]
            if any(term in s2 or s2 in term for s2 in seen_sub):
                continue
            seen_sub.append(term)
            # two of the papers' own sentences, so hover can say what the
            # term MEANS here without us writing a definition (Guardrail 2)
            exs = []
            for ix in post_by_term[t2["t"]]:
                if rows[ix]["cy"] != pc1:
                    continue
                sent = (lim0_texts[ix] or "").strip()
                if len(sent) < 40:
                    continue
                exs.append({"t": rows[ix]["t"][:90], "s": sent[:200]})
                if len(exs) == 2:
                    break
            frows.append({"t": term, "s0": t2["s0"], "s1": t2["s1"],
                          "up": 1 if t2["z"] > 0 else 0, "nw": 1 if t2["a"] <= 2 else 0,
                          "ex": exs})
        digest["fights"] = frows[:6]


        # ---- fresh: benchmarks that did not exist in the previous edition ----
        dcnt: dict[int, list[int]] = defaultdict(lambda: [0, 0])
        for r in rows:
            if r["cy"] not in (pc0, pc1):
                continue
            for di in set(r["k0"]):
                dcnt[di][0 if r["cy"] == pc0 else 1] += 1
        fresh = [{"di": di, "l": data.items[di], "b": b}
                 for di, (a, b) in dcnt.items()
                 if a == 0 and b >= 6 and not is_placeholder(data.items[di])]
        fresh.sort(key=lambda x: -x["b"])
        digest["fresh"] = fresh[:10]

    return {
        "papers": rows,
        "digest": digest,
        "vocab": {"m": meth.items, "d": data.items, "t": task.items},
        "terms": terms,
        "lims": lims,
        # PCA(128) int8 of the BGE-M3 vectors, 1.1 MB. Search is lexical and
        # always will be — a browser cannot embed a query — but these let a
        # lexical hit set reach the papers that mean the same thing without
        # sharing the words. Kept as a SEPARATE, marked group, never blended in.
        "emb": load_json(EMBED) if EMBED.exists() else None,
        "datasets": [[di, c] for c, di in ds_entry[:400]],
        # id, papers, family index, parent id (null when top level)
        "methods": payload_methods,
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
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
-webkit-font-smoothing:antialiased}
/* One column for the whole page. With the sheet at page width, a 1380px band of
   controls above it left the content looking pushed to the right. */
.wrap{max-width:980px;margin:0 auto;padding:18px 20px 60px}
header{margin:14px 0 22px;text-align:center}
/* One line, always: the sentence never wraps — fitTitle() shrinks the font
   to the content instead, so filling a slot cannot push the rest onto line 2. */
h1{font-size:40px;margin:0;font-weight:700;letter-spacing:-.025em;line-height:1.3;
white-space:nowrap}
/* The title IS the query: "What's new in [reasoning] for [healthcare]?" */
#home{cursor:pointer}
.tslot{display:inline-block;font:inherit;border:0;cursor:pointer;padding:0 8px;margin:0 1px;
border-radius:10px;background:#e8effc;color:var(--acc);border-bottom:3px solid var(--acc);
line-height:1.25}
.tslot .x{margin-left:6px;font-weight:400;opacity:.5;font-size:.75em;vertical-align:2px}
.tslot.ghost{background:none;color:#9aa0a6;border-bottom:3px dashed #c9ccd0;font-weight:500}
h1 .tand{font-weight:500;color:var(--ink2)}
.tedit{position:relative;display:inline-block}
/* the tail: free words and the limitation filter, readable as part of the sentence */
.ttail{font-size:.62em;font-weight:500;color:var(--ink2);display:block;margin-top:4px}
.ttail .tslot{border-bottom-width:2px}
.ttail .lim{background:#fbe9ef;color:#a84a68;border-bottom-color:#c76a86}
.searchrow{display:flex;gap:8px;margin-bottom:10px}
.search{display:flex;gap:8px;align-items:center;flex:1}
.limhit{background:#f4a8bd;border-radius:2px;padding:0 1px;text-decoration:none;font-style:normal}

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
.phow{font-size:10.5px;color:var(--mut);padding:6px 16px 0;font-style:italic}
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
.vcard{text-align:center;background:var(--card);border-radius:12px;padding:16px 18px}
.vcard.dim{opacity:.55}
.vname{font-size:19px;font-weight:700;letter-spacing:-.01em}
.vfull{font-size:12px;color:var(--mut);margin:2px 0 12px;min-height:28px}
.vcard{border:1px solid transparent;font:inherit;cursor:pointer}
.vcard:not(.dim):hover{border-color:var(--acc)}
.vcard.dim{cursor:default}
.vyr{font-size:12px;font-weight:600;color:var(--mut);margin-left:8px}
.vn{font-size:12.5px;color:var(--ink2)}
.vsoon{font-size:12.5px;color:var(--mut)}
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
.rh{font-size:10px;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-bottom:7px}
.rh em{font-style:normal;font-weight:500;letter-spacing:0;text-transform:none;margin-left:4px}
/* rail mini chart: one venue pair per row, log x like the landing chart */
.mrr{display:block;width:100%;text-align:left;font:inherit;background:none;border:0;
  padding:2.5px 0;cursor:pointer;border-radius:4px}
.mrr:hover .mrl{color:var(--acc)}
.mrl{font-size:12.5px;color:var(--ink2);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;
  display:block;margin-bottom:1px}
.mrl .tag{margin-left:5px}
.mrt{position:relative;height:6px;display:block}
.mrt .lane{position:absolute;inset:0;height:auto;margin:0}
/* the set card: composition of the current selection, in reach while scrolling */
.setcard{margin-top:14px;border-top:1px solid var(--ring);padding-top:10px}
.scn{font-size:22px;font-weight:700;letter-spacing:-.02em}
.scn small{font-size:11px;font-weight:500;color:var(--mut);margin-left:5px}
.scyr{margin:8px 0 2px}
.scyrow{display:grid;grid-template-columns:minmax(34px,auto) 1fr 76px;gap:8px;align-items:center;
  font-size:12px;color:var(--ink2);padding:1.5px 0}
.scyrow .yb{position:relative;height:7px}
.scyrow .yb i{position:absolute;left:0;top:0;height:100%;border-radius:2px}
.scyrow b{font-weight:600;font-size:10px;text-align:right;color:var(--ink2);white-space:nowrap}
.scsub{font-size:10.5px;color:var(--mut);margin:2px 0 0}
.scchips{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
.scchip{font:inherit;font-size:11.5px;padding:2px 8px;border-radius:20px;cursor:pointer;
  border:1px solid var(--ring);background:var(--card);color:var(--ink2)}
.scchip b{font-weight:600;color:var(--mut);margin-left:3px}
.scchip:hover{border-color:var(--acc);color:var(--acc)}
.scchip.on2{background:#e8effc;border-color:#bcd2f2;color:var(--acc);font-weight:600}
.scchip.on2:hover{border-color:var(--warm);color:var(--warm)}
.scsplit{font:inherit;font-size:11.5px;margin-top:11px;padding:5px 11px;border-radius:8px;
border:1px solid var(--ring);background:var(--card);color:var(--ink2);cursor:pointer;width:100%}
.scsplit:hover{color:var(--ink);border-color:var(--mut)}
.scsplit.on{background:var(--ink);color:#fff;border-color:var(--ink)}
/* Inside-this-set: the since-last-year grammar, with the SELECTION as the
   denominator — which methods/tasks/domains rose or fell within the field the
   reader picked. Rows appear only when the two-test rule passes on the set's
   own sizes, so a small set honestly shows only its LARGEST items. */
.insbox{background:var(--card);border-radius:12px;padding:13px 18px 11px;margin:0 0 12px}
.inshd{font-size:12px;font-weight:660;display:flex;align-items:center;gap:10px}
.inshd em{font-style:normal;font-weight:500;color:var(--mut);font-size:10.5px}
.insbox .cr{font-size:12px;padding:1.5px 0}
.insbox .chgax{height:11px}
.axtag{font-size:9px;font-weight:700;letter-spacing:.04em;padding:1px 7px;border-radius:9px;
margin-left:7px;vertical-align:1px}
.axtag.v0{background:#e8effc;color:var(--v0)}
.axtag.v1{background:#e2f3ef;color:var(--v1)}
.axtag.v2{background:#f0eafa;color:var(--v2)}
.insleg .axtag{margin-left:5px}
/* the arrival banner: the clicked claim, restated where the reader lands */
.story{background:linear-gradient(90deg,#e8effc,#f2f6fc 70%,var(--card));border-left:4px solid var(--acc);
border-radius:12px;padding:12px 16px;margin:0 0 12px;display:grid;
grid-template-columns:1fr auto;gap:4px 14px;align-items:center}
.story .sy1{font-size:11px;letter-spacing:.06em;text-transform:uppercase;color:var(--acc);font-weight:700}
.story .sy2{font-size:15.5px;font-weight:650;grid-column:1}
.story .sy2 em{font-style:normal;font-weight:500;color:var(--ink2)}
.story .sy2 b.dn{color:var(--warm)} .story .sy2 b.up{color:var(--acc)}
.story .sytrk{grid-column:1;max-width:520px;margin-top:3px}
.story .syx{grid-column:2;grid-row:1/span 3;font:inherit;font-size:16px;border:0;
background:none;cursor:pointer;color:var(--mut);padding:4px 8px}
.story .syx:hover{color:var(--warm)}
/* the digest: analysis first, selection second — every row is a door */
.digbox{background:var(--card);border-radius:12px;padding:16px 18px;margin-top:14px}
.dighd{font-size:17px;font-weight:660}
.dighd em{font-style:normal;font-weight:500;font-size:11.5px;color:var(--mut);margin-left:10px}
.digrow{display:block;width:100%;font:inherit;text-align:left;border:0;background:none;
cursor:pointer;padding:7px 9px;border-radius:8px;font-size:14px;color:var(--ink)}
.digrow:hover{background:#f2f5f9}
.digrow.open{background:#f2f5f9}
.digrow b{font-weight:650}
.digrow .n2{color:var(--mut);font-weight:400;font-size:11px;margin-left:7px}
.shift{margin-left:9px;white-space:nowrap}
.shift i{font-style:normal;font-weight:700}
.shift.up i{color:var(--acc)} .shift.dn i{color:var(--warm)}
.shift .nw2{font-size:9px;font-weight:700;letter-spacing:.04em;color:var(--acc);vertical-align:2px;margin-left:2px}
.digx{margin:2px 6px 10px;padding:10px 14px;background:#f7f8fa;border-radius:10px}
.dxr{display:grid;grid-template-columns:230px 1fr 92px;gap:12px;align-items:center;
padding:2.5px 0;font-size:12px;color:var(--ink2)}
.dxr .crl{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.dxr .ax2{color:var(--mut);font-size:10px;margin-left:6px}
.dxr b{font-weight:600;font-size:10.5px;color:var(--ink2);text-align:right;white-space:nowrap}
.digopen button{font:inherit;font-size:12px;padding:5px 13px;border-radius:8px;cursor:pointer;
border:1px solid var(--acc);background:var(--acc);color:#fff}
.fightrow{display:grid;grid-template-columns:190px 1fr 92px;gap:12px;align-items:center}
.vleg{display:flex;flex-wrap:wrap;gap:8px 22px;justify-content:center;align-items:center;
margin:12px 0 2px;font-size:14px;font-weight:600;color:var(--ink)}
.vleg i{display:inline-block;width:12px;height:12px;border-radius:3px;margin-right:7px;vertical-align:-1px}
.vleg em{font-style:normal;font-weight:500;color:var(--ink2)}
.vlone{font-size:11.5px;font-weight:500;color:var(--mut)}
.freshwrap{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
/* merged What-moved rows: field dumbbell reads big, its inner shifts as pills */
.cr.mv{grid-template-columns:215px 1fr 92px;font-size:14.5px;padding:5px 8px;border-radius:8px}
.cr.mv:hover{background:#f2f5f9}
.cr.mv .crl{font-weight:600}
.mvn{font-weight:650;font-size:12px;color:var(--ink2);text-align:right;white-space:nowrap}
.chgplot.mv .lane{height:10px}
.pillrow{display:flex;flex-wrap:wrap;gap:5px;align-items:center;margin:1px 0 8px 227px;position:relative}
.pillhd{font-size:9.5px;letter-spacing:.07em;text-transform:uppercase;color:var(--mut);margin-right:2px}
.pill{font:inherit;font-size:12px;padding:2.5px 9px;border-radius:20px;border:0;cursor:pointer;
background:#eef1f5;color:var(--ink2)}
.pill i{font-style:normal;font-weight:700}
.pill b{font-weight:650}
.pill.up{background:#e5edfa;color:#1d4f9c} .pill.up i{color:var(--acc)}
.pill.dn{background:#faece3;color:#8f3d13} .pill.dn i{color:var(--warm)}
.pill:hover{filter:brightness(.96)}
.pill[disabled]{cursor:default}
.pill .nw2{font-size:8px;font-weight:800;letter-spacing:.05em;margin-left:4px;vertical-align:1px}
@media(max-width:700px){.cr.mv{grid-template-columns:130px 1fr 70px}.pillrow{margin-left:0}}
/* the fights hover card: the papers' own sentences define the term */
.fightrow{position:relative}
.tip{display:none;position:absolute;left:180px;top:100%;z-index:50;width:min(520px,80vw);
background:var(--card);border:1px solid var(--ring);border-radius:12px;
box-shadow:0 12px 34px rgba(0,0,0,.16);padding:11px 14px;text-align:left;cursor:default}
.fightrow:hover .tip{display:block}
.tipt{display:block;font-size:10.5px;font-weight:650;color:var(--ink);margin-top:7px}
.tipt:first-child{margin-top:0}
.tips{display:block;font-size:11.5px;color:var(--ink2);font-weight:400;line-height:1.45;margin-top:1px}
.allfields{margin-top:14px;text-align:center}
.allfields>button{font:inherit;font-size:12px;color:var(--mut);background:none;border:0;
cursor:pointer;padding:6px 10px}
.allfields>button:hover{color:var(--ink)}
.afgrid{display:flex;flex-wrap:wrap;gap:5px;justify-content:center;margin-top:10px}
/* what-moved entry view */
.chgbox{background:var(--card);border-radius:12px;padding:16px 18px;margin-top:14px}
.chghd{font-size:17px;font-weight:660;display:flex;align-items:center;gap:10px}
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
.chgsec{font-size:10.5px;letter-spacing:.08em;text-transform:uppercase;color:var(--mut);
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
<header><h1 id="ttl">What's new in AI research</h1></header>

<div id="landing" hidden></div>
<nav id="rail" hidden></nav>

<div id="main">

<div class="searchrow">
<div class="search">
  <input type="search" id="q" aria-label="Search every abstract">
</div>
</div>


<div class="xref" id="xref" hidden></div>
<div class="legend" id="legend">
  <span class="sw"><span class="hl why">why it was needed</span></span>
  <span class="sw"><span class="hl new">what is new</span></span>
  <span class="sw"><span class="hl eff">what it achieved</span></span>
  <span>— the paper's own sentences, cut but never rewritten. The colours and the order are ours.</span></div>

<div class="story" id="story" hidden></div>

<div class="insbox" id="inset" hidden></div>

<div class="res" id="results"></div>

<aside class="panel" id="panel" hidden>
  <div class="phd"><span>Closest papers</span><button id="pclose">Close</button></div>
  <div class="phow">closest by abstract meaning (embedding), across both years — not citations</div>
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
let NBK=0, NBR=null;
const nbrsOf=g=>NBR&&g<D.nemb?NBR.slice(g*NBK,(g+1)*NBK):[];

// ---- PCA(128) int8 vectors, dequantised and row-normalised once ----
// A browser cannot embed a query, so search stays lexical. These are used the
// other way round: take what the words DID find, and reach the papers that mean
// the same thing without sharing the vocabulary. Those arrive as a separate,
// labelled group — never mixed into the literal matches.
let EMB=null, EMB_P=null;
function ensureEmb(){
  return EMB_P??=part('emb.json').then(d=>{
    if(!d.emb)return;
    NBK=d.neighbors?d.neighbors.k:0; NBR=d.neighbors?d.neighbors.nbr:null;
    D.nemb=d.nemb;
    const {dims,b64,scale}=d.emb;
    const bin=atob(b64), n=d.nemb;
    const v=new Float32Array(n*dims);
    for(let r=0;r<n;r++){
      let acc=0;
      for(let dd=0;dd<dims;dd++){
        const q=(bin.charCodeAt(r*dims+dd)<<24)>>24;   // byte -> signed int8
        const x=q*scale; v[r*dims+dd]=x; acc+=x*x;
      }
      acc=Math.sqrt(acc)||1;
      for(let dd=0;dd<dims;dd++) v[r*dims+dd]/=acc;
    }
    EMB={v,dims,n};
  });
}
const hasVec=g=>EMB&&g<EMB.n;

// ---- on-demand parts -------------------------------------------------------
// The page inlines only what first paint and every count need. Sentences,
// search indexes and vectors arrive from data/*.json when first touched (and
// via an idle prefetch), so the growing corpus set does not grow first paint.
const PART={};                              // name -> Promise
function part(name){
  return PART[name]??=fetch('data/'+name+'?v='+(D.pv||0)).then(r=>{
    if(!r.ok)throw new Error(name+' '+r.status);
    return r.json();});
}
// ---- term indexes: abstracts (search) and limitation sentences (V5), lazy --
let TV=null,TP=null,LV=null,LP=null;
const POST=new Map(), LPOST=new Map();
let SEARCH_P=null;
function ensureSearch(){
  return SEARCH_P??=part('search.json').then(d=>{
    TV=d.terms.v; TP=d.terms.p; LV=d.lims.v; LP=d.lims.p;
    TP.forEach((ids,i)=>{for(const t of ids){let a=POST.get(t);if(!a)POST.set(t,a=[]);a.push(i);}});
    LP.forEach((ids,i)=>{for(const t of ids){let a=LPOST.get(t);if(!a)LPOST.set(t,a=[]);a.push(i);}});
  });
}
// st.lim is a STRING matched as a substring of the interned limitation terms,
// so "hallucinat" covers hallucination / hallucinations / hallucinated at once —
// the concept, not one inflection of it.
let LSET=null, LSETID=null;
function limSet(){
  if(LSETID===st.lim)return LSET;
  LSETID=st.lim;
  if(st.lim===null||LV===null){
    LSETID=undefined; LSET=st.lim===null?null:new Set();
    if(st.lim!==null)ensureSearch().then(render);   // repaint once the index lands
    return LSET;
  }
  LSET=new Set();
  for(let t=0;t<LV.length;t++)
    if(LV[t].includes(st.lim)) for(const i of LPOST.get(t)||[])LSET.add(i);
  return LSET;
}

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
const st={corp:null,q:[],topics:new Set(),fams:new Set(),sel:null,panel:null,ds:null,meth:null,mfam:null,grouped:false,lim:null};
// The claim the reader clicked to get here — the door's face, carried to the
// destination so "why am I looking at this set?" never needs remembering.
let STORY=null;
// A conference pick alone is NOT a selection. Picking "ICML 2026" leaves 6,637
// papers, which is the problem this product exists to remove — the reader still
// has to say what their field is.
const chosen=()=>st.q.length||st.topics.size||st.fams.size||st.ds!==null||st.meth!==null||st.mfam!==null||st.lim!==null;

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
  if(!EMB){ ensureEmb().then(render); return []; }
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

let Q_EMPTY=null;
function qPending(){ return st.q.length>0&&TV===null; }
function queryHits(){
  if(!st.q.length)return null;
  // index not here yet: match NOTHING rather than flash all 6,637 as a result
  if(TV===null){ ensureSearch().then(render); return Q_EMPTY??=new Set(); }
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
    if(st.corp!==null&&st.corp>=0&&p.cy!==st.corp)continue;
    if(skip!=='ds'&&st.ds!==null&&!p.k0.includes(st.ds))continue;
    if(skip!=='meth'&&!methPass(p))continue;
    if(skip!=='topics'&&!facetPass(p,false))continue;
    if(skip!=='domains'&&!facetPass(p,true))continue;
    if(st.lim!==null&&!limSet().has(i))continue;
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
  const mu=p.mu0||[];
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
  if(cc!==null&&cc>=0&&p.cy!==cc)return false;
  if(st.ds!==null&&!p.k0.includes(st.ds))return false;
  if(!methPass(p))return false;
  // Within a facet the picks are OR — two topics widen the set. ACROSS facets
  // they are AND: "reasoning" plus "healthcare" means reasoning applied to
  // healthcare, not everything that is either. Anything else makes the two rows
  // fight each other.
  if(!facetPass(p,false)||!facetPass(p,true))return false;
  if(st.lim!==null&&!limSet().has(i))return false;
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

// The failure filter's term, marked inside the pink limitation sentence.
// Tag-safe: only text segments between tags are touched.
function markLim(html){
  if(st.lim===null)return html;
  const t=st.lim.replace(/[.*+?^${}()|[\]\\]/g,'\\$&').replace(/\s+/g,'\\s+');
  const rx=new RegExp('('+t+')','ig');
  return html.split(/(<[^>]+>)/).map(seg=>seg.startsWith('<')?seg:seg.replace(rx,'<i class="limhit">$1</i>')).join('');
}
// Card sentences live in data/spans_<corpus>.json, fetched the first time a
// card from that corpus reaches the screen; render() repaints on arrival.
const SPX={}, SP_LOADED={};
function ensureSpans(cys){
  for(const ci of cys){
    const k=CY[ci].k;
    if(SP_LOADED[k])continue;
    SP_LOADED[k]=true;
    part('spans_'+k+'.json').then(d=>{
      Object.assign(SPX,d);
      render();
    }).catch(()=>{SP_LOADED[k]=false;});
  }
}
function card(i){
  const p=P[i];
  const sx=SPX[p.i], spReady=!!sx;
  const pn=sx?sx[0]:[], pL=sx?sx[1]:null, pK=sx?sx[2]:null, pR=sx?sx[3]:null,
        pc1=sx?sx[4]:null;
  const names=(arr,fn)=>arr.map(fn).join(', ');
  // One passage, not three labelled rows. The three sentences are the paper's,
  // in the order a reader needs them, and the colour says which question each
  // answers: why it was needed, what is new, what it achieved.
  const parts=[];
  if(pL)parts.push(`<span class="hl why">${markLim(annotate(pL,p,true))}</span>`);
  const change=pK||pn[0]||'';
  if(change)parts.push(`<span class="hl new">${annotate(change,p,true)}</span>`);
  if(pR)parts.push(`<span class="hl eff">${annotate(pR,p,true)}</span>`);

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
  const c1=(pc1||[])[0];
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
      :spReady?`<div class="passage miss">no sentence in this paper states what is new</div>`
      :`<div class="passage miss">loading the paper's own sentences…</div>`}
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
  if(!NBR){
    ensureEmb().then(()=>openPanel(i));
    $('#panel').hidden=false;
    $('#pbody').innerHTML='<div style="color:var(--mut);font-size:12px;padding:10px 0">loading the embedding neighbours…</div>';
    return;
  }
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
  return `<div class="venues">${cards}</div>`+venueLegendHTML()+changedHTML()+digestHTML()+allFieldsHTML();
}
// Which hue is which conference — pinned under the cards, where the eye goes
// before the bars. Venues with a single edition are named but carry no bar.
function venueLegendHTML(){
  const prs=venuePairs();
  if(prs.length<2)return '';
  const single=VENUES.map((v,hue)=>({v,hue}))
    .filter(x=>CY.some(c=>c.v===x.v.v)&&!prs.some(p=>p.v===x.v.v))
    .map(x=>{const c=CY.find(c2=>c2.v===x.v.v);
      return `<span><i style="background:color-mix(in srgb, var(--v${x.hue}) 34%, var(--card))"></i>`+
        `${esc(x.v.v)} <em>${c.y} only</em></span>`;});
  return `<div class="vleg">`+prs.map(p=>
      `<span><i style="background:var(--v${p.hue})"></i>${esc(p.v)} <em>from ${p.y0} to ${p.y1}</em></span>`).join('')+
    single.join('')+
    `<span class="vlone">pale = the earlier edition</span></div>`;
}

// ---- the digest: what MOVED, precomputed at build time (BH-free but bar-
// consistent: the product's standing |z|>=2.576 + material rule, family size
// disclosed; appearing 0->n needs no test to be a fact). Every row applies
// itself as a selection — analysis first, selection second.
const DG=D.digest||{};
// The chosen scope lives in the URL hash: '#all' is the union, '#icml-2026'
// one corpus; back returns to the landing and reload keeps the reader put.
const corpFromHash=()=>{
  const h=location.hash.slice(1);
  if(h==='all')return -1;
  const j=CY.findIndex(c=>c.k===h);return j<0?null:j;};
function go(j){
  st.corp=j; st.sel=null; st.grouped=false;
  history.pushState(null,'',j===null?location.pathname+location.search
                             :'#'+(j===-1?'all':CY[j].k));
  render();
}
window.addEventListener('popstate',()=>{ st.corp=corpFromHash(); st.sel=null; render(); });

function digestHTML(){
  if(!DG.pair)return '';
  const hue=Math.max(VENUES.findIndex(v=>v.v===DG.pair.v),0);
  let h='';
  if((DG.fights||[]).length){
    const M=Math.max(...DG.fights.flatMap(f=>[f.s0,f.s1]),5);
    const X=v=>v<=0.5?0:Math.log(v/0.5)/Math.log(M/0.5)*100;
    h+=`<div class="digbox"><div class="dighd">What the field fights`+
      `<em>failures named in the papers' own limitation sentences · share per 1,000 papers · hover for the sentences themselves</em></div>`+
      DG.fights.map((f,fi)=>{
        const rx=new RegExp('('+f.t.replace(/[.*+?^${}()|[\]\\]/g,'\\$&')+')','ig');
        const tip=(f.ex||[]).length?`<span class="tip">`+f.ex.map(x=>
          `<span class="tipt">${esc(x.t)}</span>`+
          `<span class="tips">${esc(x.s).replace(rx,'<i class="limhit">$1</i>')}…</span>`).join('')+
          `</span>`:'';
        return `<button class="digrow fightrow" data-fight="${fi}"><b>${esc(f.t)}</b>`+
        `<span class="trk">${laneHTML({hue,s0:f.s0,s1:f.s1,w0:X(f.s0),w1:X(f.s1)})}</span>`+
        `<b>${f.s0} → ${f.s1} /1k</b>${tip}</button>`;
      }).join('')+'</div>';
  }
  if((DG.fresh||[]).length){
    h+=`<div class="digbox"><div class="dighd">New this year`+
      `<em>benchmarks no ${DG.pair.y0} paper used</em></div><div class="freshwrap">`+
      DG.fresh.map(f=>`<button class="scchip" data-fresh="${f.di}">${esc(f.l)}<b>${f.b}</b></button>`).join('')+
      '</div></div>';
  }
  return h;
}
function allFieldsHTML(){
  const rows=T.map((t,ti)=>({ti,l:t.l,n:t.n+t.e,j:t.j})).filter(x=>!x.j&&x.n>=8)
    .sort((a,b)=>a.l.localeCompare(b.l));
  return `<div class="allfields"><button id="aftog">${AF_OPEN?'hide':'browse'} all ${rows.length} fields`+
    ` <span style="opacity:.6">${AF_OPEN?'▴':'▾'}</span></button>`+
    (AF_OPEN?`<div class="afgrid">`+rows.map(x=>
      `<button class="scchip" data-af="${x.ti}">${esc(x.l)}<b>${x.n}</b></button>`).join('')+'</div>':'')+
    `</div>`;
}
let AF_OPEN=false;
function enterWith(mut){
  st.q=[]; const q=$('#q'); if(q)q.value='';
  st.sel=null; st.grouped=false;
  st.topics.clear(); st.fams.clear(); st.meth=null; st.mfam=null; st.ds=null;
  st.lim=null;
  mut();
  go(-1);
}
function mixStoryFor(r,x){
  const hue=Math.max(VENUES.findIndex(v=>v.v===DG.pair.v),0);
  return {label:`inside <b>${esc(r.l)}</b>, ${esc(x.l)} ${x.up?'rose':'fell'}`,
          sub:'share of the set',s0:x.s0,s1:x.s1,unit:'%',
          fmt:v=>((v/10).toFixed(v<100?1:0)),
          hue,y0:DG.pair.y0,y1:DG.pair.y1,ti:r.ti};
}
function wireDigest(){
  document.querySelectorAll('[data-fight]').forEach(el=>el.onclick=()=>{
    const f=DG.fights[+el.dataset.fight];
    const hue=Math.max(VENUES.findIndex(v=>v.v===DG.pair.v),0);
    STORY={label:`prior work struggles with <b>${esc(f.t)}</b>`,sub:'papers per 1,000',
           s0:f.s0,s1:f.s1,unit:'/1k',fmt:v=>v,hue,y0:DG.pair.y0,y1:DG.pair.y1,lim:f.t};
    enterWith(()=>{st.lim=f.t;});});
  document.querySelectorAll('[data-fresh]').forEach(el=>el.onclick=()=>{
    const di=+el.dataset.fresh;
    const f=DG.fresh.find(x=>x.di===di);
    STORY={label:`<b>${esc(dname(di))}</b> — a benchmark no ${DG.pair.y0} paper used`,
           sub:f?`${f.b} papers in ${DG.pair.y1}`:'',di};
    enterWith(()=>st.ds=di);});
  document.querySelectorAll('[data-af]').forEach(el=>el.onclick=()=>{
    enterWith(()=>st.topics.add(+el.dataset.af));});
  const af=$('#aftog'); if(af)af.onclick=()=>{AF_OPEN=!AF_OPEN;render();};
}
function wireLanding(){
  wireDigest();
  document.querySelectorAll('[data-corp]').forEach(el=>el.onclick=()=>go(+el.dataset.corp));
  wireChanged();
}

// ---- V1: the title is the query --------------------------------------------
// "What's new in [reasoning] for [healthcare] with [LLMs] on [GSM8K]?"
// Slots map 1:1 onto existing state, so the category menus below and the
// sentence are two views of one selection, and the sentence doubles as the
// always-current answer to "what am I looking at?".
// lazy: MS/MV are declared further down the script, and this block's
// top level runs first — touching them here would be a TDZ crash
let MTOP=null;
const mtop=id=>{ if(!MTOP){MTOP=new Map(); for(const [x,,,pa] of MS)MTOP.set(x,pa==null?x:pa);} return MTOP.get(id)??id; };
function slotState(){
  const kt=[...st.topics].filter(ti=>T[ti].f!==1), dt=[...st.topics].filter(ti=>T[ti].f===1);
  const kf=[...st.fams].filter(k=>k.startsWith('#topfams|')), df=[...st.fams].filter(k=>k.startsWith('#domfams|'));
  const lab=(picks,fams)=>{
    if(picks.length)return T[picks[0]].l+(picks.length+fams.length>1?` +${picks.length+fams.length-1}`:'');
    if(fams.length)return fams[0].split('|')[1]+(fams.length>1?` +${fams.length-1}`:'');
    return null;
  };
  return {
    k:lab(kt,kf),
    d:lab(dt,df),
    u:st.meth!==null?MV[st.meth]:(st.mfam!==null?MF[st.mfam]:null),
    b:st.ds!==null?dname(st.ds):null,
  };
}
function clearSlot(ax){
  if(ax==='k'){ for(const ti of [...st.topics]) if(T[ti].f!==1)st.topics.delete(ti);
                for(const k of [...st.fams]) if(k.startsWith('#topfams|'))st.fams.delete(k); }
  if(ax==='d'){ for(const ti of [...st.topics]) if(T[ti].f===1)st.topics.delete(ti);
                for(const k of [...st.fams]) if(k.startsWith('#domfams|'))st.fams.delete(k); }
  if(ax==='u'){ st.meth=null; st.mfam=null; }
  if(ax==='b')st.ds=null;
}
// STORY: {kind, label, sub, s0, s1, unit, hue, y0, y1, ti?, lim?, di?}
// Valid only while the filter it set is still active; cleared by its own x.
function storyValid(){
  if(!STORY)return false;
  if(STORY.ti!==undefined)return st.topics.has(STORY.ti);
  if(STORY.lim!==undefined)return st.lim===STORY.lim;
  if(STORY.di!==undefined)return st.ds===STORY.di;
  return true;
}
function drawStory(){
  const el=$('#story'); if(!el)return;
  if(!storyValid()){ el.hidden=true; if(!STORY)return; return; }
  const y=STORY.y0!==undefined?` · ${STORY.y0} → ${STORY.y1}`:'';
  const dirUp=STORY.s1>=STORY.s0;
  const nums=STORY.s0!==undefined
    ?` <b class="${dirUp?'up':'dn'}">${STORY.fmt(STORY.s0)} → ${STORY.fmt(STORY.s1)}${STORY.unit}</b>`
    :'';
  let lane='';
  if(STORY.s0!==undefined){
    const M=Math.max(STORY.s0,STORY.s1)*1.15, F=Math.min(STORY.s0,STORY.s1,M/8)/2;
    const X=v=>v<=F?0:Math.log(v/F)/Math.log(M/F)*100;
    lane=`<span class="sytrk">${laneHTML({hue:STORY.hue||0,s0:STORY.s0,s1:STORY.s1,w0:X(STORY.s0),w1:X(STORY.s1)})}</span>`;
  }
  el.innerHTML=`<div class="sy1">you picked</div>`+
    `<div class="sy2">${STORY.label}${nums} <em>${esc(STORY.sub||'')}${y}</em></div>`+
    lane+`<button class="syx" title="dismiss">×</button>`;
  el.hidden=false;
  el.querySelector('.syx').onclick=()=>{STORY=null;render();};
}

const INS_MIN=12;          // below this on either side, a year claim is noise
// One panel, all three axes at once — the tabs made readers click to compare.
// Axis is carried by hue (the validated triple, reused panel-locally: builds
// on = blue, does = teal, applied to = purple) AND by a written tag on every
// row, so colour is never the only channel. When a second venue's data lands
// this panel goes venue-laned and the axis returns to tags alone.
const INS_AX={u:{hue:0,tag:'builds on'},s:{hue:1,tag:'does'},d:{hue:2,tag:'applied to'}};
function drawInside(){
  const el=$('#inset'); if(!el)return;
  el.hidden=true;
  if(qPending())return;
  const prs=venuePairs();
  const pr=st.corp===-1?prs[0]:prs.find(p=>p.c1===st.corp||p.c0===st.corp);
  if(!pr)return;
  const hits=queryHits(), keep=st.corp; st.corp=null;
  const S=[new Set(),new Set()];
  for(let i=0;i<P.length;i++){
    if(P[i].cy!==pr.c0&&P[i].cy!==pr.c1)continue;
    if(match(i,hits,P[i].cy))S[P[i].cy===pr.c0?0:1].add(i);
  }
  st.corp=keep;
  const n0=S[0].size, n1=S[1].size;
  if(Math.min(n0,n1)<INS_MIN)return;
  const rows=[];
  for(const ax of ['u','s','d']){
    const cnt=new Map();
    for(const side of [0,1]) for(const i of S[side]){
      const p=P[i]; const seen=new Set();
      const ids=ax==='u'?(p.mu0||[]).map(m=>mtop(m))
              :ax==='s'?(p.s0||[])
              :p.g.filter(([ti])=>!T[ti].j&&T[ti].f===1).map(([ti])=>ti);
      for(const id of ids){ if(seen.has(id))continue; seen.add(id);
        let c=cnt.get(id); if(!c)cnt.set(id,c=[0,0]); c[side]++; }
    }
    const nm=ax==='u'?(id=>MV[id]):ax==='s'?(id=>tname(id)):(id=>T[id].l);
    for(const [id,[a,b]] of cnt){
      if(a+b<5)continue;
      const s0=a/n0*1000, s1=b/n1*1000;
      const pp=(a+b)/(n0+n1), se=Math.sqrt(pp*(1-pp)*(1/n0+1/n1));
      const z=se?(b/n1-a/n0)/se:0;
      const lo=Math.min(s0,s1), hi=Math.max(s0,s1);
      const mat=Math.abs(s1-s0)>=30||(lo>0&&hi/lo>=1.5);
      rows.push({ax,id,l:nm(id),a,b,s0,s1,z,mat,nw:a<=1&&b>2,gn:b<=1&&a>2});
    }
  }
  if(!rows.length)return;
  const up=rows.filter(r=>r.z>=Z_SHOW&&r.mat&&!r.nw).sort((x,y)=>y.z-x.z).slice(0,6);
  const fresh=rows.filter(r=>r.z>=Z_SHOW&&r.mat&&r.nw).sort((x,y)=>y.s1-x.s1).slice(0,6);
  const dn=rows.filter(r=>r.z<=-Z_SHOW&&r.mat).sort((x,y)=>x.z-y.z).slice(0,6);
  const secs=[['rising',up],['new',fresh],['falling',dn]].filter(x=>x[1].length);
  if(!secs.length)return;
  const all=secs.flatMap(x=>x[1]);
  const M=Math.max(...all.flatMap(r=>[r.s0,r.s1]),50);
  const F=20;
  const X=v=>v<=F?0:Math.log(v/F)/Math.log(M/F)*100;
  const ticks=[20,50,100,200,500,1000].filter(t=>t>=F&&t<=M*1.04);
  const grid='<div class="chgg">'+ticks.map(t=>`<i style="left:${X(t).toFixed(2)}%"></i>`).join('')+'</div>';
  const axis='<div class="chgax">'+ticks.map(t=>`<b style="left:${X(t).toFixed(2)}%">${t/10}%</b>`).join('')+'</div>';
  const body=secs.map(([nm2,rs])=>`<div class="chgsec">${nm2}</div>`+rs.map(r=>{
    const A=INS_AX[r.ax];
    const lane=laneHTML({hue:A.hue,s0:r.s0,s1:r.s1,w0:X(r.s0),w1:X(r.s1)});
    const tag=r.gn?'<em class="tag gone">gone</em>':'';
    const kind={u:'m',s:null,d:'t'}[r.ax];
    const attrs=kind?`data-k="${kind}" data-id="${r.id}"`:'disabled style="cursor:default"';
    return `<button class="cr" ${attrs}><span class="crl" title="${esc(r.l)}">${esc(r.l)}${tag}`+
      `<span class="axtag v${A.hue}">${A.tag}</span></span>`+
      `<span class="trk">${lane}</span></button>`;
  }).join('')).join('');
  const leg=`<span class="chgtabs insleg">`+Object.values(INS_AX).map(A=>
    `<span class="axtag v${A.hue}">${A.tag}</span>`).join('')+`</span>`;
  el.innerHTML=`<div class="inshd">Inside this set — since last year`+
    `<em>share of the ${n0} (${pr.y0}) and ${n1} (${pr.y1}) papers picked</em>${leg}</div>`+
    `<div class="chgplot">${grid}${body}</div>${axis}`;
  el.hidden=false;
  el.querySelectorAll('.cr[data-id]').forEach(b=>b.onclick=()=>{
    const k=b.dataset.k, id=+b.dataset.id;
    if(k==='m'){ st.meth=st.meth===id?null:id; st.mfam=null; }
    else st.topics.has(id)?st.topics.delete(id):st.topics.add(id);
    render();});
}

// The rail's before-a-pick view: this venue's own significant movers, the same
// two-test rule as the landing, one pair-lane each. A row IS a filter — clicking
// it applies the topic, because a trend a reader cannot act on is trivia.
function railTrend(){
  const prs=venuePairs();
  const pr=st.corp===-1?prs[0]:prs.find(p=>p.c1===st.corp||p.c0===st.corp);
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
        `<span class="mrt">${laneHTML({hue:pr.hue,s0:r.s0,s1:r.s1,w0:X(r.s0),w1:X(r.s1)})}</span></button>`;
    }).join('');
}
// After a pick: what the chosen set is MADE OF, held sticky while the list
// scrolls. Same selection counted in each edition of this venue (share of that
// year — sizes differ 2x), then the vocabulary that recurs inside the set,
// scoped tighter than the corpus-wide menus above the results.
function railSetCard(res){
  const hits=queryHits();
  const all=st.corp===-1;
  const sib=CY.map((c,j)=>({c,j})).filter(x=>all||x.c.v===CY[st.corp].v);
  const ys=sib.map(({c,j})=>{
    let n=0; for(let i=0;i<P.length;i++) if(match(i,hits,j))n++;
    return {y:c.y,v:c.v,n,sh:n/CYN[j]*1000,
            hue:Math.max(VENUES.findIndex(v2=>v2.v===c.v),0),
            last:c.y===Math.max(...CY.filter(c2=>c2.v===c.v).map(c2=>c2.y))};
  });
  const mx=Math.max(...ys.map(r=>r.sh),1e-9);
  const yrows=ys.map(r=>{
    const col=r.last?`var(--v${r.hue})`
              :`color-mix(in srgb, var(--v${r.hue}) 34%, var(--card))`;
    return `<div class="scyrow"><span>${all?esc(r.v)+' ':''}${r.y}</span>`+
      `<span class="yb"><i style="width:${Math.max(r.sh/mx*100,1.5).toFixed(1)}%;background:${col}"></i></span>`+
      `<b>${r.n} · ${(r.sh/10).toFixed(1)}%</b></div>`;
  }).join('');
  const dk=new Map(), mk=new Map();
  const par=new Map(); for(const [id,,,pa] of MS)par.set(id,pa==null?id:pa);
  for(const i of res){
    for(const d of new Set(P[i].k0)) if(DSSET.has(d)&&d!==st.ds)dk.set(d,(dk.get(d)||0)+1);
    const seen=new Set();
    for(const m of (P[i].mu0||[])){ const t2=par.get(m)??m;
      if(seen.has(t2))continue; seen.add(t2);
      if(t2!==st.meth)mk.set(t2,(mk.get(t2)||0)+1); }
  }
  const chips=(map,kind,name)=>[...map.entries()].filter(([,c])=>c>=2)
    .sort((a,b)=>b[1]-a[1]).slice(0,4)
    .map(([id,c])=>`<button class="scchip" data-ck="${kind}" data-cid="${id}">${esc(name(id))}<b>${c}</b></button>`).join('');
  const dch=chips(dk,'d',dname), mch=chips(mk,'m',i=>MV[i]);
  const nf=res.filter(i=>P[i].f).length;
  // Orals are not a filter — the tier shows as a badge, the list is already
  // ordered orals first, and this line says how many the selection holds.
  let nOral=0,nSpot=0;
  for(const i of res){ if(P[i].o===1)nOral++; else if(P[i].o===2)nSpot++; }
  const tiers=(nOral||nSpot)
    ?`<div class="scsub">${nOral?`${nOral} oral${nOral>1?'s':''}`:''}`+
     `${nSpot?`${nOral?' · ':''}${nSpot} spotlight${nSpot>1?'s':''}`:''} — listed first</div>`:'';
  const V2=slotState();
  const fch=[];
  if(V2.k)fch.push(['k',V2.k]);
  if(V2.d)fch.push(['d','for '+V2.d]);
  if(V2.u)fch.push(['u','built on '+V2.u]);
  if(V2.b)fch.push(['b','on '+V2.b]);
  if(st.lim!==null)fch.push(['l','struggles: “'+st.lim+'”']);
  const fchips=fch.length
    ?`<div class="scchips" style="margin-bottom:8px">`+fch.map(([ax,l])=>
       `<button class="scchip on2" data-fc="${ax}">${esc(l)} ×</button>`).join('')+`</div>`
    :'';
  return `<div class="setcard"><div class="rh">This set</div>`+fchips+
    `<div class="scn">${res.length.toLocaleString()}<small>papers</small></div>`+tiers+
    `<div class="scyr">${yrows}</div>`+
    (sib.length>1?`<div class="scsub">the same pick, in each edition — share of that year</div>`:'')+
    (dch?`<div class="rh" style="margin-top:11px">Tested on <em>in this set</em></div><div class="scchips">${dch}</div>`:'')+
    (mch?`<div class="rh" style="margin-top:11px">Builds on <em>in this set</em></div><div class="scchips">${mch}</div>`:'')+
    (nf?`<div class="scsub" style="margin-top:9px">${nf} of ${res.length} cards use full text</div>`:'')+
    `<button class="scsplit ${st.grouped?'on':''}" id="grptog2">${st.grouped?'Show papers':'Split into subgroups'}</button>`+
    `</div>`;
}
function render(){
  const landing=st.corp===null;
  $('#landing').hidden=!landing;
  $('#rail').hidden=landing;
  document.querySelector('.wrap').classList.toggle('withrail',!landing);
  document.querySelector('.searchrow').hidden=landing;
  if(landing){
    $('#legend').hidden=true; $('#results').innerHTML=''; $('#inset').hidden=true;
    $('#story').hidden=true;
    $('#q').placeholder='';
    $('#landing').innerHTML=landingHTML();
    wireLanding();
    return;
  }
  // The landing is used once; after that the rail moves between venues freely.
  // Filters survive a switch — the vocabulary is shared, so "robotics" means the
  // same thing at the next conference.
  $('#rail').innerHTML=
    `<button class="rv ${st.corp===-1?'on':''}" data-rv="-1">All<span>${P.length.toLocaleString()}</span></button>`+
    VENUES.map(ven=>{
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
  $('#legend').hidden=!on;
  if(!on){
    $('#results').innerHTML=`<div class="start">Search above, pick a mover on the left — or go back for the full digest.`+
      `<span>Nothing is listed until you do — ${(st.corp===-1?P.length:CY[st.corp].n).toLocaleString()} papers is the problem, not the answer.</span></div>`;
    $('#inset').hidden=true; $('#story').hidden=true;
    $('#q').placeholder='';
    const rt=$('#rtr'); if(rt){ rt.innerHTML=railTrend();
      rt.querySelectorAll('[data-tid]').forEach(el=>el.onclick=()=>{
        applyChgRow('t',+el.dataset.tid); render();}); }
    return;
  }
  { const V3=slotState();
    const scope=V3.k||V3.d||(st.lim!==null?`“${st.lim}”`:null)||V3.u||V3.b;
    $('#q').placeholder=scope?`search within ${scope}…`:''; }
  const res=results();
  if(qPending()){
    $('#inset').hidden=true;
    $('#results').innerHTML='<div class="capped">loading the search index…</div>';
    const rt0=$('#rtr'); if(rt0)rt0.innerHTML='';
    return;
  }
  drawStory();
  drawInside();
  { const rt=$('#rtr'); if(rt){ rt.innerHTML=railSetCard(res);
      const g2=rt.querySelector('#grptog2'); if(g2)g2.onclick=()=>{st.grouped=!st.grouped;render();};
      rt.querySelectorAll('[data-fc]').forEach(el=>el.onclick=()=>{
        const ax=el.dataset.fc;
        if(ax==='l')st.lim=null; else clearSlot(ax);
        render();});
      rt.querySelectorAll('[data-ck]').forEach(el=>el.onclick=()=>{
        const id=+el.dataset.cid;
        if(el.dataset.ck==='d')st.ds=st.ds===id?null:id; else st.meth=st.meth===id?null:id;
        render();}); } }

  const extra=nearby();
  const show=res.slice(0,MAX_SHOWN);
  // No "show more": nobody reads to the end of 6,637. Past the cap the answer is
  // to narrow, and the count above says how much is not on screen.
  const extraShown=extra.slice(0, res.length>=MAX_SHOWN?0:Math.min(extra.length,MAX_SHOWN-res.length));
  ensureSpans(new Set([...show,...extraShown].map(i=>P[i].cy)));
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
  if(st.grouped){ renderGrouped(res); return; }
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


function countInto(base){
  const n=new Array(T.length).fill(0);
  for(const i of base) for(const t of TOPIC_OF[i]) n[t]++;
  return n;
}


// vocabularies the chart, sentence slots and set card all draw from
const MV=D.mvocab||[], MS=D.methods||[], MF=D.mfams||[];
const DS=D.datasets, DSSET=new Set(DS.map(([d])=>d));
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
    for(const m of (p.mu0||[])){ const top=par.get(m)??m;
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
    for(const di of new Set(p.k0)){ if(!ok.has(di))continue;
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
  let tot1=0, tot0=0; for(const pr of pairs){tot1+=CYN[pr.c1]; tot0+=CYN[pr.c0];}
  const by=new Map();
  for(const pr of pairs){
    for(const r of changedRows(pr.c0,pr.c1)[tab]){
      const n0=CYN[pr.c0], n1=CYN[pr.c1];
      const pp=(r.a+r.b)/(n0+n1), se=Math.sqrt(pp*(1-pp)*(1/n0+1/n1));
      const z=se?(r.b/n1-r.a/n0)/se:0;
      const lo=Math.min(r.s0,r.s1), hi=Math.max(r.s0,r.s1);
      const mat=Math.abs(r.s1-r.s0)>=D_MIN||(lo>0?hi/lo:1e9)>=FOLD_MIN;
      let e=by.get(r.l); if(!e)by.set(r.l,e={l:r.l,id:r.id,lanes:[],a:0,b:0});
      e.a+=r.a; e.b+=r.b;
      e.lanes.push({hue:pr.hue,s0:r.s0,s1:r.s1,z,mat,nw:r.a<=2&&r.b>2,gn:r.b<=2&&r.a>2});
    }
  }
  const rows=[...by.values()].filter(e=>e.lanes.some(x=>Math.max(x.s0,x.s1)>=minS));
  for(const e of rows){
    e.up=e.lanes.filter(x=>x.z>=Z_SHOW&&x.mat).length;
    e.dn=e.lanes.filter(x=>x.z<=-Z_SHOW&&x.mat).length;
    e.nw=e.lanes.some(x=>x.nw); e.gn=e.lanes.some(x=>x.gn);
    e.maxz=Math.max(...e.lanes.map(x=>Math.abs(x.z)));
    e.s1u=e.b/tot1*1000; e.s0u=e.a/tot0*1000;
  }
  const rising=rows.filter(e=>e.up&&!e.dn&&!e.nw)
    .sort((x,y)=>y.up-x.up||y.maxz-x.maxz).slice(0,LABEL_CAP);
  const fresh=rows.filter(e=>e.up&&!e.dn&&e.nw)
    .sort((x,y)=>y.s1u-x.s1u).slice(0,LABEL_CAP);
  const falling=rows.filter(e=>e.dn&&!e.up)
    .sort((x,y)=>y.maxz-x.maxz).slice(0,LABEL_CAP);
  return {pairs,rising,fresh,falling,all:rows};
}
// Log x. The linear form could not show both size and growth: length is an
// absolute encoding, so 0.2->0.9% (a 4.5x rise) was invisible next to a big
// flat bar. On a log axis equal tails are equal FOLD changes, which is the
// comparison the sections are making — and the axis says so with plain ticks,
// not a caption. Bars start at the 0.1% floor; nothing below it is drawn.
const CHG_F=1;   // floor, per-1,000
function logX(M){ return v=>v<=CHG_F?0:Math.log(v/CHG_F)/Math.log(M/CHG_F)*100; }
function laneHTML(x){
  const dark=`var(--v${x.hue})`;
  const pale=`color-mix(in srgb, var(--v${x.hue}) 34%, var(--card))`;
  if(x.lone)
    return `<span class="lane" title="${(x.s0/10).toFixed(1)}% — single edition, no year pair">`+
      `<i style="width:${Math.max(x.w0,.6).toFixed(1)}%;background:${pale}"></i></span>`;
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
  const MIXBY={}; (DG.mix||[]).forEach(r=>MIXBY[r.ti]=r);
  const rawT=chgTab==='t'?changedRows(S.pairs[0].c0,S.pairs[0].c1).t:null;
  // fields whose own share never moved but whose INSIDE did (image generation)
  const shifted=chgTab==='t'
    ?(DG.mix||[]).filter(r=>![...S.rising,...S.fresh,...S.falling].some(e=>e.id===r.ti))
       .map(r=>{const raw=rawT.find(x=>x.id===r.ti);
                return raw&&{l:r.l,id:r.ti,lanes:[{hue:S.pairs[0].hue,s0:raw.s0,s1:raw.s1}],gn:0};})
       .filter(Boolean)
    :[];
  const secs=[['rising',S.rising],['shifting inside',shifted],['new',S.fresh],['falling',S.falling]]
    .filter(x=>x[1].length);
  if(!secs.length)return '';
  const all=secs.flatMap(x=>x[1]);
  const M=Math.max(...all.flatMap(e=>e.lanes.flatMap(x=>[x.s0,x.s1])),CHG_F*2);
  const X=logX(M);
  const ticks=[1,3,10,30,100].filter(t=>t<=M*1.04);
  const grid='<div class="chgg mv">'+ticks.map(t=>`<i style="left:${X(t).toFixed(2)}%"></i>`).join('')+'</div>';
  const axis='<div class="chgax mv">'+ticks.map(t=>`<b style="left:${X(t).toFixed(2)}%">${t/10}%</b>`).join('')+'</div>';
  const pill=(x,ti)=>{
    const kind=x.ax==='u'?'m':x.ax==='d'?'t':null;
    const attrs=kind?`data-pill="${ti}:${x.ax}:${x.id}"`:'disabled';
    return `<button class="pill ${x.up?'up':'dn'}" ${attrs}>${esc(x.l)} `+
      `<i>${x.up?'↑':'↓'}</i> <b>${(x.s0/10).toFixed(0)}→${(x.s1/10).toFixed(0)}%</b>`+
      `${x.nw?'<span class="nw2">NEW</span>':''}</button>`;
  };
  // venues holding a single edition (NeurIPS 2025) draw one pale bar: where
  // they stand, with no year tail — a position can be shown, a trend cannot
  const singles=VENUES.map((v,hue)=>({v:v.v,hue,cs:CY.map((c,j)=>({c,j})).filter(x=>x.c.v===v.v)}))
    .filter(x=>x.cs.length===1).map(x=>({hue:x.hue,ci:x.cs[0].j,n:CYN[x.cs[0].j]}));
  const singleShare=(sg,id)=>{
    let c=0;
    for(let i=0;i<P.length;i++){
      const p=P[i]; if(p.cy!==sg.ci)continue;
      if(chgTab==='t'){ for(const [ti] of p.g) if(ti===id){c++;break;} }
      else if(chgTab==='m'){ for(const m of (p.mu0||[])) if(mtop(m)===id){c++;break;} }
      else { if(p.k0.includes(id))c++; }
    }
    return c/sg.n*1000;
  };
  const body=secs.map(([name,rows])=>`<div class="chgsec">${name}</div>`+rows.map(e=>{
    let lanes=e.lanes.map(x=>laneHTML({...x,w0:X(x.s0),w1:X(x.s1)})).join('');
    for(const sg of singles){
      const sh=singleShare(sg,e.id);
      if(sh>0)lanes+=laneHTML({hue:sg.hue,s0:sh,w0:X(sh),lone:1});
    }
    const tag=e.gn?'<em class="tag gone">gone</em>':'';
    const u0=e.s0u!==undefined?e.s0u:e.lanes[0].s0;
    const u1=e.s1u!==undefined?e.s1u:e.lanes[0].s1;
    const mix=chgTab==='t'?MIXBY[e.id]:null;
    const pills=mix?`<div class="pillrow"><span class="pillhd">inside</span>`+
      mix.shifts.slice(0,3).map(x=>pill(x,e.id)).join('')+`</div>`:'';
    return `<button class="cr mv" data-k="${chgTab}" data-id="${e.id}">`+
      `<span class="crl" title="${esc(e.l)}">${esc(e.l)}${tag}</span>`+
      `<span class="trk">${lanes}</span>`+
      `<b class="mvn">${(u0/10).toFixed(1)}→${(u1/10).toFixed(1)}%</b></button>`+pills;
  }).join('')).join('');
  const leg=S.pairs.length>1?''
    :`<div class="chgleg"><span><i style="background:color-mix(in srgb, var(--v${S.pairs[0].hue}) 34%, var(--card))"></i>${S.pairs[0].y0}</span>`+
     `<span><i style="background:var(--v${S.pairs[0].hue})"></i>${S.pairs[0].y1}</span></div>`;
  const tab=(k,l)=>`<button class="chgtab ${chgTab===k?'on':''}" data-tab="${k}">${l}</button>`;
  return `<div class="chgbox"><div class="chghd">What moved — since last year`+
    `<span class="chgtabs">${tab('t','fields')}${tab('m','methods')}${tab('d','benchmarks')}</span></div>`+
    `<div class="chgplot mv">${grid}${body}</div>${axis}${leg}</div>`;
}
// One chart pick replaces the whole selection: the reader asked a new question.
function applyChgRow(k,id){
  st.q=[]; const q=$('#q'); if(q)q.value='';
  st.sel=null; st.grouped=false;
  st.topics.clear(); st.fams.clear(); st.meth=null; st.mfam=null; st.ds=null;
  st.lim=null;
  if(k==='t')st.topics.add(id); else if(k==='m')st.meth=id; else st.ds=id;
}
function wireChanged(){
  document.querySelectorAll('[data-tab]').forEach(el=>el.onclick=()=>{
    chgTab=el.dataset.tab; render();});
  document.querySelectorAll('[data-pill]').forEach(el=>el.onclick=e=>{
    e.stopPropagation();
    const [ti,ax,id]=el.dataset.pill.split(':').map((v,ix)=>ix===1?v:+v);
    const r=(DG.mix||[]).find(r2=>r2.ti===ti);
    if(r)STORY=mixStoryFor(r,r.shifts.find(x=>x.ax===ax&&x.id===id)||r.shifts[0]);
    if(st.corp===null){
      enterWith(()=>{st.topics.add(ti); if(ax==='u')st.meth=id; else if(ax==='d')st.topics.add(id);});
    }else{
      st.topics.add(ti); if(ax==='u'){st.meth=id;st.mfam=null;} else if(ax==='d')st.topics.add(id);
      render();
    }});
  document.querySelectorAll('.cr[data-id]').forEach(el=>el.onclick=()=>{
    const k=el.dataset.k, id=+el.dataset.id;
    if(st.corp===null){
      const nm=k==='t'?T[id].l:k==='m'?MV[id]:dname(id);
      STORY={label:`<b>${esc(nm)}</b>`,sub:'picked from What moved',
             ...(k==='t'?{ti:id}:k==='d'?{di:id}:{})};
    }
    applyChgRow(k,id);
    if(st.corp===null)go(-1); else render();
  });
}

function renderGrouped(res){
  if(!EMB){
    ensureEmb().then(render);
    $('#results').innerHTML='<div class="capped">loading the embedding vectors…</div>';
    return;
  }
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


$('#q').addEventListener('input',e=>{
  st.q=e.target.value.toLowerCase().split(/\s+/).filter(Boolean);
  render();});
$('#pclose').onclick=closePanel;

const c=D.coverage;
// The coverage caveats are no longer a paragraph at the foot of the page. Each
// one now sits where it actually bites: the tag share next to the topic list, the
// "looks similar" mark on the papers it applies to, and "no sentence states what
// is new" on the cards that have none. Stating them in place beats a block that
// is read once and then ignored.

$('#ttl').onclick=()=>{ if(st.corp!==null)go(null); };
st.corp=corpFromHash();
render();
// prefetch the on-demand parts once the first paint is done — a reader on the
// landing costs nothing extra, a reader who searches never notices the split
setTimeout(()=>{ ensureSearch(); ensureEmb();
  if(st.corp!==null)ensureSpans([st.corp]); },1200);

</script></body></html>"""


ROBOTS = "User-agent: *\nDisallow: /\n"

# Cloudflare Pages / Netlify read this file. The meta tag already asks crawlers
# not to index; the header says the same thing to fetchers that ignore HTML.
HEADERS = "/*\n  X-Robots-Tag: noindex, nofollow\n"


def write_dist(html: str, parts: dict[str, str]):
    """Emit a folder that can be dragged onto a static host as-is."""
    d = ROOT / "dist"
    d.mkdir(parents=True, exist_ok=True)
    (d / "index.html").write_text(html, encoding="utf-8")
    (d / "robots.txt").write_text(ROBOTS, encoding="utf-8")
    (d / "_headers").write_text(HEADERS, encoding="utf-8")
    dd = d / "data"
    dd.mkdir(exist_ok=True)
    for name, body in parts.items():
        (dd / name).write_text(body, encoding="utf-8")
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

    # ---- split: the page inlines only what the first paint and every count
    # need; card sentences, the search indexes and the vectors load on demand.
    # Sized for the six-corpus future — a single file was 21.8 MB raw already.
    dump = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":"))
    parts: dict[str, str] = {}
    spans_by: dict[str, dict] = {c["k"]: {} for c in payload["corpora"]}
    keys_by_ci = [c["k"] for c in payload["corpora"]]
    for r in payload["papers"]:
        spans_by[keys_by_ci[r["cy"]]][str(r["i"])] = [
            r.pop("n"), r.pop("L"), r.pop("K"), r.pop("R"), r.pop("c1")]
    for k, v in spans_by.items():
        parts[f"spans_{k}.json"] = dump(v)
    parts["search.json"] = dump({"terms": payload.pop("terms"),
                                 "lims": payload.pop("lims")})
    parts["emb.json"] = dump({"emb": payload.pop("emb"),
                              "neighbors": payload.pop("neighbors"),
                              "nemb": payload["nemb"]})

    import zlib
    payload["pv"] = zlib.crc32("|".join(
        f"{n}:{len(b)}" for n, b in sorted(parts.items())).encode())
    html = HTML.replace("__DATA__", dump(payload))
    out = REPORTS / "index.html" if args.out is None else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    ddir = out.parent / "data"
    ddir.mkdir(exist_ok=True)
    for name, body in parts.items():
        (ddir / name).write_text(body, encoding="utf-8")
    c = payload["coverage"]
    psz = sum(len(b) for b in parts.values())
    print(f"wrote {out} ({len(html)/1024/1024:.1f} MB core) + data/ "
          f"({len(parts)} files, {psz/1024/1024:.1f} MB, fetched on demand)")
    if args.dist:
        print(f"  dist/ ready to upload: {write_dist(html, parts)}")
    print(f"  {c['total']:,} papers · {c['with_span']:,} with a verified span "
          f"({c['with_span']/c['total']:.0%}) · {c['tagged']:,} carrying a topic "
          f"({c['tagged']/c['total']:.0%}) · {len(payload['topics'])} topics")
    if c["from_fulltext"]:
        print(f"  {c['from_fulltext']:,} papers using full-text spans")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
