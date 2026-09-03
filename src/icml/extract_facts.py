"""Full-text phase 5 — extract structured facts from sectioned full text.

Runs a local LLM under **guided JSON decoding**, so every output is schema-valid
by construction and never needs re-parsing or retrying.

The design choice that matters most is `role` on each method. Without it the idea
graph conflates "this paper proposes X" with "this paper compares against X", and
the constellation degenerates into a popularity map of well-known baselines.
Separating proposed / building-block / baseline is what makes an edge meaningful.

Every extracted item carries a **verbatim evidence span** copied from the paper,
so any node in the final graph can be traced to the sentence that produced it.
That is what keeps the atlas auditable rather than model-asserted.

Two passes, never merged — see OUT_BY_SOURCE below for why:

    # census pass (99.3%) — the only valid basis for sparsity/coverage claims
    .venv/bin/python -m icml.extract_facts --source abstract

    # enrichment pass (~70% arXiv subset) — richer methods for the Ideas layer
    .venv/bin/python -m icml.extract_facts --source fulltext

Add --limit N to smoke test. Both are resumable.

Outputs: data/interim/facts_abstract.jsonl, data/interim/facts_fulltext.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .common import FOCUS_YEAR, INTERIM, PROCESSED, RAW, read_jsonl


def corpus_default_year() -> int:
    return FOCUS_YEAR

FULLTEXT = INTERIM / "fulltext.jsonl"
RESOLVED = RAW / "arxiv" / "resolved.jsonl"

# Two passes, deliberately kept in separate files:
#
#   abstract  — every paper with an abstract (99.3% census), IDENTICAL input
#               shape for all. This is the ONLY basis allowed for sparsity,
#               connectivity or coverage claims.
#   fulltext  — the ~70% with an arXiv preprint, much richer. Enriches the Ideas
#               layer only.
#
# Never merge them into one table. Full text yields more methods per paper simply
# by having more input, so a mixed table would make preprinting subfields look
# methodologically richer — a coverage artifact, not a finding.
OUT_BY_SOURCE = {
    "abstract": INTERIM / "facts_abstract.jsonl",
    "fulltext": INTERIM / "facts_fulltext.jsonl",
}

DEFAULT_MODEL = "Qwen/Qwen3-14B"

# Section priority: keep the parts that state what the paper does, and drop
# experiment detail first when the budget binds.
SECTION_ORDER = ["abstract", "intro", "method", "background", "experiments", "conclusion"]

# Property order IS generation order under guided decoding. contribution_type is
# deliberately LAST: classifying it first made the model decide before extracting
# any evidence, which produced "unknown" alongside a list of proposed methods in
# ~44% of a sample. Generating it after tasks/methods/datasets conditions the
# label on what was actually found.
SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["tasks", "domain", "methods", "datasets",
                 "limitation", "key_change", "result_claim",
                 "novelty_spans", "contribution_type"],
    "properties": {
        "tasks": {
            "type": "array", "maxItems": 4,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "evidence"],
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "evidence": {"type": "string", "maxLength": 300},
                },
            },
        },
        # Application domain feeds level-3 multi-label tagging, where the
        # author-declared taxonomy fails: 40 autonomous-driving papers sit in 7
        # different area->subarea buckets, so "driving" must come from content.
        "domain": {
            "type": ["string", "null"], "maxLength": 40,
            "description": "application domain, or null for domain-general work",
        },
        "methods": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name", "role", "evidence"],
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    # The distinction that makes the idea graph meaningful.
                    "role": {"type": "string",
                             "enum": ["proposed", "building-block", "baseline"]},
                    "evidence": {"type": "string", "maxLength": 300},
                },
            },
        },
        "datasets": {
            "type": "array", "maxItems": 8,
            "items": {
                "type": "object", "additionalProperties": False,
                "required": ["name"],
                "properties": {
                    "name": {"type": "string", "maxLength": 60},
                    "evidence": {"type": "string", "maxLength": 300},
                },
            },
        },
        "metrics": {"type": "array", "maxItems": 6,
                    "items": {"type": "string", "maxLength": 40}},
        # The three spans that turn a novelty CLAIM into a readable DIFFERENCE.
        # A card showing only novelty_spans says "we improve X" without saying
        # what X could not do — measured on the 2026 run, that is most of them.
        # All three are verbatim and verified exactly like novelty_spans; a
        # near-miss is a rewrite, and a rewrite is not evidence.
        "limitation": {
            "type": ["string", "null"], "maxLength": 320,
            "description": "verbatim sentence stating what existing work fails at",
        },
        "key_change": {
            "type": ["string", "null"], "maxLength": 320,
            "description": "verbatim sentence stating the mechanism this paper changes",
        },
        "result_claim": {
            "type": ["string", "null"], "maxLength": 320,
            "description": "verbatim sentence stating the outcome, with numbers if given",
        },
        # THE PRIMARY ARTIFACT. Verbatim sentences stating what is new. Placed
        # late so the model selects them after having identified the actual
        # contribution, and verified against the source text after generation —
        # a span that is not a real substring is dropped, not trusted.
        "novelty_spans": {
            "type": "array", "maxItems": 3,
            "items": {"type": "string", "maxLength": 400},
        },
        # Last on purpose — see the comment above the schema.
        "contribution_type": {
            "type": "string",
            "enum": ["new-method", "theory", "empirical-study", "benchmark-or-dataset",
                     "position", "application", "survey", "tooling", "unknown"],
        },
    },
}

SYSTEM = """You extract structured facts from machine-learning papers.

Rules, in order of importance:
1. Ground every field in the provided text. Never use outside knowledge about the
   paper, its authors, or the venue.
2. Every `evidence` value must be a VERBATIM span copied from the provided text.
   Never paraphrase evidence. If you cannot find a verbatim span, omit the item.
3. `role` for each method:
   - "proposed"        = introduced or contributed by THIS paper
   - "building-block"  = existing technique this paper builds on, uses, or STUDIES
   - "baseline"        = compared against only, not used and not the object of study
   Getting this wrong is the most damaging error you can make. If the paper's
   subject IS an existing method (an analysis or stability paper), that method is
   "building-block", never "baseline".
4. A `task` is the PROBLEM being solved, never the system that solves it.
   Good: "image classification", "autonomous driving", "protein structure
   prediction", "constrained nonconvex optimization", "text-to-image generation".
   WRONG: "DiOpt", "AdaGC", "our framework" (those are the paper's method — they
   belong in `methods` with role "proposed"), or "diffusion model" (a technique,
   not a problem). If the work is purely methodological, name the general problem
   class it addresses.
5. `domain` is the real-world application area if the paper targets one
   ("autonomous driving", "healthcare", "drug discovery", "recommender systems").
   Use null for domain-general machine learning. Do not invent a domain.
6. `novelty_spans`: copy 1-3 sentences VERBATIM from the text above that state
   what is new or different about this work.
   - Copy exactly, character for character. Do not paraphrase, shorten, fix
     grammar, or join sentences. Every span is checked against the source and
     silently discarded if it is not an exact match.
   - Prefer sentences naming the specific contribution over generic claims like
     "we achieve state-of-the-art results".
   - If no sentence states what is new, return an empty array.
7. `limitation`, `key_change`, `result_claim` are single VERBATIM sentences,
   copied character for character, each checked against the source and discarded
   if not an exact match. Use null rather than writing one yourself.
   - `limitation`  : what existing work cannot do, or fails at. Usually a sentence
     beginning "However,", "Existing", "Prior work", or "Despite". If the text
     only asserts that a problem is hard, and never says what current methods get
     wrong, return null.
     The sentence must NAME what fails. A sentence whose subject is an
     unresolved reference — "this view", "these methods", "such approaches",
     "it" — reads as nothing when shown alone. If the naming sentence exists,
     pick it; if every candidate only points ("We show that this view is not
     accurate"), return null rather than an orphaned pointer.
   - `key_change`  : the sentence that says what this paper does DIFFERENTLY —
     the mechanism, not the claim. Prefer "we replace A with B" / "instead of A,
     we B" over "we propose X, a novel framework".
   - `result_claim`: the outcome sentence. Prefer one carrying a number or a
     named comparison over "achieves state-of-the-art results".
8. Use canonical short names ("LoRA", "PPO", "graph neural network"), not sentences.
9. Disambiguate terms by how the paper uses them. A word can name different things
   in different subfields: "diffusion" in a stochastic-process or SDE analysis is
   NOT a diffusion generative model; "transformer" in a power-systems paper is not
   the architecture. Name the concept the paper actually means, and if the sense is
   unclear, omit the item.
10. If a method has role "proposed", contribution_type must not be "unknown".
11. If the text is too fragmentary to judge, use contribution_type "unknown" and
    return empty arrays. Never guess."""


# ---- The DEEP pass: highlight papers only (the venue's own spotlights/orals).
# Four more questions a reader asks after "what is new", every answer a verbatim
# sentence verified like everything else. This enriches single cards (Guardrail
# 5); nothing may count, rank or filter on these fields.
DEEP_ORDER = ["abstract", "method", "experiments", "conclusion", "intro", "background"]

DEEP_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["mechanism", "numbers", "ablation", "own_limits"],
    "properties": {
        # HOW it works — the device, not the claim
        "mechanism": {"type": "array", "maxItems": 3,
                      "items": {"type": "string", "maxLength": 400}},
        # the size of the result, in the paper's own numbers
        "numbers": {"type": "array", "maxItems": 3,
                    "items": {"type": "string", "maxLength": 400}},
        # what actually carried the result
        "ablation": {"type": "array", "maxItems": 2,
                     "items": {"type": "string", "maxLength": 400}},
        # what the AUTHORS say does not work yet — about THIS paper
        "own_limits": {"type": "array", "maxItems": 2,
                       "items": {"type": "string", "maxLength": 400}},
    },
}

DEEP_SYSTEM = """You select sentences from a machine-learning paper. You never write sentences.

Every value you return must be a sentence copied VERBATIM, character for
character, from the provided text. Each one is checked against the source and
silently discarded if it is not an exact match — a paraphrase is worthless.
Return fewer items (or empty arrays) rather than paraphrasing.

- `mechanism`: 1-3 sentences from the method description that state HOW the
  approach works — the actual device or procedure, not the claim that it is
  novel or effective. Prefer sentences with concrete nouns ("we replace A with
  B", "each token attends to...", "the loss combines...") over announcements.
- `numbers`: 1-3 result sentences carrying the paper's own numbers — accuracy,
  speedup, error reduction, with the baseline or benchmark named. Skip
  sentences that claim improvement without a number.
- `ablation`: 1-2 sentences reporting what an ablation showed — which component
  mattered, what happens when it is removed or varied.
- `own_limits`: 1-2 sentences where the AUTHORS state limitations of THIS work —
  what it does not handle, where it fails, what is left open. This is about the
  paper itself, never about prior work. Usually in a Limitations or Discussion
  section. If the authors state none, return an empty array.

Every sentence must STAND ALONE for a reader who has not opened the paper:
never pick a sentence that points into the document ("Lines 11-14",
"Section 3", "Figure 2", "as described below") or that leans on an equation
to say what it does. Prefer the prose sentence that states the mechanism in
words.

If the text lacks a section (no ablation, no limitations), return an empty
array for that field. Never guess, never summarise."""


# PDF text carries the layout, not just the words: a two-column page breaks words
# across lines and leaves the hyphen behind. Measured on 80 spans pulled from
# camera-ready PDFs, 25% failed a whitespace-only comparison — and NONE of them
# were invented. 17.5% came back with the hyphen repaired, 2.5% more once
# punctuation was ignored. The old check was not strict, it was brittle: it threw
# away the paper's own sentences for how the typesetter wrapped them.
#
# Deliberately NOT relaxed to subsequence matching. A subsequence of a 24,000
# character document is nearly free to satisfy, so it would stop being evidence.
# The remaining 5% stay rejected.
_WSP = __import__("re").compile(r"\s+")
_HYPH = __import__("re").compile(r"(\w)-\s+(\w)")
_PUNCT = __import__("re").compile(r"[^\w\s]")


def _match_norm(s: str, drop_punct: bool = False) -> str:
    s = _HYPH.sub(r"\1\2", s or "")
    if drop_punct:
        s = _PUNCT.sub("", s)
    return _WSP.sub(" ", s).strip().lower()


def verify_spans(spans: list[str], source: str) -> tuple[list[str], int]:
    """Keep only spans that genuinely occur in the source text.

    This is what makes "extractive, not generative" a guarantee rather than a
    request. Guided decoding constrains structure, not truthfulness — a model can
    emit a fluent sentence that never appeared in the paper.
    """
    hay = _match_norm(source)
    hay_p = _match_norm(source, drop_punct=True)
    kept, dropped = [], 0
    for s in spans or []:
        n = _match_norm(s)
        if len(n) < 25:
            dropped += 1
            continue
        if n in hay or _match_norm(s, drop_punct=True) in hay_p:
            kept.append(s.strip())
        else:
            dropped += 1
    return kept, dropped


SPAN_FIELDS = ("limitation", "key_change", "result_claim")
DEEP_FIELDS = ("mechanism", "numbers", "ablation", "own_limits")


def finalize_deep(facts: dict, source: str) -> tuple[int, int, bool]:
    """Verify every deep field in place — same guarantee, different schema."""
    n_kept = n_drop = 0
    for f in DEEP_FIELDS:
        kept, dropped = verify_spans(facts.get(f) or [], source)
        facts[f] = kept
        n_kept += len(kept)
        n_drop += dropped
    return n_kept, n_drop, False


def finalize(facts: dict, source: str) -> tuple[int, int, bool]:
    """Verify every extractive field in place. Returns (kept, dropped, inconsistent).

    The three new single-sentence fields go through the SAME check as
    novelty_spans. They are the fields the card leans on hardest, so an
    unverified one would be the most damaging thing on the page.
    """
    kept, dropped = verify_spans(facts.get("novelty_spans"), source)
    facts["novelty_spans"] = kept
    n_kept, n_drop = len(kept), dropped
    for f in SPAN_FIELDS:
        v = facts.get(f)
        if not isinstance(v, str) or not v.strip():
            facts[f] = None
            continue
        k, d = verify_spans([v], source)
        facts[f] = k[0] if k else None
        n_kept += len(k)
        n_drop += d
    inconsistent = (facts.get("contribution_type") == "unknown"
                    and any(m.get("role") == "proposed" for m in facts.get("methods", [])))
    return n_kept, n_drop, inconsistent


def run_via_endpoint(args, jobs: list[dict], out_path) -> int:
    """Extract through an already-running vLLM OpenAI server.

    The in-process path allocates its own GPU memory, which fails when a server
    is already resident — and killing a colleague's (or your own long-running)
    server to reclaim 69 GB is not something a batch job should do. Posting to
    the endpoint reuses the weights that are already loaded.
    """
    import concurrent.futures as cf
    import threading
    import urllib.error
    import urllib.request

    url = args.endpoint.rstrip("/") + "/chat/completions"
    lock = threading.Lock()
    tally = {"ok": 0, "bad": 0, "flagged": 0, "kept": 0, "dropped": 0, "n": 0}

    def one(job: dict) -> dict:
        body = {
            "model": args.model,
            "messages": [{"role": "system", "content": DEEP_SYSTEM if args.source == "deep" else SYSTEM},
                         {"role": "user", "content": f"# {job['title']}\n\n{job['text']}"}],
            "temperature": 0.0,          # deterministic: reruns must reproduce
            "max_tokens": args.max_tokens,
            # Qwen enables thinking by default; with a JSON grammar the output must
            # be JSON from the first token. Verified: without this the server
            # returns a reasoning trace instead of an object.
            "chat_template_kwargs": {"enable_thinking": False},
            "response_format": {"type": "json_schema",
                                "json_schema": {"name": "facts",
                                                "schema": DEEP_SCHEMA if args.source == "deep" else SCHEMA}},
        }
        row = {"event_id": job["event_id"], "arxiv_base": job["arxiv_base"],
               "source": args.source, "model": args.model}
        raw = ""
        for attempt in range(3):
            try:
                req = urllib.request.Request(
                    url, data=json.dumps(body).encode(),
                    headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=args.timeout) as fh:
                    raw = json.load(fh)["choices"][0]["message"]["content"].strip()
                break
            except (urllib.error.URLError, TimeoutError, KeyError, OSError) as exc:
                if attempt == 2:
                    # A transient failure must never be written as a result — see
                    # the arxiv 429 trap in CLAUDE.md.
                    return {**row, "ok": False, "error": f"endpoint: {exc}", "transient": True}
                time.sleep(2 * (attempt + 1))
        try:
            facts = json.loads(raw, strict=False)
            fin = finalize_deep if args.source == "deep" else finalize
            kept, dropped, inconsistent = fin(facts, job["text"])
            row |= {"facts": facts, "ok": True}
            if inconsistent:
                row["flag"] = "unknown-type-with-proposed-method"
            with lock:
                tally["ok"] += 1
                tally["kept"] += kept
                tally["dropped"] += dropped
                tally["flagged"] += bool(inconsistent)
        except json.JSONDecodeError as exc:
            row |= {"ok": False, "error": f"json: {exc}", "raw": raw[:500]}
            with lock:
                tally["bad"] += 1
        return row

    start = time.time()
    print(f"posting to {url} with {args.workers} workers")
    with out_path.open("a", encoding="utf-8") as fh, \
         cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row in pool.map(one, jobs):
            # Drop transient failures rather than recording them: resume skips
            # anything already present, so a written failure is permanent.
            if row.get("transient"):
                continue
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            tally["n"] += 1
            if tally["n"] % 200 == 0:
                fh.flush()
                el = time.time() - start
                print(f"  {tally['n']}/{len(jobs)}  ok={tally['ok']} bad={tally['bad']} "
                      f"{tally['n']/max(el,1e-6)*3600:.0f}/h", flush=True)

    dur = time.time() - start
    tot = tally["kept"] + tally["dropped"]
    print(f"\ndone in {dur/60:.1f}m — {tally['ok']} ok, {tally['bad']} unparseable, "
          f"{tally['flagged']} flagged ({len(jobs)/max(dur,1e-6)*3600:.0f} papers/hour)")
    print(f"  spans: {tally['kept']} verified verbatim, {tally['dropped']} "
          f"rejected as not-in-source ({tally['dropped']/max(tot,1):.1%} hallucinated)")
    return 0


def build_text(sections: list[dict], max_chars: int,
               section_order: list[str] = SECTION_ORDER) -> str:
    """Assemble kept sections in priority order, within a character budget."""
    by_bucket: dict[str, list[str]] = {}
    for s in sections:
        by_bucket.setdefault(s["bucket"], []).append(s["text"])

    parts: list[str] = []
    used = 0
    order = section_order + [b for b in by_bucket if b not in section_order]
    for bucket in order:
        for text in by_bucket.get(bucket, []):
            if used >= max_chars:
                return "\n\n".join(parts)
            room = max_chars - used
            chunk = text[:room]
            parts.append(f"## {bucket}\n{chunk}")
            used += len(chunk)
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description="Extract structured facts with a local LLM.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--source", choices=["abstract", "fulltext", "deep"], default="fulltext",
                    help="abstract = 99.3%% census (use for sparsity); "
                         "fulltext = ~70%% arXiv subset (use for the Ideas layer)")
    ap.add_argument("--limit", type=int, default=0)
    # Scientific text with math/notation tokenizes at roughly 2 chars/token, NOT
    # the ~4 typical of prose. 24k chars is therefore up to ~12k tokens, which
    # overflowed a 12,288 context and killed the whole run. The A100 has ample KV
    # cache (42 GiB at 0.9 util), so give the window real headroom.
    ap.add_argument("--max-chars", type=int, default=24000,
                    help="paper text budget; assume ~2 chars/token for ML papers")
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--gpu-util", type=float, default=0.90)
    ap.add_argument("--max-tokens", type=int, default=1400)
    ap.add_argument("--endpoint", default=None,
                    help="OpenAI-compatible base URL of a RUNNING vLLM server, e.g. "
                         "http://127.0.0.1:8000/v1 — reuses loaded weights instead of "
                         "allocating a second copy on the GPU")
    ap.add_argument("--workers", type=int, default=16,
                    help="concurrent requests when --endpoint is used")
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--out", default=None,
                    help="write to this file instead of the default; use when "
                         "re-running a year whose old-schema output must stay readable")
    ap.add_argument("--ids", default=None,
                    help="file of event_ids, one per line — extract only these papers")
    ap.add_argument("--year", type=int, default=None,
                    help="edition year; default is the focus year")
    ap.add_argument("--venue", default="icml", choices=["icml", "neurips", "iclr"])
    ap.add_argument("--chunk", type=int, default=256,
                    help="papers per write batch; smaller = finer resume granularity")
    args = ap.parse_args()

    # Earlier years go to their own files. Merging them into facts_abstract.jsonl
    # would put three conferences in the table every corpus claim is computed
    # from, and the counts would silently become three-year totals.
    from .corpus import Corpus
    venue = {"icml": "ICML", "neurips": "NeurIPS", "iclr": "ICLR"}[args.venue]
    corpus = Corpus(venue, args.year or None or corpus_default_year())
    fulltext_path = FULLTEXT
    resolved_path = RESOLVED
    if args.source == "deep":
        # per-corpus output for every venue, focus included. Originally the
        # highlight pass; 2026-09-03 it became the corpus-wide agent-card
        # pass, so each corpus reads its RICHEST full-text file.
        if venue != "ICML":
            fulltext_path = INTERIM / f"fulltext_{corpus.key.replace('-', '_')}.jsonl"
            resolved_path = RESOLVED.with_name(f"resolved_{corpus.key}.jsonl")
        elif corpus.is_focus and not args.year:
            html = INTERIM / "fulltext_html_icml_2026.jsonl"
            if html.exists():          # the HTML re-extraction beats the PDF pass
                fulltext_path = html
        else:
            # earlier ICML years: PMLR camera-ready, rows carry event_id
            fulltext_path = INTERIM / f"fulltext_pmlr_{args.year}.jsonl"
        out_path = INTERIM / f"facts_deep_{corpus.key}.jsonl"
    elif corpus.is_focus and not args.year:
        out_path = OUT_BY_SOURCE[args.source]
    elif args.source == "fulltext":
        if venue != "ICML":
            # Other venues ride the arXiv bridge with per-corpus files
            # (resolved_<key>.jsonl from arxiv_match --venue, fulltext_<key>
            # from pdf_extract --out).
            fulltext_path = INTERIM / f"fulltext_{corpus.key.replace('-', '_')}.jsonl"
            resolved_path = RESOLVED.with_name(f"resolved_{corpus.key}.jsonl")
        else:
            # Earlier ICML years have no arXiv bridge but do have the PMLR
            # camera-ready (icml.pmlr), whose rows carry event_id directly.
            fulltext_path = INTERIM / f"fulltext_pmlr_{args.year}.jsonl"
            if not fulltext_path.exists():
                raise SystemExit(f"no {fulltext_path.name} — run `icml.pmlr text` first")
        out_path = corpus.facts_fulltext
    else:
        out_path = corpus.facts
    if not corpus.papers.exists():
        raise SystemExit(f"no {corpus.papers.name} — run `icml.normalize "
                         f"--venue {args.venue} --year {corpus.year}` first")
    papers = list(read_jsonl(corpus.papers))
    if args.out:
        out_path = Path(args.out)
    titles = {p["event_id"]: p["title"] for p in papers}
    done = {r["event_id"] for r in read_jsonl(out_path)} if out_path.exists() else set()

    jobs = []
    if args.source == "abstract":
        # Census pass: identical input shape for every paper.
        for p in papers:
            if p["event_id"] in done or not p.get("abstract"):
                continue
            jobs.append({"event_id": p["event_id"], "arxiv_base": None,
                         "title": p["title"], "text": p["abstract"][: args.max_chars]})
    else:
        if not fulltext_path.exists():
            raise SystemExit(f"no {fulltext_path.name} — run `icml.pdf_extract` first")
        # arxiv_base -> ICML event_id, so facts attach to the canonical paper
        # record. PMLR rows skip the bridge: they already carry event_id.
        to_event: dict[str, int] = {}
        if resolved_path.exists():
            for r in read_jsonl(resolved_path):
                if r.get("arxiv_base"):
                    to_event.setdefault(r["arxiv_base"], r["event_id"])
        for row in read_jsonl(fulltext_path):
            if not row.get("ok"):
                continue
            eid = row.get("event_id") or to_event.get(row["arxiv_base"])
            if eid is None or eid in done:
                continue
            text = build_text(row.get("sections") or [], args.max_chars,
                              DEEP_ORDER if args.source == "deep" else SECTION_ORDER)
            if len(text) < 500:
                continue
            jobs.append({"event_id": eid, "arxiv_base": row["arxiv_base"],
                         "title": titles.get(eid, ""), "text": text})

    if args.ids:
        only = {int(x) for x in Path(args.ids).read_text().split() if x.strip()}
        jobs = [j for j in jobs if j["event_id"] in only]
    if args.limit:
        jobs = jobs[: args.limit]
    if not jobs:
        print(f"nothing to do ({len(done)} already extracted)")
        return 0

    chars = sum(len(j["text"]) for j in jobs)
    print(f"source={args.source}  {len(jobs)} papers, ~{chars/4/1e6:.1f}M input tokens "
          f"-> {out_path.name}")

    if args.endpoint:
        return run_via_endpoint(args, jobs, out_path)

    from vllm import LLM, SamplingParams
    # vLLM 0.26 renamed guided_decoding -> structured_outputs. Support both so a
    # version bump in either direction doesn't silently drop schema enforcement.
    try:
        from vllm.sampling_params import StructuredOutputsParams
        constraint = {"structured_outputs": StructuredOutputsParams(json=DEEP_SCHEMA if args.source == "deep" else SCHEMA)}
    except ImportError:  # pragma: no cover — older vLLM
        from vllm.sampling_params import GuidedDecodingParams
        constraint = {"guided_decoding": GuidedDecodingParams(json=DEEP_SCHEMA if args.source == "deep" else SCHEMA)}

    print(f"loading {args.model}…")
    llm = LLM(model=args.model, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_util, trust_remote_code=True)

    sampling = SamplingParams(
        temperature=0.0,  # deterministic: reruns must reproduce
        max_tokens=args.max_tokens,
        **constraint,
    )

    start = time.time()
    def run_chunk(chunk: list[dict]):
        convos = [[{"role": "system", "content": SYSTEM},
                   {"role": "user", "content": f"# {j['title']}\n\n{j['text']}"}]
                  for j in chunk]
        # Qwen3 enables thinking mode by default. With a JSON grammar the output
        # must be JSON from the first token, so reasoning traces are both wasted
        # tokens and a fight against the grammar. Disable where supported.
        try:
            return llm.chat(convos, sampling,
                            chat_template_kwargs={"enable_thinking": False})
        except TypeError:  # template/vLLM without that kwarg
            return llm.chat(convos, sampling)

    ok = bad = flagged = spans_kept = spans_dropped = 0
    # Write per chunk, not once at the end: a full-text run is hours long, and
    # collecting everything in memory before the first write means a crash at 90%
    # loses the whole run. Chunked writes make resume meaningful.
    with out_path.open("a", encoding="utf-8") as fh:
      for c0 in range(0, len(jobs), args.chunk):
        chunk = jobs[c0 : c0 + args.chunk]
        for job, out in zip(chunk, run_chunk(chunk)):
              raw = out.outputs[0].text.strip()
              row = {"event_id": job["event_id"], "arxiv_base": job["arxiv_base"],
                     "source": args.source, "model": args.model}
              try:
                  # strict=False tolerates raw control characters inside strings.
                  # The grammar constrains structure but still lets a literal newline
                  # through in an evidence span, which strict JSON rejects.
                  facts = json.loads(raw, strict=False)
                  # Enforce the extractive guarantee before anything downstream
                  # can treat a span as the paper's own words.
                  # Same verification for both paths — see finalize().
                  kept, dropped, inconsistent = finalize(facts, job["text"])
                  spans_kept += kept
                  spans_dropped += dropped
                  row |= {"facts": facts, "ok": True}
                  if inconsistent:
                      row["flag"] = "unknown-type-with-proposed-method"
                      flagged += 1
                  ok += 1
              except json.JSONDecodeError as exc:
                  # Guided decoding makes this near-impossible; record rather than crash.
                  row |= {"ok": False, "error": f"json: {exc}", "raw": raw[:500]}
                  bad += 1
              fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        n=min(c0+args.chunk,len(jobs))
        print(f"  {n}/{len(jobs)} written  ok={ok} bad={bad} flagged={flagged}", flush=True)

    dur = time.time() - start
    print(f"\ndone in {dur/60:.1f}m — {ok} ok, {bad} unparseable, {flagged} flagged "
          f"({len(jobs)/max(dur,1e-6)*3600:.0f} papers/hour)")
    tot = spans_kept + spans_dropped
    print(f"  novelty spans: {spans_kept} verified verbatim, {spans_dropped} "
          f"rejected as not-in-source ({spans_dropped/max(tot,1):.1%} hallucinated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
