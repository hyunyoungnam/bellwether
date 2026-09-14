"""PaperQA2 over the SAME evidence Bellwether's cards use — evidence fixed,
harness varies.

Corpus: one .txt per paper (title, venue, abstract, the extracted
limitation / key-change / result sentences) under data/eval/pqa_corpus, with
a manifest so PaperQA2 does not spend an LLM call per document inventing a
citation string. Embeddings run locally (sentence-transformers, CPU); only
the answering LLM is paid, and every call passes scripts/eval/budget.py.

    .venv-eval/bin/python scripts/eval/run_paperqa.py --build            # corpus files
    .venv-eval/bin/python scripts/eval/run_paperqa.py --limit 50 --dry   # cost trial
    .venv-eval/bin/python scripts/eval/run_paperqa.py                    # all questions

Needs ANTHROPIC_API_KEY in ~/.bellwether/.env. First run indexes the corpus
(hours on CPU for 29.6k papers; --limit N indexes a subset for trials).
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import budget  # noqa: E402

CORPUS = ROOT / "data" / "eval" / "pqa_corpus"
INDEX = ROOT / "data" / "eval" / "pqa_index"
OUT = HERE / "answers" / "paperqa"
LLM = "anthropic/claude-opus-5"          # parity with the Bellwether agent's tier
EMBED = "st-BAAI/bge-small-en-v1.5"      # local; verify the prefix on first run


def build(limit: int | None) -> int:
    from bellwether.mcp import Store, _VENUE
    S = Store()
    CORPUS.mkdir(parents=True, exist_ok=True)
    rows = []
    n = 0
    for key in S.union["corpora"]:
        venue, year = key.rsplit("-", 1)
        for eid, rec in sorted(S.papers(key).items()):
            g = S.gid_by.get((key, int(eid)))
            if g is None:
                continue
            sp = S.span_entry(g) or [None] * 4
            body = [f"Title: {rec.get('title', '')}",
                    f"Venue: {_VENUE.get(venue, venue)} {year}",
                    "", f"Abstract: {rec.get('abstract', '')}"]
            for lab, i in (("Limitation of prior work stated", 1),
                           ("Key change", 2), ("Result claimed", 3)):
                if sp[i]:
                    body += ["", f"{lab}: {sp[i]}"]
            f = CORPUS / f"{g}.txt"
            f.write_text("\n".join(body) + "\n", encoding="utf-8")
            rows.append({"file_location": f.name, "doi": "",
                         "title": rec.get("title", ""),
                         "citation": f"{rec.get('title', '')}. "
                                     f"{_VENUE.get(venue, venue)} {year}."})
            n += 1
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    with (CORPUS / "manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["file_location", "doi", "title", "citation"])
        w.writeheader()
        w.writerows(rows)
    print(f"corpus: {n} papers -> {CORPUS}")
    return n


def _cache_sentence_transformers() -> None:
    """PaperQA2 builds a SentenceTransformer per document it indexes (measured
    4.7 s per file = model load); one instance per model name is enough."""
    import sentence_transformers as st
    if getattr(st, "_bellwether_cached", False):
        return
    orig, cache = st.SentenceTransformer, {}

    def cached(name, *a, **kw):
        if name not in cache:
            cache[name] = orig(name, *a, **kw)
        return cache[name]

    st.SentenceTransformer = cached
    st._bellwether_cached = True


def settings():
    _cache_sentence_transformers()
    from paperqa import Settings
    s = Settings(llm=LLM, summary_llm=LLM, embedding=EMBED, temperature=0.0)
    # every model slot — the agent loop and parsing enrichment default to
    # gpt-4o and would fail (or bill OpenAI) silently otherwise
    s.agent.agent_llm = LLM
    s.parsing.enrichment_llm = LLM
    s.agent.index.paper_directory = str(CORPUS)
    s.agent.index.index_directory = str(INDEX)
    # batch_size defaults to 1 and the sentence-transformer is instantiated
    # per batch — measured 10 s per file, i.e. days for the corpus
    s.agent.index.batch_size = 2000
    s.agent.index.concurrency = 2
    s.embedding_config = {"batch_size": 128, "device": "cpu"}
    s.agent.index.manifest_file = str(CORPUS / "manifest.csv")
    s.parsing.use_doc_details = False        # no Crossref/S2 lookups
    s.parsing.disable_doc_valid_check = True
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="papers (build) / questions (run)")
    ap.add_argument("--dry", action="store_true", help="index only, no questions")
    ap.add_argument("--only", nargs="*", default=None, help="question ids")
    a = ap.parse_args()

    if a.build:
        return 0 if build(a.limit) else 1

    budget.load_env()
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY missing — put it in ~/.bellwether/.env", file=sys.stderr)
        return 2
    budget.arm_litellm()          # observe every call; LiteLLM stops at what is left
    budget.assert_room()
    print("budget:", budget.status())

    from paperqa import ask
    s = settings()
    qs = json.load((HERE / "questions_resolved.json").open(encoding="utf-8"))["questions"]
    if a.only:
        qs = [q for q in qs if q["id"] in set(a.only)]
    if a.limit:
        qs = qs[:a.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    if a.dry:
        qs = qs[:1]
    for q in qs:
        f = OUT / f"{q['id']}.md"
        if f.exists():
            continue
        budget.assert_room(0.5)
        t0 = time.time()
        try:
            r = ask(q["q"], settings=s)
        except budget.BudgetExceeded as exc:
            print(f"stopped: {exc}")
            return 3
        sess = getattr(r, "session", r)
        text = getattr(sess, "formatted_answer", None) or getattr(sess, "answer", "") or str(r)
        # the authoritative ledger line: PaperQA2's own per-session cost,
        # written synchronously; anything the callbacks saw beyond it is
        # flushed at exit
        budget.charge(LLM, usd=float(getattr(sess, "cost", 0) or 0), note=f"paperqa {q['id']}")
        f.write_text(f"<!-- system: paperqa | llm: {LLM} | embed: {EMBED} | "
                     f"{time.strftime('%Y-%m-%d %H:%M')} | {time.time() - t0:.0f}s -->\n"
                     f"{text}\n", encoding="utf-8")
        print(f"{q['id']}: {len(text)} chars, {time.time() - t0:.0f}s | {budget.status()}")
        if a.dry:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
