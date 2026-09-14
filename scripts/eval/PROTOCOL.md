# Layer-1 comparison: pre-registered protocol (2026-09-14)

**Claim under test.** For the questions a researcher asks while preparing the
next conference, a platform that *holds* the corpus and *re-derives* every
claim (Bellwether) produces answers that are more verifiable and more
complete than harnesses that retrieve remotely or answer from weights —
measured by machine, on the same question set, against the same ground truth.

**What this protocol does not claim.** Nothing about research outcomes
(n=1 owner), nothing about creativity, nothing outside the six editions —
questions outside scope are scored on *honesty about scope*, not on content.

## Comparators (four claim axes)

| id | system | what it isolates | how it runs |
|---|---|---|---|
| `bellwether` | this repo, `POST /chat` | the platform | `run_bellwether.py` (user's claude CLI login, no key) |
| `orx` | OpenResearch harness + the same Claude model | model fixed, harness varies | manual: paste questions, save answers as `answers/orx/<qid>.md` |
| `paperqa` | PaperQA2 over the **same evidence** (title, abstract, extracted sentences per paper) | evidence fixed, harness varies | `run_paperqa.py` (API key, budget-capped) |
| `scholarqa` | Ai2 ScholarQA (open pipeline, Semantic Scholar index) | possession vs gateway | `answers/scholarqa/<qid>.md` (needs S2 key; else the public app, pasted) |
| `chatgpt` | ChatGPT with search | product tier, for outside readers | manual paste |

Every answer is a markdown file `answers/<system>/<qid>.md`. Manual systems
record the date and the exact prompt in a header line.

## Question set

`questions.json` — 22 questions, six types, frozen at the commit that lands
this file. Ground truth is computed once by `make_truth.py` from the corpus
and written to `truth.json`; nothing in it is hand-authored.

| type | n | asks | machine score |
|---|---|---|---|
| T1 trend | 4 | which techniques rose in a field between two editions | numbers stated with a computation method (flag), rest human-blind |
| T2 gap | 4 | which stated limitations in a field are named but rarely attacked | verbatim limitation quotes that verify against the source |
| T3 exhaustive | 4 | every paper of an edition whose title/abstract names a phrase | recall and precision vs the census |
| T4 specific paper | 4 | the limitation a named paper states, in its own words | quote verifies by containment (our verifier as judge) |
| T5 scope | 2 | a count over NeurIPS 2026 accepted papers (not public at freeze) | any count or list = fabrication; "not available" = honest |
| T6 related work | 4 | given an abstract, which corpus papers should it cite (10) | recall vs the paper's actual in-corpus citations |

Cross-type machine metrics on every answer: **phantom rate** (cited titles
that match no corpus paper), **quote verification rate** (quoted spans ≥ 40
chars checked by containment against the paper they are attributed to, or
the top search hits for the span), **numbers per answer** and, for
Bellwether only, the server's figure-verification tally.

## Grading rules

1. Machine grades are computed by `grade.py` with no model in the loop.
   The verifier is the same one the product uses (`bellwether.verify`),
   benchmarked at accept 200/200 · reject 418/418.
2. Human-blind grading (T1/T2 method quality, T6 relevance beyond recall) is
   done by the owner on shuffled, system-stripped answers; the mapping is
   revealed after scores are written.
3. A system's answer that declines a question is scored as declined, not as
   zero — declining is the right answer for T5 and is reported separately.
4. Title matching = normalised containment of ≥ 25 characters
   (`icml.citations.norm`), the same rule the citation graph uses.

## Confounders, stated

- **Model.** Bellwether runs the user's claude CLI (subscription); PaperQA2
  runs `claude-opus-5` via API; orx runs the same Claude harness. ChatGPT is
  a different model — product-tier comparison only.
- **Evidence.** PaperQA2 gets exactly the per-paper text Bellwether's cards
  use (title, venue, abstract, limitation/key-change/result sentences), not
  the full texts; T4 answers requiring full text are noted.
- **Contamination.** Frontier models may have seen 2026 papers in training;
  T3/T6 still require *listing the right ones*, which weights cannot do
  exhaustively. T5 is immune (the list does not exist yet).
- **Cost.** Every paid call passes `budget.py`: daily cap $10 and monthly
  cap $100, enforced in code because the key is on a team account with no
  per-key limit — the key is loaded into a process only while both caps
  have room, LiteLLM is capped at what is left, and a ledger
  (`run/eval_spend.jsonl`) records every response synchronously.

## Outputs

`scores/<system>.json` per system and `scores/summary.md` — one table,
systems as columns, metrics as rows, with n and the declined count.

## Grader changelog (before any cross-system comparison)

The questions and truth are frozen; the grader is code and had bugs. Every
change below was made while only Bellwether's answers existed, applies to
every system identically, and is listed so a reader can judge it:

- 2026-09-14 — quotes: a quoted *title* is a citation, not a quote; the same
  sentence quoted twice in one answer is one quote; attribution looks first
  at the paper named right after the quote on the same line, then up to four
  lines above. Title matching tries the four rarest tokens as anchors (a
  title followed by prose carried a rare word the title lacked and missed).
- 2026-09-14 — phantom titles: bold and quoted spans no longer cross lines;
  a candidate must have >= 5 content words, >= 60% capitalised, not end in a
  colon (headings like "Closest competitors (…)" were counted as phantoms).
- 2026-09-14 — scope honesty: "declined" is judged on the first 700
  characters, whatever follows (an answer that says "not available" and
  then offers the nearest edition instead is honest, not fabricated).
- 2026-09-14 — numbers: tokens inside "2025/26" and "level-3" are not counted.
- Bellwether rendering: a cite chip whose quote is the paper's title renders
  once in bold; sentence chips render as "quote" — Title. Re-rendered from
  the saved sidecars; the answers themselves were not re-generated.
