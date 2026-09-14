# Harness comparison — three systems (2026-09-14, preliminary)

Protocol: `scripts/eval/PROTOCOL.md`. Questions frozen (22, six types),
truth computed from the corpus, grader model-free. Three of the planned
systems have run; ChatGPT (product tier) and ScholarQA (gateway, needs an
S2 key) are pending. The numbers are what the machine measured, with the
caveats that apply; no human-blind grading yet.

| system | what it isolates | model | evidence | turn cap |
|---|---|---|---|---|
| bellwether | the platform | the user's claude CLI (subscription; default model claude-opus-5) | six editions, held: abstracts, extracted sentences, full text on demand, citation graph | 12 |
| paperqa (PaperQA2 2026.8) | evidence fixed, harness varies | claude-opus-5 answers/plans, claude-haiku-4-5 summarises | the same per-paper text Bellwether's cards use, indexed locally | — |
| claude-web | model fixed, platform removed | the same claude CLI, same default model | none held: built-in WebSearch/WebFetch, run from an empty directory outside the repo | 60 (see caveats) |

## Machine scores

| metric | bellwether | paperqa | claude-web | what it measures |
|---|---|---|---|---|
| answers | 21 / 22 | 22 / 22 | 21 / 22 | Bellwether: T2-01 over its 12-turn limit twice. claude-web: T1-02 hung three times (no output in 10 min) |
| quotes (≥ 40 chars) | 142 | 24 | 95 | quoted spans, titles excluded, duplicates collapsed |
| … verified against the paper named | **120 (85%)** | 4 (17%) | 22 (23%) | containment check against the attributed paper |
| … existing in *any* corpus paper | **128 (90%)** | 11 (46%) | 44 (46%) | the sentence is some paper's sentence, whoever it was attributed to |
| titles outside the corpus | 2.3% | 0.8% | 18% | title-like strings matching none of the 29,605 papers — for a web system mostly real papers from other venues, not inventions |
| exhaustive recall | 0.33 | 0.04 | **0.46** | share of the census list actually named |
| exhaustive precision | 0.73 | 0.40 | 0.73 | named papers that are in the census |
| related-work recall | 0.10 | 0.05 | **0.21** | overlap of 10 recommendations with the paper's actual in-corpus citations |
| paper's own limitation verifies (T4) | 4 / 4 | 4 / 4 | 4 / 4 | |
| scope honest (T5) | 2 / 2 | 2 / 2 | 2 / 2 | NeurIPS 2026 does not exist yet; all three said so |
| turns per answer | ≤ 12 | — | 22.5 avg | |

Cost: PaperQA2 8.33 USD for the day; Bellwether and claude-web 0 USD API
(subscription CLI).

## What the numbers say

1. **Honesty about scope and finding a paper's own sentence are not
   platform properties.** All three refused the NeurIPS 2026 questions and
   all three found the limitation sentence when asked for it by title. The
   evidence was reachable either way (held locally, or one search away).
2. **Quoting is where the platform shows.** Nine in ten sentences
   Bellwether puts in quotation marks are sentences some paper wrote, and
   85% verify against the paper they are attached to. For the same model
   with a browser, fewer than half of the quoted spans exist in any of the
   29,605 papers — the rest are paraphrases in quotation marks or text from
   pages outside the corpus — and PaperQA2, which summarises evidence before
   answering, is at the same 46%. Verification-before-display is the
   measurable difference, and it is not explained by data possession: the
   web baseline could reach every one of these papers online.
3. **Enumeration and related work go the other way, and the reason is
   specific.** The web baseline names 46% of a census list against
   Bellwether's 33% because it reads the proceedings pages, while
   Bellwether's `search_papers` returns top-k and there is no enumerate
   tool. It doubles Bellwether's related-work recall (0.21 vs 0.10) because
   given the abstract it finds the paper itself online and reads its
   reference list (9–17 proceedings/arXiv/OpenReview mentions per T6 answer)
   — while Bellwether holds the citation graph and its agent did not use it
   for the question. Two concrete product changes follow: an `enumerate`
   tool over the search index, and `get_citations`-driven related work.
4. **"Outside the corpus" is not "invented".** 18% of the web baseline's
   titles are not among the six editions; spot-checked, they are CVPR and
   arXiv-only papers, not fabrications. Within the corpus its precision on
   the census question equals Bellwether's (0.73).

## Caveats, stated

- n = 21–22 per system, one question edition, one grader. The quote gaps
  (85% vs 23% vs 17%; 90% vs 46% vs 46%) are large enough to survive the
  noise; the recall gaps are 2–4 hits and should be read as direction.
- Turn caps differ: Bellwether 12, claude-web 60. At 12 the web baseline
  failed 5 of its first 6 questions and a trend question measured 41 turns
  — one Bellwether tool call returns a census, one web search returns ten
  links. The cap was measuring itself; the difference is a confounder in
  Bellwether's disfavour on recall (it stops earlier) and is stated.
- The web baseline ran on the same machine and network; one question hung
  on a page fetch and is counted as failed after three tries.
- The grader was corrected during the day as each new system's answer
  format exposed a bug (PROTOCOL.md changelog); every fix applies to all
  systems. Human-blind grading of T1/T2 (method quality) has not been done.
- PaperQA2 ran with Haiku evidence summaries to fit the approved daily
  cap; its one Opus-summarised answer scored the same on its question.
- orx (OpenResearch on the same CLI) could not run on this network — its
  Rust binary trusts only bundled roots behind the TLS-inspecting gateway.

## Next

ChatGPT (product tier, manual paste) and ScholarQA (gateway, S2 key), then
the human-blind pass on T1/T2. On the product side, the two changes the
evaluation asks for: `enumerate` and citation-graph related work — and a
re-run of the same 22 questions after them.
