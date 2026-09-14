# Harness comparison — first two systems (2026-09-14, preliminary)

Protocol: `scripts/eval/PROTOCOL.md`. Questions frozen (22, six types),
truth computed from the corpus, grader model-free. This page reports the
first two of five planned systems; orx, ScholarQA and ChatGPT are pending
(manual paste / S2 key). Nothing here is a cross-system verdict yet — the
numbers are what the machine measured, with the caveats that apply.

| system | what it isolates | model | evidence |
|---|---|---|---|
| bellwether | the platform | user's claude CLI (subscription) | six editions, held: abstracts, extracted sentences, full text on demand, citation graph |
| paperqa (PaperQA2 2026.8) | evidence fixed, harness varies | claude-opus-5 answers/plans, claude-haiku-4-5 summarises evidence | the same per-paper text Bellwether's cards use (title, venue, abstract, L/K/R sentences), indexed locally |

## Machine scores

| metric | bellwether | paperqa | what it measures |
|---|---|---|---|
| answers | 21 / 22 | 22 / 22 | T2-01 exceeded Bellwether's 12-turn agent limit twice (retry included) — a product limit, counted as missing |
| declined | 8 | 7 | answers that open by saying the question cannot be answered as posed |
| phantom rate | 0.023 | 0.008 | title-like strings matching no corpus paper (Bellwether's: papers named from model memory in T6) |
| quotes / verified | 142 / **119 (84%)** | 24 / **4 (17%)** | quoted spans ≥ 40 chars checked by containment against the attributed paper |
| exhaustive recall | **0.33** | 0.04 | share of the census list (10–102 papers) actually named |
| exhaustive precision | 0.73 | 0.40 | named papers that are in the census |
| related-work recall | 0.10 | 0.05 | overlap of 10 recommendations with the paper's actual in-corpus citations (8–15) |
| paper's own limitation verifies (T4) | 4 / 4 | 4 / 4 | the quoted sentence is in that paper |
| scope honest (T5) | 2 / 2 | 2 / 2 | NeurIPS 2026 does not exist yet; both said so |
| numbers per answer | 17.7 | 14.6 | counted, not judged |

Cost: PaperQA2 22 questions = 8.33 USD total for the day (0.68 per question
on the first two, then ~0.25–1.0 depending on type; T6 questions cost most).
Bellwether: 0 USD API (subscription CLI).

## What the numbers say

1. **Where the evidence is the same, honesty is the same.** Both systems
   refused the NeurIPS 2026 questions and both found the paper's own
   limitation sentence when asked for it. Holding the corpus is not what
   makes a system honest about scope; the question type is answerable
   from the evidence either way.
2. **Quoting is where the architectures separate.** PaperQA2 summarises
   each evidence chunk with an LLM before answering, so what it puts in
   quotation marks is often the summary's wording, not the paper's — 4 of
   24 verify. Bellwether's chips are checked before display, and its
   unverified 23 are the sentences the agent quoted *without* an anchor
   (a known gap: anchor coverage is not 100%). This is the measurable
   form of "verification by re-execution vs by summary".
3. **Neither can enumerate — but by different amounts.** The census
   question ("every paper naming X") is answered at 33% by Bellwether and
   4% by PaperQA2. PaperQA2 says so itself ("the context does not support
   an exhaustive enumeration"). Bellwether's 33% is its own product limit:
   `search_papers` returns top-k, and there is no enumerate tool. That is
   the first concrete feature this evaluation asks for.
4. **Related-work recall is low for both** (0.10 / 0.05). Ten suggestions
   against 8–15 actual citations is a hard target; the metric mostly says
   "recommendation ≠ what the authors cited". Keep it, don't read much
   into it until the human-blind relevance pass.

## Caveats, stated

- n = 21–22 per system, one edition of the questions, one grader. The
  quote-verification gap (84% vs 17%) is large enough to survive the
  noise; the related-work gap is not.
- The grader was corrected several times today while only these two
  systems' answers existed (PROTOCOL.md changelog) — every correction was
  a format bug (italic titles, reference-list prefixes, decline phrasing,
  duplicate quotes) and applies to every system alike.
- PaperQA2 ran with Haiku evidence summaries to fit the approved daily
  cap; the one Opus-summarised answer scored the same on its question.
- Human-blind grading of T1/T2 (method quality) has not been done; those
  types contribute only their machine columns here.
- Bellwether's missing T2-01 is a real limit (`--max-turns 12`) and is
  left as measured; raising the limit is a product change for after the
  comparison.

## Next

orx (same Claude, different harness — the cleanest single comparison),
ChatGPT (product tier), ScholarQA (gateway) — answers to
`scripts/eval/answers/<system>/`, then `grade.py` adds the columns.
