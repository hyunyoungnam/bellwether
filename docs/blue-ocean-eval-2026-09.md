# Blue-ocean instrument: retrospective validation (2026-09-14)

**Question.** Can the corpus say which *(application field, technique)* pairs
nobody has combined yet are worth combining — and can that claim be checked?

**Answer, measured.** Yes to the check, modestly to the claim. Ranking empty
(field, technique) cells by how the field's own stated failures match the
technique's claims elsewhere predicts which cells the next year's editions
fill at **1.3–1.5× the precision of popularity** at the top of the list
(P@25 0.36 vs 0.24, P@50 0.30 vs 0.22), and 16× a random pick. Over the whole
ranking (average precision) evidence-matching and popularity are equal
(0.11 vs 0.12). The instrument is worth shipping *with those numbers printed
on it*, and not as a discovery engine.

## Method

This is literature-based discovery (Swanson 1986; Tshitoyan et al., Nature
2019) done as temporal link prediction on a fully-held corpus:

- **Fields** = the domain a paper declares (`card_terms.dom`; 7,553 of
  29,595 papers declare one — 26%). ML-internal subfields ("computer
  vision", "large language models", …) are excluded: they are where
  techniques come from, not where they land. Three spellings merged.
- **Techniques** = what a paper builds on or proposes (`b` + `p`, alias-
  normalised); model product names (gpt-*, llama-*, …) excluded — running an
  LLM on a field is not a method crossing over.
- **Cell (F, T)** = number of F papers using T. Empty = 0.
- **Past** = NeurIPS 2024, ICML 2025, NeurIPS 2025, ICLR 2025.
  **Future** = ICML 2026, ICLR 2026 (held out; already on disk).
- **Truth** = cells empty in the past and filled in the future:
  **195 of 10,569** empty cells (base rate 1.8%), over 30 fields × 378
  techniques (field ≥ 12 past papers, technique ≥ 8).

Scorers rank the past's empty cells knowing only the past:

| scorer | what it knows | P@10 | P@25 | P@50 | P@100 | AP |
|---|---|---|---|---|---|---|
| random | nothing | 0.00 | 0.00 | 0.00 | 0.01 | 0.020 |
| pa (popularity) | size(F) × size(T) | 0.30 | 0.24 | 0.22 | 0.22 | 0.117 |
| cf (collaborative) | fields sharing techniques with F already use T, and vice-versa | 0.20 | 0.24 | 0.18 | 0.16 | 0.079 |
| lim (evidence) | cosine of F's failure-term profile (limitation sentences) vs T's claim-term profile (key-change/result sentences), IDF-weighted, terms in >2% of sentences dropped | 0.20 | 0.32 | 0.30 | 0.22 | 0.101 |
| **lim_cf** (shipped, "established") | lim × cf | **0.40** | **0.36** | **0.30** | **0.26** | 0.112 |
| lim_spec (shipped, "novel") | lim ÷ log(2 + fields already using T) | 0.10 | 0.12 | 0.12 | 0.07 | 0.049 |

Reproduce: `PYTHONPATH=src python -m icml.blue_ocean --eval`
(deterministic, <2 s). Emit: `--emit` → `data/processed/blue_ocean.json`.

## What the numbers say

1. **Evidence matching carries signal beyond popularity, at the top.** The
   top-25 of lim_cf are filled at 36% within a year vs 24% for "big field ×
   big technique". That is the part of the list a reader actually looks at.
2. **Over the full ranking it is a wash.** AP 0.112 vs 0.117. The
   instrument's value is concentrated in its first page — which is fine for a
   tool, and disqualifying for any grander claim.
3. **Novelty costs precision, predictably.** Discounting techniques already
   spread across many fields halves the hit rate (P@50 0.12). That is what an
   ocean is: fewer ships. The tool ships both tiers with both numbers so the
   reader chooses the risk.
4. **What a hit looks like** (past-only ranking; the 2026 paper that filled
   the cell): *protein design ← discrete diffusion model* → "Transition-
   Directed Discrete Diffusion for Allosteric Binder Generation";
   *drug discovery ← flow matching* → "Interpolation-Based Conditioning of
   Flow Matching Models for Bioisosteric …"; *autonomous driving ← mixture
   of experts* → "DroneDINO"; *healthcare ← DPO* → "StethoLM".

## Limits (state them with the tool)

- **Emptiness is lexical.** "embodied ai ← vision-language-action model" is
  an empty cell only because those papers name the technique differently.
  The evidence sentences are shown so a reader sees the artefact; the count
  is not a fact about the world, it is a fact about the card terms.
- **Filled ≠ good.** A cell filled in 2026 means someone did it and got
  accepted — a community judgment, not an outcome. Rejected attempts are
  invisible; the truth set has survivorship bias.
- **26% of papers declare a domain.** Fields are undercounted where authors
  do not name one; the domain vocabulary is the authors', not ours.
- **One split.** Only one clean temporal split exists (≤2025 → 2026). The
  gap to popularity is 4–7 hits at P@50 out of 195 truths; treat the ordering
  of scorers as robust and the magnitudes as ±0.05.
- **Six ML editions.** "Global" here means global within
  ICML/NeurIPS/ICLR. AAAI/CVPR/ECCV extend the same instrument without
  changing it (owner's plan once the skeleton stands).
- **Lag.** Ideas filled in 2026 were conceived in 2025; the instrument's
  horizon is about one edition cycle.

## Pre-registration

`data/processed/blue_ocean.json` built over all six editions is the
**frozen prediction for NeurIPS 2026**: when its accepted list is collected,
score both tiers' top-50 against the cells NeurIPS 2026 fills, with `pa` and
`random` as the same baselines. The commit that lands this file is the
timestamp; the scorer must not be re-tuned after the list is public.
