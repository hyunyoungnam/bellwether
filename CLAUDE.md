# ICML 2026 — Find the papers in my field, and what is new in them

## Why this exists

People go to a conference to find what is **new in their field**. Everything here
serves three questions a researcher actually asks, in this order:

1. **"There are 6,600 papers. Which ones are in my field?"**
   Banal to state, and the whole product fails without it.
2. **"I now have 40. What is in them, and which do I read?"**
   Answered by **visualizing the set the reader just selected** — arranged so the
   shape of their field is visible and the choice of what to open is obvious.
3. **"What is the new idea in this paper?"**
   Answered by showing **the paper's own sentences**, so they can skip the rest.

Everyone has the same 6,637 papers. The papers are not the product. **Getting a
researcher from 6,637 to the 30 that matter to them, showing them the shape of
those 30, and getting them from a paper to its idea in ten seconds, is the
product.**

**Reading the papers is the reader's job and we never do it for them. Deciding
which ones to read is ours** — that decision is what every view here must serve.
A list that merely proves the 40 papers exist has not done it.

**Read this whole file before building.** The second half records data facts and
traps that cost real time to discover.

---

## The product is processing, not access

Access is already solved. icml.cc lists every paper, arXiv serves the PDFs, and
nobody needs a fourth link list. **We do not ship access to text — we ship the
result of working that text over.**

This settles what the 26 GB of PDFs are for. Full text is an **input to
processing**, never a thing to serve. Do not build a PDF reader, a full-text
search box, or a "read it here" pane; the paper's own page is one click away and
always better. Full text exists to make our processed view carry more than a link
would.

It also settles a question an abstract cannot answer. **An abstract frequently
does not let a reader tell what a paper actually does** — which is a large part
of why a reader is stuck in front of 6,600 of them. A card that reprints abstract
prose has added nothing. The processed view must beat the abstract, or it has no
reason to exist.

**What "processing" is allowed to mean.** Selecting, decomposing, arranging,
counting, and placing — over the paper's own sentences. Not writing new ones.
Guardrail 2 is the hard edge here and is unchanged: every displayed string is
either extracted verbatim or computed from `data/processed/`. "Better than the
abstract" is earned by **structure** — the right sentence pulled out, the claim
split into proposes / builds on / compares against / data, the paper placed among
its neighbors — and never by paraphrase. **A fluent model-written summary is a
regression here, not an improvement**, however much better it reads.

---

## Who does what — the model never writes, it only picks

Two engines run in this project and the boundary between them is the guarantee.

| Work | Runs on | Why |
|---|---|---|
| Collection, parsing, normalization, schema design | **Claude Code** | written once, executed repeatedly; 6,637 papers need no LLM to be fetched or reshaped |
| **Selecting** sentences (limitation / key_change / result_claim) | **local model** | bulk and repeated — 17k papers/hour on the A100 |
| **Editing** those sentences (cutting the opening frame) | **rules** | a regex is auditable; a model here would end the extractive guarantee |
| Rule and prompt design, failure analysis | **Claude Code** | work like "why did 31.5% fail verification" |

**The local model chooses sentences; it never composes them.** The moment that
line moves, `verify_spans` has nothing to check and every claim on the page
becomes unfalsifiable.

Cost is the smaller reason for keeping extraction local. The larger one is that
the extraction was re-run three times in a single day while tuning the schema —
per-call pricing would have made that experiment unaffordable, and the schema
would still be wrong.

---

## Start of session

```bash
cd ~/whatsnewai
python3 scripts/status.py          # what exists, what's stale, what to run next
```

Stdlib-only stages run on system `python3` with `PYTHONPATH=src`. Anything using
embeddings or a local LLM runs on `.venv/bin/python` (see *Environment*).

---

## Problem 1 — Categorization

### The taxonomy has three levels, and only two exist

| Level | Example | Source | Coverage |
|---|---|---|---|
| 1. Area (9) | "Deep Learning" | author-declared | 90% |
| 2. Subarea (70) | "Robotics", "Health / Medicine" | author-declared | 89.9% |
| 3. **Topic** | **"autonomous driving"** | **must be derived** | not built |

**Level 1 is nearly useless here and should not drive navigation.** At a machine
learning conference "Deep Learning" is close to tautological — it holds 2,257
papers. Areas are paradigm labels, not fields anyone works in.

**Level 2 is free, author-declared, and genuinely useful.** Use it as the default
entry scaffold. It is the highest-trust signal in the dataset: no inference, no
model, straight from the submission form.

**Level 3 is the level that actually matters and does not exist.** Measured:
40 autonomous-driving papers are scattered across **7** different area→subarea
combinations — Computer Vision (15), Robotics (8), no-area (4), Generative Models
(2), and others. A researcher in that field must check four subareas and will
still miss papers. That failure is the reason this project exists.

### Level 3 must be MULTI-LABEL, not a partition

A driving paper is legitimately also computer vision, also video generation, also
world models. **Forcing one bucket per paper recreates the scattering under a new
name.** So:

- **Hierarchy (level 1→2) is for navigation** — browsing when you arrive cold.
- **Tags (level 3) are for retrieval** — "show me autonomous driving" must sweep
  across every subarea at once.

A paper carries as many topic tags as it earns. Tags overlap; counts do not sum
to the corpus size, and no view may imply they do.

### The vocabulary is a shared artifact, not a per-run computation

**Added 2026-08-14.** `icml.taxonomy` owns the level-3 vocabulary and every rule
that decides what may become a label. Three separate steps, and the middle one is
a decision a human makes once:

```bash
python3 -m icml.taxonomy discover --stamp 2026-08-14   # propose, from all corpora
#                       -> config/taxonomy.json         # freeze: the shared vocabulary
python3 -m icml.taxonomy label --all                    # apply to every corpus
```

**Deriving topics from each corpus separately makes them incomparable.** Measured
with the same ≥8-paper rule applied to each year alone:

| Corpus | Papers | Topics it derives on its own |
|---|---|---|
| ICML 2024 | 2,622 | 33 |
| ICML 2025 | 3,323 | 45 |
| ICML 2026 | 6,590 | **91** |
| shared by all three | | **21** |

`object detection` "disappears" by 2026 and `code generation` "appears" — neither
is a finding. A fixed paper count is far easier to clear in a corpus 2.5× larger,
so most of that gap is corpus size. Any trend read off per-corpus vocabularies is
an artifact. **Only `label` runs for a new conference; the vocabulary does not
move underneath the numbers.**

### The domain menu has one authored level, and it is marked as such

The 156 topics split by which extracted field produced them — `domain` (24) or
`tasks` (132). That split is derived. What is **not** derived is the grouping of
those 24 domains into five families (Life sciences & medicine, Physical sciences
& earth, Robotics & autonomy, Software & security, Society & economy).

Four groupings were tried before authoring one, and all four failed:

| Axis | Why it failed |
|---|---|
| topic-to-topic containment | 0 pairs at 80% overlap |
| author area | 76 of 156 land under "Deep Learning" |
| author subarea | 37 fragments, median 2 topics each |
| k-means on topic centroids | put "software engineering" inside a "reasoning" group |

Embeddings cannot adjudicate it either: **every topic-centroid pair sits between
0.85 and 0.92 cosine**, so the space has no resolving power at this level.
`DOMAIN_FAMILIES` in `taxonomy.py` is therefore a curation decision, written down
so it can be argued with. Every member label is still data-derived; only the
family names are ours, and they label a menu, never a paper. A domain nobody has
grouped shows under "Other" rather than disappearing.

`NOT_A_DOMAIN` is the companion rule: papers write "time series analysis",
"image processing", "real world" into the `domain` field, but those name a
modality or a task. They stay in the vocabulary and move to the task side.

### Cross-corpus numbers are shares, never counts

Anything spanning two corpora is stated per 1,000 papers. This is not pedantry —
it reverses conclusions:

| Topic | 2024 | 2025 | 2026 |
|---|---|---|---|
| drug discovery | 41 (15.6/1k) | 43 (12.9/1k) | 39 (**5.9**/1k) |
| federated learning | 26 (9.9/1k) | 26 (7.8/1k) | 30 (**4.5**/1k) |
| healthcare | 35 (13.3/1k) | 64 (19.2/1k) | 178 (**27.0**/1k) |

Raw counts say drug discovery is flat and federated learning grew. Both fell by
more than half as a share of the conference.

### Adding a new conference

`Corpus(venue, year)` in `icml.corpus` resolves every path from the name, so no
stage hardcodes the focus year again (`normalize --year 2024` used to overwrite
the 2026 canonical file). The sequence is: collect → abstracts → normalize →
extract_facts `--source abstract` → `taxonomy label`. Re-run `taxonomy discover`
**only** when deliberately revising the vocabulary, and re-label every corpus
together when you do.

### Deriving level 3

Ranked by trust, highest first:

1. **Datasets** are the strongest domain anchor. nuScenes/Waymo/KITTI signal
   driving far more reliably than keywords, because using the dataset is a
   commitment, not a word choice. 3,973 distinct datasets are already extracted.
2. **Extracted tasks and methods** from the census pass.
3. **Embedding neighborhoods** — papers that sit together usually belong
   together; useful for pulling in papers that avoid the expected vocabulary.
4. **Keyword rules** — cheapest, most brittle, and the easiest to over-trust.

Always report coverage: how many papers a tag caught, and that papers avoiding
the expected vocabulary are missed.

**Let the user define their field by example too.** A field is often
cross-cutting ("efficient attention for long context") and matches no node in any
taxonomy. Seed papers → nearest neighbors is a first-class entry path, not a
nice-to-have.

---

## Problem 2 — Novelty

### Extract sentences. Do not write them.

The novelty of a paper is shown as **verbatim spans from that paper**. Never a
model-written summary in their place.

Why this is not a stylistic preference:

- An extracted sentence provably exists in the paper and can be checked in two
  seconds.
- It carries the authors' own words and hedges.
- There is nothing to hallucinate.

Structured fields (proposes / builds on / compares against / datasets) are a
**navigational index over the spans**, not a replacement for them. Every
structured claim must be traceable to a span.

### The card is one edited passage, not a form (2026-08-18)

The three spans are no longer three labeled rows. They run together as a single
passage in the order a reader needs, and **color carries the role** instead of a
label:

| Highlighter | The question it answers |
|---|---|
| pink | why the work was needed — what prior work could not do |
| yellow | what is new |
| sky blue | what it achieved, with the paper's own numbers |

Text stays near-black on a translucent wash, so contrast is ~14:1 under every
color. The extracted terms (proposes / builds on / compared with / data / tasks)
sit inside the passage block, not in a footnote under it — they are part of the
summary, not metadata about it.

**Elision now also cuts the announcement.** "We introduce a framework that…"
became "a framework that…". Without it the card reads as scraped text; with it,
as an edited sentence. Still pure deletion, still verified as an ordered
subsequence — `_ANNOUNCE` in `site.py` lists the verbs, and it never fires if
fewer than 30 characters would remain.

### Who to write to comes from the paper, or not at all

The feed has no email field — checked, it does not exist. The ICML template
prints `Correspondence to: Name <addr>` on page one, and `icml.contacts` parses
exactly that: **3,077 of 4,770 full texts (65%), 46% of the corpus**. Papers that
do not print the line show their first author's name with no address. Guessing
that the first or last author is the corresponding one would invent a fact the
paper states plainly whenever it states it at all.

These addresses are published in the papers themselves, but republishing them in
bulk is a different act from printing them once — worth a deliberate decision
before the page is made public rather than unlisted. **Decided 2026-09-03
(owner): published as-is with the public release** — they are the papers' own
printed facts and contacting authors is normal academic practice. Revisit on
complaint.

### The three difference spans (added 2026-08-13)

A novelty span alone states a *claim*: "we provide an improved analysis under
much milder conditions". Milder than what? The reader cannot tell, so they open
the paper — and the card has done nothing. Measured on the 2026 census: 31% of
spans open with "we propose/introduce", and median title-word overlap is 25%.

The schema therefore carries three more **verbatim, verified** single sentences:

| Field | The sentence that says |
|---|---|
| `limitation` | what existing work fails at |
| `key_change` | what this paper does differently — the mechanism, not the claim |
| `result_claim` | the outcome, with numbers where the paper gives them |

They go through the same `verify_spans()` check as `novelty_spans`; a near-miss
is a rewrite and a rewrite is not evidence. **This is not license to write
summaries** — the fields are cut from the paper and rearranged, and the reader
gets structure, not prose. See *The product is processing, not access*.

**That reversed once, then reversed back — and the second correction matters
more.** The full-text pass looked worse: 40.7% of its "verbatim" quotes failed
verification against 5.5% for abstracts. The obvious reading was that the model
invents more when given more text.

**It does not.** Measured on 80 spans pulled from camera-ready PDFs:

| | |
|---|---|
| passed the old whitespace-only check | 75.0% |
| passed once the line-break hyphen was repaired | +17.5% |
| passed once punctuation was ignored | +2.5% |
| **actually absent from the source** | **0.0%** |

The verifier was not strict, it was **brittle**. It rejected the paper's own
sentences because of how the typesetter wrapped them, and that rejection was
being read as a hallucination rate. `verify_spans` now normalizes line-break
hyphens and punctuation on both sides. It is deliberately NOT relaxed to
subsequence matching: a subsequence of a 24,000-character document is nearly free
to satisfy and would stop being evidence.

**Full text is the better source for these three fields, not the worse one.**
Where the abstract says "existing methods are limited", the paper says "prior
work completes around 20 classes"; where the abstract says "we evaluate on
standard benchmarks", the paper says "~50% of PaSCo on SemanticKITTI". Papers
whose abstract states no limitation at all frequently state one in the intro.

So the passes are merged **field by field, not record by record**: the abstract
leads, full text fills only the fields it leaves empty. That union reaches 94% /
93% / 88% for limitation / key_change / result_claim, and 2,898 papers take
something from full text. `_PDF_HYPHEN` in `site.py` repairs the artifact at
display time, distinguishing a split word (`condition- ing` → conditioning) from
a real compound (`token- level` → token-level).

### Known and accepted: claims do not discriminate

Measured on this corpus:

- **82.9%** of abstracts contain an explicit novelty claim
  ("we propose", "novel", "first", "outperform")
- **82.0%** are classified `new-method`
- **11,515** distinct proposed names, of which **98% appear exactly once**

So "extract the novelty claim" returns ~5,400 structurally identical claims:
*we propose X, it is new, it beats Y*. Extraction is easy; discrimination is not.

**Decision (2026-07-31): we do not attempt relational novelty for now** —
novelty measured against what a reader already knows, or against a paper's
neighbors. The product shows what each paper states about itself. Revisit only
on explicit request.

Consequence to respect: **do not rank papers by novelty.** With 83% claiming it,
any ranking would be an artifact of phrasing. Show what a paper says is new; let
the reader judge.

---

## Problem 3 — Showing the set

Filtering 6,637 down to 40 is not the finish line. **Forty rows in a list is
still forty papers of reading**, and a reader who cannot tell which to open first
has been handed the original problem at 1/166 scale.

**The unit of the view is the selected set.** Once a reader has picked a field —
by tag, by subarea, by search, or by seed paper — the view's job is to make that
set legible at a glance: where its papers clump and where one sits alone, which
methods and datasets recur across it, which subareas it straddles, which are
orals. That is a visualization problem, and it is the part of this product that
is least built.

**This is not the corpus atlas that was rejected on 2026-07-31.** The distinction
is the scope, and it is the whole difference:

- The atlas drew all 6,637 papers and answered *"what is this conference like"* —
  a question no attendee asked, producing a picture nobody could act on.
- A field view draws **the 40 the reader chose** and answers *"which of mine do I
  open first"* — the decision they are actually stuck on.

If you find yourself drawing the whole corpus again, you have slipped back.

**Guardrail 1 still binds, and this is exactly where it gets tested.** "Which do
I read first" must be answered with *structure*, never with a score: cluster,
spread, recurrence, overlap, and relation are measurable and may be shown. A
computed "importance", "impact", or "read this first" ranking is not, and must
not be built — see also *do not rank papers by novelty* above. The reader ranks;
we make ranking possible.

Material already available for this: `landscape.json` coordinates (6,592 of
6,637 — the 45 papers without abstracts have none, and must be shown as
uncounted, not dropped silently), `neighbors.json`, the extracted method/dataset
/task vocabulary, topic tags, and the oral/spotlight flags.

---

## Guardrails

1. **Show what is different; never claim what is important.** Differences are
   measurable, significance is not. No "promising", "overlooked", "breakthrough".
2. **Every displayed string is either extracted verbatim or computed from
   `data/processed/`.** Nothing hand-written into a view.
3. **Never invent or estimate paper data.** Missing is shown as missing.
4. **Coverage is always stated.** A tag over 40 papers, a field with 89.9%
   coverage, a concept list showing 78 of 16,485 — say so in the UI.
5. **Anything that compares papers must be built on uniform extraction input.**
   Counts, rankings, sizes, and any cross-field statement come from the abstract
   pass only. Enriching a single paper's own view with full text is allowed and
   is the point (see *The product is processing, not access*) — but it is marked,
   and nothing sorts or scores on it. See *Two extraction passes*.

---

## What exists today

Built and validated. Reusable under the new framing.

| Asset | Size | Use |
|---|---|---|
| `data/processed/papers.jsonl` | 6,637 | canonical records; area/subarea, 99.3% abstracts |
| `data/processed/emb_6592_*.npy` | 6,592 × 1024 | BGE-M3 embeddings → neighbors, field-by-example |
| `data/interim/facts_abstract.jsonl` | 6,590 | census extraction: tasks/methods/datasets + evidence |
| `data/interim/facts_fulltext.jsonl` | 4,579 | richer extraction from arXiv PDFs |
| `data/interim/fulltext*.jsonl` | 286 MB | parsed full text (PDFs deleted after parsing) |
| `config/term_aliases.json` | 2,446 acronyms | deterministic vocabulary normalization |
| `data/processed/topics.json` | 91 topics | level-3 multi-label tags, explicit + expanded |
| `data/processed/neighbors.json` | 6,592 × 20 | precomputed "more like this" (1.6 MB) |
| `data/processed/embed_compact.json` | 128-dim int8 | multi-seed queries in-browser (1.1 MB) |
| `reports/index.html` + `reports/data/` | 1.4 MB gz core + 16 MB on-demand | the interface; `--dist` writes an uploadable `dist/` |

**Removed 2026-07-31** (served the abandoned "corpus atlas / trend report"
framing — do not recreate): `report.py`, `viz.py`, `trends.py`, `metrics.py`,
`sparsity.py`, `ideas.py`, `demo.py`, `analyze.py`, `config/lexicon.json`, and
their outputs. If you find yourself rebuilding a co-occurrence matrix, a trend
chart, an institution ranking, or a corpus-wide "atlas", stop — that is the
framing this project moved away from.

The surviving modules all serve the three questions. `landscape.py` survives for
its **embeddings** (field-by-example retrieval) and its layout, which is the raw
material for *Problem 3* — but only ever drawn over a selected set, never over
the whole corpus.

---

## Pipeline

| # | Command | Produces |
|---|---|---|
| 1 | `python3 -m icml.collect` | `data/raw/virtual_feed_<year>_<date>.json` |
| — | `python3 -m icml.normalize --venue neurips --year 2025` | other venues: same infra, same schema; feed-inline abstracts for past editions, `scripts/scrape_abstracts_generic.py` for the current one. `Corpus` paths are venue-qualified (`papers_neurips_2025.jsonl`); ICML keeps its historical names |
| 2 | `python3 -m icml.abstracts` | abstracts, 99.3% (~20 min) |
| 3 | `python3 -m icml.normalize` | `papers.jsonl` ← canonical |
| 4a | `python3 -m icml.arxiv_harvest` | 495k-record arXiv index (OAI-PMH) |
| 4b | `python3 -m icml.arxiv_match` | title→arXiv, offline, seconds |
| 4c | `python3 -m icml.arxiv_fetch` | PDFs (~26 GB) |
| 4d | `.venv/bin/python -m icml.pdf_extract` | sectioned text, 60% dropped |
| 5 | `.venv/bin/python -m icml.extract_facts --source {abstract,fulltext}` | structured facts + evidence |
| 6 | `.venv/bin/python -m icml.landscape` | embeddings + layout (cached `.npy`) |
| 7 | `.venv/bin/python -m icml.topics` | `topics.json` — level-3 multi-label tags |
| 8 | `.venv/bin/python -m icml.embed` | union embeddings + `neighbors_union.json` + `embed_union.json` — "more like this" across every active corpus (42% of nearest neighbors cross corpora; per-corpus lists could not return those). `union.json` defines **gid**, the only global paper key (now 25,068 vectors across five corpora; 76% of nearest neighbors cross corpora) — `event_id` collides across venues and tracks |
| 9 | `.venv/bin/python -m icml.site --dist` | `reports/index.html` + uploadable `dist/` |

**Earlier years, for trends only** (added 2026-08-13). One year cannot say what is
*growing* — only what is *big*. 2024 and 2025 are collected so "what is rising"
has an honest answer:

```bash
python3 -m icml.collect --years 2024 2025
python3 -m icml.abstracts --year 2024        # ~5 req/s, ~10 min per year
python3 -m icml.normalize --year 2024        # -> papers_2024.jsonl
.venv/bin/python -m icml.extract_facts --source abstract --year 2024 \
    --endpoint http://127.0.0.1:8000/v1 --model qwen3.6-27b --workers 40
```

| Year | Papers | Abstracts |
|---|---|---|
| 2024 | 2,634 | 99.7% |
| 2025 | 3,339 | 99.6% |
| 2026 | 6,637 | 99.3% |

Cross-year comparison is **abstract pass only** — 2024/25 have no PDFs at all, so
any full-text field would make 2026 look richer purely by year. Guardrail 5.

---

## Two extraction passes — never merge them

- `--source abstract` → **6,590 papers**, identical input shape for all. The only
  valid basis for any claim about the whole corpus.
- `--source fulltext` → **4,579 papers** (71.9%), ~2× the methods per paper.

Full text yields **5.98 methods/paper vs 3.04** from abstracts. Merged, every
paper with a preprint would look twice as methodologically rich — and preprinting
correlates with subfield, so that artifact would masquerade as a finding.

**Full text is unavailable for ~29% and the gap is not random** (LLM/vision
communities preprint far more than parts of theory and statistics). Use full text
to enrich a paper's own page; never to compare fields.

### Where each pass is allowed to surface (decided 2026-08-11)

| Surface | Source | Why |
|---|---|---|
| Anything counted, sized, ranked, or compared | abstract pass only | uniform input; see Guardrail 5 |
| A paper's own detail view | best available | the reader is looking at one paper, comparing nothing |
| A result card in a list | **best available, marked** | see below |

The result card is the arguable one, and the trade was made deliberately. Cards
sit next to each other, so a paper with a preprint shows visibly more — roughly
2× the methods. That asymmetry is real and is **disclosed on the card** (`· full
text`) rather than hidden by starving every card down to abstract level. The
alternative was worse: a uniformly thin card fails the test in *The product is
processing, not access*, and does so for all 6,637 papers to avoid an artifact
that affects the appearance of 4,601.

One rule keeps this from becoming the artifact CLAUDE.md warns about: **nothing
sorts, scores, or filters on how full a card is**. Build the flag
`--spans {abstract,fulltext}` into anything new so the uniform view stays one
argument away. (2026-08-27, owner decision: the per-card "read from the full
paper" tag and the sidebar "N of M cards use full text" line were removed —
with ~77% coverage on every corpus the asymmetry is small and the labels read
as clutter; the highlight screen's caption still states its coverage.)

---

## Data sources — verified 2026-07-28

**Feed:** `https://icml.cc/static/virtual/data/icml-<year>-orals-posters.json`
No auth. 2023–2026 confirmed. No abstracts, no keywords (`keywords` is empty for
every record).

**Abstracts:** server-rendered at `https://icml.cc/virtual/<year>/poster/<id>`
inside `<div class="abstract-content">`. ~5 req/s. 45 pages fail persistently.

**Search AND similar are bought, not built (2026-08-25).** Meilisearch (single
binary in `bin/`, DB in `data/meili/`, master key file beside it) serves
typo-tolerant, relevance-ranked search over title+abstract of every corpus,
and `/similar` over the same BGE-M3 vectors uploaded as a `userProvided`
embedder; documents are keyed by gid. `.venv/bin/python
scripts/search_index.py` (re)indexes both text and vectors — rerun it whenever
a corpus is added, re-normalized, or re-embedded (vectors ride the SAME
document upload; a separate vector pass would be dropped by the next full
reindex, since POST /documents replaces). Registering the embedder on an index
whose documents lack `_vectors` fails validation — papers without an abstract
carry `_vectors: {bge: null}` to opt out. The page reaches the engine through
`scripts/serve.py`, which serves reports/ AND proxies exactly two paths, POST
`/meili/search` and POST `/meili/similar`, injecting a search-only key
server-side — no other Meilisearch endpoint is exposed through the tunnel.
Both features degrade the same way: engine down → search falls back to the
shipped abstract-term index (word-AND, no typo tolerance), the Similar panel
falls back to the shipped 20-neighbor list. Result order is engine relevance
while a query is live, the tiers order otherwise. Restart after reboot:
`setsid nohup ./bin/meilisearch --db-path data/meili/db --http-addr
127.0.0.1:7700 --master-key "$(cat data/meili/master_key)" --no-analytics &`
then `setsid nohup python3 scripts/serve.py &`.

**Full text routes (decided 2026-08-25, measured):**
- **HTML is archived; PDFs are not (revised 2026-08-28).** The whole HTML
  corpus gzips to ~1.2 GB — after paying the refetch cost three times
  (sentences, then references, then author emails), keeping
  `data/raw/html/*.html.gz` is cheap insurance against the next field.
  The PDF rule below stands unchanged.
- **PDFs are an intermediate, never an archive.** Parse -> keep the sectioned
  JSONL (286 MB for both ICML passes) -> delete the PDF. The 39 GB of raw
  PDFs were deleted after integrity-checking the parsed text; re-download is
  the recovery path if the parser ever changes materially.
- **arXiv HTML is the primary route for new venues, PDF the fallback.**
  Sampled 50 papers against their own verified spans: HTML available 98%,
  line-break-hyphen artifacts 264 -> 3 per 100k chars (the noise class that
  faked the 40.7% hallucination rate), richer text (82k vs 34k chars). The
  crude test parser refound only 84% of PDF-verified spans (math and inline
  tags), so the switch is GATED: the production HTML parser must refind >=95%
  before it becomes primary.
  **Gate measured 2026-08-26 (production parser, 50 papers, 256 spans):**
  strict refind 89.8% — but the misses are NOT parser loss. Classified: 4
  spans truly gone and ~19 lightly edited (arXiv VERSION DRIFT — the PDFs
  were an older version and the authors revised the sentences; e.g. "total
  variation distance" became "trace distance"), 3 spans carrying PDF
  footnote-glue artifacts ("OrchJail1"). Spans present in the raw HTML were
  refound at 231/232 = **99.6%**, which is the number the gate's intent
  (don't lose content by switching parsers) actually asks for. Declared
  PASSED on that basis; references gate passed 50/50 (median 47 entries).
  A moving source means strict refind against old-version spans can never
  reach 95% — do not re-litigate this with the same metric.
  **Second requirement (added 2026-08-26): the HTML parser must PRESERVE the
  references section** — as a `references` bucket with one entry per cited
  work, not prose. The PDF pipeline's 60% filter dropped references entirely,
  which is why no citation data exists. Citations unlock intra-corpus edges
  (A cites B across our six editions): reading-order structure in a selected
  set, a "both cite" strip in the compare view, and cite-graph entry paths —
  all structure, never a citation-count ranking (guardrail 1; incoming counts
  are also near-empty for a fresh corpus).

**Dead ends — do not retry:**
- **OpenReview API** (`api2.openreview.net`) → 403 ChallengeRequiredError. A bot
  challenge, not a rate limit; headers do not help.
- **OpenReview PDFs** (`openreview.net/pdf?id=`) → 403, serves HTML.
- **`icml.cc/api/miniconf/...`** → requires authentication.
- **PMLR** → no ICML 2026 volume yet.
- **Per-paper arXiv search API** → HTTP 429 even at 4.5s spacing. Use the
  OAI-PMH bulk harvest + offline matching instead (~1300 records/request).

---

## Traps — each cost real time

- **gid order is SORTED corpus key, not ACTIVE declaration order.** `icml.embed`
  stacked per-corpus embedding blocks in `available()` order (ICML first) while
  gid sorts rows by key string ("iclr" < "icml" < "neurips"). The two orders
  coincided for four corpora and silently diverged when ICLR joined — every
  paper wore another paper's neighbors, and nothing crashed. Rows are now
  placed by index (`emb[idx] = block`), never stacked. Anything new that joins
  per-corpus arrays onto gid must be checked against `union.json` order, not
  against the corpus loop.
- **NeurIPS and ICLR give an oral's second listing a SYNTHETIC OpenReview id**
  (`2025-Oral--451-87f1fe27`) where ICML repeats the real one — so keying
  dedup on the id left every ICLR/NeurIPS oral double-counted (210 surviving
  pairs at ICLR 2025). `normalize.dedupe_key()` now keys on the normalized
  title, which is safe by measurement: across all five collected feeds, no two
  distinct real papers share one.
- **Orals are listed twice** — once as Oral, once as Poster, different ids.
  Counting feed rows inflates the corpus by ~160 and double-counts every oral.
  `normalize.dedupe_key()` collapses them: 6,796 rows → **6,637 papers**. Always
  read `papers.jsonl`, never the raw feed.
- **`uid` is not an identity** — it collides across tracks. Key on OpenReview id,
  falling back to normalized title.
- **ML papers tokenize at ~2 chars/token**, not the ~4 typical of prose. A 24k-char
  budget is ~12k tokens and overflowed a 12,288 context, killing a whole run.
- **Guided-JSON property order is generation order.** `contribution_type` first
  made the model classify before extracting evidence — 44% returned "unknown"
  beside a list of proposed methods. Moving it last cut that to 12.5%.
- **Qwen3 thinking mode fights a JSON grammar** — pass
  `chat_template_kwargs={"enable_thinking": False}` or every paper returns an
  empty object.
- **`json.loads(..., strict=False)`** — the grammar still lets a literal newline
  into an evidence span.
- **`ninja` must be on PATH** or vLLM dies with a bare `FileNotFoundError`.
- **`datasets` contains counts, not only names** — "three datasets", "multiple
  benchmarks", "real-world data". Charted, these outrank every real dataset
  because they are the one phrase many abstracts share. `taxonomy.is_placeholder()`
  rejects them; reuse it anywhere dataset names are counted.
- **A fixed cluster count fakes splits.** `k = n/18` cut 31 federated learning
  papers into 28 and 3, which tells the reader nothing. Some fields are genuinely
  homogeneous. `site.js:groupsFor()` rejects any k that leaves a group under 3 or
  one group holding over 60%, and returns `{split:false}` instead of inventing
  groups; the UI then says the set does not split. (This rule and the percentile
  rescale below were the whole content of the `setview.py` draft, which was folded
  into `icml.site` on 2026-08-19 and removed — the draft could only show six
  hardcoded selections, the site groups whatever the reader picked.)
- **Corpus coordinates collapse a focused selection to a dot** — 31 federated
  learning papers occupied a few pixels. Rescale to the selection, and use the
  5th–95th percentile, not min/max: one distant paper owns the whole box and
  squeezes the rest into a corner. State how many fall outside the frame.
- **The GPU may already be full of your own vLLM server.** A server left running
  from an earlier session held 69 of 80 GB on both A100s. `extract_facts` loads
  vLLM *in-process* and would OOM. Do not kill the server — pass
  `--endpoint http://127.0.0.1:8000/v1 --model <served-name> --workers 40`, which
  reuses the loaded weights. Measured: 4,403 papers/hour at 40 workers vs 973 at
  8. The endpoint path needs `chat_template_kwargs={"enable_thinking": false}`
  exactly like the in-process path — without it the server returns a reasoning
  trace instead of an object.
- **`normalize --year` used to overwrite `papers.jsonl`.** The output path is now
  year-aware (`papers_2024.jsonl`, `papers_2025.jsonl`); only the focus year
  writes the canonical file and the manifest. Earlier-year facts go to
  `facts_abstract_<year>.jsonl` — never merged into `facts_abstract.jsonl`, or
  every corpus count silently becomes a three-year total.
- **The extractor writes absent values as the *string* `"null"`.** It survived
  canonicalization and shipped as a topic chip labeled `null` holding 15 papers
  across 9 subareas. `topics.NULL_LITERALS` filters it. Any new field that
  becomes a label needs the same guard — a length check does not catch it,
  `"null"` is four characters.
- **The `decision` vocabulary changes per year, so presentation flags must not
  be summed raw.** 2025 writes `Accept (poster / spotlight poster / oral)` as
  parallel labels; 2026 writes `Accept (regular / spotlight)` and picks its 168
  orals FROM the spotlights — its feed marks every oral a spotlight, which looks
  like a bug and is the source data. Adding raw `is_spotlight` across years adds
  two different quantities.
  **Decided 2026-08-26: the interface marks one distinction per venue — the
  venue's own decision, in the venue's own word.** At ICML 2026 the review
  decisions are regular/spotlight only; oral is stage programming (chosen from
  spotlights that committed to present in person), so an "Oral" badge there
  would rank logistics. Where spotlights exist the badge is Spotlight
  (`is_spotlight`); at ICLR 2026 the decision string itself is `Accept (Oral)`
  and the badge is Oral. `hl_field` in site.py derives this per corpus and
  `corpora[].hw` carries the word; results order highlight-first.
- **Institutions are free text** — `config/institution_aliases.json` merges known
  variants; extend conservatively.
- **A rate-limited batch must never be recorded as a result.** Writing
  `arxiv_id: null` on a 429 baked a transient failure into the dataset
  permanently, since resume skips anything already present.

---

## Measured quality (census pass, 2026-07-31)

6,566 papers in 22 min (~17.7k/hour), 2 unparseable.

| | |
|---|---|
| Papers with ≥1 **verified** novelty span | **96.6%** (6,363/6,590) |
| Spans **rejected as fabricated** | **11.7%** (2,006 of 17,204) |
| Papers with a `domain` | 35.4% |
| `contribution_type: unknown` | 1.0% (was 25%) |
| Internally inconsistent (flagged) | 11 (was 281) |

**The 11.7% matters more than any other number here.** Guided decoding constrains
*structure*, not *truthfulness* — the model produced 2,006 fluent sentences that
never appeared in the paper, while being asked for verbatim quotes.
`verify_spans()` checks every span against the source and drops non-matches.
**Never present an unverified span as the paper's words.** If you add a new
extractive field, verify it the same way.

Fixed by the 2026-07-31 re-run: `tasks` no longer contaminated with method names
(now "vision-language-action control", not "Any3D-VLA"); `contribution_type`
moved last in the schema; `domain` added.

Still weak:
- **`metrics` is frequently empty** — do not build a feature on it.
- **Topic coverage is 29%** — only 1,919/6,590 papers carry an explicit level-3
  tag, because many papers declare no domain and only generic tasks. State this
  in the UI; do not imply the tag set is exhaustive.
- **The topics overlap, and there is no hierarchy in the data to lean on.**
  Measured 2026-08-14: lexical nesting covers only 15 of 91 labels (`reasoning`
  has 6 children; four other parents have one each), and **no topic pair shows
  80% paper-set containment — zero**. `healthcare` is not above or below
  `robotics`. Building a tree would invent structure rather than reveal it.
  What works instead is **conditioning each list on the current selection**:
  picking `robotics` cuts 139 benchmarks to 9, picking `GSM8K` cuts 91 topics to
  12, and nothing is invented — a chip disappears only when no selected paper
  uses it. Aliasing is still unresolved (`LIBERO` vs `LIBERO benchmark`).
  For TASKS it is now half-resolved (2026-09-01): a task containing a topic
  label at a word boundary ("efficient llm inference") joins that topic's
  via/kind-2 class — `declared_with_folded` in taxonomy.py; measured 5,743
  foldable mentions, "llm inference" went 4→23 at ICML alone. Labels under
  8 chars stay exact-only ("llm" is contained in half the vocabulary).
- Rows with a `proposed` method but `unknown` type carry
  `flag: "unknown-type-with-proposed-method"`. The flag records the
  inconsistency; never silently auto-repair it.

---

## Environment

- **A100 80GB**, 48 cores, 444 GB RAM. Model size is not a constraint.
- `.venv/` holds torch/vLLM/sentence-transformers. Stages 1–3 stay stdlib-only so
  collection never breaks.
- `HF_HOME=~/whatsnewai/.cache/hf` — models stay inside the project.
- **Keep every generated file inside this project directory.** Firefox is a snap
  and cannot write to `/tmp`; screenshot to `reports/.preview/` (gitignored),
  never to `~/`.
- Embeddings are cached to `data/processed/emb_*.npy` — layout and clustering are
  cheap to re-run, embedding is not.
- Docker requires a password this session, so GROBID is unavailable; PyMuPDF with
  font-weight heading detection is the validated substitute.

---

## Interface principles

- **The conversation is the front door (2026-09-03, owner — supersedes the
  briefs-landing earlier the same day).** `/` serves the chat shell
  (`reports/chat.html`, shown name: **Frontier**, provisional): the reader's
  question spawns their own logged-in coding agent (Claude CLI headless, no
  API key) armed with only the wnai MCP tools; every anchor `⟦gid|quote⟧` is
  verified server-side BEFORE display (`wnai/verify.py`), tool calls stream
  live as the trail, conversations persist in `data/chats/`. The built corpus
  view moved to `/browse` and remains the evidence surface — cite chips open
  `/browse#p<gid>`, that paper's card selected and unfolded. Guardrails
  transfer: the agent's prose may be wrong and is never dressed as the
  paper's words; a green check means the quote provably exists in the paper.
- **The landing is a research workspace, not an analytics page (2026-09-03,
  owner decision — supersedes the "analysis first" landing of 2026-08-24).**
  The landing shows: search, the RESEARCH BRIEFS list, venue cards, and the
  quiet all-fields index. The digest/since-last-year/struggles charts left the
  landing — do not restore them there; their computations remain build-time
  material destined for agent tools. A brief is written by the reader's own
  coding agent (Claude Code/Codex via the wnai MCP server, `.claude/skills/
  research-brief`), saved to `reports/briefs/`, and verified by
  `scripts/verify_brief.py`: prose is the agent's, every cite carries a
  verbatim quote checked against the paper, unverified quotes are marked on
  the page, never hidden. Cite chips open the paper's card on the ALL screen.
  Briefs cite by **gid**; the page's `P` is row-indexed with the gid in `.i`
  — `rowOfGid()` is the only legal bridge (the gid trap, again).

- **No self-describing blurbs (2026-09-03, owner).** The interface never
  explains its own mechanism or virtues in copy — no "answers with your
  agent's account", no "every quote verified before display", no legend
  sentences decoding ✓/✗. Affordances teach by use (a tooltip on the mark, an
  example chip); a caption may state a unit or coverage figure, nothing more.
  If a screen seems to need such a sentence, that is the redesign signal
  below, not a licence to write it.
- **If a view needs a sentence to be understood, redesign the view.** This is the
  stated goal of the interface: intuitively readable without auxiliary
  explanation. Captions may state a unit or a coverage figure; they must never
  carry reading instructions. (The "Since last year" chart went slopegraph →
  paired bars for exactly this reason: a slopegraph needs to be explained, two
  bars of different length do not.)
- **The main page is every conference, analysis only (2026-08-24).** Headline
  tiles (one rule-picked fact per angle) lead; the sections follow; the venue
  cards sit at the BOTTOM as scope filters ("one conference at a time"), and
  papers are never browsed on the main page itself. Scope -1 ("#all") is a
  results screen over the union — every digest door lands there, with the
  venue-year stamp telling papers apart; a venue card scopes the same screen
  to one conference. The ladder from analysis to papers: a digest row expands
  in place (lanes + the papers CARRYING the shift, orals first) → clicking an
  evidence title opens the ALL screen scrolled to that paper's card → "open
  the N papers" opens the whole set.
- **Analysis first, selection second (2026-08-24).** The landing leads with
  what MOVED, and every printed fact is a door that applies itself as a
  selection. Below the since-last-year chart: WHERE THE MIX SHIFTED (inside a
  field, which building blocks / tasks / co-domains took a different share of
  the set — "image generation: diffusion 48→25%, autoregressive new"), WHAT
  THE FIELD FIGHTS (failures named in limitation sentences, share per 1,000),
  NEW THIS YEAR (benchmarks no earlier-edition paper used), and a quiet
  "browse all fields" index for the reader whose field did not move. Rows
  expand in place to lanes plus an "open the N papers" button — preview before
  commitment. All of it is precomputed at build time in `build_digest`-style
  code inside `site.py`, so first paint scans nothing.
- **The digest bar is the product's standing rule, disclosed — not FDR.**
  Measured: the mix family is ~120 tests whose strongest true shifts sit at
  z 2.6–3.2; Benjamini-Hochberg at q=.05 with that m zeroes the digest while
  the top of the list is unmistakably real. So rows pass |z| ≥ 2.576 +
  materiality, the caption states the family size and that at most about one
  shown row could be chance. Appearing is its own evidence: 0 → 8+ papers
  (GRPO, RLVR) earns a row with no test at all, as does vanishing.
- **Limitation text is register, not only content — guard it.** "rely",
  "significant", "typically" shift across editions with huge z while naming no
  failure; a discourse stoplist plus a df ceiling (a term in >5% of either
  edition is the genre's phrasing) and a topic-word echo guard (a term that is
  a field's own name tracks the field, not a failure) keep the fights section
  about failures. At topic granularity the failure axis stayed register noise
  even after the guards — the mix section therefore scans builds-on / tasks /
  co-domains only, and the failure axis lives at corpus level where the sample
  carries it. The abstract-pass limitation index (`lims0`) is a build-time
  instrument only and is not shipped.
- **The title is a title again (2026-08-24).** The slot grammar ("What's new
  in [topic] for [field]…") tested badly with its first reader and is gone;
  the current selection still reads as removable chips under the title.
- **The failure glossary is OURS, and says so (2026-08-25).** Struggles rows
  define their term on hover with a curated one-line gloss (FIGHT_GLOSS in
  site.py) — interface copy in the same class as a caption stating a unit, not
  paper data — visually separated and labeled "our gloss", with the papers'
  own sentences beneath as evidence. Guardrail 2 still governs everything
  attributed to a paper; the gloss never is.
- **A venue's cold screen leads with what the venue put forward (2026-08-26).**
  The corpus empty state lists the venue-declared highlight set (spotlights /
  ICLR orals) captioned "the venue's own selection, not ours", with coverage.
  Highlight papers carry a DEEP pass — `extract_facts --source deep` pulls
  mechanism / numbers / ablation / own_limits as verbatim verified sentences
  into `facts_deep_<key>.jsonl`. Display (revised 2026-08-27): full text
  serves the PASSAGE, never a side ledger — the mechanism sentences join the
  yellow wash inside the passage (deduped by token overlap); numbers /
  ablation / own_limits stay on disk unused for now. Every body sentence
  passes the STANDALONE gate (site.py `standalone()`): no document deixis
  ("Lines 11-14", "Figure 2", "as described below"), notation below a small
  density bar — an equation on a card is the reading the reader came to
  skip. A failing sentence leaves the field to the abstract pass. Guardrail
  5 still binds: nothing counts, sorts or filters on deep fields.
  Cards fold top/middle/bottom (2026-08-27): title-venue-year / passage /
  terms; the list default hides only the middle, a click unfolds it — the
  terms alone answer "which problem, what name, on what, which data".
- **Citations serve two views, neither a shared-bibliography list (decided
  2026-08-28).** A pairwise shared bibliography restates similarity and was
  removed. Instead: the compare view states the RELATION — "the left paper
  cites the right one" / each cites the other / neither — one line above the
  cards. And the set sidebar shows STANDS ON: the papers most of the
  selection cites, counted within the selection (>=3 citers, top 5) — the
  field's load-bearing ancestors; every row is a door narrowing the set to
  the papers citing that ancestor (chip x to leave). Data: citations.json
  edges ride the spans parts per paper (slot 6). A retained governor-cite
  name in a passage still links: in-corpus -> compare ("the paper it
  cites"), else arXiv/DOI/S2 via the paper's own or Semantic Scholar's
  parsed references — never a guess.
- **Picking a similar paper opens a side-by-side compare (2026-08-25).** The
  Similar panel's neighbor click no longer scrolls-or-walks: it opens a
  two-card split view — the read paper left, the picked one right, a "both
  papers" strip naming their shared methods/data/tasks above. The sidebars
  leave (the question at that moment is "what differs between these two",
  not "what is in the set"); selection state is untouched and × restores the
  list exactly. No signal may say which side is better — left is only "what
  you were reading". Similar on either card exits to that paper's panel, so
  the walk continues.
- **"Who fights my problem" is an entry path (2026-08-20).** A second input
  searches ONLY the extracted limitation sentences (89% of papers state one; a
  salient-term index of unigrams + df>=5 bigrams ships in the payload). The
  chosen string matches as a substring of the interned terms, so "hallucinat"
  is hallucination/-s/-ed at once; suggestions show the union count first and
  the narrower inflections under it. Matches mark the phrase inside the card's
  pink limitation span. This axis is orthogonal to every taxonomy — it groups
  papers by the failure they attack, in their own words.
- **The landing is conference-first (2026-08-19).** One card per venue, opening
  the newest year. An earlier year is never a browsing category — nobody goes
  back to browse 2025 — it exists as the baseline: it feeds the "Since last
  year" chart and appears in a paper's neighbors, stamped with its year, and
  nowhere else.
- **"Since last year" shows what a rule selects, not a top-N.** A row is
  colored and labeled only when its share change clears a two-proportion
  z-test at |z| ≥ 2.576 (99%) — the corpus sizes decide what is noise, not a
  hand-picked cutoff. The two largest current shares are kept as gray anchors.
  Everything else is not drawn: the gray context mass was tried and is what
  made the chart unreadable.
- **Search first, then show the set, then the paper.** Those are the three
  questions in order. A picture of the whole corpus is not a step in that path
  (see *Problem 3*); a picture of the reader's 40 papers is the missing one.
- **Every view must help decide what to open.** That is the test to apply before
  adding anything. A view that only proves the papers exist has failed it, no
  matter how much it displays.
- **Results must be pre-decomposed** — a result row shows *proposes → builds on →
  data*, not just a title. Scanning 50 results should take a minute.
- **Minimize prose, but never at the cost of a caveat.** Coverage and method
  notes stay.
- **One core page + on-demand parts (2026-08-24).** Still no build step and no
  Node, but no longer one file: index.html inlines only what first paint and
  every count need (5.3 MB raw, 1.4 MB gz); card sentences (per corpus), the
  search/limitation indexes and the vectors live in `data/*.json`, fetched on
  first touch plus an idle prefetch. Cards render instantly with a "loading the
  paper's own sentences…" line that fills on arrival. Consequence: file://
  preview no longer works — use the `python3 -m http.server` that already backs
  the tunnel. Sized for six corpora; a single file was 21.8 MB raw at two.
- Charts follow the project dataviz standard; **load the `dataviz` skill before
  touching one.** A scatter is an all-pairs form and caps at three categorical
  hues — research areas are never eight colors; use emphasis instead.
- **Look at what you built** before calling it done:
  ```bash
  firefox --headless --window-size=1400,1200 \
    --screenshot ~/whatsnewai/reports/.preview/shot.png \
    file:///home/hyunyoungnam/whatsnewai/reports/<file>.html
  ```

---

## Non-goals

- A narrative report or trend article
- An institution ranking
- A novelty score or "most important paper" ranking
- Model-written paper summaries presented in place of the paper's own words
- An automated future-research recommender
- A corpus-wide picture of all 6,637 papers, in any chart form
- A reader for the papers themselves — PDF pane, full-text search, "read it here"

**Not a non-goal, and easy to confuse with the first and sixth items:
visualizing the set a reader has selected.** That is *Problem 3* and it is wanted.
What was abandoned is the picture of the conference; what is wanted is the
picture of your forty papers.
