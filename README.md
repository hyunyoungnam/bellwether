# What's new in AI research

A reading tool for major AI conference proceedings. One edition of one
conference is ~6,600 accepted papers; nobody reads a list that long. This
project turns the proceedings into three answerable questions:

1. **Which papers are in my field?** — search, topics, methods, and benchmarks
   filter the corpus down to the few dozen that are yours.
2. **What is in the set I picked?** — the selection is counted, grouped, and
   compared, so deciding what to open is possible at a glance.
3. **What is new in this paper?** — each card shows the paper's **own
   sentences**, colour-marked: pink for why the work was needed, yellow for
   what is new, blue for what it achieved. Nothing is paraphrased; every
   displayed sentence exists verbatim in the paper and is machine-verified
   against it.

Currently loaded: six editions across three conferences — ICML 2025/2026,
NeurIPS 2024/2025, ICLR 2025/2026 — 29,605 papers in one cross-venue index.

## Install & run

Runs locally and serves on your network: install on one machine, browse from
any device on the same LAN. Requirements: git and Python 3.10+.

**Linux / macOS** — one line:

```bash
curl -fsSL https://raw.githubusercontent.com/hyunyoungnam/whatsnewai/main/install.sh | bash
```

**Windows** — install WSL once (PowerShell: `wsl --install`, then reboot),
open the Ubuntu terminal, and run the same line. Everything below happens
inside WSL; the browser on Windows reaches it at the printed address.

The script clones into `~/whatsnewai`, installs the `wnai` command into its
own venv, and fetches the search-engine binary and keys. Then:

```bash
wnai fetch-data --file <bundle>       # the data bundle (site + search index, ~450 MB)
wnai serve
```

(While this repo is private the one-liner needs an authenticated clone —
`git clone` it yourself and run `bash install.sh`; the curl form works from
the first public release. Set `WNAI_BUNDLE=<file-or-url>` before the
installer to fold `fetch-data` in.)

`wnai serve` starts everything on one port and prints both addresses:

```
  local:    http://127.0.0.1:8001
  network:  http://192.168.0.180:8001   <- other devices on this network
```

Ctrl+C stops everything. `wnai status` shows what is running and what data
exists. The data bundle is produced by `wnai bundle` on a build machine; a
downloadable release URL (for `wnai fetch-data --url`) will accompany the
first public release.

### Connect a coding agent (no API key)

The repo ships an MCP server over stdio — `wnai mcp` — with read-only tools
for search, similarity, topics, citations, and each paper's verified
sentences. Claude Code picks it up automatically from `.mcp.json` when opened
in this directory; the agent brings its own model, so no API key is involved.

## How the site is organised

- **Landing** — one card per conference, opening its newest edition, plus
  *Since last year*: which topics, methods, and benchmarks take a
  significantly different share of the conference than in the previous
  edition.
- **Inside a conference** — search plus three category rows (topic / method /
  benchmark). Nothing is listed until the reader narrows: showing all 6,637
  papers is the problem, not the answer. Result cards carry the highlighted
  passage, the extracted terms (proposes / builds on / compared with / data),
  the corresponding author where the paper names one, and links out.
- **An earlier year is never a browsing category.** Last year's edition exists
  as a baseline: it powers *Since last year* and appears among a paper's
  nearest neighbours (stamped with its year), nowhere else.

## How "Since last year" picks its rows

The chart answers "what is rising?" without ranking by opinion. Shares are
per-edition (papers per 1,000), because editions differ ~2× in size and raw
counts would only restate that. A category is shown when it passes **both**
tests, or is an anchor:

| Test | Rule | Why |
|---|---|---|
| Real | two-proportion z ≥ 2.576 (99%) | the corpus sizes decide what is sampling noise, not a hand-picked cutoff |
| Material | share moved ≥ 2 per 1,000, **or** ≥ 1.5× | statistically real but tiny drifts are not worth a row |
| Anchor | top-2 by current share, any change | the chart must also say what the biggest things are, or "rose" has no context |

At most 5 risers and 5 fallers are shown, ordered by current share. A category
with ≤ 2 papers in one year is tagged **new** or **gone** — at that count,
presence is indistinguishable from noise. In the bar, the pale band is last
year's share and the dark band is this year's, the longer drawn underneath so
the tail stays visible — a dark tail grew, a pale tail shrank. The figures are
"was → is", each as that year's share of its conference.

The comparison is computed from the same abstract-level extraction for both
years — never from full text, which only exists for part of one year and
would make that year look artificially richer.

## Principles

- **Extractive, never generative.** The local model selects sentences; rules
  cut them (pure deletion, verified as an ordered subsequence); nothing on
  screen is model-written prose. A span that fails verification against the
  paper is dropped, not repaired.
- **Show differences, never importance.** Cluster, count, share, and change
  are measurable and shown; "promising" and "breakthrough" are not. With 83%
  of papers claiming novelty, ranking by it would rank phrasing.
- **Coverage is stated.** Topic tags reach about half the corpus; full text
  exists for ~72% of 2026; contact lines for 65% of full texts. Gaps are
  shown as gaps.
- **If a view needs a sentence to be understood, redesign the view.**

## Pipeline (summary)

Feed collection → abstract scrape → normalisation (dedup: orals are listed
twice) → arXiv/PMLR full-text fetch and sectioning → structured extraction
with verbatim-span verification (local vLLM) → shared frozen taxonomy →
union embeddings and cross-venue neighbours → the site (`reports/`) plus a
Meilisearch index, packed by `wnai bundle` for installs. The pipeline needs a
GPU machine; an install only serves its output.

See `CLAUDE.md` for the full build documentation, data traps, and measured
quality numbers.
