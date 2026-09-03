---
name: research-brief
description: Answer a research question from the conference corpus and save a span-cited brief the web page can display. Use when the user asks a question about research trends, fields, methods, or papers and wants a durable, verifiable answer.
---

# Research brief

Turn the user's question (`$ARGUMENTS`) into a saved brief: prose you write,
resting on citations you did NOT write — verbatim quotes from the papers,
machine-verified after saving.

## Rules that make the brief trustworthy

1. **Evidence comes only from the bellwether MCP tools** (search_papers,
   similar_papers, get_paper, list_topics, topic_papers, get_citations). If
   they are unavailable, run the same lookups via
   `PYTHONPATH=src python3` with `bellwether.mcp.Store` — never from memory.
2. **Every factual claim carries at least one cite**, and every cite's
   `quote` is copied EXACTLY from a tool response — a `verified_sentences`
   field or an abstract sentence. Never trim, bridge, or paraphrase inside a
   quote; pick a different sentence instead. Quotes under ~20 characters
   fail verification by design.
3. **Scope honesty.** The corpus is six editions (ICML 2025/26,
   NeurIPS 2024/25, ICLR 2025/26). If the question reaches beyond it, say so
   in the brief's final block instead of answering from your own knowledge.
4. **No importance ranking.** Report what differs, recurs, or is claimed —
   never which paper "matters most".

## Steps

1. Explore: 2–5 tool calls (search by phrasing variants, topic_papers when a
   topic label fits, similar/citations to widen from the best hits).
2. Write the brief JSON to `reports/briefs/<id>.json` — `<id>` is a short
   kebab-case slug of the question:

```json
{
  "id": "kv-cache-2026",
  "question": "the user's question, verbatim",
  "date": "YYYY-MM-DD",
  "agent": "claude-code",
  "answer": [
    {"text": "One paragraph of your prose making one point.",
     "cites": [{"gid": 14086, "quote": "the paper's exact sentence"}]},
    {"text": "Closing block: coverage and what the corpus cannot answer.",
     "cites": []}
  ]
}
```

3. Verify — this is not optional:

```bash
python3 scripts/verify_brief.py reports/briefs/<id>.json
```

4. If any quote fails, fix it (re-quote from get_paper) and re-run until the
   count is clean, or leave it failing only when the mismatch is itself the
   finding. The landing page lists the brief automatically; tell the user the
   verified count and the page to open.
