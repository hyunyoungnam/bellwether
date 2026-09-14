# scripts/eval — the layer-1 comparison, in practice

Protocol and rules: `PROTOCOL.md`. Questions: `questions_resolved.json`
(22, frozen). Truth: `truth.json` (corpus-computed). Everything below writes
into `answers/<system>/<qid>.md`; `grade.py` reads them all.

## Automated systems

```bash
# Bellwether — needs the server up (bellwether serve); uses the claude CLI login
PYTHONPATH=src .venv-serve/bin/python scripts/eval/run_bellwether.py
PYTHONPATH=src .venv-serve/bin/python scripts/eval/run_bellwether.py --rerender   # after a rendering change

# PaperQA2 over the same evidence — needs ANTHROPIC_API_KEY in ~/.bellwether/.env
.venv-eval/bin/python scripts/eval/run_paperqa.py --build          # once: corpus files + manifest
.venv-eval/bin/python scripts/eval/run_paperqa.py --dry --only T3-01   # index (first run) + one question
.venv-eval/bin/python scripts/eval/run_paperqa.py                  # all questions, skips existing
.venv-eval/bin/python scripts/eval/budget.py                       # spend today / month
```

## Manual systems (paste the questions, save the answers)

For `orx` and `chatgpt` (and `scholarqa` via the public app if no S2 key):

1. Open `questions_resolved.json`; paste each `q` verbatim, one fresh
   conversation per question. Do not add instructions, hints, or follow-ups.
2. Save the full reply as `answers/<system>/<qid>.md` with a first line:
   `<!-- system: chatgpt | model: <what the UI says> | 2026-09-14 -->`
3. Keep the reply's markdown (lists, bold, quotes) — the grader reads
   quoted spans and list items.
4. If the system refuses or says it cannot, save that too; declines are a
   result, not a gap.

orx: `orx up` in WSL (Claude harness; no key). ChatGPT: web search on.

## Grade

```bash
PYTHONPATH=src .venv-serve/bin/python scripts/eval/grade.py             # every system with answers
PYTHONPATH=src .venv-serve/bin/python scripts/eval/grade.py bellwether paperqa
```

Writes `scores/<system>.json` and `scores/summary.md`. Human-blind items
(T1/T2 method quality) are graded afterwards on shuffled, system-stripped
answers; the machine columns are final as written.
