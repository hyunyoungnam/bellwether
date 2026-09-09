"""Korean rendering of an answer's prose, and nothing else.

The answer is written and VERIFIED in English. What a Korean reader sees is
that same answer with its prose put into Korean — the quote chips and the
figure chips are never handed to a translator, so no verdict on this page can
be changed by translation. The language is a view; one canonical text is
written, verified and stored.

Who renders it: the reader's own agent CLI (claude -p, the same no-API-key
account that answers questions), on a small model. Local small models were
measured for this job and DISCARDED (owner, 2026-09-09): every candidate at
or under 4B garbled this register beyond a glossary's reach — Midm turned
"states" into 국가들 (countries), fall short into "충분히 짧습니다"; NAVER's
1.5B flipped 2.576 into 0.576. The agent was the only renderer whose output a
Korean reader accepted, and it costs no install weight.

One call renders a whole turn: the prose segments go in as numbered blocks
and come back the same way, so a 22-paragraph answer is one process spawn,
not twenty-two.

The guard that matters is unchanged: a segment is only shown in Korean when
every figure the English printed comes back unchanged (the one forgiven
change is "per 1k" written out as 1,000). We cannot check whether the Korean
says the right thing — only that it says the same numbers — so the English
original stays one toggle away.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import os
from pathlib import Path

from . import figures

ROOT = Path(__file__).resolve().parents[2]
TERMS = ROOT / "config" / "ko_terms.json"
MODEL = os.environ.get("WNAI_MT_AGENT_MODEL", "claude-haiku-4-5-20251001")
TIMEOUT = float(os.environ.get("WNAI_MT_TIMEOUT", "240"))

_SYSTEM: str | None = None
_BLOCK = re.compile(r"⟪(\d+)⟫\s*(.*?)(?=⟪\d+⟫|\Z)", re.S)


def system_prompt() -> str:
    """The glossary is a repo artifact (config/ko_terms.json), not a string
    here. `derived_from_ui` are the words the interface already shows;
    `authored` is our own decision and reads as one."""
    global _SYSTEM
    if _SYSTEM is not None:
        return _SYSTEM
    try:
        d = json.loads(TERMS.read_text())
    except OSError:
        d = {"derived_from_ui": {}, "authored": {}, "rules": []}
    pairs = {**d.get("derived_from_ui", {}), **d.get("authored", {})}
    gloss = "; ".join(f"{k}={v}" for k, v in pairs.items())
    rules = " ".join(d.get("rules", []))
    _SYSTEM = (
        "You are a translation engine. The user's text is a sequence of "
        "blocks, each starting with a marker like ⟪3⟫. The blocks are "
        "consecutive fragments of ONE document, split where citation chips "
        "sit between them — a block may begin or end mid-sentence, with a "
        "bare period or colon. Translate each fragment's English into "
        "natural Korean for a machine-learning researcher, exactly as it "
        "stands: keep its leading and trailing punctuation, and never "
        "import words (such as a paper or benchmark name) from outside the "
        "block. Output ALL blocks, each starting with its own marker, in "
        "the same order, and nothing else. Never merge, reorder, drop or "
        "answer blocks. Keep every figure exactly as written and keep the "
        "unit it belongs to. Keep markdown as it is. "
        f"Use exactly these renderings: {gloss}. {rules}")
    return _SYSTEM


def up() -> bool:
    return shutil.which("claude") is not None


def _figures_survive(src: str, out: str) -> bool:
    """Every number the English printed must come back, by value."""
    got = [v for _, v in figures.claimed(out)]
    for raw, v in figures.claimed(src):
        if v in got:
            continue
        if re.search(re.escape(raw) + r"\s*[kK]\b", src) and v * 1000 in got:
            continue
        return False
    return True


def _ask(blocks: list[str]) -> dict[int, str]:
    msg = "\n".join(f"⟪{i}⟫\n{t}" for i, t in enumerate(blocks))
    try:
        p = subprocess.run(
            ["claude", "-p", msg, "--model", MODEL,
             "--append-system-prompt", system_prompt()],
            capture_output=True, text=True, timeout=TIMEOUT,
            stdin=subprocess.DEVNULL, cwd=ROOT)
    except (subprocess.TimeoutExpired, OSError):
        return {}
    if p.returncode != 0:
        return {}
    return {int(m.group(1)): m.group(2).strip()
            for m in _BLOCK.finditer(p.stdout or "")}


def prose(texts: list[str]) -> list[str | None]:
    """Translate the segments in one agent call; None where English stays."""
    idx = [i for i, t in enumerate(texts) if t.strip()]
    got = _ask([texts[i] for i in idx]) if idx else {}
    out: list[str | None] = [None] * len(texts)
    for k, i in enumerate(idx):
        g = got.get(k)
        # a rendering that loses a figure loses the segment: the reader sees
        # the English sentence rather than a Korean one with a different number
        if g and _figures_survive(texts[i], g):
            out[i] = g
    return out


def turn(segs: list[dict]) -> list[dict] | None:
    """A turn's segments with prose in Korean, quotes and figures untouched."""
    idx = [i for i, s in enumerate(segs)
           if s.get("t") == "p" and s.get("s", "").strip()]
    if not idx:
        return None
    got = prose([segs[i]["s"] if i in idx else "" for i in range(len(segs))])
    if not any(got):
        return None
    ko = [dict(s) for s in segs]
    kept = 0
    for i in idx:
        if got[i]:
            ko[i]["s"] = got[i]
            kept += 1
    return ko + [{"t": "meta", "translated": kept, "of": len(idx),
                  "by": MODEL.split("-2025")[0]}]
