"""Korean rendering of an answer's prose, and nothing else.

The answer is written and VERIFIED in English. What a Korean reader sees is
that same answer with its prose put into Korean — the quote chips and the
figure chips are never handed to a translator, so no verdict on this page can
be changed by translation. That is the whole point of doing it this way rather
than asking the agent to answer in Korean: the language becomes a view, the
same conversation reads either way, and verification always runs on one
canonical text.

What runs it: a small local model served by vLLM (measured pick,
K-intelligence/Midm-2.0-Mini-Instruct — 2.3B, MIT, 44/44 figures preserved on
our own text). If no engine answers, the reader gets English. That is a
degradation, not an error.

The guard that matters: a translation is only shown when every figure the
English printed comes back unchanged. A model that drops or rounds a number
loses that segment, and the segment stays English. We cannot check whether the
Korean says the right thing — only that it says the same numbers — so the
English original stays one toggle away.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path

from . import figures

ROOT = Path(__file__).resolve().parents[2]
TERMS = ROOT / "config" / "ko_terms.json"
ENDPOINT = os.environ.get("WNAI_MT_ENDPOINT", "http://127.0.0.1:8002/v1")
MODEL = os.environ.get("WNAI_MT_MODEL", "K-intelligence/Midm-2.0-Mini-Instruct")
TIMEOUT = float(os.environ.get("WNAI_MT_TIMEOUT", "90"))

_SYSTEM: str | None = None


def system_prompt() -> str:
    """The glossary is a repo artifact, not a string in this file.

    `derived_from_ui` are the words the interface already shows — prose that
    renders them differently contradicts the tree drawn beside it. `authored`
    is our own decision, and reads as one.
    """
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
        "You are a translation engine. Translate the user's English text into "
        "Korean for a machine-learning researcher. Output ONLY the "
        "translation. Do not answer, explain, summarise or add anything. Keep "
        "every figure exactly as written and keep the unit it belongs to. Keep "
        "markdown exactly as it is. Use exactly these renderings: "
        f"{gloss}. {rules}")
    return _SYSTEM


def _post(path: str, body: dict) -> dict | None:
    req = urllib.request.Request(
        ENDPOINT.rstrip("/") + path, data=json.dumps(body).encode(),
        method="POST", headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as fh:
            return json.loads(fh.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError, TimeoutError):
        return None


def up() -> bool:
    try:
        with urllib.request.urlopen(ENDPOINT.rstrip("/") + "/models",
                                    timeout=3) as fh:
            return fh.status == 200
    except Exception:  # noqa: BLE001
        return False


def _one(text: str) -> str | None:
    out = _post("/chat/completions", {
        "model": MODEL, "temperature": 0, "max_tokens": 768,
        "messages": [{"role": "system", "content": system_prompt()},
                     {"role": "user", "content": text}]})
    if not out:
        return None
    try:
        return (out["choices"][0]["message"]["content"] or "").strip()
    except (KeyError, IndexError):
        return None


def _figures_survive(src: str, out: str) -> bool:
    """Every number the English printed must come back, by value.

    One expansion is legitimate and common: "per 1k" written out as
    "1,000편당". A value missing from the Korean is forgiven only when the
    source printed it with a k-suffix and the expanded value is present."""
    import re
    got = [v for _, v in figures.claimed(out)]
    for raw, v in figures.claimed(src):
        if v in got:
            continue
        if re.search(re.escape(raw) + r"\s*[kK]\b", src) and v * 1000 in got:
            continue
        return False
    return True


def prose(texts: list[str]) -> list[str | None]:
    """Translate each prose segment; None where it must stay English."""
    out: list[str | None] = []
    for t in texts:
        if not t.strip():
            out.append(None)
            continue
        got = _one(t)
        # a model that loses a figure loses the segment: the reader sees the
        # English sentence rather than a Korean one with a different number
        out.append(got if got and _figures_survive(t, got) else None)
    return out


def turn(segs: list[dict]) -> list[dict] | None:
    """A turn's segments with prose in Korean, quotes and figures untouched."""
    idx = [i for i, s in enumerate(segs)
           if s.get("t") == "p" and s.get("s", "").strip()]
    if not idx:
        return None
    got = prose([segs[i]["s"] for i in idx])
    if not any(got):
        return None
    ko = [dict(s) for s in segs]
    kept = 0
    for i, g in zip(idx, got):
        if g:
            ko[i]["s"] = g
            kept += 1
    ko_meta = {"t": "meta", "translated": kept, "of": len(idx),
               "by": MODEL}
    return ko + [ko_meta]
