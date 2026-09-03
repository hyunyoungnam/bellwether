"""Quote verification against the corpus — the one shared implementation.

Both the brief verifier (scripts/verify_brief.py) and the chat loop stamp
agent-cited quotes with this: a quote passes only if its normalized text
appears verbatim in what we hold for that paper (title, abstract, the
verified card sentences, deep-pass sentences). Prose is the agent's and may
be wrong; a PASSED quote provably exists in the paper.
"""
from __future__ import annotations

import re

from .mcp import Store

_NORM = re.compile(r"[^a-z0-9]+")
MIN_QUOTE = 20      # shorter matches are nearly free to satisfy — not evidence


def norm(s: str) -> str:
    return _NORM.sub(" ", (s or "").lower()).strip()


def source_text(store: Store, gid: int) -> str:
    parts = []
    r = store.rec(gid)
    if r:
        parts += [r.get("title") or "", r.get("abstract") or ""]
    sp = store.span_entry(gid)
    if sp:
        n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
        parts += list(n) + [L or "", K or "", R or ""]
        if isinstance(sp[5], list):          # deep-pass sentences, when present
            for arr in sp[5]:
                if isinstance(arr, list):
                    parts += [x for x in arr if isinstance(x, str)]
    return " " + norm(" ".join(parts)) + " "


class Verifier:
    """Caches per-paper source text across many checks."""

    def __init__(self, store: Store | None = None):
        self.store = store or Store()
        self._src: dict[int, str] = {}

    def check(self, gid: int, quote: str) -> bool:
        q = norm(quote)
        if len(q) < MIN_QUOTE:
            return False
        if gid not in self._src:
            try:
                self._src[gid] = source_text(self.store, gid)
            except Exception:  # noqa: BLE001 — a bad gid is just unverified
                self._src[gid] = ""
        return (" " + q + " ") in self._src[gid]
