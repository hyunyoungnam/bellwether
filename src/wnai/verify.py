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
        self._fields: dict[int, list[tuple[str, str]]] = {}

    def _sources(self, gid: int) -> list[tuple[str, str]]:
        """[(role, normalized text)] — roles are the card's own colours:
        L(pink), K/n(yellow), R(blue), d(deep), a(abstract/title)."""
        if gid not in self._fields:
            out: list[tuple[str, str]] = []
            try:
                r = self.store.rec(gid)
                sp = self.store.span_entry(gid)
                if sp:
                    n, L, K, R = sp[0] or [], sp[1], sp[2], sp[3]
                    out.append(("L", norm(L or "")))
                    out.append(("K", norm(" ".join([K or ""] + list(n)))))
                    out.append(("R", norm(R or "")))
                    if isinstance(sp[5], list):
                        deep = " ".join(x for arr in sp[5] if isinstance(arr, list)
                                        for x in arr if isinstance(x, str))
                        out.append(("d", norm(deep)))
                if r:
                    out.append(("a", norm((r.get("title") or "") + " "
                                          + (r.get("abstract") or ""))))
            except Exception:  # noqa: BLE001 — a bad gid is just unverified
                pass
            self._fields[gid] = [(role, " " + t + " ") for role, t in out if t]
        return self._fields[gid]

    def role(self, gid: int, quote: str) -> str | None:
        """The matched field's role, or None when the quote is not verbatim."""
        q = norm(quote)
        if len(q) < MIN_QUOTE:
            return None
        for role, src in self._sources(gid):
            if (" " + q + " ") in src:
                return role
        return None

    def check(self, gid: int, quote: str) -> bool:
        return self.role(gid, quote) is not None
