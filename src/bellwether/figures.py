"""Figures in an answer, recomputed before the reader sees them.

A quote is checked by matching it against the paper. A NUMBER cannot be
matched that way — it was not written by anyone, it was computed. So it is
checked the only honest way available: the tool that produced it is run
again on this machine and every number in the claim must appear in the
result.

This is deliberately mechanical. It decides nothing about whether the
sentence around the number is fair; it decides whether the number survives
recomputation. Three outcomes, and the third is not a failure:

  ok  every number in the claim appears in the recomputed result
  no  at least one does not — it was mistyped or invented
  na  the claim could not be recomputed (a tool outside the list below, or
      an argument that no longer resolves), so nothing is claimed about it

Only deterministic tools are on the list. `search_papers` is NOT: its ranking
and its estimated total come from the search engine, and marking a figure
verified when the check is not reproducible would be worse than not marking
it at all.
"""
from __future__ import annotations

import re

# tool -> the kind of argument it takes
RECOMPUTABLE = {"field_trend": "topic", "gap_scan": "topic",
                "topic_papers": "topic", "field_cards": "topic",
                "get_citations": "gid", "get_paper": "gid"}

# Ids are not figures. A gid is a five-digit number that would make almost any
# claimed count match something, so paper lists never enter the pool.
SKIP_KEYS = {"gid", "gids", "attackers", "namers", "results", "cards",
             "citing", "cited_by", "cites", "neighbors", "id"}

_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def pool(obj, out: list | None = None) -> list[float]:
    """Every figure in a tool result, ids excluded."""
    out = [] if out is None else out
    if isinstance(obj, dict):
        for k, v in obj.items():
            # a numeric KEY is a figure too: by_year is {"2025": 13}, and a
            # claim that names the edition is quoting the result, not inventing
            # (in-process the key is an int; through JSON the agent saw "2025")
            if isinstance(k, int) and not isinstance(k, bool):
                out.append(float(k))
            elif isinstance(k, str) and _NUM.fullmatch(k):
                out.append(float(k))
            if k in SKIP_KEYS:
                # the LIST is not a figure, but HOW MANY is — "28 papers cite
                # this one" is the count of a list this pool otherwise skips
                if isinstance(v, list):
                    out.append(float(len(v)))
                continue
            pool(v, out)
    elif isinstance(obj, list):
        for v in obj:
            pool(v, out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, str):
        for m in _NUM.findall(obj):
            out.append(float(m))
    return out


def claimed(claim: str) -> list[tuple[str, float]]:
    return [(m, float(m)) for m in _NUM.findall(claim)]


def _matches(raw: str, val: float, vals: list[float]) -> bool:
    """A figure matches if some computed value rounds to it at ITS precision.

    The claim's own precision sets the tolerance: "4.4" accepts 4.43, "1.15"
    does not accept 1.2. Rounding is where honest transcription differs from
    invention, and this is the line between them.
    """
    dp = len(raw.split(".")[1]) if "." in raw else 0
    return any(round(x, dp) == round(val, dp) for x in vals)


def recompute(tool: str, arg: str, cache: dict, run=None) -> list[float] | None:
    """The tool's figures, or None when the claim cannot be recomputed."""
    kind = RECOMPUTABLE.get(tool)
    if kind is None:
        return None
    key = (tool, arg)
    if key in cache:
        return cache[key]
    if run is None:                       # imported here: mcp loads the corpus
        from . import mcp
        run = {t["name"]: t["fn"] for t in mcp.TOOLS}
    fn = run.get(tool)
    if fn is None:
        cache[key] = None
        return None
    try:
        if kind == "gid":
            out = fn({"gid": int(arg)})
        else:
            out = fn({"topic": arg})
        vals = None if (isinstance(out, dict) and out.get("error")) else pool(out)
    except Exception:                     # noqa: BLE001 — an unresolvable arg
        vals = None
    cache[key] = vals
    return vals


def check(tool: str, arg: str, claim: str, cache: dict | None = None) -> dict:
    cache = {} if cache is None else cache
    vals = recompute(tool, arg, cache)
    nums = claimed(claim)
    if vals is None:
        return {"state": "na", "tool": tool, "arg": arg, "n": len(nums),
                "why": "not recomputable"}
    if not nums:
        return {"state": "na", "tool": tool, "arg": arg, "n": 0,
                "why": "no figure in the claim"}
    missing = [raw for raw, v in nums if not _matches(raw, v, vals)]
    return {"state": "no" if missing else "ok", "tool": tool, "arg": arg,
            "n": len(nums), "missing": missing}
