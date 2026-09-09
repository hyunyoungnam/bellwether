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

# 3,324 is one number. The comma-group form must be tried first, or the scan
# reports "324" as an unexplained figure — measured on a real answer.
_NUM = re.compile(r"-?\d{1,3}(?:,\d{3})+(?:\.\d+)?|-?\d+(?:\.\d+)?")
_YEAR = re.compile(r"^(?:19|20)\d\d$")


def _val(raw: str) -> float:
    return float(raw.replace(",", ""))


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
            out.append(_val(m))          # tool notes print "19,071" too
    return out


def claimed(claim: str) -> list[tuple[str, float]]:
    return [(m, _val(m)) for m in _NUM.findall(claim)]


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


# ---------------------------------------------------------------- the auto pass
# Requiring the agent to anchor every figure does not work: measured on three
# real answers, only 38% of the numbers it printed sat inside an anchor, and it
# re-printed bare copies of figures it had just anchored. Verification cannot
# depend on the writer's cooperation — so the server checks EVERY number against
# the tools this turn actually called, anchor or no anchor.
#
# Of the 45 bare numbers in that sample: 84% appeared verbatim in a tool result,
# 7% were a simple ratio or difference of two of them, and the remaining 9% were
# a thousands-separator parsing bug (since fixed), the project's own significance
# threshold, and a gid the agent printed. None was invented.

def derived_set(vals: list[float]) -> set:
    """Figures a reader can get from two computed ones — a share, a difference.

    An agent that writes "31 of 705" and then "4.4%" did arithmetic, not
    invention, and marking that unexplained would be wrong. The combinations
    are fixed and mechanical; nothing here judges whether the arithmetic was
    the RIGHT thing to compute.
    """
    out: set = set()
    uniq = sorted(set(vals))[:120]                   # bounded: pairs are O(n^2)
    for i, a in enumerate(uniq):
        for b in uniq[i + 1:]:
            for c in (a - b, b - a, a + b,
                      (a / b * 100) if b else None, (b / a * 100) if a else None,
                      (a / b) if b else None, (b / a) if a else None):
                if c is None or c < 0 or c > 1e7:
                    continue
                out.add(round(c, 0))
                out.add(round(c, 1))
                out.add(round(c, 2))
    return out


def turn_pool(trail: list, cache: dict | None = None) -> tuple[list[float], list[str]]:
    """Everything the deterministic tools of THIS turn return, recomputed."""
    cache = {} if cache is None else cache
    vals: list[float] = []
    used: list[str] = []
    for step in trail or []:
        name = step.get("name") if isinstance(step, dict) else None
        arg = (step.get("arg") if isinstance(step, dict) else "") or ""
        if name in RECOMPUTABLE and arg:
            got = recompute(name, arg, cache)
            if got:
                vals += got
                used.append(f"{name}:{arg}")
    return vals, used


def scan(text: str, vals: list[float], deriv: set, skip: set) -> list[tuple]:
    """(start, end, raw, verdict) for every figure printed in this text.

    Verdicts: 'ok' (in a tool result), 'derived' (arithmetic of two of them),
    'no' (in neither). Years and the answer's own gids are not figures.
    """
    out = []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        if _YEAR.fullmatch(raw):
            continue
        v = _val(raw)
        if v in skip:
            continue
        dp = len(raw.split(".")[1]) if "." in raw else 0
        if any(round(x, dp) == round(v, dp) for x in vals):
            verdict = "ok"
        elif round(v, dp) in deriv:
            verdict = "derived"
        else:
            verdict = "no"
        out.append((m.start(), m.end(), raw, verdict))
    return out
