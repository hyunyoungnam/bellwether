"""A daily dollar cap for every eval run that spends API money.

The comparison harnesses (PaperQA2, ScholarQA, ...) call paid models. This
ledger is the one place their spend is counted, and the one gate every runner
must pass before it starts a question:

    from budget import load_env, assert_room, charge, litellm_callback
    load_env()                 # ~/.bellwether/.env -> os.environ (never logged)
    assert_room()              # raises BudgetExceeded once today's spend >= cap
    ... run one question ...
    charge("claude-sonnet-5", in_tok, out_tok, note="paperqa q07")

Cap: WNAI_EVAL_DAILY_USD (default 10). Ledger: run/eval_spend.jsonl, one
line per charge, local day boundaries. Prices are Anthropic first-party
rates per million tokens (cached 2026-06 from the API reference); a model not
in the table is charged at the Opus rate so an unknown model can only
over-count, never slip under the cap. This is the soft, per-day guard — the
hard backstop is a spend limit on the API workspace itself, set in the
Anthropic Console, which no local bug can bypass.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "run" / "eval_spend.jsonl"
ENV = ROOT / ".env"
CAP = float(os.environ.get("WNAI_EVAL_DAILY_USD", "10"))

# $ per 1M tokens: (input, output, cache write, cache read)
PRICES = {
    "claude-haiku-4-5": (1.0, 5.0, 1.25, 0.10),
    "claude-sonnet-5": (2.0, 10.0, 2.50, 0.20),
    "claude-sonnet-4-6": (3.0, 15.0, 3.75, 0.30),
    "claude-opus-5": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-8": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-7": (5.0, 25.0, 6.25, 0.50),
    "claude-opus-4-6": (5.0, 25.0, 6.25, 0.50),
    "claude-fable-5": (10.0, 50.0, 12.5, 1.0),
    "claude-fable-5-1": (10.0, 50.0, 12.5, 1.0),
}
_UNKNOWN = PRICES["claude-opus-5"]


class BudgetExceeded(RuntimeError):
    pass


def load_env(path: pathlib.Path = ENV) -> None:
    """KEY=VALUE lines into the environment; existing variables win."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _price(model: str):
    m = (model or "").lower()
    for k, p in PRICES.items():
        if k in m:
            return p
    return _UNKNOWN


def cost_of(model: str, in_tok: int, out_tok: int,
            cache_write: int = 0, cache_read: int = 0) -> float:
    pi, po, pw, pr = _price(model)
    return (in_tok * pi + out_tok * po + cache_write * pw + cache_read * pr) / 1e6


def today_spend(day: str | None = None) -> float:
    day = day or _dt.date.today().isoformat()
    if not LEDGER.exists():
        return 0.0
    total = 0.0
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("day") == day:
            total += float(r.get("usd", 0))
    return total


def remaining() -> float:
    return max(0.0, CAP - today_spend())


def assert_room(estimate_usd: float = 0.0) -> None:
    spent = today_spend()
    if spent + estimate_usd >= CAP:
        raise BudgetExceeded(
            f"daily cap ${CAP:.2f} reached (spent ${spent:.2f} today"
            + (f", next call ~${estimate_usd:.2f}" if estimate_usd else "") + ")")


def charge(model: str, in_tok: int = 0, out_tok: int = 0, *, usd: float | None = None,
           cache_write: int = 0, cache_read: int = 0, note: str = "") -> float:
    usd = cost_of(model, in_tok, out_tok, cache_write, cache_read) if usd is None else usd
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": _dt.datetime.now().isoformat(timespec="seconds"),
            "day": _dt.date.today().isoformat(), "model": model,
            "in": in_tok, "out": out_tok, "cache_write": cache_write,
            "cache_read": cache_read, "usd": round(usd, 6), "note": note,
        }) + "\n")
    return usd


def litellm_callback(kwargs, completion_response, start_time, end_time) -> None:
    """litellm.success_callback entry: records what LiteLLM computed, else our
    own price table from the usage block."""
    model = kwargs.get("model") or getattr(completion_response, "model", "") or ""
    usd = kwargs.get("response_cost")
    usage = getattr(completion_response, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    charge(model, in_tok, out_tok, usd=usd, note="litellm")
    if today_spend() >= CAP:
        raise BudgetExceeded(f"daily cap ${CAP:.2f} reached mid-run")


if __name__ == "__main__":
    print(f"cap ${CAP:.2f} | spent today ${today_spend():.4f} | remaining ${remaining():.4f}")
