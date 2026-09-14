"""Dollar caps for every eval run that spends API money — enforced in code,
because the key lives on a team account where no per-key limit can be set.

Two caps, both counted from one ledger, both must hold:

    WNAI_EVAL_DAILY_USD    default 10    (local calendar day)
    WNAI_EVAL_MONTHLY_USD  default 100   (local calendar month)

Three lines of defence, in order:

1. **Key gate.** `load_env()` is the only thing that puts the API key into
   the environment, and it refuses to when either cap is spent. A script
   that never calls it never sees the key; a script that calls it after the
   cap is reached has no key to spend.
2. **Pre-call gate.** `assert_room(estimate)` before each question; the
   LiteLLM process budget (`litellm.max_budget`) is set to what is left, so
   a runaway loop inside one run is cut by the library too.
3. **Post-call ledger.** Every paid response is appended to
   run/eval_spend.jsonl (LiteLLM's own cost figure when it has one, else our
   price table); the callback raises the moment the ledger crosses a cap.

What this cannot see: spend made with the same key by code that bypasses
this module. Keep the key in ~/.bellwether/.env only, load it only through
`load_env()`, and the ledger is the truth. Overshoot is bounded by one
call (an Opus call with 30k input is ~$0.15–0.50).

    from budget import load_env, assert_room, charge, litellm_callback
    load_env()                 # ~/.bellwether/.env -> os.environ, gated
    assert_room()              # raises BudgetExceeded
    ... one question ...
    charge("claude-opus-5", in_tok, out_tok, note="paperqa q07")

Prices: Anthropic first-party per-million rates (cached 2026-06); a model
not in the table is charged at the highest rate so an unknown model can
only over-count. `python scripts/eval/budget.py` prints the status.
"""
from __future__ import annotations

import atexit
import datetime as _dt
import json
import os
import pathlib
import threading

import tlsfix  # noqa: F401  — relax strict X.509 checks before any client is built

ROOT = pathlib.Path(__file__).resolve().parents[2]
LEDGER = ROOT / "run" / "eval_spend.jsonl"
ENV = ROOT / ".env"
DAILY = float(os.environ.get("WNAI_EVAL_DAILY_USD", "10"))
MONTHLY = float(os.environ.get("WNAI_EVAL_MONTHLY_USD", "100"))
KEY_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY")

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
_UNKNOWN = PRICES["claude-fable-5-1"]


class BudgetExceeded(RuntimeError):
    pass


# ----------------------------------------------------------------- ledger
def _rows():
    if not LEDGER.exists():
        return
    for line in LEDGER.read_text(encoding="utf-8").splitlines():
        try:
            yield json.loads(line)
        except ValueError:
            continue


def spend(day: str | None = None, month: str | None = None) -> float:
    today = _dt.date.today()
    day = day or today.isoformat()
    month = month or today.strftime("%Y-%m")
    total = 0.0
    for r in _rows():
        d = str(r.get("day", ""))
        if (day and d == day) or (month and not day and d.startswith(month)):
            total += float(r.get("usd", 0))
    return total


def today_spend() -> float:
    return spend(day=_dt.date.today().isoformat())


def month_spend() -> float:
    return spend(day=None, month=_dt.date.today().strftime("%Y-%m"))


def remaining() -> float:
    u = unflushed()
    return max(0.0, min(DAILY - today_spend() - u, MONTHLY - month_spend() - u))


def status() -> str:
    return (f"day ${today_spend() + unflushed():.4f} / ${DAILY:.2f} | "
            f"month ${month_spend() + unflushed():.4f} / ${MONTHLY:.2f} | "
            f"remaining ${remaining():.4f}")


def assert_room(estimate_usd: float = 0.0) -> None:
    u = unflushed()
    d, m = today_spend() + u, month_spend() + u
    if d + estimate_usd >= DAILY:
        raise BudgetExceeded(f"daily cap ${DAILY:.2f} reached (${d:.2f} today)")
    if m + estimate_usd >= MONTHLY:
        raise BudgetExceeded(f"monthly cap ${MONTHLY:.2f} reached (${m:.2f} this month)")


# ----------------------------------------------------------------- key gate
def load_env(path: pathlib.Path = ENV) -> None:
    """KEY=VALUE lines into the environment; existing variables win. API keys
    are loaded ONLY while both caps have room — past a cap, the key stays on
    disk and the process simply has none."""
    if not path.exists():
        return
    room = remaining() > 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        if k in KEY_VARS and not room:
            continue
        os.environ.setdefault(k, v.strip().strip('"').strip("'"))
    if not room:
        raise BudgetExceeded("cap reached — API keys not loaded: " + status())


# ----------------------------------------------------------------- charging
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


_lock = threading.Lock()
_seen = 0.0      # cost observed by LiteLLM callbacks in this process
_charged = 0.0   # cost this process has written to the ledger


def unflushed() -> float:
    with _lock:
        return max(0.0, _seen - _charged)


def charge(model: str, in_tok: int = 0, out_tok: int = 0, *, usd: float | None = None,
           cache_write: int = 0, cache_read: int = 0, note: str = "") -> float:
    """Append one paid event to the ledger — the synchronous, authoritative
    record. Runners call this per question (PaperQA2 reports session.cost)."""
    global _charged
    usd = cost_of(model, in_tok, out_tok, cache_write, cache_read) if usd is None else float(usd)
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    now = _dt.datetime.now()
    with LEDGER.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "ts": now.isoformat(timespec="seconds"), "day": now.date().isoformat(),
            "model": model, "in": in_tok, "out": out_tok,
            "cache_write": cache_write, "cache_read": cache_read,
            "usd": round(usd, 6), "note": note,
        }) + "\n")
    with _lock:
        _charged += usd
    return usd


def litellm_callback(kwargs, completion_response, start_time, end_time) -> None:
    """litellm.success_callback entry. Runs on LiteLLM's callback thread, so
    it only OBSERVES: LiteLLM's own cost figure when it has one, else our
    table from the usage block. `flush()` (also at exit) writes whatever the
    runner did not record itself — the ledger ends up >= either view."""
    global _seen
    model = kwargs.get("model") or getattr(completion_response, "model", "") or ""
    usd = kwargs.get("response_cost")
    usage = getattr(completion_response, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0)
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0)
    if usd is None:
        usd = cost_of(model, in_tok, out_tok)
    with _lock:
        _seen += float(usd)


def flush(note: str = "unflushed") -> float:
    """Write the observed-but-unrecorded remainder, if any."""
    delta = unflushed()
    if delta > 1e-9:
        charge("litellm", usd=delta, note=note)
    return delta


atexit.register(flush, "unflushed-at-exit")


def arm_litellm() -> None:
    """Point LiteLLM at the ledger and cap this process at what is left —
    LiteLLM raises BudgetExceededError in the call path, which is the real
    mid-run stop (a raise inside the callback thread would not reach the
    caller)."""
    import litellm
    if litellm_callback not in (litellm.success_callback or []):
        litellm.success_callback = list(litellm.success_callback or []) + [litellm_callback]
    litellm.max_budget = remaining()


if __name__ == "__main__":
    print(status())
