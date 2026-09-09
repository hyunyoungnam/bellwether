"""Which local model may translate an answer — decided on OUR text, not WMT.

The plan is: answers are written in English once, verified in English, and a
local model renders the PROSE into Korean at display time. Quotes and figure
chips are never handed to it. This measures whether a candidate can do that
job without damaging what the answer says.

What it scores, all of it automatic except the last:

  numbers      every figure the source printed must survive, unchanged. This
               is the one that cannot be traded away: a translated share that
               drifts is worse than an English sentence.
  placeholders our masking survives (terms and figures go in as ⟪n⟫ tokens)
  shape        sentence count and length ratio — a dropped clause or an added
               one both show here
  discipline   did it translate, or did it start answering / commenting?
  speed        seconds per segment, on this machine

Run:  .venv/bin/python scripts/mt_bakeoff.py [--models a,b] [--limit N]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("HF_HOME", str(ROOT / ".cache/hf"))

from bellwether import figures  # noqa: E402

SRC = ROOT / "run/mt/en_answers.jsonl"
OUT = ROOT / "run/mt"

CANDIDATES = {
    # id -> (hf repo, how to build the prompt)
    "motif-2.6b": ("Motif-Technologies/Motif-2.6B", "chat"),
    "translategemma-4b": ("google/translategemma-4b-it", "translategemma"),
    "qwen3-4b": ("Qwen/Qwen3-4B-Instruct-2507", "chat"),
}

# ---------------------------------------------------------------- masking
# The production rule, tested here rather than assumed: a figure and a term of
# our own vocabulary go to the model as an opaque token and come back untouched.
_TERMS: list[str] = []


def load_terms() -> list[str]:
    """Names the corpus itself uses — never translated, never inflected."""
    global _TERMS
    if _TERMS:
        return _TERMS
    seen: set[str] = set()
    try:
        ct = json.loads((ROOT / "data/processed/card_terms.json").read_text())
        for corpus in ct.values():
            for rec in corpus.values():
                for k in ("p", "b", "d"):
                    for name in (rec.get(k) or []):
                        if 3 < len(name) < 40:
                            seen.add(name)
    except OSError:
        pass
    _TERMS = sorted(seen, key=len, reverse=True)[:4000]
    return _TERMS


def mask(text: str) -> tuple[str, list[str]]:
    keep: list[str] = []

    def put(m: re.Match) -> str:
        keep.append(m.group(0))
        return f"⟪{len(keep)-1}⟫"

    # order matters: the longest things first, or a term eats a number
    out = re.sub(r"`[^`]+`|\*\*[^*]+\*\*", put, text)
    names = [t for t in load_terms() if t.lower() in out.lower()][:40]
    for n in sorted(names, key=len, reverse=True):
        out = re.sub(re.escape(n), put, out, flags=re.I)
    out = re.sub(figures._NUM, put, out)
    return out, keep


def unmask(text: str, keep: list[str]) -> str:
    def take(m: re.Match) -> str:
        i = int(m.group(1))
        return keep[i] if i < len(keep) else m.group(0)
    return re.sub(r"⟪\s*(\d+)\s*⟫", take, text)


# ---------------------------------------------------------------- scoring
_SENT = re.compile(r"[.!?。][\s\n]|\n")


def score(src: str, out: str, keep: list[str], masked_out: str) -> dict:
    s_nums = [m for m, _ in figures.claimed(src)]
    o_nums = [m for m, _ in figures.claimed(out)]
    kept = len(re.findall(r"⟪\s*\d+\s*⟫", masked_out))
    hangul = len(re.findall(r"[가-힣]", out))
    latin = len(re.findall(r"[A-Za-z]", out))
    return {
        "numbers_src": len(s_nums),
        "numbers_kept": sum(1 for n in set(s_nums) if n in o_nums),
        "numbers_lost": sorted(set(s_nums) - set(o_nums)),
        "ph_src": len(keep),
        "ph_kept": kept,
        "sent_src": len(_SENT.findall(src)) + 1,
        "sent_out": len(_SENT.findall(out)) + 1,
        "len_ratio": round(len(out) / max(len(src), 1), 2),
        "korean": round(hangul / max(hangul + latin, 1), 2),
    }


# ---------------------------------------------------------------- models
def build_prompt(kind: str, tok, text: str):
    if kind == "translategemma":
        msg = [{"role": "user", "content": [{
            "type": "text", "source_lang_code": "en", "target_lang_code": "ko",
            "text": text}]}]
        return tok.apply_chat_template(msg, tokenize=False,
                                       add_generation_prompt=True)
    sysmsg = ("You are a translation engine. Translate the user's English text "
              "into Korean. Output ONLY the translation. Keep every ⟪n⟫ token "
              "exactly as it appears, in the same order. Do not answer, "
              "explain, summarise or add anything.")
    msg = [{"role": "system", "content": sysmsg},
           {"role": "user", "content": text}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def run_model(name: str, repo: str, kind: str, items: list[str]) -> list[dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    # no device_map: that path wants `accelerate`, and one .to() is enough here
    model = AutoModelForCausalLM.from_pretrained(
        repo, dtype=torch.bfloat16, trust_remote_code=True).to("cuda").eval()
    load = time.time() - t0
    rows = []
    for text in items:
        masked, keep = mask(text)
        prompt = build_prompt(kind, tok, masked)
        enc = tok(prompt, return_tensors="pt").to(model.device)
        t1 = time.time()
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=768, do_sample=False,
                                 pad_token_id=tok.eos_token_id)
        dt = time.time() - t1
        raw = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        rows.append({"src": text, "masked_out": raw,
                     "out": unmask(raw, keep), "keep": keep, "secs": round(dt, 1)})
    del model
    torch.cuda.empty_cache()
    print(f"  {name}: loaded in {load:.0f}s, "
          f"{sum(r['secs'] for r in rows)/max(len(rows),1):.1f}s/segment", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="motif-2.6b,translategemma-4b")
    ap.add_argument("--limit", type=int, default=24)
    a = ap.parse_args()

    items: list[str] = []
    for line in SRC.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        for s in d["segs"]:
            if s.get("t") == "p" and len(s.get("s", "").strip()) > 80:
                items.append(s["s"].strip())
    items = items[:a.limit]
    print(f"{len(items)} prose segments, "
          f"{sum(len(x) for x in items):,} chars\n")

    report = {}
    for name in a.models.split(","):
        repo, kind = CANDIDATES[name]
        print(f"== {name} ({repo})", flush=True)
        try:
            rows = run_model(name, repo, kind, items)
        except Exception as exc:  # noqa: BLE001
            print(f"  UNAVAILABLE: {type(exc).__name__}: {str(exc)[:200]}\n")
            report[name] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
            continue
        scores = [score(r["src"], r["out"], r["keep"], r["masked_out"]) for r in rows]
        agg = {
            "segments": len(rows),
            "numbers": f"{sum(s['numbers_kept'] for s in scores)}/"
                       f"{sum(s['numbers_src'] for s in scores)}",
            "placeholders": f"{sum(s['ph_kept'] for s in scores)}/"
                            f"{sum(s['ph_src'] for s in scores)}",
            "len_ratio_med": sorted(s["len_ratio"] for s in scores)[len(scores)//2],
            "sent_delta": sum(abs(s["sent_out"] - s["sent_src"]) for s in scores),
            "korean_med": sorted(s["korean"] for s in scores)[len(scores)//2],
            "secs_per_seg": round(sum(r["secs"] for r in rows)/max(len(rows),1), 1),
        }
        report[name] = agg
        (OUT / f"out_{name}.json").write_text(
            json.dumps({"agg": agg, "rows": rows, "scores": scores},
                       ensure_ascii=False, indent=1))
        print("  " + json.dumps(agg, ensure_ascii=False) + "\n", flush=True)

    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
