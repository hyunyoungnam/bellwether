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
    # id -> (hf repo, prompt shape, backend)
    # vLLM is the deployment target: a model it cannot serve is a model we
    # would have to keep a second runtime alive for (Motif needs transformers
    # 4.x pinned and has neither a vLLM path nor a GGUF).
    "hunyuan-mt-1.5-7b": ("tencent/HY-MT1.5-7B", "chat", "vllm"),
    "qwen3-4b": ("Qwen/Qwen3-4B-Instruct-2507", "chat", "vllm"),
    "kanana-2.1b": ("kakaocorp/kanana-1.5-2.1b-instruct-2505", "chat", "vllm"),
    # KISTI's Korean model, continued-pretrained on science and technology
    # text — the register our answers are actually written in
    "koni-4b": ("KISTI-KONI/KONI-4B-instruct-20250901", "chat", "vllm"),
    # the Korean labs' small models, the register our readers write in
    "midm-2.0-mini": ("K-intelligence/Midm-2.0-Mini-Instruct", "chat", "vllm"),
    "exaone-4.0-1.2b": ("LGAI-EXAONE/EXAONE-4.0-1.2B", "chat", "vllm"),
    "hyperclovax-1.5b": ("naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                         "chat", "vllm"),
    "motif-2.6b": ("Motif-Technologies/Motif-2.6B", "chat", "hf"),
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


NOMASK = os.environ.get("MT_NOMASK") == "1"
USE_GLOSSARY = os.environ.get("MT_GLOSSARY") == "1"
FEWSHOT = os.environ.get("MT_FEWSHOT") == "1"

# The words this corpus keeps using, and the senses a general model gets wrong.
# Measured failures: reasoning->논리, register->등록, branch->분야, and
# "per 1k" read as pages rather than papers.
GLOSSARY = (
    " Use exactly these renderings: reasoning=추론; register=(언어) 사용역, "
    "즉 상투적 표현; branch=가지 (단, 수량은 '가지 10개'처럼 쓰고 '10가지'라고 "
    "쓰지 말 것 — '10가지'는 열 종류로 읽힌다); named=지적; attacked=공략; "
    "gap=공백; share=점유율; per 1k=1,000편당 (편은 논문이며 페이지가 아니다); "
    "paper count=논문 수; corpus=코퍼스; limitation=한계; spotlight=스포트라이트; "
    "card=카드; edition=에디션; bar(임계값)=기준선 ('바'라고 쓰지 말 것); "
    "roughly=대략; branches=가지들; attack rate=공략률.")


def mask(text: str) -> tuple[str, list[str]]:
    """Masking is meant to protect figures — but a model that does not respect
    the token is worse than none, so the harness can also hand over the plain
    text and simply check what came back."""
    if NOMASK:
        return text, []
    keep: list[str] = []

    def put(m: re.Match) -> str:
        keep.append(m.group(0))
        return f"⟪{len(keep)-1}⟫"

    # order matters: the longest things first, or a term eats a number.
    # NOT bold spans: in our answers ** ** wraps whole clauses, and masking
    # those hands the model a sentence with nothing to translate — measured,
    # it swallowed entire segments and flattered every score.
    out = re.sub(r"`[^`]+`", put, text)
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
    # echo: a long verbatim run of the SOURCE inside the translation means the
    # model repeated the input instead of (or after) translating it
    echo = max((len(w2) for w2 in re.findall(r"[A-Za-z][A-Za-z ,.:%()\-]{39,}", out)
                if w2.strip()[:40] in src), default=0)
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
        "echo_chars": echo,
    }


# ---------------------------------------------------------------- models
def build_prompt(kind: str, tok, text: str):
    sysmsg = ("You are a translation engine. Translate the user's English text "
              "into Korean for a machine-learning researcher. Output ONLY the "
              "translation. Do not answer, explain, summarise or add anything. "
              "Keep every figure exactly as written and keep the unit it "
              "belongs to. Keep method, dataset and benchmark names in English."
              + (GLOSSARY if USE_GLOSSARY else ""))
    msg = [{"role": "system", "content": sysmsg}]
    if FEWSHOT:
        # some instruct models answer or annotate however the system prompt is
        # worded; one worked example is the standard cure
        msg += [{"role": "user", "content":
                 "Reasoning papers rose 4.4 -> 6.2 per 1k, but z = 1.15 does "
                 "not clear the bar."},
                {"role": "assistant", "content":
                 "추론 논문은 1,000편당 4.4 → 6.2로 늘었지만, z = 1.15는 "
                 "기준선을 넘지 못합니다."}]
    msg += [{"role": "user", "content": text}]
    return tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)


def run_vllm(name: str, repo: str, kind: str, items: list[str]) -> list[dict]:
    """The deployment path: one engine, batched, chat template from the repo."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    t0 = time.time()
    llm = LLM(model=repo, dtype="bfloat16", gpu_memory_utilization=0.42,
              max_model_len=4096, enforce_eager=False, trust_remote_code=True)
    tok = AutoTokenizer.from_pretrained(repo, trust_remote_code=True)
    load = time.time() - t0
    masked, keeps = zip(*(mask(t) for t in items))
    prompts = [build_prompt(kind, tok, m) for m in masked]
    # Stop where the chat template says a turn ends. KONI-4B ships
    # generation_config with eos_token_id 1 (<eos>) while its template closes
    # turns with <end_of_turn> (106) — so generation runs straight past the
    # boundary and the model starts writing the next turn itself. This is the
    # repo's packaging, not the model's ability; stopping on the token the
    # template itself uses is the honest fix, not a guess at where to cut.
    ends = [tok.eos_token_id]
    for t2 in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>"):
        i2 = tok.convert_tokens_to_ids(t2)
        if isinstance(i2, int) and i2 >= 0 and i2 != tok.unk_token_id:
            ends.append(i2)
    sp = SamplingParams(temperature=0.0, max_tokens=768,
                        stop_token_ids=sorted(set(ends)),
                        stop=["\nuser\n", "\nUser:", "<start_of_turn>"])
    t1 = time.time()
    outs = llm.generate(prompts, sp)
    dt = (time.time() - t1) / max(len(items), 1)
    rows = [{"src": src, "masked_out": o.outputs[0].text.strip(),
             "out": unmask(o.outputs[0].text.strip(), keep), "keep": list(keep),
             "secs": round(dt, 1)}
            for src, keep, o in zip(items, keeps, outs)]
    del llm
    print(f"  {name}: engine up in {load:.0f}s, {dt:.1f}s/segment (batched)", flush=True)
    return rows


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
    ap.add_argument("--models", default="hunyuan-mt-1.5-7b,qwen3-4b,kanana-2.1b")
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
        repo, kind, backend = CANDIDATES[name]
        print(f"== {name} ({repo}, {backend})", flush=True)
        try:
            runner = run_vllm if backend == "vllm" else run_model
            rows = runner(name, repo, kind, items)
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
        "echoed_source": sum(1 for s in scores if s["echo_chars"] >= 40),
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
