"""The level-3 topic vocabulary, shared across every conference and year.

Why this is a module and not a per-run computation
--------------------------------------------------
Deriving topics from each corpus separately makes them incomparable. Measured on
the three ICML years, with the same >=8-paper rule applied to each corpus alone:

    2024   33 topics   (2,622 papers)
    2025   45 topics   (3,323 papers)
    2026   91 topics   (6,590 papers)
    shared by all three: 21

`object detection` "disappears" by 2026 and `code generation` "appears" — neither
is a finding. A fixed paper count is far easier to clear in a corpus 2.5x larger,
so most of that difference is corpus size, not research shifting. Any trend read
off per-corpus vocabularies would be an artifact.

So the pipeline is three separate steps, and the middle one is a decision:

    discover  — propose labels from one or more corpora, ranked by SHARE
    freeze    — write config/taxonomy.json; this is the shared vocabulary
    label     — apply that fixed vocabulary to any corpus, old or new

Only `label` runs for a new conference. The vocabulary does not move underneath
the numbers, which is what makes two years comparable at all.

Shares, not counts
------------------
Everything crossing a corpus boundary is expressed per 1,000 papers. A topic with
40 papers in 2024 and 60 in 2026 did not grow by half; it shrank, because the
conference grew more.

    python3 -m icml.taxonomy discover --min-share 1.2
    python3 -m icml.taxonomy label --venue ICML --year 2024
"""
from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict

from .common import ROOT, dump_json, load_json
from .corpus import Corpus, available
from .normalize_terms import basic

TAXONOMY = ROOT / "config" / "taxonomy.json"

# ---------------------------------------------------------------- rules
# One home for every "this string is not a label" rule. They used to live in
# topics.py and setview.py separately, which is how `null` shipped as a topic
# holding 15 papers: the guard existed in one module and not the other.

TOO_GENERIC = {
    "machine learning", "deep learning", "artificial intelligence", "ai",
    "optimization", "classification", "regression", "prediction", "generation",
    "representation learning", "supervised learning", "unsupervised learning",
    "computer vision", "natural language processing", "nlp", "general",
    "none", "n/a", "other", "everything else", "various", "multiple",
}

# The extractor writes an absent value as the *string* "null" often enough that a
# length check does not catch it — "null" is four characters.
NULL_LITERALS = {"null", "nil", "nan", "na", "unknown", "unspecified",
                 "not specified", "not applicable", "-", "--"}

# "three datasets", "multiple benchmarks": a count wearing a name.
_PLACEHOLDER = re.compile(
    r"^(a|an|the)?\s*"
    r"(\d+|two|three|four|five|six|several|multiple|various|many|numerous|all|some|"
    r"other|another|standard|common|public|open|open-source|large[- ]scale|diverse|"
    r"real[- ]world|real|synthetic|simulated|benchmark|in[- ]domain|out[- ]of[- ]domain|"
    r"downstream|widely[- ]used|popular|different)?\s*"
    r"(datasets?|benchmarks?|data|corpora|corpus|tasks?|environments?|settings?)$")

# A plural category word at the end is a category, not a name: "image datasets".
# Singular is left alone — "LoCoMo benchmark" and "SCOPE dataset" are real names.
_CATEGORY = re.compile(r"\b(benchmarks|datasets|corpora|suites|environments|tasks)$")

# Papers write these into the `domain` field, but they name a MODALITY or a TASK,
# not a place the work is applied: "time series analysis", "image processing".
# Left in, the domain menu stops being a list of application areas — which is the
# only thing that menu is for. They stay in the vocabulary; they just sit on the
# task side of the split.
NOT_A_DOMAIN = {
    "real world", "time series analysis", "image processing", "video analysis",
    "audio processing", "computer vision and natural language", "signal processing",
    "scientific computing", "ai safety", "data analysis", "pattern recognition",
    "natural language understanding", "information retrieval",
}

# The one level here that is AUTHORED rather than derived, and it is marked as
# such. Measured first: topic-centroid cosines all sit between 0.85 and 0.92, so
# the embeddings can neither confirm nor refute a grouping — they have no
# resolving power at this level. Grouping five life-science labels under one
# heading is therefore a curation decision, recorded here so it can be argued
# with. Every MEMBER label is still data-derived; only the family names are ours,
# and they label a menu, never a paper.
DOMAIN_FAMILIES = [
    ("Life sciences & medicine",
     ["healthcare", "biology", "drug discovery", "neuroscience", "genomics",
      "medical imaging", "bioinformatics", "protein design", "single cell analysis",
      "biotechnology", "neuroimaging"]),
    ("Physical sciences & earth",
     ["chemistry", "computational chemistry", "materials science", "physics",
      "weather forecasting", "remote sensing", "astronomy", "quantum computing",
      "energy", "agriculture"]),
    ("Robotics & autonomy",
     ["robotics", "robotic manipulation", "autonomous driving", "autonomous system",
      "autonomous agent", "embodied ai", "navigation"]),
    ("Software & security",
     ["software engineering", "software development", "cybersecurity",
      "code generation", "edge computing"]),
    ("Society & economy",
     ["finance", "education", "recommender system", "law", "social network",
      "e commerce", "social science", "psychology", "operations research",
      "human computer interaction", "transportation", "manufacturing"]),
]
_FAMILY_OF = {lab: name for name, labs in DOMAIN_FAMILIES for lab in labs}

# Different words for the same application area. Folding them is worth ~22 papers
# on 2026 — small, but "meteorology" and "weather forecasting" being two chips is
# simply wrong, and the size of the fix is not the test of whether it is right.
DOMAIN_ALIASES = {
    "meteorology": "weather forecasting", "climate": "weather forecasting",
    "climate science": "weather forecasting",
    "recommendation system": "recommender system", "recommendation": "recommender system",
    "public health": "healthcare", "medicine": "healthcare", "clinical": "healthcare",
    "medical": "healthcare", "healthcare policy": "healthcare",
    "software security": "cybersecurity", "security": "cybersecurity",
    "machine learning security": "cybersecurity",
    "protein engineering": "protein design", "proteomics": "genomics",
    "molecular biology": "biology", "computational biology": "bioinformatics",
    "material science": "materials science",
    "autonomous vehicle": "autonomous driving", "self driving": "autonomous driving",
    "robot": "robotics", "robot learning": "robotics",
    "finance and economics": "finance", "economics": "finance",
    "quantitative finance": "finance",
}

# Application areas the papers name that sit just under the share bar. Promoting
# them adds only ~49 papers, because the real limit is that two thirds of papers
# name no domain at all — but a menu of application areas that omits "physics"
# reads as broken even when the arithmetic says it barely matters.
EXTRA_DOMAINS = {
    "physics", "quantum computing", "human computer interaction", "edge computing",
    "biotechnology", "operations research", "social science", "astronomy",
    "neuroimaging", "agriculture", "energy", "transportation", "manufacturing",
    "law", "psychology",
}

# The task side is far too large to curate label by label (132 and growing), so it
# is grouped by ORDERED keyword rules instead: first match wins, so every label
# lands in exactly one family and the assignment can be read off the label itself.
# Order carries meaning — "multi agent system" must reach Decision before
# Reasoning claims it on "agent", and "causal inference" must reach Theory before
# Efficiency claims it on "inference". Measured: 129 of 131 labels match.
TASK_FAMILIES = [
    ("Trust & safety",
     r"safety|align|robust|adversarial|privacy|fair|interpretab|explainab|hallucinat|"
     r"unlearn|watermark|attack|poison|jailbreak|toxic|copyright|membership inference|"
     r"feature attribution"),
    ("Decision & control",
     r"reinforcement|policy|control|planning|decision|bandit|game|agent|manipulation|"
     r"navigation|exploration|reward|imitation"),
    ("Reasoning & language",
     r"reason|question answering|language model|llm|text generation|in context|"
     r"chain of thought|dialogue|translation|summar|instruction|prompt|next token"),
    ("Generation & synthesis",
     r"generation|synthesis|generative|diffusion|editing|inpaint|super resolution|"
     r"restoration|render|text to|decoding"),
    ("Perception & multimodal",
     r"vision|image|video|visual|segmentation|object detection|recognition|depth|3d|"
     r"point cloud|audio|speech|multimodal|scene"),
    ("Data types & structure",
     r"time series|tabular|graph|node |link |sequence|spatio|geometric|molecul|protein|"
     r"partial differential|symbolic regression|clustering|feature selection"),
    ("Learning settings",
     r"federated|continual|lifelong|transfer|meta |few shot|zero shot|self supervised|"
     r"semi supervised|active learning|curriculum|distribution|domain adaptation|"
     r"test time|imbalanc|multi learning|model merging|anomaly"),
    ("Efficiency & systems",
     r"quantiz|prun|distill|compress|efficien|llm inference|long context|kv cache|memory|"
     r"parameter efficient|fine tuning|peft|lora|sparsit|accelerat|latency|scaling law"),
    ("Theory & estimation",
     r"optimization|convergence|generalization|bound|complexity|causal|inference|"
     r"estimation|uncertainty|conformal|bayesian|statistic|sampling|kernel|matrix|convex|"
     r"gradient|stochastic|classification|regression|selection|inverse|computing"),
]
_TASK_RX = [(n, re.compile(p)) for n, p in TASK_FAMILIES]

# A label that names nothing. Same class as the placeholder dataset names.
JUNK_LABELS = {"real world", "real world application", "real world scenario"}


def task_family(label: str) -> str:
    for name, rx in _TASK_RX:
        if rx.search(label):
            return name
    return "Other"


# Never strip the "s" from -ics/-ss/-us/-sis: robotics, analysis, bias.
_PLURAL = re.compile(r"(?<=[a-z])s$")
_NO_STRIP = re.compile(r"(ics|ss|us|sis|is|as)$")


def canon(label: str) -> str:
    """Light canonicalisation. Deliberately conservative about merging."""
    s = basic(label or "")
    s = re.sub(r"\b(task|tasks|problem|problems|application|applications)\b", "", s).strip()
    s = re.sub(r"\s+", " ", s)
    if len(s) > 3 and not _NO_STRIP.search(s):
        s = _PLURAL.sub("", s)
    return s


def dataset_key(name: str) -> str:
    """Fold key for one benchmark spelled many ways.

    Measured on both ICML corpora: 138 groups, 187 redundant surfaces —
    CIFAR-10/CIFAR10, MATH-500/MATH500/"MATH 500", LIBERO/"LIBERO benchmark",
    AIME24/"AIME \'24". Case, punctuation and the words benchmark/dataset/
    corpus/suite carry no identity. `+` DOES and is kept: HumanEval+ and MBPP+
    are extended sets, not respellings of HumanEval and MBPP.
    """
    s = name.lower()
    s = _re_ds_lead.sub("", s)
    s = _re_ds_tail.sub("", s)
    return _re_ds_keep.sub("", s)


_re_ds_lead = re.compile(r"^(the|a)\s+")
_re_ds_tail = re.compile(r"\s+(benchmark|dataset|corpus|suite)s?$")
_re_ds_keep = re.compile(r"[^a-z0-9+]")


def is_placeholder(name: str) -> bool:
    """True for 'three datasets', 'image datasets', 'real-world data'."""
    s = basic(name or "")
    return bool(_PLACEHOLDER.match(s) or _CATEGORY.search(s))


def is_label(s: str) -> bool:
    """Whether a canonicalised string may become a topic at all."""
    return bool(s) and len(s) >= 4 and s not in TOO_GENERIC and s not in NULL_LITERALS


# ---------------------------------------------------------------- methods
# The `building-block` role is the closest thing the extraction has to "the idea
# this paper stands on": 9,761 mentions across 6,338 distinct names. The names
# arrive unnormalised, so "reinforcement learning" and "reinforcement learning
# (RL)" were two entries of 148 and 97 rather than one of 245.
# An acronym gloss can sit anywhere, not only at the end: "Bradley-Terry (BT)
# model" kept the "(bt)" and became a separate entry from "Bradley-Terry model".
_ACR_ANY = re.compile(r"\s*\(([A-Za-z][A-Za-z0-9\-/]{1,17})\)\s*")
# Authors mix hyphen, en dash, em dash and the LaTeX "--". They are the same
# character to a reader, and were four different keys here.
_DASHES = re.compile(r"[\u2010-\u2015\u2212]|--+")
_ALIAS_FILE = ROOT / "config" / "term_aliases.json"
_TERM_ALIASES: dict | None = None


def _term_aliases() -> dict:
    global _TERM_ALIASES
    if _TERM_ALIASES is None:
        d = load_json(_ALIAS_FILE) if _ALIAS_FILE.exists() else {}
        _TERM_ALIASES = d.get("aliases", d)
    return _TERM_ALIASES


def _fold(s: str) -> str:
    k = basic(_DASHES.sub("-", s or ""))
    k = _term_aliases().get(k, k)
    if len(k) > 3 and not _NO_STRIP.search(k):
        k = _PLURAL.sub("", k)
    return k


def method_key(name: str) -> tuple[str, str | None]:
    """Canonical key for a method name, plus the acronym that aliases to it.

    "Group Relative Policy Optimization (GRPO)" -> ("group relative policy
    optimization", "grpo"), so a paper that wrote only "GRPO" lands in the same
    bucket. The gloss is removed wherever it appears, so "Bradley-Terry (BT)
    model" and "Bradley–Terry model" are one entry rather than three.
    Returns ("", None) for anything too short or placeholder-like.
    """
    n = _DASHES.sub("-", (name or "").strip())
    if not n or is_placeholder(n):
        return "", None
    acr = None
    m = _ACR_ANY.search(n)
    if m:
        rest = (n[:m.start()] + " " + n[m.end():]).strip()
        if rest:                       # a gloss beside a name
            acr = _fold(m.group(1))
            n = rest
        else:                          # the whole name was "(GRPO)"
            n = m.group(1)
    key = _fold(n)
    return (key, acr if acr and acr != key else None)


# Families for the method row. Ordered, first match wins, exactly like the task
# families — and with the same lesson learned the hard way: these patterns are
# WORD-ANCHORED. An unanchored "ode" matched "m-ode-l" and swept LLMs, CLIP and
# Mamba into "Diffusion & flow"; "bert" matched "Hil-bert-Space".
METHOD_FAMILIES = [
    ("Diffusion & flow",
     r"\bdiffusion|\bflow matching|score.?based|\bdenois|rectified flow|\bsde\b|"
     r"langevin|\bddpm\b|\bddim\b|classifier.?free guidance"),
    ("Reinforcement learning",
     r"\breinforcement|\bpolicy\b|policy gradient|policy optimi|actor.?critic|"
     r"q.?learning|\bbandit|monte carlo tree|\bmdp\b|\bppo\b|\bgrpo\b|\bucb\b|"
     r"bellman|imitation"),
    ("LLM adaptation & alignment",
     r"\blora\b|\bpeft\b|fine.?tun|\badapter|prompt tun|instruction tun|\brlhf\b|"
     r"\bdpo\b|preference optimi|reward model|\brft\b|in.?context learning|"
     r"chain.?of.?thought|retrieval.?augmented"),
    ("Transformers & sequence",
     r"\btransformer|attention|\bmamba\b|state space|\brnn\b|\blstm\b|recurrent|"
     r"token|positional|\bkv cache\b|\brope\b"),
    ("Foundation models",
     r"language model|\bllm|\bclip\b|\bvlm|vision.?language|multimodal large|"
     r"foundation model|\bgpt|\bbert\b|llama|qwen|judge"),
    ("Graph & geometry",
     r"\bgraph|\bgnn\b|\bgcn\b|message passing|equivariant|manifold|geometric|"
     r"topolog|optimal transport|gaussian splatting"),
    ("Self-supervised & representation",
     r"contrastive|self.?supervised|\bmasked|autoencoder|\bvae\b|embedding|"
     r"representation learning|clustering|prototyp|\bjepa\b"),
    ("Neural architectures",
     r"neural network|\bmlp\b|\bcnn\b|convolution|\bsnn\b|spiking|neural ode|"
     r"neural operator|\bpinn|architecture|\bresnet\b|\bunet\b|autoregressive|"
     r"generative model|world model|\bgflownet"),
    ("Privacy & robustness",
     r"differential privacy|adversarial train|watermark|\brobust|certif|"
     r"membership inference|unlearn"),
    ("Probabilistic & Bayesian",
     r"bayesian|gaussian process|variational|posterior|\bmcmc\b|sampling|uncertainty|"
     r"conformal|probabilistic|\bsmc\b|normalizing flow|\bmmd\b|wasserstein|"
     r"gaussian mixture|\bgmm\b|\bkernel|density"),
    ("Optimization & training",
     r"gradient|\badam|\bsgd\b|optimizer|regulariz|normaliz|schedul|curriculum|"
     r"convex|newton|momentum|\bsam\b|\bsvd\b|\bpca\b|meta.?learning|\bmuon\b|"
     r"eigen|spectral|matrix factoriz"),
    ("Efficiency",
     r"quantiz|prun|sparsit|compress|mixture.?of.?experts|\bmoe\b|speculative|"
     r"\bcache\b|low.?rank|distill"),
]
_METHOD_RX = [(n, re.compile(p, re.I)) for n, p in METHOD_FAMILIES]


def method_family(display: str, key: str) -> str:
    for name, rx in _METHOD_RX:
        if rx.search(display) or rx.search(key):
            return name
    return "Other"


def building_blocks(facts: dict[int, dict], min_papers: int = 3):
    """Normalised building-block methods -> (rows, papers_per_key, display names).

    An acronym seen on its own is folded into its expansion when some other paper
    spelled the expansion out; otherwise it stands alone, because inventing the
    expansion would be a guess.

    `facts` is a paper-key -> facts mapping, not a Corpus: the method menu spans
    every corpus on screen, and the keys must already be global (see
    `site.gid_map`) or two conferences would collide in `ids`.
    """
    ids: dict[str, set[int]] = defaultdict(set)
    surface: dict[str, Counter] = defaultdict(Counter)
    acr_to_full: dict[str, str] = {}
    for eid, f in facts.items():
        for m in f.get("methods") or []:
            if m.get("role") != "building-block":
                continue
            key, acr = method_key(m["name"])
            if not key or len(key) < 3:
                continue
            if acr and len(acr) >= 2:
                acr_to_full[acr] = key
            ids[key].add(eid)
            surface[key][m["name"].strip()] += 1
    for acr, full in acr_to_full.items():
        if acr in ids and acr != full:
            ids[full] |= ids.pop(acr)
            surface[full].update(surface.pop(acr, Counter()))
    rows = sorted(((len(v), k) for k, v in ids.items() if len(v) >= min_papers),
                  reverse=True)
    disp = {k: c.most_common(1)[0][0] for k, c in surface.items() if c}

    # A method that specialises another rolls under it — but only if the parent is
    # at least as large, so a rare bare word ("diffusion") cannot swallow the
    # common specific one ("diffusion model", 153 papers).
    size = {k: n for n, k in rows}
    parent: dict[str, str] = {}
    for a in size:
        best = None
        for b in size:
            if a == b or size[b] < size[a]:
                continue
            if a.endswith(" " + b) or a.endswith("-" + b) or a.startswith(b + " "):
                if best is None or len(b) > len(best):
                    best = b
        if best:
            parent[a] = best
    return rows, ids, disp, parent


# ---------------------------------------------------------------- membership
def declared_split(corpus: Corpus) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """Membership kept apart by which field produced it: `domain` or `tasks`.

    This is the only first level the data actually supports. Topic-to-topic
    containment is zero; grouping by the author taxonomy puts 76 of 156 topics
    under "Deep Learning"; clustering the topic centroids produced groups holding
    both "reasoning" and "software engineering". But the extractor already
    distinguishes WHERE a paper is applied from WHAT it does, and that split is
    clean: healthcare / robotics / genomics on one side, image generation /
    planning / forecasting on the other.
    """
    dom: dict[str, set[int]] = defaultdict(set)
    tsk: dict[str, set[int]] = defaultdict(set)
    for eid, f in corpus.read_facts().items():
        if f.get("domain"):
            lab = DOMAIN_ALIASES.get(canon(f["domain"]), canon(f["domain"]))
            if is_label(lab):
                dom[lab].add(eid)
        for t in f.get("tasks") or []:
            lab = canon(t["name"])
            if is_label(lab):
                tsk[lab].add(eid)
    return dom, tsk


def declared(corpus: Corpus) -> dict[str, set[int]]:
    """label -> the papers whose OWN extracted fields put them there.

    Only `domain` and `tasks` count. Method and dataset names are what a paper
    uses, not what it is about, and letting them vote turns the vocabulary into a
    list of popular techniques.
    """
    members: dict[str, set[int]] = defaultdict(set)
    for eid, f in corpus.read_facts().items():
        cands = []
        if f.get("domain"):
            lab = canon(f["domain"])
            cands.append(DOMAIN_ALIASES.get(lab, lab))
        for t in f.get("tasks") or []:
            cands.append(canon(t["name"]))
        for lab in cands:
            if is_label(lab):
                members[lab].add(eid)
    return members


def declared_with_folded(corpus: Corpus, vocab: set[str]) -> tuple[dict[str, set[int]], dict[str, set[int]]]:
    """(exact members, folded members) per label.

    Folded (added 2026-09-01): a task that CONTAINS a label at a word boundary
    — "efficient llm inference", "long-context llm inference" — is that label
    said NARROWER, and joins the via/kind-2 class. Without this, "llm
    inference" held 6 papers while dozens sat one adjective away (measured:
    5,743 foldable mentions corpus-wide). Labels under 8 chars stay exact-only:
    "llm" is contained in half the vocabulary.
    """
    members: dict[str, set[int]] = defaultdict(set)
    folded: dict[str, set[int]] = defaultdict(set)
    fold_labels = [lb for lb in vocab if len(lb) >= 8]
    # phrase aliases the containment fold cannot see: the contained phrase on
    # the left names the SAME task as the vocabulary label on the right
    fold_alias = {"language generation": "text generation",
                  "language modelling": "language modeling"}
    fold_alias = {k2: v2 for k2, v2 in fold_alias.items() if v2 in vocab}
    for eid, f in corpus.read_facts().items():
        cands = []
        if f.get("domain"):
            lab = canon(f["domain"])
            cands.append(DOMAIN_ALIASES.get(lab, lab))
        for t in f.get("tasks") or []:
            cands.append(canon(t["name"]))
        for lab in cands:
            if is_label(lab):
                members[lab].add(eid)
            if lab in vocab:
                continue           # exact vocabulary hit needs no folding
            pad = " " + lab + " "
            for lb in fold_labels:
                if " " + lb + " " in pad:
                    folded[lb].add(eid)
            for ph, lb in fold_alias.items():
                if " " + ph + " " in pad:
                    folded[lb].add(eid)
    for lb in folded:
        folded[lb] -= members.get(lb, set())
    return members, folded


def share(n: int, total: int) -> float:
    """Papers per 1,000 — the only count comparable across corpus sizes."""
    return 1000.0 * n / max(total, 1)


# ---------------------------------------------------------------- discover
def find_parent(label: str, vocab) -> str | None:
    """The longest vocabulary label this one specialises, or None.

    Purely lexical and therefore checkable: "medical reasoning" ends with
    "reasoning", so it is a kind of reasoning. Measured on ICML 2026, 919 tail
    labels specialise a vocabulary label and doing this rolls 911 more papers into
    a topic — coverage 30% -> 44% without inventing a single label. It is also the
    ONLY hierarchy the data supports: no topic pair shows 80% paper-set
    containment, so a parent/child tree over the vocabulary itself would be made
    up, not found.
    """
    best = None
    for v in vocab:
        if label == v:
            return None
        if label.endswith(" " + v) or label.endswith("-" + v) or label.startswith(v + " "):
            if best is None or len(v) > len(best):
                best = v
    return best


def discover(corpora: list[Corpus], min_share: float, min_papers: int) -> dict:
    """Propose a vocabulary from several corpora at once.

    A label qualifies on its share in ANY corpus, not on its total. A topic that
    is 2% of a small older conference belongs in the vocabulary even if the newest
    and largest corpus has moved on — otherwise the taxonomy can only ever
    describe the present, and "declined" becomes indistinguishable from "never
    existed".
    """
    per: dict[str, dict[str, float]] = defaultdict(dict)
    counts: dict[str, dict[str, int]] = defaultdict(dict)
    totals = {}
    for c in corpora:
        m = declared(c)
        totals[c.key] = len(c.read_papers(with_abstract=True))
        for lab, ids in m.items():
            per[lab][c.key] = share(len(ids), totals[c.key])
            counts[lab][c.key] = len(ids)

    topics = []
    for lab, shares in per.items():
        best = max(shares.values())
        peak = max(counts[lab].values())
        if (best >= min_share and peak >= min_papers) or lab in EXTRA_DOMAINS:
            topics.append({
                "label": lab,
                "peak_share": round(best, 2),
                "seen_in": sorted(counts[lab]),
                "counts": counts[lab],
            })
    topics.sort(key=lambda t: -t["peak_share"])

    # Everything that did not clear the bar gets one more chance: if it is a
    # lexical specialisation of a label that did, it becomes that label's child.
    vocab = {t["label"] for t in topics}
    children: dict[str, list[str]] = defaultdict(list)
    orphans = 0
    for lab in per:
        if lab in vocab:
            continue
        parent = find_parent(lab, vocab)
        if parent:
            children[parent].append(lab)
        else:
            orphans += 1
    # A vocabulary label can also specialise another one ("visual question
    # answering" under "question answering"): 13 parents, 25 children. Those stay
    # selectable in their own right — they are real topics — but the parent's
    # count includes them, so picking "reasoning" does not silently exclude the
    # six kinds of reasoning that cleared the bar separately.
    nested = {}
    for a in vocab:
        par = find_parent(a, vocab - {a})
        if par:
            nested[a] = par
            children[par].append(a)
    for t in topics:
        t["children"] = sorted(children.get(t["label"], ()))
        if t["label"] in nested:
            t["nested_under"] = nested[t["label"]]
    return {
        "version": "",                       # stamped by the caller
        "unit": "papers per 1,000",
        "min_share": min_share,
        "min_papers": min_papers,
        "derived_from": [{"corpus": c.key,
                          "papers": totals[c.key]} for c in corpora],
        "child_labels": sum(len(t["children"]) for t in topics),
        "nested_labels": len(nested),
        "labels_with_no_parent": orphans,
        "topics": topics,
    }


def label_corpus(corpus: Corpus, tax: dict) -> dict:
    """Apply a FIXED vocabulary to one corpus. No new labels are created here."""
    vocab = {t["label"] for t in tax["topics"]}
    m, m_fold = declared_with_folded(corpus, vocab)
    total = len(corpus.read_papers(with_abstract=True))
    out = []
    for t in tax["topics"]:
        direct = set(m.get(t["label"], ()))
        via: set[int] = set()
        for kid in t.get("children", ()):
            via |= m.get(kid, set())
        via |= m_fold.get(t["label"], set())
        via -= direct
        # Direct and rolled-up are reported apart, like explicit and expanded
        # membership: "said so" and "said something narrower" are not the same
        # claim, and merging them hides which one a count rests on.
        out.append({
            "label": t["label"],
            "n_explicit": len(direct),
            "n_via_child": len(via),
            "share": round(share(len(direct) + len(via), total), 2),
            "explicit": sorted(direct),
            "via_child": sorted(via),
            "children_seen": sorted(k for k in t.get("children", ()) if m.get(k)),
            "nested_under": t.get("nested_under"),
        })
    # facet: 1 = the papers name it as a domain, 0 = as a task, 2 = both equally
    dom, tsk = declared_split(corpus)
    for t, row in zip(tax["topics"], out):
        fam = [t["label"]] + list(t.get("children", ()))
        d = len(set().union(*[dom.get(x, set()) for x in fam]) or set())
        k = len(set().union(*[tsk.get(x, set()) for x in fam]) or set())
        lab = t["label"]
        if lab in NOT_A_DOMAIN:
            # a task wearing a domain's clothes — it still needs a task family
            row["facet"], row["family"] = 0, task_family(lab)
            row["junk"] = lab in JUNK_LABELS
            continue
        if lab in EXTRA_DOMAINS:
            row["facet"] = 1
            row["family"] = _FAMILY_OF.get(lab) or "Other"
            row["junk"] = False
            continue
        row["facet"] = 2 if d + k == 0 else (1 if d / (d + k) >= 0.6
                                             else (0 if d / (d + k) <= 0.4 else 2))
        # A curated family name, or "Other" for a domain nobody has grouped yet —
        # shown rather than hidden, so the gap in the curation is visible.
        row["family"] = (_FAMILY_OF.get(lab) or "Other") if row["facet"] == 1 \
            else task_family(lab)
        row["junk"] = lab in JUNK_LABELS

    child_of = {k: t["label"] for t in tax["topics"] for k in t.get("children", ())}
    unmatched = sum(1 for lab in m if lab not in vocab and lab not in child_of)
    return {
        "corpus": corpus.key, "venue": corpus.venue, "year": corpus.year,
        "papers": total,
        "taxonomy_version": tax.get("version", ""),
        "topics": out,
        "labels_seen_outside_taxonomy": unmatched,
    }


# ---------------------------------------------------------------- cli
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("discover", help="propose a shared vocabulary")
    d.add_argument("--min-share", type=float, default=1.2,
                   help="papers per 1,000 in at least one corpus")
    d.add_argument("--min-papers", type=int, default=5,
                   help="absolute floor, so a tiny corpus cannot invent a topic")
    d.add_argument("--stamp", default="", help="version string to record")

    l = sub.add_parser("label", help="apply the frozen vocabulary to a corpus")
    l.add_argument("--venue", default="ICML")
    l.add_argument("--year", type=int, default=None)
    l.add_argument("--all", action="store_true", help="every corpus that exists")

    args = ap.parse_args()
    corpora = available()
    if not corpora:
        raise SystemExit("no corpus has both papers and extracted facts")

    if args.cmd == "discover":
        tax = discover(corpora, args.min_share, args.min_papers)
        tax["version"] = args.stamp or "unstamped"
        dump_json(TAXONOMY, tax)
        print(f"wrote {TAXONOMY.relative_to(ROOT)}: {len(tax['topics'])} topics "
              f"from {len(corpora)} corpora")
        for t in tax["topics"][:12]:
            per = "  ".join(f"{k.split('-')[1]}:{v}" for k, v in sorted(t["counts"].items()))
            print(f"   {t['label']:34s} peak {t['peak_share']:5.2f}/1k   {per}")
        return 0

    tax = load_json(TAXONOMY)
    targets = corpora if args.all else [Corpus(args.venue, args.year or corpora[-1].year)]
    for c in targets:
        doc = label_corpus(c, tax)
        out = c.topics
        dump_json(out, doc)
        hit = sum(1 for t in doc["topics"] if t["n_explicit"] or t["n_via_child"])
        cov = len(set().union(*(set(t["explicit"]) | set(t["via_child"])
                                for t in doc["topics"])) or ())
        print(f"{c.key}: {hit}/{len(doc['topics'])} topics present · "
              f"{cov:,}/{doc['papers']:,} papers covered ({cov/doc['papers']:.0%}) · "
              f"{doc['labels_seen_outside_taxonomy']:,} labels outside -> {out.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
