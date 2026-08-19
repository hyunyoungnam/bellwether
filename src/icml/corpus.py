"""One conference-year, and where its files live.

Every stage used to hardcode `papers.jsonl` and `facts_abstract.jsonl`, which
silently meant "the focus year". Adding ICML 2024 exposed that: `normalize --year
2024` overwrote the 2026 canonical file. A corpus is now named, and its paths are
derived from the name rather than assumed.

    from .corpus import Corpus
    Corpus("ICML", 2026).papers        # data/processed/papers.jsonl
    Corpus("ICML", 2024).papers        # data/processed/papers_2024.jsonl

The focus year keeps the unsuffixed filenames so nothing downstream breaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .common import FOCUS_YEAR, INTERIM, PROCESSED, read_jsonl

# Which years the product actually uses. 2024 was collected and extracted, then
# dropped: three years made the vocabulary drift for no gain, and the two most
# recent conferences are what a reader is asking about. The files stay on disk —
# this is a scope decision, not a deletion.
ACTIVE_YEARS = (2025, 2026)


@dataclass(frozen=True)
class Corpus:
    venue: str = "ICML"
    year: int = FOCUS_YEAR

    @property
    def key(self) -> str:
        return f"{self.venue.lower()}-{self.year}"

    @property
    def is_focus(self) -> bool:
        return self.year == FOCUS_YEAR

    def _p(self, stem: str, ext: str, base: Path) -> Path:
        return base / (f"{stem}{ext}" if self.is_focus else f"{stem}_{self.year}{ext}")

    @property
    def papers(self) -> Path:
        return self._p("papers", ".jsonl", PROCESSED)

    @property
    def facts(self) -> Path:
        """Abstract pass only. Earlier years have no PDFs, so full text would make
        the focus year look richer purely by being the focus year."""
        return self._p("facts_abstract", ".jsonl", INTERIM)

    @property
    def topics(self) -> Path:
        """Named by corpus key, not year — the labels come from the shared frozen
        vocabulary, so ICML 2025 and a future NeurIPS 2025 are different files."""
        return PROCESSED / f"topics_{self.key}.json"

    def embeddings(self, n: int, model: str) -> Path:
        return PROCESSED / f"emb_{n}_{model.replace('/', '_')}.npy"

    def exists(self) -> bool:
        return self.papers.exists() and self.facts.exists()

    def read_papers(self, with_abstract: bool = False) -> list[dict]:
        rows = list(read_jsonl(self.papers))
        return [p for p in rows if p.get("abstract")] if with_abstract else rows

    def read_facts(self) -> dict[int, dict]:
        return {r["event_id"]: r["facts"] for r in read_jsonl(self.facts) if r.get("ok")}


def available(venue: str = "ICML") -> list[Corpus]:
    """Every corpus with both a papers file and an extraction, oldest first."""
    years = set()
    for p in PROCESSED.glob("papers_*.jsonl"):
        try:
            years.add(int(p.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    years.add(FOCUS_YEAR)
    years &= set(ACTIVE_YEARS)
    return sorted((c for c in (Corpus(venue, y) for y in years) if c.exists()),
                  key=lambda c: c.year)
