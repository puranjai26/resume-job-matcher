"""Skill vocabulary and normalisation.

Loads ``data/skills.csv`` (canonical skills) and ``data/aliases.csv``
(alias -> canonical) and exposes a :class:`SkillVocabulary` that answers one
question: *given a surface form found in text, which canonical skill is it?*

Normalisation is case-insensitive and punctuation-insensitive, so ``NODE JS``,
``Node.js`` and ``nodejs`` all resolve to ``Node.js``.
"""

from __future__ import annotations

import csv
import difflib
import functools
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from .config import ALIASES_CSV, SKILLS_CSV

LOGGER = logging.getLogger(__name__)

_PUNCT_RE = re.compile(r"[^\w+#/]+")


def normalize_surface(text: str) -> str:
    """Reduce a surface form to a comparable key.

    ``"Node.js "`` -> ``"node js"``; ``"scikit-learn"`` -> ``"scikit learn"``;
    ``"C++"`` -> ``"c++"``.
    """
    if not text:
        return ""
    lowered = text.strip().lower()
    lowered = lowered.replace("&", " and ")
    lowered = _PUNCT_RE.sub(" ", lowered)
    return " ".join(lowered.split())


@dataclass(frozen=True)
class SkillEntry:
    """One canonical skill in the vocabulary."""

    name: str
    category: str
    match_mode: str = "default"  # "default" | "strict"

    @property
    def is_strict(self) -> bool:
        return self.match_mode == "strict"


class SkillVocabulary:
    """Canonical skills plus every alias that maps onto them."""

    def __init__(self, entries: list[SkillEntry], aliases: dict[str, str]) -> None:
        self.entries: dict[str, SkillEntry] = {e.name: e for e in entries}

        #: normalised surface form -> canonical skill name
        self.lookup: dict[str, str] = {}
        for entry in entries:
            self.lookup[normalize_surface(entry.name)] = entry.name

        self.alias_targets: dict[str, str] = {}
        for alias, canonical in aliases.items():
            if canonical not in self.entries:
                LOGGER.warning("Alias '%s' targets unknown skill '%s'", alias, canonical)
                continue
            key = normalize_surface(alias)
            if key and key not in self.lookup:
                self.lookup[key] = canonical
            self.alias_targets[key] = canonical

        #: tuple-of-tokens -> canonical, used by the phrase matcher
        self.phrase_index: dict[tuple[str, ...], str] = {
            tuple(surface.split()): canonical
            for surface, canonical in self.lookup.items()
            if surface
        }
        self.max_phrase_len: int = max((len(p) for p in self.phrase_index), default=1)
        self._fuzzy_pool: list[str] = [s for s in self.lookup if len(s) >= 5]

    # ------------------------------------------------------------------ #

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return normalize_surface(name) in self.lookup

    @property
    def skill_names(self) -> list[str]:
        return sorted(self.entries)

    def category_of(self, canonical: str) -> str:
        entry = self.entries.get(canonical)
        return entry.category if entry else "Other"

    def is_strict(self, canonical: str) -> bool:
        entry = self.entries.get(canonical)
        return bool(entry and entry.is_strict)

    def normalize(self, surface: str) -> str | None:
        """Map a surface form to its canonical skill, or ``None`` if unknown."""
        return self.lookup.get(normalize_surface(surface))

    def normalize_many(self, surfaces: list[str]) -> list[str]:
        """Normalise a list, dropping unknowns and preserving first-seen order."""
        out: list[str] = []
        for surface in surfaces:
            canonical = self.normalize(surface)
            if canonical and canonical not in out:
                out.append(canonical)
        return out

    def fuzzy_normalize(self, surface: str, threshold: float = 0.88) -> str | None:
        """Resolve near-miss spellings such as ``"kubernets"`` -> ``Kubernetes``.

        Used only as a fallback after exact lookup fails, and never for very
        short strings where edit distance is meaningless.
        """
        key = normalize_surface(surface)
        if not key:
            return None
        exact = self.lookup.get(key)
        if exact:
            return exact
        if len(key) < 5:
            return None
        matches = difflib.get_close_matches(key, self._fuzzy_pool, n=1, cutoff=threshold)
        return self.lookup[matches[0]] if matches else None


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [{k: (v or "").strip() for k, v in row.items() if k} for row in csv.DictReader(fh)]


def load_skill_vocabulary(
    skills_csv: Path | str = SKILLS_CSV,
    aliases_csv: Path | str = ALIASES_CSV,
) -> SkillVocabulary:
    """Build a :class:`SkillVocabulary` from the CSV data files."""
    skills_path, aliases_path = Path(skills_csv), Path(aliases_csv)
    if not skills_path.exists():
        raise FileNotFoundError(
            f"Skill database not found at {skills_path}. "
            "The data/ directory must be present next to src/."
        )

    entries: list[SkillEntry] = []
    seen: set[str] = set()
    for row in _read_csv(skills_path):
        name = row.get("skill", "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        entries.append(
            SkillEntry(
                name=name,
                category=row.get("category") or "Other",
                match_mode=row.get("match_mode") or "default",
            )
        )

    aliases: dict[str, str] = {}
    if aliases_path.exists():
        for row in _read_csv(aliases_path):
            alias, canonical = row.get("alias", ""), row.get("canonical", "")
            if alias and canonical:
                aliases[alias] = canonical
    else:
        LOGGER.warning("Alias file not found at %s - continuing without aliases.", aliases_path)

    LOGGER.info("Loaded %d skills and %d aliases", len(entries), len(aliases))
    return SkillVocabulary(entries, aliases)


@functools.lru_cache(maxsize=1)
def get_vocabulary() -> SkillVocabulary:
    """Cached default vocabulary (loaded once per process)."""
    return load_skill_vocabulary()


def normalize_skill(surface: str) -> str | None:
    """Module-level convenience wrapper around the default vocabulary."""
    return get_vocabulary().normalize(surface)


def normalize_skills(surfaces: list[str]) -> list[str]:
    return get_vocabulary().normalize_many(surfaces)


__all__ = [
    "SkillEntry",
    "SkillVocabulary",
    "get_vocabulary",
    "load_skill_vocabulary",
    "normalize_skill",
    "normalize_skills",
    "normalize_surface",
]
