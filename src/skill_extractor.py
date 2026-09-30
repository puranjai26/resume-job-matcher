"""Skill extraction.

Skills are found with **phrase matching**, not single-token matching, so
``machine learning`` is recognised as one skill and never as ``machine`` plus
``learning``. The matcher is a longest-match-first scan over a token index built
from ``data/skills.csv`` + ``data/aliases.csv``.

Why a hand-written matcher instead of spaCy's ``PhraseMatcher``: it is exact,
dependency-free, runs in a few milliseconds, and keeps the app working on a
plain ``pip install`` with no model download. spaCy is still used elsewhere
(lemmatisation, NER for organisations and dates) when it is available.

Ambiguous one- and two-character skills (``R``, ``C``, ``Go``, ``SAS``) are
marked ``strict`` in the CSV: they only match as standalone tokens with their
original capitalisation, which prevents "go to the C-suite" from becoming two
programming languages.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sections import get_section, split_sections
from .skill_normalizer import SkillVocabulary, get_vocabulary, normalize_surface

# Tokens keep +, # and / so that "c++", "c#" and "ci/cd" survive.
_TOKEN_RE = re.compile(r"[\w+#/]+")
_BAD_NEIGHBOURS = set("&/+.\\-_")


@dataclass(frozen=True)
class SkillMention:
    """One occurrence of a skill in a document."""

    skill: str  # canonical name, e.g. "Machine Learning"
    surface: str  # text as it appeared, e.g. "ML"
    start: int  # character offset in the source text
    end: int
    category: str = "Other"

    @property
    def is_alias(self) -> bool:
        return normalize_surface(self.surface) != normalize_surface(self.skill)


@dataclass
class SkillExtractionResult:
    """All skills found in one document."""

    skills: list[str] = field(default_factory=list)
    mentions: list[SkillMention] = field(default_factory=list)
    evidence: dict[str, str] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def __contains__(self, skill: str) -> bool:
        return skill in self.skills

    def __len__(self) -> int:
        return len(self.skills)

    def by_category(self) -> dict[str, list[str]]:
        grouped: dict[str, list[str]] = {}
        for mention in self.mentions:
            bucket = grouped.setdefault(mention.category, [])
            if mention.skill not in bucket:
                bucket.append(mention.skill)
        return dict(sorted(grouped.items()))

    def surfaces_for(self, skill: str) -> list[str]:
        seen: list[str] = []
        for mention in self.mentions:
            if mention.skill == skill and mention.surface not in seen:
                seen.append(mention.surface)
        return seen


# --------------------------------------------------------------------------- #
# Tokenisation
# --------------------------------------------------------------------------- #


def _tokenize_with_offsets(
    text: str, slash_phrases: frozenset[str]
) -> list[tuple[str, str, int, int]]:
    """Return ``(normalised, original, start, end)`` for each token.

    A token containing ``/`` is split further unless the whole token is a known
    skill (``ci/cd``, ``a/b``), so "Python/Java" yields two tokens.
    """
    tokens: list[tuple[str, str, int, int]] = []
    for match in _TOKEN_RE.finditer(text):
        raw = match.group(0)
        lowered = raw.lower()
        if "/" in lowered and lowered not in slash_phrases:
            offset = match.start()
            for part in re.split(r"(/)", raw):
                if part and part != "/":
                    tokens.append((part.lower(), part, offset, offset + len(part)))
                offset += len(part)
            continue
        tokens.append((lowered, raw, match.start(), match.end()))
    return tokens


def _strict_match_ok(text: str, original_token: str, canonical: str, start: int, end: int) -> bool:
    """Guard for ambiguous short skills such as ``R``, ``C``, ``Go``, ``SAS``."""
    if original_token != canonical and original_token != canonical.upper():
        return False
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    if before.isalnum() or after.isalnum():
        return False
    if before in _BAD_NEIGHBOURS:
        return False
    if after == ".":
        # A sentence-final full stop is fine ("...experience in C."), but an
        # abbreviation such as "C.V." or "B.Tech" is not.
        following = text[end + 1] if end + 1 < len(text) else " "
        return not following.isalnum()
    return after not in _BAD_NEIGHBOURS


def _line_at(text: str, position: int) -> str:
    start = text.rfind("\n", 0, position) + 1
    end = text.find("\n", position)
    if end == -1:
        end = len(text)
    return text[start:end].strip(" \t-*\u2022")[:240]


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


class SkillExtractor:
    """Phrase-matching skill extractor over a :class:`SkillVocabulary`."""

    def __init__(self, vocabulary: SkillVocabulary | None = None) -> None:
        self.vocab = vocabulary or get_vocabulary()
        self._slash_phrases = frozenset(surface for surface in self.vocab.lookup if "/" in surface)

    # -- core matcher ---------------------------------------------------- #

    def find_mentions(self, text: str) -> list[SkillMention]:
        """Longest-match-first scan returning every skill occurrence."""
        if not text or not text.strip():
            return []

        tokens = _tokenize_with_offsets(text, self._slash_phrases)
        index = self.vocab.phrase_index
        max_len = self.vocab.max_phrase_len
        mentions: list[SkillMention] = []

        position = 0
        total = len(tokens)
        while position < total:
            matched = False
            upper = min(max_len, total - position)
            for size in range(upper, 0, -1):
                window = tokens[position : position + size]
                key = tuple(tok[0] for tok in window)
                canonical = index.get(key)
                if canonical is None:
                    continue
                start, end = window[0][2], window[-1][3]
                if self.vocab.is_strict(canonical) and not _strict_match_ok(
                    text, window[0][1], canonical, start, end
                ):
                    continue
                mentions.append(
                    SkillMention(
                        skill=canonical,
                        surface=text[start:end],
                        start=start,
                        end=end,
                        category=self.vocab.category_of(canonical),
                    )
                )
                position += size
                matched = True
                break
            if not matched:
                position += 1
        return mentions

    # -- fuzzy pass on explicit skill lists ------------------------------ #

    def _fuzzy_from_skill_section(self, text: str, threshold: float) -> list[SkillMention]:
        """Catch misspellings inside an explicit *Skills* section.

        Only comma/pipe separated items in a skills block are considered, so a
        typo in prose can never invent a skill.
        """
        block = get_section(text, "skills")
        if not block:
            return []
        found: list[SkillMention] = []
        for item in re.split(r"[,;|\u2022\n]+", block):
            candidate = item.strip(" .:-\u2013()")
            if not (2 < len(candidate) <= 30) or not candidate:
                continue
            if self.vocab.normalize(candidate):
                continue  # exact matcher already has it
            canonical = self.vocab.fuzzy_normalize(candidate, threshold)
            if canonical:
                position = text.find(item)
                found.append(
                    SkillMention(
                        skill=canonical,
                        surface=candidate,
                        start=max(position, 0),
                        end=max(position, 0) + len(item),
                        category=self.vocab.category_of(canonical),
                    )
                )
        return found

    # -- public API ------------------------------------------------------ #

    def extract(
        self,
        text: str,
        *,
        use_fuzzy: bool = True,
        fuzzy_threshold: float = 0.88,
    ) -> SkillExtractionResult:
        """Extract skills from a document.

        Args:
            text: Original (not lowercased) document text.
            use_fuzzy: Also try close-spelling matches inside a Skills section.
            fuzzy_threshold: Similarity cut-off for that fallback.

        Returns:
            A :class:`SkillExtractionResult` with de-duplicated canonical skills,
            every mention, an evidence line per skill, and mention counts.
        """
        mentions = self.find_mentions(text or "")
        if use_fuzzy and text:
            known = {m.skill for m in mentions}
            mentions.extend(
                m
                for m in self._fuzzy_from_skill_section(text, fuzzy_threshold)
                if m.skill not in known
            )

        skills: list[str] = []
        evidence: dict[str, str] = {}
        counts: dict[str, int] = {}
        for mention in mentions:
            counts[mention.skill] = counts.get(mention.skill, 0) + 1
            if mention.skill not in evidence:
                skills.append(mention.skill)
                line = _line_at(text, mention.start)
                evidence[mention.skill] = line or mention.surface

        return SkillExtractionResult(
            skills=sorted(skills),
            mentions=mentions,
            evidence=evidence,
            counts=counts,
        )

    def extract_by_section(self, text: str) -> dict[str, list[str]]:
        """Return ``{section_name: [skills]}`` for a document."""
        return {
            name: self.extract(block, use_fuzzy=False).skills
            for name, block in split_sections(text).items()
            if block.strip()
        }


# --------------------------------------------------------------------------- #
# Module-level convenience API
# --------------------------------------------------------------------------- #

_DEFAULT_EXTRACTOR: SkillExtractor | None = None


def get_extractor() -> SkillExtractor:
    """Return the process-wide extractor (built once)."""
    global _DEFAULT_EXTRACTOR
    if _DEFAULT_EXTRACTOR is None:
        _DEFAULT_EXTRACTOR = SkillExtractor()
    return _DEFAULT_EXTRACTOR


def extract_skills(text: str, **kwargs) -> list[str]:
    """Return the sorted list of canonical skills found in ``text``."""
    return get_extractor().extract(text, **kwargs).skills


def extract_skills_detailed(text: str, **kwargs) -> SkillExtractionResult:
    return get_extractor().extract(text, **kwargs)


__all__ = [
    "SkillExtractionResult",
    "SkillExtractor",
    "SkillMention",
    "extract_skills",
    "extract_skills_detailed",
    "get_extractor",
]
