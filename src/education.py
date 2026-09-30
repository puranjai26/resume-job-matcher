"""Education extraction, normalisation and matching.

Degrees are normalised onto an ordered level scale so that "B.Tech CSE" and
"Bachelor of Engineering in Computer Science" compare correctly against a
requirement of "Bachelor's degree in Computer Science or related field".

Level scale::

    0  none / unknown
    1  diploma or certificate
    2  bachelor's
    3  master's
    4  doctorate

Fields are normalised to a small controlled set (``computer science``,
``information technology``, ``data science``, ...) with a relatedness map, so a
job asking for Computer Science accepts an IT graduate at a small discount.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sections import get_section

LEVEL_NAMES: dict[int, str] = {
    0: "Not specified",
    1: "Diploma / Certificate",
    2: "Bachelor's",
    3: "Master's",
    4: "Doctorate",
}

# (regex, level, canonical label). Order matters: longest/most specific first.
_DEGREE_PATTERNS: tuple[tuple[str, int, str], ...] = (
    (r"\bph\.?\s?d\b|\bdoctorate\b|\bdoctoral\b|\bd\.?phil\b", 4, "PhD"),
    (r"\bm\.?\s?tech\b|\bmaster of technology\b", 3, "M.Tech"),
    (r"\bm\.?\s?e\b(?![a-z])|\bmaster of engineering\b", 3, "M.E."),
    (r"\bm\.?\s?c\.?\s?a\b|\bmaster of computer applications?\b", 3, "MCA"),
    (r"\bm\.?\s?sc\b|\bmaster of science\b", 3, "M.Sc"),
    (r"\bm\.?\s?b\.?\s?a\b|\bmaster of business administration\b", 3, "MBA"),
    (
        r"\bm\.?\s?s\b(?![a-z])(?!\s?office)|\bmaster'?s?\s+(?:degree|of|in)\b"
        r"|\bmasters?\b|\bpost[- ]?graduat\w*\b",
        3,
        "Master's",
    ),
    (r"\bb\.?\s?tech\b|\bbachelor of technology\b", 2, "B.Tech"),
    (r"\bb\.?\s?e\b(?![a-z])|\bbachelor of engineering\b", 2, "B.E."),
    (r"\bb\.?\s?c\.?\s?a\b|\bbachelor of computer applications?\b", 2, "BCA"),
    (r"\bb\.?\s?sc\b|\bbachelor of science\b", 2, "B.Sc"),
    (r"\bb\.?\s?com\b|\bbachelor of commerce\b", 2, "B.Com"),
    (r"\bb\.?\s?a\b(?![a-z])|\bbachelor of arts\b", 2, "B.A."),
    (
        r"\bbachelor'?s?\s*(?:degree|of|in)?\b|\bbachelors?\b"
        r"|\bundergraduate degree\b|\bgraduation\b|\bunder[- ]?graduate\b",
        2,
        "Bachelor's",
    ),
    (r"\bdiploma\b|\bpolytechnic\b|\bassociate degree\b|\bcertificate program\b", 1, "Diploma"),
)

_COMPILED_DEGREES = tuple(
    (re.compile(pattern, re.IGNORECASE), level, label) for pattern, level, label in _DEGREE_PATTERNS
)

_FIELD_PATTERNS: tuple[tuple[str, str], ...] = (
    (
        r"computer science(?: (?:and )?engineering)?|\bcse\b|\bc\.?s\.?e\b|comp\.? sci",
        "Computer Science",
    ),
    (r"information technology|\bi\.?t\b(?![a-z])|information systems", "Information Technology"),
    (r"data science|data analytics|\bdata\b(?= (?:science|analytics))", "Data Science"),
    (
        r"artificial intelligence|machine learning|\bai\s?(?:&|and)?\s?ml\b",
        "Artificial Intelligence",
    ),
    (r"software engineering|software development", "Software Engineering"),
    (r"electronics(?: and communication)?|\bece\b|electrical", "Electronics"),
    (r"mechanical engineering|\bmech\b", "Mechanical Engineering"),
    (r"civil engineering", "Civil Engineering"),
    (r"mathematics|statistics|applied math", "Mathematics / Statistics"),
    (r"business administration|management studies|\bcommerce\b|\bfinance\b", "Business"),
    (r"cyber ?security|information security", "Cybersecurity"),
)

_COMPILED_FIELDS = tuple(
    (re.compile(pattern, re.IGNORECASE), label) for pattern, label in _FIELD_PATTERNS
)

#: Fields treated as acceptable substitutes for one another (partial credit).
_RELATED_FIELDS: dict[str, set[str]] = {
    "Computer Science": {
        "Information Technology",
        "Software Engineering",
        "Data Science",
        "Artificial Intelligence",
        "Electronics",
        "Mathematics / Statistics",
        "Cybersecurity",
    },
    "Information Technology": {
        "Computer Science",
        "Software Engineering",
        "Data Science",
        "Cybersecurity",
    },
    "Data Science": {
        "Computer Science",
        "Mathematics / Statistics",
        "Artificial Intelligence",
        "Information Technology",
    },
    "Artificial Intelligence": {
        "Computer Science",
        "Data Science",
        "Mathematics / Statistics",
    },
    "Software Engineering": {"Computer Science", "Information Technology"},
    "Cybersecurity": {"Computer Science", "Information Technology"},
    "Mathematics / Statistics": {"Data Science", "Computer Science"},
    "Electronics": {"Computer Science", "Information Technology"},
}

_RELATED_FIELD_PHRASE = re.compile(
    r"\bor (?:a )?(?:related|similar|equivalent|relevant)(?: field| discipline| degree)?\b"
    r"|\bequivalent (?:experience|qualification)\b|\bany (?:degree|graduate|discipline)\b",
    re.IGNORECASE,
)


@dataclass
class EducationInfo:
    """Education parsed from a resume or a job description."""

    level: int = 0
    level_name: str = "Not specified"
    degrees: list[str] = field(default_factory=list)
    fields: list[str] = field(default_factory=list)
    accepts_related: bool = False
    evidence: str | None = None

    @property
    def specified(self) -> bool:
        return self.level > 0 or bool(self.fields)

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_name": self.level_name,
            "degrees": self.degrees,
            "fields": self.fields,
            "accepts_related": self.accepts_related,
            "evidence": self.evidence,
        }

    def describe(self) -> str:
        if not self.specified:
            return "Not explicitly specified"
        parts = [self.level_name] if self.level else []
        if self.fields:
            parts.append(" / ".join(self.fields))
        return " - ".join(parts) if parts else "Not explicitly specified"


@dataclass
class EducationMatch:
    """Comparison of candidate education against the requirement."""

    score: float | None
    status: str
    requirement: EducationInfo
    candidate: EducationInfo

    @property
    def available(self) -> bool:
        return self.score is not None


def _detect_degrees(text: str) -> tuple[int, list[str], str | None]:
    level, labels, evidence = 0, [], None
    for pattern, degree_level, label in _COMPILED_DEGREES:
        match = pattern.search(text)
        if not match:
            continue
        if label not in labels:
            labels.append(label)
        if degree_level > level:
            level = degree_level
            start = max(0, match.start() - 40)
            evidence = " ".join(text[start : match.end() + 60].split())
    return level, labels, evidence


def _detect_fields(text: str) -> list[str]:
    found: list[str] = []
    for pattern, label in _COMPILED_FIELDS:
        if pattern.search(text) and label not in found:
            found.append(label)
    return found


def extract_education(text: str, *, prefer_section: bool = True) -> EducationInfo:
    """Extract education information from a resume or job description."""
    if not text or not text.strip():
        return EducationInfo()

    scope = text
    if prefer_section:
        section = get_section(text, "education")
        if section and len(section) > 10:
            scope = section

    level, degrees, evidence = _detect_degrees(scope)
    fields = _detect_fields(scope)

    # Fall back to the whole document if the section yielded nothing.
    if not level and not fields and scope is not text:
        level, degrees, evidence = _detect_degrees(text)
        fields = _detect_fields(text)

    return EducationInfo(
        level=level,
        level_name=LEVEL_NAMES.get(level, "Not specified"),
        degrees=degrees,
        fields=fields,
        accepts_related=bool(_RELATED_FIELD_PHRASE.search(text)),
        evidence=evidence,
    )


def extract_education_requirement(text: str) -> EducationInfo:
    """Extract the education requirement stated in a job description."""
    info = extract_education(text, prefer_section=True)
    if not info.accepts_related and _RELATED_FIELD_PHRASE.search(text or ""):
        info.accepts_related = True
    return info


def _field_score(required: list[str], candidate: list[str], accepts_related: bool) -> float:
    if not required:
        return 1.0
    if not candidate:
        return 0.5  # unknown field: neither rewarded nor heavily punished
    if set(required) & set(candidate):
        return 1.0
    related = set()
    for name in required:
        related |= _RELATED_FIELDS.get(name, set())
    if set(candidate) & related:
        return 1.0 if accepts_related else 0.75
    return 0.6 if accepts_related else 0.3


def match_education(requirement: EducationInfo, candidate: EducationInfo) -> EducationMatch:
    """Compare candidate education with the requirement.

    Returns ``None`` for the score when either side states nothing, so that the
    component's weight can be redistributed instead of scoring a false zero.
    """
    if not requirement.specified:
        return EducationMatch(
            None, "Job does not state an education requirement", requirement, candidate
        )
    if not candidate.specified:
        return EducationMatch(
            None, "No education information found in the resume", requirement, candidate
        )

    required_level = requirement.level
    candidate_level = candidate.level

    if required_level == 0:
        level_score = 1.0
        level_note = "No specific degree level required"
    elif candidate_level >= required_level:
        level_score = 1.0
        level_note = f"{candidate.level_name} meets the {requirement.level_name} requirement"
    elif candidate_level == 0:
        level_score = 0.4
        level_note = "Degree level could not be identified in the resume"
    else:
        gap = required_level - candidate_level
        level_score = max(0.3, 1.0 - 0.35 * gap)
        level_note = f"{candidate.level_name} is below the required {requirement.level_name}"

    field_score = _field_score(requirement.fields, candidate.fields, requirement.accepts_related)
    if requirement.fields:
        if field_score >= 1.0:
            field_note = "field of study matches"
        elif field_score >= 0.7:
            field_note = "field of study is related"
        else:
            field_note = f"field of study differs from {', '.join(requirement.fields)}"
    else:
        field_note = "no specific field required"

    score = round(0.6 * level_score + 0.4 * field_score, 3)
    return EducationMatch(score, f"{level_note}; {field_note}", requirement, candidate)


__all__ = [
    "EducationInfo",
    "EducationMatch",
    "LEVEL_NAMES",
    "extract_education",
    "extract_education_requirement",
    "match_education",
]
