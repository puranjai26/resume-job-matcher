"""Experience extraction and comparison.

Two things are extracted:

* **Requirements** from a job description -- "3+ years", "2 to 4 years",
  "minimum 1 year", "freshers welcome" -- as a structured
  ``{"min_years": float | None, "max_years": float | None}``.
* **Candidate experience** from a resume, from explicit statements
  ("2 years of experience") and from employment date ranges
  ("Jan 2021 - Present").

If nothing can be extracted the value stays ``None``. The module never invents
a number, and the scoring engine treats ``None`` as "unknown" rather than zero.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from .sections import get_section, split_sections

_NUMBER_WORDS = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "twelve": 12,
    "fifteen": 15,
    "half": 0.5,
    "a": 1,
    "an": 1,
}

_MONTHS = {
    "jan": 1,
    "january": 1,
    "feb": 2,
    "february": 2,
    "mar": 3,
    "march": 3,
    "apr": 4,
    "april": 4,
    "may": 5,
    "jun": 6,
    "june": 6,
    "jul": 7,
    "july": 7,
    "aug": 8,
    "august": 8,
    "sep": 9,
    "sept": 9,
    "september": 9,
    "oct": 10,
    "october": 10,
    "nov": 11,
    "november": 11,
    "dec": 12,
    "december": 12,
}

_NUM = r"(?:\d+(?:\.\d+)?|" + "|".join(_NUMBER_WORDS) + r")"
_YEAR_UNIT = r"(?:\+\s*)?(?:years?|yrs?|year's)"

#: Ordered: the first pattern that matches a sentence wins.
_RANGE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("range", re.compile(rf"\b({_NUM})\s*(?:-|\u2013|to|and)\s*({_NUM})\s*{_YEAR_UNIT}\b", re.I)),
    ("plus", re.compile(rf"\b({_NUM})\s*\+\s*{_YEAR_UNIT}\b", re.I)),
    (
        "plus_word",
        re.compile(rf"\b({_NUM})\s*{_YEAR_UNIT}\s*(?:\+|or more|and above|and up)", re.I),
    ),
    (
        "minimum",
        re.compile(
            rf"\b(?:min(?:imum)?|at least|no less than|over|more than)\s*"
            rf"(?:of\s*)?({_NUM})\s*{_YEAR_UNIT}\b",
            re.I,
        ),
    ),
    (
        "maximum",
        re.compile(rf"\b(?:up to|max(?:imum)?|less than|under)\s*({_NUM})\s*{_YEAR_UNIT}\b", re.I),
    ),
    ("exact", re.compile(rf"\b({_NUM})\s*{_YEAR_UNIT}\b", re.I)),
)

_MONTHS_PATTERN = re.compile(rf"\b({_NUM})\s*(?:\+\s*)?months?\b", re.I)

_FRESHER_PATTERN = re.compile(
    r"\b(?:freshers?|fresh graduates?|entry[- ]level|"
    r"no (?:prior |previous )?experience (?:is )?(?:required|necessary|needed)|"
    r"0[-\u2013 ]?(?:to|-)?\s?1\s*years?|graduate trainee|final[- ]year students?)\b",
    re.I,
)

_DATE_RANGE = re.compile(
    r"\b(?:(?P<m1>[A-Za-z]{3,9})[\s.,/-]+)?(?P<y1>(?:19|20)\d{2})\s*"
    r"(?:-|\u2013|\u2014|to|until|through)\s*"
    r"(?:(?P<m2>[A-Za-z]{3,9})[\s.,/-]+)?(?P<y2>(?:19|20)\d{2}|present|current|now|till date|date)",
    re.I,
)

#: Lines describing a degree must not contribute employment date ranges:
#: "B.Tech, Example University, 2017 - 2021" is four years of study, not work.
_ACADEMIC_LINE = re.compile(
    r"\b(?:b\.?\s?tech|m\.?\s?tech|b\.?\s?e\b|m\.?\s?e\b|b\.?\s?sc|m\.?\s?sc|"
    r"b\.?\s?c\.?\s?a|m\.?\s?c\.?\s?a|b\.?\s?com|mba|ph\.?\s?d|bachelor|master|"
    r"diploma|university|college|institute|school|cgpa|gpa|percentage|semester)\b",
    re.IGNORECASE,
)

#: Sections whose date ranges are never employment periods.
_NON_WORK_SECTIONS = frozenset(
    {"education", "certifications", "achievements", "summary", "benefits"}
)

_EXPERIENCE_CONTEXT = re.compile(
    r"\b(?:experience|exp\.?|working|worked|professional|industry|hands[- ]on|"
    r"background|expertise|practice|tenure)\b",
    re.I,
)


@dataclass
class ExperienceRequirement:
    """Structured experience requirement from a job description."""

    min_years: float | None = None
    max_years: float | None = None
    raw_text: str | None = None
    accepts_freshers: bool = False

    @property
    def specified(self) -> bool:
        return self.min_years is not None or self.max_years is not None or self.accepts_freshers

    def to_dict(self) -> dict:
        return {
            "min_years": self.min_years,
            "max_years": self.max_years,
            "raw_text": self.raw_text,
            "accepts_freshers": self.accepts_freshers,
        }

    def describe(self) -> str:
        if not self.specified:
            return "Not explicitly specified"
        if self.accepts_freshers and not self.min_years:
            return "Open to freshers / entry level"
        if self.min_years is not None and self.max_years is not None:
            return f"{_fmt(self.min_years)}-{_fmt(self.max_years)} years"
        if self.min_years is not None:
            return f"{_fmt(self.min_years)}+ years"
        return f"Up to {_fmt(self.max_years)} years"


@dataclass
class CandidateExperience:
    """Experience detected in a resume."""

    total_years: float | None = None
    source: str = "none"  # "statement" | "date_ranges" | "none"
    evidence: str | None = None

    @property
    def specified(self) -> bool:
        return self.total_years is not None

    def to_dict(self) -> dict:
        return {
            "total_years": self.total_years,
            "source": self.source,
            "evidence": self.evidence,
        }

    def describe(self) -> str:
        if self.total_years is None:
            return "Not explicitly specified"
        return f"{_fmt(self.total_years)} years"


@dataclass
class ExperienceMatch:
    """Comparison of candidate experience against the requirement."""

    score: float | None  # 0-1, or None when it cannot be judged
    status: str  # human-readable verdict
    requirement: ExperienceRequirement
    candidate: CandidateExperience

    @property
    def available(self) -> bool:
        return self.score is not None


def _fmt(value: float | None) -> str:
    if value is None:
        return "?"
    return str(int(value)) if float(value).is_integer() else f"{value:.1f}"


def _to_number(token: str) -> float | None:
    token = token.strip().lower()
    if token in _NUMBER_WORDS:
        return float(_NUMBER_WORDS[token])
    try:
        return float(token)
    except ValueError:
        return None


def parse_experience_requirement(text: str) -> ExperienceRequirement:
    """Parse the experience requirement out of a job description."""
    if not text or not text.strip():
        return ExperienceRequirement()

    scoped = get_section(text, "experience") or ""
    haystacks = [scoped, text] if scoped else [text]

    for haystack in haystacks:
        for line in haystack.splitlines():
            if not _EXPERIENCE_CONTEXT.search(line) and not re.search(r"\byears?\b", line, re.I):
                continue
            for kind, pattern in _RANGE_PATTERNS:
                match = pattern.search(line)
                if not match:
                    continue
                first = _to_number(match.group(1))
                if first is None:
                    continue
                raw = match.group(0).strip()
                if kind == "range":
                    second = _to_number(match.group(2))
                    return ExperienceRequirement(first, second, raw)
                if kind in ("plus", "plus_word", "minimum"):
                    return ExperienceRequirement(first, None, raw)
                if kind == "maximum":
                    return ExperienceRequirement(None, first, raw)
                return ExperienceRequirement(first, None, raw)

            months = _MONTHS_PATTERN.search(line)
            if months:
                value = _to_number(months.group(1))
                if value:
                    return ExperienceRequirement(round(value / 12.0, 1), None, months.group(0))

    fresher = _FRESHER_PATTERN.search(text)
    if fresher:
        return ExperienceRequirement(0.0, None, fresher.group(0), accepts_freshers=True)
    return ExperienceRequirement()


def _work_history_scope(text: str) -> str:
    """Return only the parts of a resume where employment dates can appear.

    Education and certification blocks are excluded, because a degree spanning
    2017-2021 would otherwise be counted as four years of work experience.
    """
    sections = split_sections(text)
    if not sections:
        return text
    kept = [
        block
        for name, block in sections.items()
        if name not in _NON_WORK_SECTIONS and block.strip()
    ]
    scope = "\n".join(kept)
    return scope if scope.strip() else text


def _years_from_date_ranges(text: str) -> tuple[float | None, str | None]:
    """Sum non-overlapping employment periods found as date ranges."""
    today = date.today()
    periods: list[tuple[float, float, str]] = []
    text = _work_history_scope(text)

    for match in _DATE_RANGE.finditer(text):
        line_start = text.rfind("\n", 0, match.start()) + 1
        line_end = text.find("\n", match.end())
        line = text[line_start : line_end if line_end != -1 else len(text)]
        if _ACADEMIC_LINE.search(line):
            continue
        start_year = int(match.group("y1"))
        start_month = _MONTHS.get((match.group("m1") or "").lower()[:4].rstrip("."), 1)
        end_raw = match.group("y2").lower()
        if end_raw.isdigit():
            end_year = int(end_raw)
            end_month = _MONTHS.get((match.group("m2") or "").lower()[:4].rstrip("."), 12)
        else:
            end_year, end_month = today.year, today.month
        if not (1970 <= start_year <= today.year + 1):
            continue
        start_value = start_year + (start_month - 1) / 12
        end_value = end_year + end_month / 12
        if end_value <= start_value or end_value - start_value > 45:
            continue
        periods.append((start_value, end_value, match.group(0)))

    if not periods:
        return None, None

    periods.sort()
    merged: list[list[float]] = []
    for start_value, end_value, _ in periods:
        if merged and start_value <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end_value)
        else:
            merged.append([start_value, end_value])

    total = sum(end - start for start, end in merged)
    return round(total, 1), "; ".join(p[2] for p in periods[:3])


def extract_candidate_experience(text: str) -> CandidateExperience:
    """Extract total years of experience from a resume."""
    if not text or not text.strip():
        return CandidateExperience()

    statement_years: float | None = None
    statement_line: str | None = None
    for line in text.splitlines():
        if not _EXPERIENCE_CONTEXT.search(line):
            continue
        # Skip job *requirements* accidentally pasted into a resume.
        if re.search(r"\b(?:required|must have|we are looking|candidate should)\b", line, re.I):
            continue
        for kind, pattern in _RANGE_PATTERNS:
            match = pattern.search(line)
            if not match:
                continue
            value = _to_number(match.group(1))
            if value is None or value > 60:
                continue
            if kind == "range":
                second = _to_number(match.group(2))
                value = second if second is not None else value
            if statement_years is None or value > statement_years:
                statement_years, statement_line = value, line.strip()[:200]
            break

    if statement_years is None:
        months = _MONTHS_PATTERN.search(text)
        if months and _EXPERIENCE_CONTEXT.search(text):
            value = _to_number(months.group(1))
            if value and value <= 120:
                statement_years = round(value / 12.0, 1)
                statement_line = months.group(0)

    range_years, range_evidence = _years_from_date_ranges(text)

    if statement_years is not None and range_years is not None:
        # Trust the larger of the two, but keep both visible in the evidence.
        if range_years >= statement_years:
            return CandidateExperience(range_years, "date_ranges", range_evidence)
        return CandidateExperience(statement_years, "statement", statement_line)
    if statement_years is not None:
        return CandidateExperience(statement_years, "statement", statement_line)
    if range_years is not None:
        return CandidateExperience(range_years, "date_ranges", range_evidence)
    if _FRESHER_PATTERN.search(text):
        return CandidateExperience(0.0, "statement", "Fresher / entry level")
    return CandidateExperience()


def match_experience(
    requirement: ExperienceRequirement, candidate: CandidateExperience
) -> ExperienceMatch:
    """Compare candidate experience with the requirement.

    Returns a score in ``[0, 1]``, or ``None`` when either side is unknown --
    in which case the scoring engine redistributes this component's weight.
    """
    if not requirement.specified:
        return ExperienceMatch(
            None, "Job does not state an experience requirement", requirement, candidate
        )
    if not candidate.specified:
        return ExperienceMatch(
            None, "No experience information found in the resume", requirement, candidate
        )

    years = candidate.total_years or 0.0
    minimum = requirement.min_years
    maximum = requirement.max_years

    if minimum is None and maximum is not None:
        if years <= maximum:
            return ExperienceMatch(
                1.0,
                f"{_fmt(years)} years is within the {_fmt(maximum)}-year ceiling",
                requirement,
                candidate,
            )
        overshoot = years - maximum
        return ExperienceMatch(
            max(0.5, 1.0 - overshoot * 0.1),
            f"{_fmt(years)} years exceeds the stated maximum of {_fmt(maximum)}",
            requirement,
            candidate,
        )

    minimum = minimum or 0.0
    if years >= minimum:
        if maximum is not None and years > maximum + 2:
            return ExperienceMatch(
                0.85,
                f"{_fmt(years)} years is above the {_fmt(minimum)}-{_fmt(maximum)} year band",
                requirement,
                candidate,
            )
        return ExperienceMatch(
            1.0,
            f"{_fmt(years)} years meets the {_fmt(minimum)}-year requirement",
            requirement,
            candidate,
        )

    if minimum <= 0:
        return ExperienceMatch(1.0, "Entry level requirement satisfied", requirement, candidate)

    ratio = max(0.0, years / minimum)
    gap = minimum - years
    return ExperienceMatch(
        round(ratio, 3),
        f"{_fmt(years)} years found, {_fmt(gap)} year(s) short of the "
        f"{_fmt(minimum)}-year requirement",
        requirement,
        candidate,
    )


__all__ = [
    "CandidateExperience",
    "ExperienceMatch",
    "ExperienceRequirement",
    "extract_candidate_experience",
    "match_experience",
    "parse_experience_requirement",
]
