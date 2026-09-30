"""Section detection for resumes and job descriptions.

Documents do not follow one layout, so this module uses a tolerant rule set:
a line is treated as a heading when it is short, contains a known section
keyword, and looks like a heading (title case / upper case / ends with a colon).

Everything until the next heading belongs to that section. Text that appears
before the first heading is kept under ``"header"``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: canonical section -> keyword patterns that introduce it
SECTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "required_skills": (
        r"required skills?",
        r"requirements?",
        r"must[- ]haves?",
        r"minimum qualifications?",
        r"essential skills?",
        r"basic qualifications?",
        r"key skills required",
        r"what you(?:'ll| will) need",
        r"who you are",
        r"mandatory skills?",
        r"required qualifications?",
        r"skills? required",
        r"eligibility",
    ),
    "preferred_skills": (
        r"preferred skills?",
        r"preferred qualifications?",
        r"nice[- ]to[- ]haves?",
        r"good to have",
        r"bonus points?",
        r"desirable",
        r"desired skills?",
        r"advantageous",
        r"plus points?",
        r"optional skills?",
        r"we'd love",
    ),
    "responsibilities": (
        r"responsibilit(?:y|ies)",
        r"what you(?:'ll| will) do",
        r"role overview",
        r"job description",
        r"duties",
        r"day[- ]to[- ]day",
        r"about the role",
        r"key deliverables?",
    ),
    "skills": (
        r"skills?",
        r"technical skills?",
        r"core competenc(?:y|ies)",
        r"technologies",
        r"tech stack",
        r"areas of expertise",
        r"proficienc(?:y|ies)",
        r"toolkit",
        r"technical proficienc(?:y|ies)",
    ),
    "experience": (
        r"experience",
        r"work experience",
        r"professional experience",
        r"employment(?: history)?",
        r"internships?",
        r"work history",
        r"career history",
    ),
    "education": (
        r"education",
        r"academic(?:s| background| qualifications?)?",
        r"educational qualifications?",
        r"qualifications?",
    ),
    "projects": (r"projects?", r"academic projects?", r"personal projects?", r"portfolio"),
    "certifications": (
        r"certifications?",
        r"certificates?",
        r"licenses?",
        r"courses?",
        r"training",
        r"credentials",
    ),
    "summary": (
        r"summary",
        r"objective",
        r"profile",
        r"about(?: me| us)?",
        r"career objective",
    ),
    "achievements": (r"achievements?", r"awards?", r"honou?rs", r"accomplishments?"),
    "benefits": (r"benefits?", r"what we offer", r"perks", r"compensation", r"salary"),
}

# Longest keywords first so "required skills" wins over "skills".
_COMPILED: list[tuple[str, re.Pattern[str]]] = sorted(
    (
        (name, re.compile(rf"^[\s\W]*{pattern}\b[\s:.\-\u2013]*$", re.IGNORECASE))
        for name, patterns in SECTION_PATTERNS.items()
        for pattern in patterns
    ),
    key=lambda item: -len(item[1].pattern),
)

#: Strong inline markers: an unambiguous "this is optional" signal. Trusted even
#: inside a Requirements section.
PREFERRED_STRONG = re.compile(
    r"\b(?:preferred|nice to have|good to have|a plus|is a plus|are a plus|bonus|"
    r"desirable|desired|would be an advantage|is an advantage|optional|"
    r"not mandatory|added advantage)\b",
    re.IGNORECASE,
)

#: Weak markers: they suggest optionality but appear in genuine requirements too
#: ("Familiarity with Git" under Requirements). Only trusted outside a
#: Requirements section.
PREFERRED_WEAK = re.compile(
    r"\b(?:familiarity with|exposure to|awareness of|some experience with|"
    r"basic (?:knowledge|understanding) of|advantage)\b",
    re.IGNORECASE,
)

#: Any preferred signal (strong or weak).
PREFERRED_INLINE = re.compile(
    PREFERRED_STRONG.pattern + "|" + PREFERRED_WEAK.pattern, re.IGNORECASE
)

REQUIRED_INLINE = re.compile(
    r"\b(?:required|must have|must possess|should have|strong (?:knowledge|experience|"
    r"command)|proficien(?:t|cy) in|hands[- ]on experience|expertise in|"
    r"solid (?:understanding|experience)|demonstrated experience)\b",
    re.IGNORECASE,
)

_MAX_HEADING_WORDS = 6
_MAX_HEADING_CHARS = 60

#: Lowercase words allowed inside an otherwise title-cased heading.
_MINOR_WORDS = frozenset(
    "a an and the to of or for in on at with be is are you we will do have has"
    " from into over".split()
)


@dataclass
class Section:
    """A detected block of a document."""

    name: str
    heading: str
    text: str
    start_line: int


def _looks_like_heading(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_CHARS:
        return False
    if len(stripped.split()) > _MAX_HEADING_WORDS:
        return False
    if stripped.endswith((".", ",")) and not stripped.endswith(":"):
        return False
    core = stripped.strip(" :*#-\u2013\u2014_=")
    if not core:
        return False
    # Headings are colon-terminated, upper case, or title case.
    if stripped.endswith(":"):
        return True
    if core.isupper():
        return True
    # Title case, allowing the lowercase function words that real headings use
    # ("Nice to Have", "What You Will Do", "Good to Have").
    words = [w for w in core.split() if w[:1].isalpha()]
    if not words:
        return False
    return (
        all(
            word[:1].isupper() or word.lower() in _MINOR_WORDS
            for index, word in enumerate(words)
            if index > 0 or word[:1].isupper()
        )
        and words[0][:1].isupper()
    )


def classify_heading(line: str) -> str | None:
    """Return the canonical section name for a heading line, or ``None``."""
    if not _looks_like_heading(line):
        return None
    core = line.strip().strip(" :*#-\u2013\u2014_=")
    for name, pattern in _COMPILED:
        if pattern.match(core):
            return name
    return None


def split_sections(text: str) -> dict[str, str]:
    """Split a document into ``{section_name: text}``.

    Repeated headings of the same type are concatenated. Text before the first
    recognised heading is returned under ``"header"``.
    """
    sections = detect_sections(text)
    merged: dict[str, list[str]] = {}
    for section in sections:
        merged.setdefault(section.name, []).append(section.text)
    return {name: "\n".join(chunks).strip() for name, chunks in merged.items()}


def detect_sections(text: str) -> list[Section]:
    """Return the ordered list of detected sections."""
    if not text or not text.strip():
        return []

    lines = text.splitlines()
    found: list[Section] = []
    current_name, current_heading, buffer, start = "header", "", [], 0

    for index, line in enumerate(lines):
        name = classify_heading(line)
        if name:
            body = "\n".join(buffer).strip()
            if body:
                found.append(Section(current_name, current_heading, body, start))
            current_name, current_heading, buffer, start = name, line.strip(), [], index
        else:
            buffer.append(line)

    body = "\n".join(buffer).strip()
    if body:
        found.append(Section(current_name, current_heading, body, start))
    return found


def get_section(text: str, *names: str) -> str:
    """Return the concatenated text of the first matching section names."""
    sections = split_sections(text)
    parts = [sections[name] for name in names if sections.get(name)]
    return "\n".join(parts).strip()


def iter_bullets(text: str) -> list[str]:
    """Split a block into bullet-sized units (one requirement per line)."""
    bullets: list[str] = []
    for raw in (text or "").splitlines():
        line = raw.strip().lstrip("-*\u2022\u25cf\u25aa\u2023\u2043>+ ").strip()
        if not line:
            continue
        # "Python, SQL and Docker" style lines stay whole: splitting them would
        # lose the required/preferred marker that applies to the entire line.
        bullets.append(line)
    return bullets


__all__ = [
    "PREFERRED_INLINE",
    "PREFERRED_STRONG",
    "PREFERRED_WEAK",
    "REQUIRED_INLINE",
    "SECTION_PATTERNS",
    "Section",
    "classify_heading",
    "detect_sections",
    "get_section",
    "iter_bullets",
    "split_sections",
]
