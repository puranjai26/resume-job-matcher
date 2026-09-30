"""Job description analysis.

Splits a job description into title, required skills, preferred skills,
responsibilities, and experience/education requirements.

Job descriptions do not share one format, so three signals are combined:

1. **Section headings** -- "Required Skills", "Nice to have", ...
2. **Inline bullet markers** -- a bullet containing "preferred", "a plus" or
   "bonus" is preferred even when it sits inside a Requirements section.
3. **Fallback** -- when neither signal is present, every extracted skill is
   reported as required-by-inference and the preferred list is explicitly
   reported as ``"Not explicitly specified"``. Nothing is invented.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .education import EducationInfo, extract_education_requirement
from .experience import ExperienceRequirement, parse_experience_requirement
from .sections import (
    PREFERRED_STRONG,
    PREFERRED_WEAK,
    REQUIRED_INLINE,
    iter_bullets,
    split_sections,
)
from .skill_extractor import SkillExtractor, get_extractor

NOT_SPECIFIED = "Not explicitly specified"

_TITLE_HINT = re.compile(
    r"\b(engineer|developer|analyst|scientist|manager|intern|designer|architect|"
    r"administrator|consultant|specialist|lead|associate|executive|programmer|"
    r"tester|researcher|officer|trainee)\b",
    re.IGNORECASE,
)

_TITLE_LABEL = re.compile(
    r"^\s*(?:job\s*)?(?:title|position|role|designation)\s*[:\-\u2013]\s*(.+)$",
    re.IGNORECASE,
)

_NOISE_PREFIX = re.compile(r"^\s*(?:about us|company|location|salary|ctc|apply)\b", re.IGNORECASE)


@dataclass
class JobAnalysis:
    """Structured view of a job description."""

    title: str = NOT_SPECIFIED
    required_skills: list[str] = field(default_factory=list)
    preferred_skills: list[str] = field(default_factory=list)
    all_skills: list[str] = field(default_factory=list)
    #: Skills mentioned elsewhere (responsibilities, company blurb) that were
    #: never stated as a requirement. Reported, but not scored as required.
    other_skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    experience: ExperienceRequirement = field(default_factory=ExperienceRequirement)
    education: EducationInfo = field(default_factory=EducationInfo)
    sections_detected: list[str] = field(default_factory=list)
    #: True when required/preferred came from explicit signals rather than inference.
    explicit_requirements: bool = False
    evidence: dict[str, str] = field(default_factory=dict)

    @property
    def preferred_specified(self) -> bool:
        return bool(self.preferred_skills)

    def describe_preferred(self) -> str:
        return ", ".join(self.preferred_skills) if self.preferred_skills else NOT_SPECIFIED

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills or NOT_SPECIFIED,
            "all_skills": self.all_skills,
            "other_skills": self.other_skills,
            "responsibilities": self.responsibilities,
            "experience": self.experience.to_dict(),
            "education": self.education.to_dict(),
            "sections_detected": self.sections_detected,
            "explicit_requirements": self.explicit_requirements,
        }


def extract_job_title(text: str) -> str:
    """Find the job title using a labelled line, then a heading heuristic."""
    if not text or not text.strip():
        return NOT_SPECIFIED

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:25]:
        labelled = _TITLE_LABEL.match(line)
        if labelled:
            title = labelled.group(1).strip(" .\u2013-")
            if 2 <= len(title) <= 80:
                return title

    for line in lines[:8]:
        if _NOISE_PREFIX.match(line) or line.endswith(":"):
            continue
        words = line.split()
        if 1 < len(words) <= 9 and _TITLE_HINT.search(line):
            return line.strip(" .\u2013-*#")

    for line in lines[:15]:
        if _TITLE_HINT.search(line) and len(line.split()) <= 12:
            return line.strip(" .\u2013-*#")
    return NOT_SPECIFIED


def _skills_in(text: str, extractor: SkillExtractor) -> list[str]:
    return extractor.extract(text, use_fuzzy=False).skills if text.strip() else []


def analyze_job_description(text: str, extractor: SkillExtractor | None = None) -> JobAnalysis:
    """Analyse a job description into structured requirements.

    Args:
        text: Raw job description text.
        extractor: Optional shared :class:`SkillExtractor`.

    Returns:
        A :class:`JobAnalysis`. When required/preferred sections cannot be
        identified confidently, ``explicit_requirements`` is ``False`` and the
        preferred list is left empty (reported as "Not explicitly specified").
    """
    extractor = extractor or get_extractor()
    analysis = JobAnalysis()
    if not text or not text.strip():
        return analysis

    detailed = extractor.extract(text, use_fuzzy=False)
    analysis.all_skills = detailed.skills
    analysis.evidence = detailed.evidence
    analysis.title = extract_job_title(text)
    analysis.experience = parse_experience_requirement(text)
    analysis.education = extract_education_requirement(text)

    sections = split_sections(text)
    analysis.sections_detected = sorted(sections)

    required: list[str] = []
    preferred: list[str] = []

    def add(target: list[str], skills: list[str]) -> None:
        for skill in skills:
            if skill not in target:
                target.append(skill)

    # 1. Explicit sections.
    required_block = sections.get("required_skills", "")
    preferred_block = sections.get("preferred_skills", "")
    skills_block = sections.get("skills", "")

    # Inside a Requirements block only a *strong* marker can demote a skill:
    # "Familiarity with Git" listed under Requirements is still required.
    if required_block:
        for bullet in iter_bullets(required_block):
            bucket = preferred if PREFERRED_STRONG.search(bullet) else required
            add(bucket, _skills_in(bullet, extractor))
    if preferred_block:
        add(preferred, _skills_in(preferred_block, extractor))
    if skills_block and not required_block:
        for bullet in iter_bullets(skills_block):
            bucket = preferred if PREFERRED_STRONG.search(bullet) else required
            add(bucket, _skills_in(bullet, extractor))

    explicit = bool(required_block or preferred_block or skills_block)

    # 2. Inline markers anywhere else in the document. Outside a Requirements
    #    block, weak markers ("exposure to", "familiarity with") also count.
    requirement_block_text = "\n".join(filter(None, (required_block, skills_block)))
    for bullet in iter_bullets(text):
        if bullet in requirement_block_text:
            continue
        if PREFERRED_STRONG.search(bullet) or PREFERRED_WEAK.search(bullet):
            found = _skills_in(bullet, extractor)
            if found:
                add(preferred, found)
                explicit = True
        elif REQUIRED_INLINE.search(bullet):
            found = _skills_in(bullet, extractor)
            if found:
                add(required, found)
                explicit = True

    # A skill named as required anywhere wins over a preferred mention.
    preferred = [s for s in preferred if s not in required]

    # 3. Fallback: no usable signal at all.
    if not required and not preferred:
        required = list(analysis.all_skills)
        explicit = False
    elif not required:
        required = [s for s in analysis.all_skills if s not in preferred]
        explicit = bool(preferred_block)
    else:
        # Skills that appear only in responsibilities or prose were never stated
        # as requirements. They are reported separately rather than being
        # promoted into the required list, which would invent requirements.
        analysis.other_skills = sorted(
            s for s in analysis.all_skills if s not in required and s not in preferred
        )

    responsibilities_block = sections.get("responsibilities", "")
    analysis.responsibilities = iter_bullets(responsibilities_block)[:12]

    analysis.required_skills = sorted(required)
    analysis.preferred_skills = sorted(preferred)
    analysis.explicit_requirements = explicit
    return analysis


__all__ = ["JobAnalysis", "NOT_SPECIFIED", "analyze_job_description", "extract_job_title"]
