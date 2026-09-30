"""Scoring engine and skill gap analysis.

The final score is a weighted sum of six components, each normalised to
``[0, 1]``. Weights live in :class:`src.config.ScoringWeights` and are
**configurable project assumptions** -- no claim is made that they are optimal.

Unavailable components (for example an experience requirement the job never
states) are not scored as zero. Their weight is redistributed proportionally
over the components that *could* be computed, and the report says which ones
were skipped. That keeps a resume from being punished for information a job
description never asked for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .config import DEFAULT_SETTINGS, MatchThresholds, ScoringWeights


@dataclass
class SkillGap:
    """Matched / missing / extra skills for one requirement group."""

    matched: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    total: int = 0

    @property
    def coverage(self) -> float:
        """Fraction of the group's skills present in the resume (0-1)."""
        return len(self.matched) / self.total if self.total else 0.0

    @property
    def coverage_percent(self) -> float:
        return round(self.coverage * 100, 1)

    def summary(self) -> str:
        if not self.total:
            return "No skills in this group"
        return f"{len(self.matched)}/{self.total} skills matched"


@dataclass
class ScoreComponent:
    """One weighted contribution to the final score."""

    name: str
    label: str
    value: float | None  # 0-1, or None when unavailable
    weight: float  # configured weight
    effective_weight: float = 0.0  # weight after redistribution
    detail: str = ""

    @property
    def available(self) -> bool:
        return self.value is not None

    @property
    def contribution(self) -> float:
        """Points this component adds to the final 0-100 score."""
        return 0.0 if self.value is None else self.value * self.effective_weight * 100

    @property
    def percent(self) -> float:
        return round((self.value or 0.0) * 100, 1)


@dataclass
class MatchResult:
    """Complete, serialisable outcome of one resume-job comparison."""

    overall_score: float = 0.0
    classification: str = "Weak Match"
    components: list[ScoreComponent] = field(default_factory=list)
    required_gap: SkillGap = field(default_factory=SkillGap)
    preferred_gap: SkillGap = field(default_factory=SkillGap)
    additional_skills: list[str] = field(default_factory=list)
    skipped_components: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    explanations: list[str] = field(default_factory=list)
    gaps: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    @property
    def matched_skills(self) -> list[str]:
        return sorted(set(self.required_gap.matched) | set(self.preferred_gap.matched))

    @property
    def missing_skills(self) -> list[str]:
        return sorted(set(self.required_gap.missing) | set(self.preferred_gap.missing))

    def component(self, name: str) -> ScoreComponent | None:
        return next((c for c in self.components if c.name == name), None)

    def component_percent(self, name: str) -> float | None:
        component = self.component(name)
        return None if component is None or not component.available else component.percent

    def to_dict(self) -> dict:
        return {
            "overall_score": self.overall_score,
            "classification": self.classification,
            "components": [
                {
                    "name": c.name,
                    "label": c.label,
                    "value": c.value,
                    "percent": c.percent if c.available else None,
                    "weight": c.weight,
                    "effective_weight": round(c.effective_weight, 4),
                    "contribution": round(c.contribution, 2),
                    "available": c.available,
                    "detail": c.detail,
                }
                for c in self.components
            ],
            "required": {
                "matched": self.required_gap.matched,
                "missing": self.required_gap.missing,
                "coverage_percent": self.required_gap.coverage_percent,
            },
            "preferred": {
                "matched": self.preferred_gap.matched,
                "missing": self.preferred_gap.missing,
                "coverage_percent": self.preferred_gap.coverage_percent,
            },
            "additional_skills": self.additional_skills,
            "skipped_components": self.skipped_components,
            "warnings": self.warnings,
            "explanations": self.explanations,
            "gaps": self.gaps,
            "details": self.details,
        }


# --------------------------------------------------------------------------- #
# Skill gap analysis
# --------------------------------------------------------------------------- #


def analyze_skill_gap(resume_skills: list[str], job_skills: list[str]) -> SkillGap:
    """Compare two canonical skill lists.

    Both inputs must already be normalised (see :mod:`src.skill_normalizer`), so
    ``ML`` and ``Machine Learning`` have collapsed to one name before arriving.
    """
    resume_set = set(resume_skills)
    unique_job_skills = list(dict.fromkeys(job_skills))
    matched = sorted(s for s in unique_job_skills if s in resume_set)
    missing = sorted(s for s in unique_job_skills if s not in resume_set)
    return SkillGap(matched=matched, missing=missing, total=len(unique_job_skills))


def find_additional_skills(resume_skills: list[str], job_skills: list[str]) -> list[str]:
    """Skills the candidate has that the job never mentions."""
    return sorted(set(resume_skills) - set(job_skills))


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def _redistribute(components: list[ScoreComponent]) -> None:
    """Spread the weight of unavailable components over the available ones."""
    available = [c for c in components if c.available]
    total_available = sum(c.weight for c in available)
    if not available or total_available <= 0:
        for component in components:
            component.effective_weight = 0.0
        return
    for component in components:
        component.effective_weight = (
            component.weight / total_available if component.available else 0.0
        )


def calculate_score(
    *,
    required_gap: SkillGap,
    preferred_gap: SkillGap,
    semantic_similarity: float,
    tfidf_similarity: float,
    experience_score: float | None,
    education_score: float | None,
    weights: ScoringWeights | None = None,
    thresholds: MatchThresholds | None = None,
    experience_detail: str = "",
    education_detail: str = "",
) -> MatchResult:
    """Combine the components into a final interpretable score.

    Args:
        required_gap: Result of :func:`analyze_skill_gap` on required skills.
        preferred_gap: Same, for preferred skills.
        semantic_similarity: 0-1 semantic score.
        tfidf_similarity: 0-1 TF-IDF score.
        experience_score: 0-1, or ``None`` when it cannot be judged.
        education_score: 0-1, or ``None`` when it cannot be judged.
        weights: Override the default weights.
        thresholds: Override the default classification bands.

    Returns:
        A :class:`MatchResult` with the score, the per-component breakdown and
        the list of components that were skipped.
    """
    weights = weights or DEFAULT_SETTINGS.weights
    thresholds = thresholds or DEFAULT_SETTINGS.thresholds

    components = [
        ScoreComponent(
            "required_skills",
            "Required skill coverage",
            required_gap.coverage if required_gap.total else None,
            weights.required_skills,
            detail=required_gap.summary(),
        ),
        ScoreComponent(
            "preferred_skills",
            "Preferred skill coverage",
            preferred_gap.coverage if preferred_gap.total else None,
            weights.preferred_skills,
            detail=preferred_gap.summary() if preferred_gap.total else "Not explicitly specified",
        ),
        ScoreComponent(
            "semantic_similarity",
            "Semantic similarity",
            max(0.0, min(1.0, float(semantic_similarity))),
            weights.semantic_similarity,
            detail="Sentence-level meaning overlap",
        ),
        ScoreComponent(
            "tfidf_similarity",
            "TF-IDF similarity",
            max(0.0, min(1.0, float(tfidf_similarity))),
            weights.tfidf_similarity,
            detail="Weighted keyword overlap",
        ),
        ScoreComponent(
            "experience",
            "Experience match",
            None if experience_score is None else max(0.0, min(1.0, float(experience_score))),
            weights.experience,
            detail=experience_detail,
        ),
        ScoreComponent(
            "education",
            "Education match",
            None if education_score is None else max(0.0, min(1.0, float(education_score))),
            weights.education,
            detail=education_detail,
        ),
    ]

    _redistribute(components)
    total = sum(component.contribution for component in components)
    total = round(max(0.0, min(100.0, total)), 1)

    return MatchResult(
        overall_score=total,
        classification=thresholds.classify(total),
        components=components,
        required_gap=required_gap,
        preferred_gap=preferred_gap,
        skipped_components=[c.label for c in components if not c.available],
    )


def build_explanations(
    result: MatchResult, *, semantic_backend: str = ""
) -> tuple[list[str], list[str]]:
    """Generate the "why" bullets from structured results only.

    No language model is involved anywhere: every sentence below is assembled
    from the numbers already computed, so the same inputs always produce the
    same explanation.

    Returns:
        ``(reasons, gaps)`` -- positive findings and shortfalls.
    """
    reasons: list[str] = []
    gaps: list[str] = []

    required = result.required_gap
    if required.total:
        line = f"{len(required.matched)}/{required.total} required skills found"
        if required.coverage >= 0.8:
            reasons.append(line)
        elif required.coverage >= 0.5:
            reasons.append(f"{line} ({required.coverage_percent:.0f}% coverage)")
        else:
            gaps.append(f"Only {line}")
        if required.matched:
            preview = ", ".join(required.matched[:6])
            extra = "..." if len(required.matched) > 6 else ""
            reasons.append(f"Matched: {preview}{extra}")
        for skill in required.missing[:8]:
            gaps.append(f"Missing required skill: {skill}")
        if len(required.missing) > 8:
            gaps.append(f"...and {len(required.missing) - 8} more required skills")
    else:
        gaps.append("No skills could be extracted from the job description")

    preferred = result.preferred_gap
    if preferred.total:
        if preferred.matched:
            reasons.append(
                f"{len(preferred.matched)}/{preferred.total} preferred skills found: "
                + ", ".join(preferred.matched[:5])
            )
        for skill in preferred.missing[:5]:
            gaps.append(f"Missing preferred skill: {skill}")

    semantic = result.component("semantic_similarity")
    if semantic and semantic.available:
        backend_note = f" ({semantic_backend})" if semantic_backend else ""
        if semantic.value >= 0.6:
            reasons.append(f"Strong semantic similarity: {semantic.percent:.0f}%{backend_note}")
        elif semantic.value >= 0.35:
            reasons.append(f"Moderate semantic similarity: {semantic.percent:.0f}%{backend_note}")
        else:
            gaps.append(
                f"Low semantic similarity ({semantic.percent:.0f}%) - resume wording differs "
                "considerably from the job description"
            )

    tfidf = result.component("tfidf_similarity")
    if tfidf and tfidf.available and tfidf.value >= 0.25:
        reasons.append(f"Shared vocabulary with the job description: {tfidf.percent:.0f}%")

    experience = result.component("experience")
    if experience and experience.available:
        (reasons if experience.value >= 0.9 else gaps).append(
            experience.detail or "Experience compared"
        )

    education = result.component("education")
    if education and education.available:
        (reasons if education.value >= 0.75 else gaps).append(
            education.detail or "Education compared"
        )

    if result.additional_skills:
        reasons.append(
            f"{len(result.additional_skills)} additional skills beyond the job description: "
            + ", ".join(result.additional_skills[:5])
        )

    for label in result.skipped_components:
        gaps.append(f"{label}: not enough information, weight redistributed")

    return reasons, gaps


__all__ = [
    "MatchResult",
    "ScoreComponent",
    "SkillGap",
    "analyze_skill_gap",
    "build_explanations",
    "calculate_score",
    "find_additional_skills",
]
