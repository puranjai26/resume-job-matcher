"""End-to-end matching pipeline.

``match_resume_to_job(resume_text, job_text)`` runs every stage and returns a
single :class:`MatchReport`. The Streamlit app, the tests and the evaluation
script all go through this one entry point, so there is exactly one definition
of "how a match is computed".

Stages::

    text -> preprocessing -> skill extraction -> normalisation
         -> job requirement analysis
         -> TF-IDF + semantic similarity
         -> experience / education matching
         -> scoring -> explanations -> report
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from .config import DEFAULT_SETTINGS, Settings
from .education import EducationInfo, extract_education, match_education
from .experience import (
    CandidateExperience,
    ExperienceMatch,
    extract_candidate_experience,
    match_experience,
)
from .job_analyzer import JobAnalysis, analyze_job_description
from .preprocessing import ProcessedText, preprocess_text
from .scoring import (
    MatchResult,
    analyze_skill_gap,
    build_explanations,
    calculate_score,
    find_additional_skills,
)
from .similarity import (
    SimilarityResult,
    calculate_keyword_overlap,
    calculate_semantic_similarity,
    calculate_tfidf_similarity,
)
from .skill_extractor import SkillExtractionResult, SkillExtractor, get_extractor

LOGGER = logging.getLogger(__name__)

MIN_RESUME_WORDS = 20
MIN_JOB_WORDS = 10


@dataclass
class MatchReport:
    """Everything the UI needs to render a full report."""

    result: MatchResult
    job: JobAnalysis
    resume_skills: SkillExtractionResult
    resume_experience: CandidateExperience
    resume_education: EducationInfo
    experience_match: ExperienceMatch
    education_match: object
    semantic: SimilarityResult
    tfidf_score: float
    keyword_score: float
    resume_processed: ProcessedText
    job_processed: ProcessedText
    warnings: list[str] = field(default_factory=list)

    # -- convenience accessors used by the UI ---------------------------- #

    @property
    def overall_score(self) -> float:
        return self.result.overall_score

    @property
    def classification(self) -> str:
        return self.result.classification

    @property
    def matched_skills(self) -> list[str]:
        return self.result.matched_skills

    @property
    def missing_skills(self) -> list[str]:
        return self.result.missing_skills

    @property
    def additional_skills(self) -> list[str]:
        return self.result.additional_skills

    def to_dict(self) -> dict:
        return {
            "overall_score": self.result.overall_score,
            "classification": self.result.classification,
            "job": self.job.to_dict(),
            "resume": {
                "skills": self.resume_skills.skills,
                "skills_by_category": self.resume_skills.by_category(),
                "experience": self.resume_experience.to_dict(),
                "education": self.resume_education.to_dict(),
                "word_count": self.resume_processed.word_count,
            },
            "similarity": {
                "tfidf": round(self.tfidf_score, 4),
                "semantic": round(self.semantic.score, 4),
                "semantic_backend": self.semantic.backend,
                "keyword_overlap": round(self.keyword_score, 4),
            },
            "score": self.result.to_dict(),
            "warnings": self.warnings,
        }


def _collect_warnings(
    resume_processed: ProcessedText,
    job_processed: ProcessedText,
    resume_skills: SkillExtractionResult,
    job: JobAnalysis,
) -> list[str]:
    """Surface data-quality problems instead of silently scoring bad input."""
    warnings: list[str] = []

    if resume_processed.is_empty:
        warnings.append("The resume is empty. No analysis is possible.")
    elif resume_processed.word_count < MIN_RESUME_WORDS:
        warnings.append(
            f"The resume contains only {resume_processed.word_count} words. "
            "Results will be unreliable."
        )

    if job_processed.is_empty:
        warnings.append("The job description is empty. No analysis is possible.")
    elif job_processed.word_count < MIN_JOB_WORDS:
        warnings.append(
            f"The job description contains only {job_processed.word_count} words. "
            "Results will be unreliable."
        )

    if not resume_skills.skills and not resume_processed.is_empty:
        warnings.append(
            "No known skills were detected in the resume. It may be image-based, or it "
            "may use terms that are not in data/skills.csv - you can add them there."
        )
    if not job.all_skills and not job_processed.is_empty:
        warnings.append(
            "No known skills were detected in the job description. "
            "The score falls back to text similarity only."
        )
    if job.all_skills and not job.explicit_requirements:
        warnings.append(
            "No explicit 'Required' or 'Preferred' section was found. All detected "
            "skills are treated as required, and preferred skills are reported as "
            "'Not explicitly specified'."
        )
    return warnings


def match_resume_to_job(
    resume_text: str,
    job_text: str,
    *,
    settings: Settings | None = None,
    extractor: SkillExtractor | None = None,
) -> MatchReport:
    """Run the complete matching pipeline on two documents.

    Args:
        resume_text: Plain text of the resume (see :func:`src.parser.extract_text`).
        job_text: Plain text of the job description.
        settings: Optional overrides for weights, thresholds and backend.
        extractor: Optional shared :class:`SkillExtractor` (avoids reloading the
            vocabulary in batch jobs).

    Returns:
        A :class:`MatchReport`. Empty or unusable input produces a report with
        a zero score and populated ``warnings`` rather than an exception.
    """
    settings = settings or DEFAULT_SETTINGS
    extractor = extractor or get_extractor()
    resume_text = resume_text or ""
    job_text = job_text or ""

    # 1. Preprocessing (keeps original text intact)
    resume_processed = preprocess_text(resume_text)
    job_processed = preprocess_text(job_text)

    # 2. Extraction
    resume_skills = extractor.extract(
        resume_text, use_fuzzy=True, fuzzy_threshold=settings.fuzzy_threshold
    )
    job = analyze_job_description(job_text, extractor=extractor)
    resume_experience = extract_candidate_experience(resume_text)
    resume_education = extract_education(resume_text)

    # 3. Skill gaps
    required_gap = analyze_skill_gap(resume_skills.skills, job.required_skills)
    preferred_gap = analyze_skill_gap(resume_skills.skills, job.preferred_skills)
    additional = find_additional_skills(resume_skills.skills, job.all_skills)

    # 4. Similarity
    tfidf_score = calculate_tfidf_similarity(resume_text, job_text)
    semantic = calculate_semantic_similarity(
        resume_text,
        job_text,
        backend=settings.semantic_backend,
        model_name=settings.semantic_model_name,
        return_result=True,
    )
    assert isinstance(semantic, SimilarityResult)
    keyword_score = calculate_keyword_overlap(resume_skills.skills, job.all_skills)

    # 5. Experience and education
    experience_match = match_experience(job.experience, resume_experience)
    education_match = match_education(job.education, resume_education)

    # 6. Scoring
    result = calculate_score(
        required_gap=required_gap,
        preferred_gap=preferred_gap,
        semantic_similarity=semantic.score,
        tfidf_similarity=tfidf_score,
        experience_score=experience_match.score,
        education_score=education_match.score,
        weights=settings.weights,
        thresholds=settings.thresholds,
        experience_detail=experience_match.status,
        education_detail=education_match.status,
    )
    result.additional_skills = additional

    warnings = _collect_warnings(resume_processed, job_processed, resume_skills, job)
    if resume_processed.is_empty or job_processed.is_empty:
        result.overall_score = 0.0
        result.classification = settings.thresholds.classify(0.0)
    result.warnings = warnings

    # 7. Deterministic explanations
    reasons, gaps = build_explanations(result, semantic_backend=semantic.backend)
    result.explanations = reasons
    result.gaps = gaps
    result.details = {
        "keyword_overlap": round(keyword_score, 4),
        "semantic_backend": semantic.backend,
        "resume_word_count": resume_processed.word_count,
        "job_word_count": job_processed.word_count,
    }

    return MatchReport(
        result=result,
        job=job,
        resume_skills=resume_skills,
        resume_experience=resume_experience,
        resume_education=resume_education,
        experience_match=experience_match,
        education_match=education_match,
        semantic=semantic,
        tfidf_score=tfidf_score,
        keyword_score=keyword_score,
        resume_processed=resume_processed,
        job_processed=job_processed,
        warnings=warnings,
    )


__all__ = ["MatchReport", "match_resume_to_job"]
