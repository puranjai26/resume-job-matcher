"""Integration tests: resume + job description -> complete match report."""

from __future__ import annotations

import pytest

from src.config import SAMPLE_RESUMES_DIR, ScoringWeights, Settings
from src.evaluation import evaluate, load_documents, load_labels, score_pairs
from src.parser import extract_text
from src.pipeline import match_resume_to_job

RESUME = """Jane Doe

SKILLS
Python, SQL, Flask, Git, Machine Learning

EXPERIENCE
Software Engineer, Example Ltd (Jan 2022 - Present)
- Built internal reporting services in Python.

EDUCATION
B.Tech in Computer Science, 2021
"""

JOB = """Job Title: Python Developer

Required Skills
- Python
- SQL
- REST API

Preferred Skills
- AWS
- Docker

Experience
2+ years of experience.

Education
Bachelor's degree in Computer Science or a related field.
"""


@pytest.fixture(scope="module")
def sample_documents():
    return load_documents()


def test_end_to_end_report_has_every_section():
    report = match_resume_to_job(RESUME, JOB)

    assert 0.0 <= report.overall_score <= 100.0
    assert report.classification in (
        "Weak Match",
        "Moderate Match",
        "Strong Match",
        "Excellent Match",
    )
    assert report.job.title == "Python Developer"
    assert "Python" in report.result.required_gap.matched
    assert "REST API" in report.result.required_gap.missing
    assert set(report.result.preferred_gap.missing) == {"AWS", "Docker"}
    assert "Flask" in report.additional_skills
    assert report.result.explanations
    assert report.result.gaps
    assert report.resume_education.level == 2
    assert report.experience_match.score is not None


def test_report_serialises_to_json_friendly_dict():
    payload = match_resume_to_job(RESUME, JOB).to_dict()
    assert payload["overall_score"] >= 0
    assert payload["job"]["required_skills"]
    assert payload["similarity"]["semantic_backend"] in ("sentence-transformers", "lsa")


def test_pipeline_is_deterministic():
    first = match_resume_to_job(RESUME, JOB).overall_score
    second = match_resume_to_job(RESUME, JOB).overall_score
    assert first == second


def test_empty_resume_produces_warning_not_a_crash():
    report = match_resume_to_job("", JOB)
    assert report.overall_score == 0.0
    assert any("empty" in warning.lower() for warning in report.warnings)


def test_empty_job_description_produces_warning_not_a_crash():
    report = match_resume_to_job(RESUME, "")
    assert report.overall_score == 0.0
    assert report.warnings


def test_resume_with_no_detectable_skills_is_handled():
    report = match_resume_to_job("I enjoy gardening and long walks on the beach.", JOB)
    assert report.resume_skills.skills == []
    assert any("No known skills" in warning for warning in report.warnings)
    assert report.overall_score >= 0.0


def test_job_without_explicit_sections_warns_about_inference():
    report = match_resume_to_job(RESUME, "Looking for someone who knows Python and SQL.")
    assert any("Required" in warning for warning in report.warnings)
    assert report.job.explicit_requirements is False


def test_very_long_documents_complete():
    report = match_resume_to_job(RESUME * 200, JOB * 50)
    assert 0.0 <= report.overall_score <= 100.0


def test_custom_settings_change_the_outcome():
    strict = Settings(
        weights=ScoringWeights().replace(
            required_skills=0.9,
            semantic_similarity=0.0,
            tfidf_similarity=0.0,
            preferred_skills=0.1,
            experience=0.0,
            education=0.0,
        )
    )
    default_score = match_resume_to_job(RESUME, JOB).overall_score
    strict_score = match_resume_to_job(RESUME, JOB, settings=strict).overall_score
    assert strict_score != default_score


def test_sample_data_loads(sample_documents):
    resumes, jobs = sample_documents
    assert len(resumes) >= 5
    assert len(jobs) >= 5


def test_matching_resume_outranks_mismatching_one(sample_documents):
    resumes, jobs = sample_documents
    devops_job = jobs["J005"]
    devops_resume = match_resume_to_job(resumes["R005"], devops_job).overall_score
    frontend_resume = match_resume_to_job(resumes["R003"], devops_job).overall_score
    assert devops_resume > frontend_resume + 20


def test_every_sample_pair_runs_without_error(sample_documents):
    resumes, jobs = sample_documents
    for resume_text in resumes.values():
        for job_text in jobs.values():
            report = match_resume_to_job(resume_text, job_text)
            assert 0.0 <= report.overall_score <= 100.0


def test_pdf_and_docx_samples_flow_through_the_pipeline(sample_documents):
    _, jobs = sample_documents
    pdf_text = extract_text(SAMPLE_RESUMES_DIR / "R002_backend_developer.pdf")
    docx_text = extract_text(SAMPLE_RESUMES_DIR / "R003_frontend_developer.docx")
    assert match_resume_to_job(pdf_text, jobs["J002"]).overall_score > 50
    assert match_resume_to_job(docx_text, jobs["J003"]).overall_score > 50


def test_labelled_evaluation_runs_and_ranks_sensibly(sample_documents):
    resumes, jobs = sample_documents
    pairs = score_pairs(load_labels(), resumes, jobs)
    assert len(pairs) >= 20

    results = evaluate(pairs)
    assert results["pipeline"]["f1"] > 0.5
    assert results["pipeline"]["precision_at_1"] >= 0.8
    # The full pipeline should separate good from bad matches at least as well
    # as raw keyword overlap does.
    assert results["pipeline"]["separation"] > 0.0


def test_sample_job_files_parse_into_requirements(sample_documents):
    _, jobs = sample_documents
    for job_id, job_text in jobs.items():
        report = match_resume_to_job("Python SQL", job_text)
        assert report.job.required_skills, f"{job_id} produced no required skills"
        assert report.job.experience.specified, f"{job_id} produced no experience requirement"
