"""Tests for the similarity engines."""

from __future__ import annotations

import pytest

from src.similarity import (
    SimilarityResult,
    calculate_keyword_overlap,
    calculate_semantic_similarity,
    calculate_tfidf_similarity,
    chunk_text,
    compare_methods,
    semantic_backend_available,
)

RESUME = (
    "Developed RESTful APIs using Flask and Python. Built machine learning models "
    "with scikit-learn and queried PostgreSQL databases for reporting."
)
RELATED_JOB = (
    "Experience building web APIs using Python frameworks. Knowledge of ML "
    "techniques and relational databases is required."
)
UNRELATED_JOB = (
    "We are hiring a graphic designer skilled in Adobe Photoshop, Illustrator and "
    "print layout for a monthly magazine."
)


# --------------------------------------------------------------------------- #
# TF-IDF
# --------------------------------------------------------------------------- #


def test_tfidf_similarity_is_bounded():
    score = calculate_tfidf_similarity(RESUME, RELATED_JOB)
    assert 0.0 <= score <= 1.0


def test_tfidf_identical_documents_score_near_one():
    assert calculate_tfidf_similarity(RESUME, RESUME) > 0.95


def test_tfidf_prefers_related_over_unrelated():
    related = calculate_tfidf_similarity(RESUME, RELATED_JOB)
    unrelated = calculate_tfidf_similarity(RESUME, UNRELATED_JOB)
    assert related > unrelated


def test_tfidf_with_empty_input_returns_zero():
    assert calculate_tfidf_similarity("", RELATED_JOB) == 0.0
    assert calculate_tfidf_similarity(RESUME, "") == 0.0
    assert calculate_tfidf_similarity("", "") == 0.0


def test_tfidf_handles_stopword_only_documents():
    assert calculate_tfidf_similarity("the and of", "a an the") == 0.0


# --------------------------------------------------------------------------- #
# Semantic
# --------------------------------------------------------------------------- #


def test_semantic_similarity_is_bounded():
    score = calculate_semantic_similarity(RESUME, RELATED_JOB)
    assert 0.0 <= score <= 1.0


def test_semantic_prefers_related_over_unrelated():
    related = calculate_semantic_similarity(RESUME, RELATED_JOB)
    unrelated = calculate_semantic_similarity(RESUME, UNRELATED_JOB)
    assert related > unrelated


def test_semantic_with_empty_input_returns_zero():
    assert calculate_semantic_similarity("", RELATED_JOB) == 0.0


def test_semantic_result_reports_its_backend():
    result = calculate_semantic_similarity(RESUME, RELATED_JOB, return_result=True)
    assert isinstance(result, SimilarityResult)
    assert result.backend in ("sentence-transformers", "lsa")
    assert 0.0 <= result.score <= 1.0


def test_lsa_backend_works_without_any_model_download():
    score = calculate_semantic_similarity(RESUME, RELATED_JOB, backend="lsa")
    assert 0.0 < score <= 1.0


def test_semantic_backend_available_reports_a_known_value():
    assert semantic_backend_available() in ("st", "lsa")
    assert semantic_backend_available("lsa") == "lsa"


def test_requesting_missing_sentence_transformers_raises_clearly():
    pytest.importorskip  # keep import used
    try:
        import sentence_transformers  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError) as info:
            calculate_semantic_similarity(RESUME, RELATED_JOB, backend="st")
        assert "sentence-transformers" in str(info.value)


# --------------------------------------------------------------------------- #
# Keyword baseline and helpers
# --------------------------------------------------------------------------- #


def test_keyword_overlap_is_jaccard():
    assert calculate_keyword_overlap(["Python", "SQL"], ["Python", "SQL"]) == 1.0
    assert calculate_keyword_overlap(["Python"], ["Docker"]) == 0.0
    assert calculate_keyword_overlap([], []) == 0.0
    assert 0.0 < calculate_keyword_overlap(["Python", "SQL"], ["Python", "AWS"]) < 1.0


def test_chunk_text_splits_long_documents():
    long_text = ". ".join(f"Sentence number {index} about Python" for index in range(80))
    chunks = chunk_text(long_text, words_per_chunk=30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)


def test_chunk_text_on_empty_input():
    assert chunk_text("") == []


def test_compare_methods_returns_every_score():
    output = compare_methods(RESUME, RELATED_JOB, ["Python", "Flask"], ["Python", "AWS"])
    assert set(output) >= {"tfidf", "semantic", "keyword", "semantic_backend"}
    assert all(0.0 <= output[key] <= 1.0 for key in ("tfidf", "semantic", "keyword"))


def test_very_long_documents_do_not_blow_up():
    long_resume = RESUME * 400
    score = calculate_semantic_similarity(long_resume, RELATED_JOB)
    assert 0.0 <= score <= 1.0
