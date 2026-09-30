"""Tests for the scoring engine, skill gap analysis and the extractors it uses."""

from __future__ import annotations

from src.config import MatchThresholds, ScoringWeights
from src.education import (
    extract_education,
    extract_education_requirement,
    match_education,
)
from src.experience import (
    extract_candidate_experience,
    match_experience,
    parse_experience_requirement,
)
from src.scoring import (
    analyze_skill_gap,
    build_explanations,
    calculate_score,
    find_additional_skills,
)


def _score(**overrides):
    defaults = dict(
        required_gap=analyze_skill_gap(["Python", "SQL"], ["Python", "SQL", "AWS"]),
        preferred_gap=analyze_skill_gap(["Python"], ["Docker"]),
        semantic_similarity=0.5,
        tfidf_similarity=0.3,
        experience_score=1.0,
        education_score=1.0,
    )
    defaults.update(overrides)
    return calculate_score(**defaults)


# --------------------------------------------------------------------------- #
# Skill gap
# --------------------------------------------------------------------------- #


def test_skill_gap_splits_matched_and_missing():
    gap = analyze_skill_gap(["Python", "SQL", "Git"], ["Python", "SQL", "AWS", "Docker"])
    assert gap.matched == ["Python", "SQL"]
    assert gap.missing == ["AWS", "Docker"]
    assert gap.total == 4
    assert gap.coverage == 0.5
    assert gap.coverage_percent == 50.0


def test_skill_gap_with_no_job_skills_has_zero_coverage():
    gap = analyze_skill_gap(["Python"], [])
    assert gap.total == 0
    assert gap.coverage == 0.0
    assert "No skills" in gap.summary()


def test_skill_gap_ignores_duplicate_job_skills():
    gap = analyze_skill_gap(["Python"], ["Python", "Python", "AWS"])
    assert gap.total == 2


def test_additional_skills_are_those_the_job_never_mentions():
    extra = find_additional_skills(["Python", "Flask", "Git"], ["Python", "AWS"])
    assert extra == ["Flask", "Git"]


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def test_score_is_between_zero_and_one_hundred():
    result = _score()
    assert 0.0 <= result.overall_score <= 100.0


def test_perfect_match_scores_one_hundred():
    result = calculate_score(
        required_gap=analyze_skill_gap(["Python", "SQL"], ["Python", "SQL"]),
        preferred_gap=analyze_skill_gap(["Docker"], ["Docker"]),
        semantic_similarity=1.0,
        tfidf_similarity=1.0,
        experience_score=1.0,
        education_score=1.0,
    )
    assert result.overall_score == 100.0
    assert result.classification == "Excellent Match"


def test_nothing_matching_scores_zero():
    result = calculate_score(
        required_gap=analyze_skill_gap([], ["Python", "SQL"]),
        preferred_gap=analyze_skill_gap([], ["Docker"]),
        semantic_similarity=0.0,
        tfidf_similarity=0.0,
        experience_score=0.0,
        education_score=0.0,
    )
    assert result.overall_score == 0.0
    assert result.classification == "Weak Match"


def test_unavailable_components_have_their_weight_redistributed():
    result = _score(experience_score=None, education_score=None)
    available = [c for c in result.components if c.available]
    assert abs(sum(c.effective_weight for c in available) - 1.0) < 1e-6
    assert all(c.effective_weight == 0.0 for c in result.components if not c.available)
    assert "Experience match" in result.skipped_components


def test_unknown_component_is_better_than_scoring_it_zero():
    # Redistribution must not act as a penalty: an unknown experience component
    # should never be treated as a zero score.
    unknown = _score(experience_score=None, education_score=None)
    scored_as_zero = _score(experience_score=0.0, education_score=0.0)
    assert unknown.overall_score > scored_as_zero.overall_score


def test_redistribution_preserves_a_uniform_score():
    # When every available component scores the same, dropping one changes
    # nothing -- the weights simply re-normalise around the rest.
    uniform = dict(
        required_gap=analyze_skill_gap(["Python", "SQL"], ["Python", "SQL"]),
        preferred_gap=analyze_skill_gap(["Docker"], ["Docker"]),
        semantic_similarity=1.0,
        tfidf_similarity=1.0,
    )
    with_all = calculate_score(experience_score=1.0, education_score=1.0, **uniform)
    without_two = calculate_score(experience_score=None, education_score=None, **uniform)
    assert with_all.overall_score == without_two.overall_score == 100.0


def test_classification_bands_follow_the_thresholds():
    thresholds = MatchThresholds()
    assert thresholds.classify(10) == "Weak Match"
    assert thresholds.classify(50) == "Moderate Match"
    assert thresholds.classify(75) == "Strong Match"
    assert thresholds.classify(95) == "Excellent Match"


def test_thresholds_are_configurable():
    strict = MatchThresholds(weak_max=60, moderate_max=80, strong_max=95)
    assert strict.classify(50) == "Weak Match"
    assert strict.classify(90) == "Strong Match"


def test_weights_are_configurable_and_change_the_score():
    skill_heavy = ScoringWeights().replace(required_skills=0.8, semantic_similarity=0.0)
    baseline = _score()
    adjusted = _score(weights=skill_heavy)
    assert adjusted.overall_score != baseline.overall_score


def test_unknown_weight_name_is_rejected():
    try:
        ScoringWeights().replace(not_a_weight=1.0)
    except ValueError as exc:
        assert "not_a_weight" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_component_contributions_sum_to_the_overall_score():
    result = _score()
    total = sum(component.contribution for component in result.components)
    assert abs(total - result.overall_score) < 0.1


def test_result_serialises_to_a_dictionary():
    payload = _score().to_dict()
    assert payload["overall_score"] >= 0
    assert "required" in payload and "components" in payload


# --------------------------------------------------------------------------- #
# Explanations
# --------------------------------------------------------------------------- #


def test_explanations_are_deterministic_and_reference_real_numbers():
    result = _score()
    result.additional_skills = ["Flask"]
    first_reasons, first_gaps = build_explanations(result)
    second_reasons, second_gaps = build_explanations(result)
    assert first_reasons == second_reasons
    assert first_gaps == second_gaps
    assert any("2/3 required skills" in reason for reason in first_reasons)
    assert any("AWS" in gap for gap in first_gaps)


# --------------------------------------------------------------------------- #
# Experience
# --------------------------------------------------------------------------- #


def test_experience_requirement_patterns():
    assert parse_experience_requirement("3+ years of experience").min_years == 3
    assert parse_experience_requirement("2 to 4 years of experience").max_years == 4
    assert parse_experience_requirement("minimum 1 year of experience").min_years == 1
    assert parse_experience_requirement("Freshers are welcome").accepts_freshers is True


def test_unstated_experience_requirement_is_not_invented():
    requirement = parse_experience_requirement("We build retail software in Bengaluru.")
    assert requirement.specified is False
    assert requirement.min_years is None
    assert requirement.describe() == "Not explicitly specified"


def test_candidate_experience_from_statement_and_date_ranges():
    from_statement = extract_candidate_experience("Backend developer with 3 years of experience.")
    assert from_statement.total_years == 3

    from_dates = extract_candidate_experience("Engineer, Acme (Jan 2020 - Dec 2022)")
    assert from_dates.total_years is not None
    assert 2.5 <= from_dates.total_years <= 3.5


def test_candidate_experience_absent_returns_none():
    candidate = extract_candidate_experience("Skills: Python, SQL")
    assert candidate.total_years is None
    assert candidate.describe() == "Not explicitly specified"


def test_experience_match_scores_and_skips_correctly():
    requirement = parse_experience_requirement("3+ years of experience")
    enough = extract_candidate_experience("I have 5 years of experience.")
    assert match_experience(requirement, enough).score == 1.0

    short = extract_candidate_experience("I have 1 year of experience.")
    match = match_experience(requirement, short)
    assert 0.0 < match.score < 1.0

    unknown = extract_candidate_experience("Skills: Python")
    assert match_experience(requirement, unknown).score is None


# --------------------------------------------------------------------------- #
# Education
# --------------------------------------------------------------------------- #


def test_education_normalisation_maps_btech_to_bachelors_and_field():
    info = extract_education("EDUCATION\nB.Tech CSE, Example University, 2022")
    assert info.level == 2
    assert info.level_name == "Bachelor's"
    assert "Computer Science" in info.fields


def test_education_levels_are_ordered():
    assert extract_education("PhD in Computer Science").level == 4
    assert extract_education("M.Tech in Computer Science").level == 3
    assert extract_education("Diploma in Computer Engineering").level == 1


def test_education_match_accepts_related_fields():
    requirement = extract_education_requirement(
        "Education: Bachelor's degree in Computer Science or a related field."
    )
    candidate = extract_education("EDUCATION\nB.E. in Information Technology, 2021")
    match = match_education(requirement, candidate)
    assert match.score is not None and match.score >= 0.9


def test_education_match_is_skipped_when_unknown():
    requirement = extract_education_requirement("We build retail software.")
    candidate = extract_education("Skills: Python")
    assert match_education(requirement, candidate).score is None
