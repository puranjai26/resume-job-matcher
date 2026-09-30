"""Tests for skill extraction, normalisation and section detection."""

from __future__ import annotations

from src.job_analyzer import NOT_SPECIFIED, analyze_job_description, extract_job_title
from src.preprocessing import preprocess_text, simple_lemmatize, tokenize
from src.sections import classify_heading, split_sections
from src.skill_extractor import extract_skills, extract_skills_detailed
from src.skill_normalizer import get_vocabulary, normalize_skill, normalize_surface

# --------------------------------------------------------------------------- #
# Vocabulary
# --------------------------------------------------------------------------- #


def test_vocabulary_has_at_least_200_skills():
    vocab = get_vocabulary()
    assert len(vocab) >= 200


def test_vocabulary_covers_the_required_categories():
    categories = {vocab_entry.category for vocab_entry in get_vocabulary().entries.values()}
    for expected in (
        "Programming",
        "Data Science",
        "Machine Learning",
        "Web Development",
        "Databases",
        "Cloud",
        "DevOps",
        "Cybersecurity",
        "Testing",
        "Mobile Development",
        "Business Analytics",
        "Tools",
    ):
        assert expected in categories


def test_normalize_surface_is_punctuation_insensitive():
    assert normalize_surface("Node.js") == normalize_surface("NODE JS") == "node js"
    assert normalize_surface("scikit-learn") == "scikit learn"


def test_skill_normalization_resolves_abbreviations():
    assert normalize_skill("ML") == "Machine Learning"
    assert normalize_skill("nlp") == "Natural Language Processing"
    assert normalize_skill("JS") == "JavaScript"
    assert normalize_skill("k8s") == "Kubernetes"
    assert normalize_skill("Postgres") == "PostgreSQL"
    assert normalize_skill("aws cloud") == "AWS"


def test_skill_normalization_is_case_insensitive():
    assert normalize_skill("PYTHON") == normalize_skill("python") == "Python"


def test_unknown_skill_returns_none():
    assert normalize_skill("underwater basket weaving") is None


def test_fuzzy_normalization_handles_close_spellings():
    vocab = get_vocabulary()
    assert vocab.fuzzy_normalize("kubernets") == "Kubernetes"
    assert vocab.fuzzy_normalize("javscript") == "JavaScript"
    # Too short to judge by edit distance.
    assert vocab.fuzzy_normalize("xy") is None


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #


def test_multi_word_skills_are_matched_as_phrases():
    skills = extract_skills("I have experience with machine learning and deep learning.")
    assert "Machine Learning" in skills
    assert "Deep Learning" in skills


def test_aliases_are_normalised_during_extraction():
    skills = extract_skills("Worked on ML and NLP projects using JS and k8s.")
    assert "Machine Learning" in skills
    assert "Natural Language Processing" in skills
    assert "JavaScript" in skills
    assert "Kubernetes" in skills


def test_duplicate_and_differently_cased_mentions_collapse():
    result = extract_skills_detailed("python PYTHON Python and more python")
    assert result.skills.count("Python") == 1
    assert result.counts["Python"] >= 4


def test_punctuated_technology_names_survive():
    skills = extract_skills("Strong in C++, C#, Node.js, CI/CD and ASP.NET")
    for expected in ("C++", "C#", "Node.js", "CI/CD", "ASP.NET"):
        assert expected in skills


def test_slash_separated_list_is_split():
    skills = extract_skills("Languages: Python/Java/Go")
    assert "Python" in skills
    assert "Java" in skills


def test_ambiguous_short_skills_need_exact_casing():
    # "go" in prose must not become the Go language.
    assert "Go" not in extract_skills("I will go to the office and see the c-suite.")
    # Standalone capitalised mentions do count.
    assert "R" in extract_skills("Programming languages: Python, R, SQL")


def test_extraction_on_empty_text_returns_nothing():
    result = extract_skills_detailed("")
    assert result.skills == []
    assert result.mentions == []


def test_evidence_line_is_captured_for_each_skill():
    text = "SKILLS\nPython, SQL\nEXPERIENCE\nBuilt dashboards in Power BI for finance."
    result = extract_skills_detailed(text)
    assert "Power BI" in result.evidence
    assert "finance" in result.evidence["Power BI"]


def test_skills_are_grouped_by_category():
    grouped = extract_skills_detailed("Python, Docker, PostgreSQL").by_category()
    assert "Programming" in grouped
    assert "DevOps" in grouped
    assert "Databases" in grouped


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #


def test_preprocessing_keeps_the_original_text():
    original = "Built REST APIs using Flask at Acme Corp."
    processed = preprocess_text(original)
    assert processed.original_text == original
    assert processed.processed_text != original
    assert processed.processed_text == processed.processed_text.lower()


def test_preprocessing_of_empty_text_is_safe():
    processed = preprocess_text("")
    assert processed.is_empty
    assert processed.processed_text == ""
    assert processed.word_count == 0


def test_tokenizer_keeps_technology_punctuation():
    tokens = tokenize("C++ and Node.js and CI/CD")
    assert "c++" in tokens
    assert "node.js" in tokens or "node" in tokens


def test_simple_lemmatizer_is_conservative():
    assert simple_lemmatize("running") == "run"
    assert simple_lemmatize("apis") == "api"
    assert simple_lemmatize("data") == "data"


# --------------------------------------------------------------------------- #
# Sections and job analysis
# --------------------------------------------------------------------------- #


def test_headings_with_lowercase_function_words_are_detected():
    assert classify_heading("Nice to Have") == "preferred_skills"
    assert classify_heading("What You Will Do") == "responsibilities"
    assert classify_heading("Required Skills") == "required_skills"
    assert classify_heading("we build order management software") is None


def test_split_sections_returns_named_blocks():
    text = "SKILLS\nPython\n\nEDUCATION\nB.Tech Computer Science"
    sections = split_sections(text)
    assert "Python" in sections["skills"]
    assert "B.Tech" in sections["education"]


def test_job_title_extraction():
    assert extract_job_title("Job Title: Backend Engineer\nWe are hiring.") == "Backend Engineer"
    assert extract_job_title("") == NOT_SPECIFIED


def test_required_and_preferred_skills_are_separated():
    job = """Job Title: Backend Engineer

Requirements
- Strong command of Python
- Proficiency in SQL

Nice to Have
- Docker and AWS experience
"""
    analysis = analyze_job_description(job)
    assert "Python" in analysis.required_skills
    assert "SQL" in analysis.required_skills
    assert "Docker" in analysis.preferred_skills
    assert "AWS" in analysis.preferred_skills
    assert analysis.explicit_requirements is True


def test_unstructured_job_falls_back_without_inventing_preferences():
    job = "We need someone who knows Python and SQL to help our team ship features."
    analysis = analyze_job_description(job)
    assert "Python" in analysis.required_skills
    assert analysis.preferred_skills == []
    assert analysis.explicit_requirements is False
    assert analysis.describe_preferred() == NOT_SPECIFIED


def test_responsibility_only_skills_are_not_promoted_to_requirements():
    job = """Job Title: Data Analyst

Responsibilities
- Maintain dashboards in Tableau

Required Skills
- Advanced SQL
"""
    analysis = analyze_job_description(job)
    assert "SQL" in analysis.required_skills
    assert "Tableau" not in analysis.required_skills
    assert "Tableau" in analysis.other_skills


def test_empty_job_description_is_handled():
    analysis = analyze_job_description("")
    assert analysis.required_skills == []
    assert analysis.title == NOT_SPECIFIED
