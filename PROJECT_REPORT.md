# NLP-Based Resume-Job Matching & Skill Gap Analyzer

**Project Report**

---

## 1. Abstract

This project presents a resume-job matching system built with Natural Language
Processing techniques. Given a resume and a job description, the system extracts skills,
experience and education from both documents, normalises equivalent terminology,
measures similarity using both a lexical method (TF-IDF with cosine similarity) and a
semantic method (sentence-level embeddings), and combines these signals into an
interpretable match score with a skill gap report.

The work is deliberately positioned as an **NLP project rather than a Generative AI
project**. No large language model is used at any stage. No external API is called, no
API key is required, and no retrieval-augmented generation is involved. Every output
the system produces — including the natural-language explanations of the score — is
assembled deterministically from structured intermediate results, so the same inputs
always yield the same report.

The system is evaluated on 35 hand-labelled resume-job pairs, comparing exact keyword
matching, TF-IDF similarity, semantic similarity and the combined pipeline using
classification metrics (precision, recall, F1), rank correlation and ranking metrics
(Precision@1, MRR, NDCG@3). The complete application deploys to Streamlit Community
Cloud with no configuration beyond selecting the repository.

---

## 2. Introduction

Screening resumes against job descriptions is repetitive, high-volume work. A recruiter
reads a job description once and then reads dozens of resumes against it, holding the
requirements in working memory. Candidates face the mirror image of the problem: they
cannot easily tell which of their skills a given posting actually values, or what they
are missing.

Software has addressed this for decades through Applicant Tracking Systems, most of
which rely on keyword search. Keyword search is fast and explainable, but it is brittle
in exactly the way natural language is flexible. A resume that says "developed RESTful
APIs using Flask" and a job description that asks for "experience building web APIs with
Python frameworks" describe the same competence with almost no shared vocabulary.

Natural Language Processing offers tools for this gap: normalisation to collapse
different surface forms of one concept, phrase matching to treat multi-word terms as
units, and distributional semantics to compare meaning rather than characters. This
project applies those tools to resume-job matching and measures whether they help.

---

## 3. Problem Statement

Build a system that, given a resume and a job description in unstructured text form:

1. extracts the text reliably from PDF, DOCX and TXT sources,
2. identifies the skills, experience and education each document contains,
3. recognises that differently-worded mentions refer to the same skill,
4. distinguishes a job's required skills from its preferred skills,
5. quantifies how well the resume matches the job,
6. reports which specific skills are missing, and
7. explains the score in terms a non-technical user can act on.

The system must not fabricate requirements that a job description does not state, and it
must not fail on malformed input.

**Research question.** Can semantic NLP techniques improve resume-job matching compared
with traditional keyword-based matching?

**Hypothesis.** Semantic methods will identify relationships between differently-worded
resume and job statements that exact keyword matching misses, and a combination of
lexical and semantic evidence will rank candidates more reliably than either alone.

---

## 4. Existing System

Three approaches dominate current practice.

**Manual screening.** A human reads both documents. Accurate in principle, but slow,
inconsistent across reviewers, and impractical at volume.

**Keyword-based Applicant Tracking Systems.** The dominant automated approach. A job's
keywords are searched for in resume text. Fast and transparent, but:

- it misses synonyms and abbreviations (`ML` vs `Machine Learning`, `Postgres` vs
  `PostgreSQL`),
- it misses paraphrase entirely (`RESTful APIs in Flask` vs `web APIs in Python`),
- it treats multi-word terms as separate tokens unless specially handled, so a resume
  containing "machine" and "learning" in unrelated sentences can score a false match,
- it can be gamed by keyword stuffing.

**Commercial AI matching platforms.** Increasingly built on large language models.
These handle paraphrase well but introduce dependencies that make them unsuitable for
this project's constraints: they require paid API access and network connectivity, their
outputs are non-deterministic, and their reasoning cannot be audited — an LLM's stated
justification for a score is generated text, not a record of the computation.

**Gap addressed.** This project targets the space between the second and third
approaches: better-than-keyword matching that remains fully local, deterministic and
auditable.

---

## 5. Proposed System

A modular NLP pipeline in which each stage produces inspectable structured output.

Key design decisions:

1. **Dictionary-driven skill extraction with phrase matching.** A curated vocabulary of
   252 skills and 223 aliases, matched longest-first over a token index. Multi-word
   skills are matched as units. Because the vocabulary is an editable CSV, the system's
   knowledge is visible and extensible rather than opaque.

2. **Explicit normalisation.** Aliases collapse to canonical names before any
   comparison, so matching operates on concepts rather than strings.

3. **Two similarity signals.** TF-IDF cosine similarity as the lexical baseline, and
   sentence embeddings for semantic similarity. Both are reported separately so their
   contributions stay visible.

4. **Graceful degradation.** Every heavy dependency is optional. Without spaCy, a
   rule-based lemmatiser is used. Without sentence-transformers, semantic similarity
   falls back to Latent Semantic Analysis. The full test suite passes in both
   configurations, which is what makes the project deployable on a free tier.

5. **Honest handling of missing information.** When a job states no experience
   requirement, that component is not scored zero — its weight is redistributed across
   the components that could be computed, and the report says so.

6. **Deterministic explanation.** Explanations are assembled from computed values by
   template. This is a smaller capability than LLM-generated prose, and a more
   trustworthy one: the explanation cannot disagree with the score.

---

## 6. Objectives

1. Extract text from PDF, DOCX and TXT resumes without crashing on malformed input.
2. Preprocess text while preserving the original for context-dependent extraction.
3. Extract skills, education, experience and certifications.
4. Normalise equivalent skill names case-insensitively.
5. Separate required from preferred skills in job descriptions.
6. Implement TF-IDF cosine similarity as a baseline.
7. Implement semantic similarity using sentence embeddings.
8. Produce an interpretable weighted score with configurable weights.
9. Identify matched, missing and additional skills.
10. Explain every score deterministically.
11. Evaluate semantic against keyword matching on labelled data.
12. Deploy to a free hosting platform with no API keys.

---

## 7. Methodology

### 7.1 Pipeline

```
Resume                                    Job description
  |                                              |
  v                                              v
Text extraction (PyMuPDF / python-docx)   Text extraction
  |                                              |
  v                                              v
Cleaning: unicode folding, ligature repair, whitespace normalisation
  |                                              |
  v                                              v
Preprocessing: tokenise, stop-words, lemmatise   Section detection
(original_text preserved)                        |
  |                                              v
  v                                     Requirement extraction
Skill extraction (phrase matching)      (required / preferred /
Experience extraction                    experience / education)
Education extraction                             |
  |                                              |
  +----------------------+-----------------------+
                         v
                Skill normalisation
                         |
         +---------------+---------------+
         v                               v
  TF-IDF similarity            Semantic similarity
         |                               |
         +---------------+---------------+
                         v
                  Scoring engine
                         |
                         v
              Skill gap analysis
                         |
                         v
                  Final report
```

### 7.2 Handling ambiguity

Two problems required specific treatment.

**Short ambiguous skill names.** `R`, `C`, `Go` and `SAS` are legitimate technologies
and also common English words or letters. Naive matching turns "I will go to the office"
into a Go programming match. These entries are marked `strict` in the vocabulary and
match only as standalone tokens with their original capitalisation, with neighbouring
characters checked to reject `R&D` and `C.V.`.

**Slash-joined tokens.** `CI/CD` and `A/B testing` require the slash to be part of the
token, but `Python/Java` requires it to be a separator. The tokeniser keeps a slash only
when the resulting token is itself a known skill, and splits otherwise.

### 7.3 Required versus preferred

Three signals are combined, in priority order:

1. **Section headings.** "Required Skills", "Nice to Have", "Preferred Qualifications".
   Heading detection tolerates the lowercase function words real headings contain — an
   early version failed on "Nice to Have" because "to" is not capitalised.
2. **Inline markers.** Markers are split into strong ("a plus", "bonus", "preferred")
   and weak ("familiarity with", "exposure to"). Inside a Requirements section only
   strong markers can demote a skill, because "Familiarity with Git" listed under
   Requirements is still a requirement.
3. **Fallback.** With no usable signal, all detected skills are treated as required by
   inference, preferred is reported as "Not explicitly specified", and the interface
   warns the user that inference occurred.

Skills appearing only in responsibilities or company description are recorded separately
rather than promoted into the required list. Promoting them would invent requirements
and penalise candidates for them.

---

## 8. NLP Techniques Used

| Technique | Where | Purpose |
|---|---|---|
| Text extraction | `parser.py` | PDF/DOCX/TXT to plain text |
| Unicode normalisation | `preprocessing.py` | Fold accents, repair PDF ligatures |
| Tokenisation | `preprocessing.py` | Split text, preserving `c++`, `node.js` |
| Stop-word removal | `preprocessing.py` | NLTK list, with built-in fallback |
| Lemmatisation | `preprocessing.py` | spaCy, with rule-based fallback |
| Sentence segmentation | `preprocessing.py` | Bullet-aware splitting |
| Phrase matching | `skill_extractor.py` | Longest-match multi-word skills |
| Fuzzy string matching | `skill_normalizer.py` | Misspellings in Skills sections |
| Named entity recognition | `preprocessing.py` | spaCy NER for organisations/dates |
| Pattern extraction | `experience.py` | `3+ years`, `2 to 4 years`, `freshers` |
| Ontology normalisation | `education.py` | `B.Tech CSE` to Bachelor's + Comp Sci |
| TF-IDF vectorisation | `similarity.py` | Word unigrams and bigrams, sublinear tf |
| Cosine similarity | `similarity.py` | Vector comparison |
| Sentence embeddings | `similarity.py` | Semantic similarity (MiniLM) |
| Latent Semantic Analysis | `similarity.py` | Truncated SVD fallback |
| Jaccard similarity | `similarity.py` | Exact keyword baseline |

**Explicitly not used:** large language models, generative text, RAG, chatbot
interfaces, external AI APIs.

---

## 9. System Architecture

| Module | Responsibility |
|---|---|
| `config.py` | Paths, limits, weights, thresholds |
| `parser.py` | Document text extraction and error handling |
| `preprocessing.py` | Cleaning, tokenisation, lemmatisation |
| `sections.py` | Heading classification and document segmentation |
| `skill_normalizer.py` | Vocabulary loading, alias resolution |
| `skill_extractor.py` | Phrase-matching skill extraction |
| `job_analyzer.py` | Required/preferred/experience/education extraction |
| `experience.py` | Experience parsing and comparison |
| `education.py` | Education parsing and comparison |
| `similarity.py` | TF-IDF and semantic similarity |
| `scoring.py` | Weighted scoring, skill gap, explanations |
| `pipeline.py` | Orchestration; the single entry point |
| `evaluation.py` | Method comparison and metrics |
| `app/streamlit_app.py` | Web interface |

The user interface, the tests and the evaluation script all call `pipeline.py`, so there
is exactly one definition of how a match is computed.

---

## 10. Dataset

| Asset | Count | Notes |
|---|---|---|
| Canonical skills | 252 | 13 categories |
| Aliases | 223 | Abbreviations and spelling variants |
| Sample resumes | 7 | 5 TXT, 1 PDF, 1 DOCX |
| Job descriptions | 5 | Varied formatting |
| Labelled pairs | 35 | Graded relevance 0/1/2 |

The seven resumes cover a data science fresher, a backend developer, a frontend
developer, a data analyst, a DevOps engineer, a full-stack developer and a QA automation
engineer. The last two were added deliberately after an initial version of the dataset
proved too easy: with five clearly separated profiles, every method scored near
ceiling and the comparison was uninformative. The full-stack profile is a genuine strong
match for two different jobs, and the QA profile shares substantial vocabulary
(Python, SQL, pytest, Jenkins, CI/CD) with jobs it is only a partial fit for — which is
exactly the case where keyword overlap should mislead.

All data is fictional. Names, companies and institutions are invented.

---

## 11. Algorithms

### 11.1 Phrase matching

Longest-match-first scan over a token index built from the vocabulary:

```
for position in token_stream:
    for size in range(max_phrase_length, 0, -1):
        if tokens[position : position + size] in phrase_index:
            emit match; position += size; break
    else:
        position += 1
```

Complexity is O(n × k) for n tokens and maximum phrase length k (k = 4 here), so a
long resume is processed in milliseconds with no model loading.

### 11.2 TF-IDF cosine similarity

Both documents are vectorised with word unigrams and bigrams and sublinear term
frequency scaling. Bigrams matter here: they let "machine learning" contribute as a
single feature.

$$\text{sim}(A,B) = \frac{\vec{A} \cdot \vec{B}}{\|\vec{A}\| \|\vec{B}\|}$$

### 11.3 Semantic similarity

Both documents are split into sentence-aware chunks of roughly 60 words and each chunk
is embedded. Embedding a whole document as one vector averages distinct competences into
an uninformative centroid; chunking preserves local meaning.

Given the chunk-similarity matrix M:

$$\text{score} = 0.5\,\overline{\max_{\text{rows}} M} + 0.25\,\overline{\max_{\text{cols}} M} + 0.25\,\overline{M}$$

The first term measures how well the resume covers each job requirement — the quantity
that matters most. The second measures the reverse direction, and the third is overall
document similarity. Weighting job coverage highest while still including the reverse
direction prevents a padded resume, which would trivially cover everything, from scoring
well on length alone.

**Fallback backend.** When sentence-transformers is unavailable, chunks are vectorised
with word and character n-gram TF-IDF, reduced by Truncated SVD to at most 128
dimensions, and compared with the same formula. This is Latent Semantic Analysis: it
captures loose wording overlap without any download. Raw LSA cosine values occupy a
narrower range than transformer values (roughly 0.15 versus 0.55 for the same clearly
relevant pair), so a documented monotonic exponent rescales them for comparability.
The transformation preserves ranking and changes only spread.

### 11.4 Scoring

$$\text{Score} = 100 \sum_{i \in \text{available}} w_i' \cdot v_i, \qquad w_i' = \frac{w_i}{\sum_{j \in \text{available}} w_j}$$

| Component | Default weight |
|---|---|
| Required skill coverage | 0.40 |
| Preferred skill coverage | 0.15 |
| Semantic similarity | 0.25 |
| TF-IDF similarity | 0.10 |
| Experience match | 0.05 |
| Education match | 0.05 |

**These weights are configurable project assumptions.** They were chosen so that
concrete skill evidence outweighs free-text similarity. They have not been shown to be
optimal, and no such claim is made. They are exposed as sidebar sliders precisely
because they are a judgement call.

The renormalisation over available components is the important detail: when a job states
no experience requirement, that component is skipped and its weight redistributed rather
than scored zero. Otherwise a resume would be penalised for information the job never
requested.

Bands: 0-39 Weak, 40-69 Moderate, 70-84 Strong, 85-100 Excellent. Also configurable.

---

## 12. Implementation

Roughly 3,000 lines of Python across 13 source modules, a Streamlit application and 100
tests. Every public function carries type hints and a docstring. `black` and `flake8`
configurations are included.

**Error handling.** All document failures are converted to `DocumentParsingError` with a
message written for a user rather than a developer. A scanned PDF produces "Almost no
text could be read from this PDF. It is probably a scanned image" rather than a stack
trace. Handled cases: empty files, corrupted PDFs, password-protected PDFs, scanned
PDFs, legacy `.doc` files, unsupported extensions, oversized uploads and encoding
problems.

**Security.** Uploaded files are never executed, never written to disk and never stored.
Text is processed in memory for the duration of one analysis. No personal data is
collected and no credentials are required.

**Testing.** 100 tests covering parsing (14), extraction and normalisation (28),
similarity (17), scoring (26) and end-to-end integration (15). The integration suite
runs all 35 resume-job combinations. The whole suite passes both with and without the
optional dependencies installed, which is how the fallback claims are verified rather
than assumed.

---

## 13. Evaluation

### 13.1 Method

Four methods were compared on the 35 labelled pairs:

1. **keyword** — Jaccard overlap of extracted skill sets
2. **tfidf** — TF-IDF cosine similarity
3. **semantic** — chunked semantic similarity (LSA backend in these runs)
4. **pipeline** — the full weighted score

Because the four produce values on different scales, each is evaluated at the threshold
maximising its own F1. Ranking metrics, which are threshold-free, are reported alongside.

### 13.2 Results

| Method | Threshold | Accuracy | Precision | Recall | F1 | Spearman | Separation |
|---|---|---|---|---|---|---|---|
| keyword | 0.19 | 0.97 | 0.88 | 1.00 | 0.93 | 0.82 | 0.41 |
| tfidf | 0.07 | 0.97 | 0.88 | 1.00 | 0.93 | 0.84 | 0.09 |
| semantic | 0.51 | 0.97 | 0.88 | 1.00 | 0.93 | 0.76 | 0.14 |
| pipeline | 0.58 | 1.00 | 1.00 | 1.00 | 1.00 | 0.84 | 0.40 |

Ranking quality (resumes ranked per job):

| Method | Precision@1 | MRR | NDCG@3 |
|---|---|---|---|
| keyword | 0.80 | 0.90 | 0.95 |
| tfidf | 1.00 | 1.00 | 1.00 |
| semantic | 1.00 | 1.00 | 0.93 |
| pipeline | 1.00 | 1.00 | 1.00 |

Score band agreement with human labels: **86% exact**, **100% within one band**.

### 13.3 Interpretation

The results support the hypothesis in part, and the honest reading matters more than a
favourable one.

**Where semantic matching helps.** Exact keyword overlap puts the wrong resume first for
one of the five jobs (Precision@1 0.80), while semantic similarity, TF-IDF and the full
pipeline all rank the correct resume first every time. This is the predicted failure
mode of keyword matching: the QA automation resume shares heavy surface vocabulary
(Python, SQL, pytest, Jenkins, CI/CD) with the backend and DevOps jobs without being a
strong fit for either, and pure skill-set overlap over-rewards it.

**Where the hypothesis is not supported.** Semantic similarity does not beat TF-IDF
here. The two are tied on every classification metric, and TF-IDF is slightly ahead on
NDCG@3. This is a real result and should not be explained away. Two factors account for
it: the sample documents use conventional technical vocabulary, which is precisely the
regime where lexical overlap works well; and the semantic scores come from the LSA
fallback rather than a transformer model, which limits the paraphrase sensitivity the
hypothesis is about. Testing the hypothesis properly requires the transformer backend
and document pairs with deliberately divergent phrasing.

**Where the pipeline earns its complexity.** The combined pipeline is the only method
that classifies all 35 pairs correctly (F1 1.00) while also ranking perfectly, and its
separation — the gap between mean scores on relevant and irrelevant pairs — is 0.40
against 0.09 for TF-IDF. That difference is the practical one. TF-IDF ranks correctly but
compresses every pair into a narrow band around 0.1, so no fixed threshold generalises
and the raw number means little to a user. The pipeline produces values that are
interpretable as scores, which is what makes a threshold like "70 = Strong Match"
possible at all.

### 13.4 Threats to validity

- **Dataset size.** 35 pairs from 7 resumes and 5 jobs. Small enough that a single
  changed label moves the metrics visibly.
- **Author-labelled.** The same person built the system and assigned the labels.
  Independent labelling would be stronger.
- **In-sample thresholds.** Each method's threshold is selected on the same pairs it is
  scored on, with no held-out split. Every number above is optimistic.
- **Backend.** Semantic results use the LSA fallback, not the transformer model, so the
  semantic row is a lower bound on what the technique can do.

These are limitations of the evaluation, not caveats added for form. The correct summary
is that the method is demonstrated and the comparison is suggestive, not established.

---

## 14. Results

The delivered system meets its objectives:

- Text extraction works for PDF, DOCX and TXT, with tested failure handling
- 252 skills and 223 aliases, with phrase matching for multi-word terms
- Required/preferred separation from headings and inline markers
- Structured experience and education extraction that returns "not specified" rather
  than a fabricated value
- Both similarity methods implemented and reported separately
- Interpretable score with per-component breakdown and effective weights shown
- Deterministic explanations
- 100 passing tests, in two dependency configurations
- Deploys to Streamlit Community Cloud with no API key

Example (backend developer resume against backend engineer job):

```
Overall match: 81.7%              Strong Match
Required skill coverage:  100%    (8/8 matched)
Preferred skill coverage: 100%    (5/5 matched)
Semantic similarity: 60.9%        TF-IDF similarity: 14.4%
Experience: 3.1 years meets the 2-year requirement
Education: Bachelor's meets the Bachelor's requirement; field of study matches
Missing skills: none
```

The score stops short of 100 because the two free-text similarity components are
inherently partial: a resume and a job description are different genres of writing and
never reach high lexical overlap. This is the intended behaviour of the weighting, not a
deficiency — a perfect 100 should require near-identical documents, not merely a good
candidate.

---

## 15. Limitations

1. **Dictionary-bound extraction.** Skills absent from `skills.csv` cannot be found.
   Coverage is good for common technical skills, not exhaustive.
2. **Unvalidated weights.** The weighting reflects a judgement about what should matter,
   not a fitted model.
3. **Small evaluation set.** See section 13.4.
4. **English only.**
5. **No OCR.** Scanned PDFs are detected and rejected with a clear message, not read.
6. **Layout sensitivity.** Multi-column PDF layouts can interleave text during
   extraction.
7. **Semantic similarity is not suitability.** Textual similarity is a proxy for
   relevance, and a lossy one.
8. **Fairness.** Automated screening can systematically disadvantage candidates whose
   vocabulary differs from the dominant pattern in the training vocabulary — non-native
   speakers, career changers, candidates from less conventional backgrounds. A
   dictionary-based system makes this visible (the dictionary can be inspected) but does
   not eliminate it. The system is a screening aid that identifies skill gaps; it should
   not be used to reject candidates automatically.

---

## 16. Future Scope

1. Learn weights from a larger labelled dataset by regression, replacing assumption with
   estimation.
2. Grow the skill dictionary automatically by mining job-posting corpora.
3. Add OCR for scanned documents.
4. Weight skills by section, so a skill demonstrated in Experience counts more than one
   listed under Interests.
5. Batch ranking of many resumes against one job.
6. Multilingual support.
7. Bias auditing: measure score differences across resumes that are equivalent in
   substance but differ in phrasing style.
8. A proper held-out evaluation split once the dataset is large enough to support one.

---

## 17. Conclusion

This project delivers a working, deployable resume-job matching system built entirely
from traditional and semantic NLP techniques, with no dependence on generative AI. It
extracts and normalises structured information from unstructured documents, compares
them by two complementary similarity methods, and produces an interpretable score with a
specific, actionable skill gap report.

The evaluation shows the combined pipeline ranks candidates correctly on the labelled
set and separates good from poor matches far more cleanly than either similarity method
alone. It does not show that semantic similarity beats TF-IDF — on this small dataset,
with the fallback backend, it does not. That result is reported as measured rather than
adjusted, and section 13.4 states what a stronger test would require.

The engineering priority throughout was *simple, reliable, explainable, deployable* over
*complex and impressive*. The clearest expression of that priority is the fallback
design: the application runs with no model downloads at all, degrading capability rather
than failing, which is what makes it deployable on free hosting by a student with no
budget and no API key.

---

## 18. References

1. Jurafsky, D. and Martin, J. H. *Speech and Language Processing*, 3rd edition draft.
2. Manning, C. D., Raghavan, P. and Schütze, H. *Introduction to Information Retrieval*.
   Cambridge University Press, 2008.
3. Salton, G. and Buckley, C. "Term-weighting approaches in automatic text retrieval."
   *Information Processing & Management*, 24(5), 1988.
4. Deerwester, S. et al. "Indexing by latent semantic analysis." *Journal of the American
   Society for Information Science*, 41(6), 1990.
5. Reimers, N. and Gurevych, I. "Sentence-BERT: Sentence Embeddings using Siamese
   BERT-Networks." *EMNLP-IJCNLP*, 2019.
6. Devlin, J. et al. "BERT: Pre-training of Deep Bidirectional Transformers for Language
   Understanding." *NAACL-HLT*, 2019.
7. Pedregosa, F. et al. "Scikit-learn: Machine Learning in Python." *JMLR*, 12, 2011.
8. Honnibal, M. and Montani, I. spaCy: Industrial-strength Natural Language Processing.
9. Bird, S., Klein, E. and Loper, E. *Natural Language Processing with Python*.
   O'Reilly, 2009.
10. Järvelin, K. and Kekäläinen, J. "Cumulated gain-based evaluation of IR techniques."
    *ACM TOIS*, 20(4), 2002.
11. Streamlit documentation. https://docs.streamlit.io
12. PyMuPDF documentation. https://pymupdf.readthedocs.io
