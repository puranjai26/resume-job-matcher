# NLP-Based Resume-Job Matching & Skill Gap Analyzer

A web application that compares a resume against a job description, scores how well
they match, and reports exactly which skills are missing.

This is an **NLP project, not a Generative AI project**. There is no chatbot, no RAG,
no LLM and no API key anywhere in it. Every number and every sentence the application
shows is computed deterministically from the extracted text, so the same two documents
always produce the same report.

---

## Project overview

Upload a resume (PDF, DOCX or TXT), paste a job description, and the application:

1. extracts text from the document,
2. finds skills, experience and education in both documents,
3. normalises equivalent skill names (`ML` and `Machine Learning` become one skill),
4. separates the job's **required** skills from its **preferred** skills,
5. measures similarity with TF-IDF and with sentence-level semantic similarity,
6. combines the signals into a weighted score with an explanation of every component.

Real output from the bundled full-stack resume against the bundled backend job:

```
Overall match: 57.6%                     MODERATE MATCH

Matched required (6/8)                   Missing required
  Python, SQL, Flask                       Django
  PostgreSQL, REST API, Git                Code Review

                                         Missing preferred
Required skill coverage:  75%              AWS, Microservices
Preferred skill coverage: 20%              Unit Testing, pytest
Semantic similarity:      55%
TF-IDF similarity:         9%
Experience: 2.1 years meets the 2-year requirement
Education: Bachelor's meets the Bachelor's requirement; field of study matches
```

The candidate covers the core stack but misses Django specifically, which is what the
skill gap list is for. A score in the high 50s for a genuinely decent candidate is
expected: the two free-text similarity components never reach high values, because a
resume and a job description are different genres of writing.

## Problem statement

Recruiters and candidates compare resumes with job descriptions by hand. Keyword search
fails on this task because the same idea is written in different words:

| Resume says | Job says |
|---|---|
| `Developed RESTful APIs using Flask` | `Experience building web APIs with Python frameworks` |
| `ML` | `Machine Learning` |
| `Postgres` | `PostgreSQL` |
| `B.Tech CSE` | `Bachelor's degree in Computer Science` |

The research question this project addresses:

> Can semantic NLP techniques improve resume-job matching compared with traditional
> keyword-based matching?

`python -m src.evaluation` answers it on a labelled dataset and writes a report.

## Features

- Upload PDF, DOCX or TXT; or paste text; or load one of the bundled samples
- 252 canonical skills across 13 categories, with 223 aliases and abbreviations
- Phrase matching, so `machine learning` is one skill and never `machine` + `learning`
- Required vs preferred skill detection from section headings *and* inline markers
- Structured experience parsing (`3+ years`, `2 to 4 years`, `freshers`)
- Education normalisation (`B.Tech CSE` maps to Bachelor's + Computer Science)
- Configurable scoring weights and score bands, adjustable live in the sidebar
- Deterministic explanations of every score component
- An evaluation script comparing four matching methods on labelled pairs
- 100 automated tests
- Runs with no model download at all if you want it to

## NLP techniques used

| Stage | Technique |
|---|---|
| Text extraction | PyMuPDF, python-docx, encoding detection, ligature repair |
| Preprocessing | Unicode folding, tokenisation, stop-word removal, lemmatisation |
| Skill extraction | Longest-match phrase matching over a token index |
| Skill normalisation | Alias dictionary + fuzzy matching (`difflib`) for misspellings |
| Section detection | Heuristic heading classification with regex patterns |
| Requirement extraction | Section rules + inline marker detection |
| Experience/education | Pattern extraction, date-range merging, level normalisation |
| Baseline similarity | TF-IDF vectors (word unigrams + bigrams) + cosine similarity |
| Semantic similarity | Sentence embeddings, or LSA (TF-IDF + Truncated SVD) as a fallback |
| Named entity recognition | spaCy NER for organisations and dates, when spaCy is installed |
| Scoring | Weighted linear combination with weight redistribution |
| Evaluation | Precision, recall, F1, Spearman, Precision@1, MRR, NDCG@3 |

## Architecture

```
        Resume (PDF/DOCX/TXT)              Job description (text)
                 |                                  |
                 v                                  v
          src/parser.py                      src/parser.py
                 |                                  |
                 v                                  v
       src/preprocessing.py               src/preprocessing.py
       (original_text kept)                        |
                 |                                  v
                 v                        src/sections.py
      src/skill_extractor.py              src/job_analyzer.py
      src/experience.py                   (required / preferred /
      src/education.py                     experience / education)
                 |                                  |
                 +----------------+-----------------+
                                  |
                                  v
                         src/skill_normalizer.py
                                  |
                 +----------------+-----------------+
                 v                                  v
      TF-IDF similarity                   Semantic similarity
      (src/similarity.py)                 (src/similarity.py)
                 |                                  |
                 +----------------+-----------------+
                                  v
                          src/scoring.py
                    (weighted score + skill gap)
                                  |
                                  v
                          src/pipeline.py
                          -> MatchReport
                                  |
                                  v
                      app/streamlit_app.py
```

Each module has one responsibility, and `src/pipeline.py` is the only place that knows
how they fit together. The UI, the tests and the evaluation script all call it.

## Technology stack

Python 3.11+, Streamlit, scikit-learn, pandas, NumPy, SciPy, PyMuPDF, python-docx.
Optional: spaCy (lemmatisation, NER) and sentence-transformers (embeddings).
Tooling: pytest, black, flake8.

## Installation

```bash
git clone <your-repository-url>
cd resume-job-matcher

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

That is the complete install. The application runs at this point.

### Optional extras

```bash
# Better lemmatisation and sentence segmentation (~15 MB)
pip install spacy
python -m spacy download en_core_web_sm

# Embedding-based semantic similarity (pulls in PyTorch, ~800 MB installed)
pip install sentence-transformers
```

Both are genuinely optional. Without spaCy, a rule-based lemmatiser and a built-in
stop-word list take over. Without sentence-transformers, semantic similarity uses
Latent Semantic Analysis. The full test suite passes in both configurations.

## Running locally

```bash
streamlit run app/streamlit_app.py
```

Then open <http://localhost:8501>. Pick "Use a sample" for both inputs to see a full
report immediately without uploading anything.

```bash
# Run the tests
pytest

# Compare the matching methods on the labelled dataset
python -m src.evaluation

# Format and lint
black src app tests
flake8 src app tests
```

## Dataset

`data/` contains everything the application needs, all of it fictional.

| File | Contents |
|---|---|
| `skills.csv` | 252 canonical skills with category and match mode |
| `aliases.csv` | 223 alias -> canonical mappings |
| `sample_resumes/` | 7 resumes (5 TXT, 1 PDF, 1 DOCX) |
| `sample_jobs/` | 5 job descriptions |
| `labels.csv` | 35 hand-labelled resume-job pairs |

The resumes describe invented people at invented companies. No real personal data is
included, and none should be committed to the repository.

Add your own skills by appending a row to `data/skills.csv`
(`skill,category,match_mode`). Use `match_mode=strict` for short or ambiguous names
such as `R` or `Go`, which then only match as standalone capitalised tokens.

## Evaluation

```bash
python -m src.evaluation
```

Four methods are compared on the labelled pairs: exact keyword overlap, TF-IDF cosine
similarity, semantic similarity, and the full weighted pipeline. Results are written to
`reports/evaluation_report.md`.

Two honest caveats, which the generated report repeats: the dataset is small (35 pairs
labelled by the project author), and each method's threshold is chosen on the same pairs
it is then scored on, with no held-out split. The numbers show the *relative* behaviour
of the methods, not a general accuracy claim.

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md). The short version: push to GitHub, point Streamlit
Community Cloud at `app/streamlit_app.py`, deploy. No Docker, no API keys, no database.

## Limitations

Worth stating plainly, because a match score can look more authoritative than it is:

- **Skill extraction is dictionary-based.** A skill that is not in `data/skills.csv`
  cannot be found. The dictionary covers common technical skills, not every niche tool.
- **The weights are assumptions.** The 40/15/25/10/5/5 split was chosen so that concrete
  skill evidence outweighs free-text similarity. It is not tuned or validated as optimal,
  and the sidebar lets you change it precisely because it is a judgement call.
- **The score bands are assumptions too.** On the bundled labels they agree with human
  labels 86% of the time and are within one band 100% of the time, on 35 pairs.
- **Semantic similarity is not job suitability.** Two documents can read similarly while
  the candidate is still wrong for the role.
- **Resume layout affects extraction.** Multi-column PDFs can interleave text, and
  scanned PDFs contain no text at all (the app detects this and says so rather than
  scoring an empty document).
- **A screening aid, not a hiring decision.** Automated resume screening can encode bias
  and disadvantage candidates whose wording differs from the norm. Use this to find
  skill gaps and prioritise reading, not to reject people.

## Future improvements

- Learn the component weights from a larger labelled dataset instead of assuming them
- Expand the skill dictionary automatically from job-posting corpora
- Add OCR (Tesseract) so scanned PDFs can be read
- Section-aware weighting, so a skill in the Experience section counts more than one in
  an Interests list
- Rank many resumes against one job description in a batch view
- Multilingual support

## Screenshots

Add screenshots here after your first local run:

| View | File |
|---|---|
| Upload and analyse | `docs/screenshot-upload.png` |
| Score dashboard | `docs/screenshot-dashboard.png` |
| Skill gap analysis | `docs/screenshot-skills.png` |

## Contributors

| Name | Role |
|---|---|
| _Your name_ | _Your contribution_ |

## License

MIT. See [LICENSE](LICENSE).
