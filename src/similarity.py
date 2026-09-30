"""Similarity engines: TF-IDF (baseline) and semantic embeddings (advanced).

Two backends are available for semantic similarity:

``sentence-transformers``
    A small local model (``all-MiniLM-L6-v2``, ~90 MB, CPU only). Loaded lazily
    and cached process-wide so it is initialised at most once. No API key and no
    external service are involved -- the model runs locally.

``lsa`` (fallback)
    Latent Semantic Analysis: TF-IDF over word *and* character n-grams reduced
    with Truncated SVD. It captures loose wording overlap ("RESTful APIs in
    Flask" vs "web APIs with Python frameworks") without any download, so the
    application still works if the model cannot be fetched on first run.

The active backend is always reported in :class:`SimilarityResult.backend` so
the UI and the evaluation report can state which one produced a number.
"""

from __future__ import annotations

import logging
import re
import threading
from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .config import (
    LSA_CALIBRATION_EXPONENT,
    SEMANTIC_BACKEND,
    SEMANTIC_CHUNK_WORDS,
    SEMANTIC_MODEL_NAME,
)
from .preprocessing import preprocess_text

LOGGER = logging.getLogger(__name__)

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: dict[str, object] = {}


@dataclass
class SimilarityResult:
    """Outcome of one similarity computation."""

    score: float
    method: str
    backend: str = ""
    detail: str = ""

    def as_percent(self) -> float:
        return round(self.score * 100, 1)


def _clamp(value: float) -> float:
    """Clamp to [0, 1] and drop NaN, which cosine can produce on empty input."""
    if value is None or np.isnan(value):
        return 0.0
    return float(min(1.0, max(0.0, value)))


def _prepare(text: str) -> str:
    """Preprocess text for bag-of-words similarity."""
    if not text or not text.strip():
        return ""
    return preprocess_text(text).processed_text


# --------------------------------------------------------------------------- #
# TF-IDF baseline
# --------------------------------------------------------------------------- #


def calculate_tfidf_similarity(
    resume_text: str,
    job_text: str,
    *,
    ngram_range: tuple[int, int] = (1, 2),
    preprocess: bool = True,
) -> float:
    """Cosine similarity between TF-IDF vectors of the two documents.

    Args:
        resume_text: Resume text (raw or preprocessed).
        job_text: Job description text.
        ngram_range: Word n-gram range; bigrams let "machine learning" count as
            a single feature.
        preprocess: Run the preprocessing pipeline first.

    Returns:
        A score in ``[0, 1]``. Returns ``0.0`` when either document is empty.
    """
    left = _prepare(resume_text) if preprocess else (resume_text or "")
    right = _prepare(job_text) if preprocess else (job_text or "")
    if not left.strip() or not right.strip():
        return 0.0

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b[\w+#./-]{1,}\b",
        )
        matrix = vectorizer.fit_transform([left, right])
    except ValueError:
        # Happens when both documents contain only stop-words.
        return 0.0
    if matrix.shape[1] == 0:
        return 0.0
    return _clamp(cosine_similarity(matrix[0], matrix[1])[0][0])


def calculate_keyword_overlap(resume_skills: list[str], job_skills: list[str]) -> float:
    """Jaccard overlap of two skill lists -- the exact-keyword baseline."""
    left, right = set(resume_skills), set(job_skills)
    if not left and not right:
        return 0.0
    union = left | right
    return _clamp(len(left & right) / len(union)) if union else 0.0


# --------------------------------------------------------------------------- #
# Semantic similarity
# --------------------------------------------------------------------------- #


def chunk_text(text: str, words_per_chunk: int = SEMANTIC_CHUNK_WORDS) -> list[str]:
    """Split text into sentence-aware chunks for embedding.

    Embedding one giant string averages everything into mush; embedding chunks
    and taking the best-matching pairs preserves local meaning.
    """
    if not text or not text.strip():
        return []
    units = [u.strip() for u in re.split(r"[\n.;]+", text) if u.strip()]
    chunks: list[str] = []
    buffer: list[str] = []
    length = 0
    for unit in units:
        count = len(unit.split())
        if length + count > words_per_chunk and buffer:
            chunks.append(" ".join(buffer))
            buffer, length = [], 0
        buffer.append(unit)
        length += count
    if buffer:
        chunks.append(" ".join(buffer))
    return chunks[:120] or [text[:2000]]


def _try_load_sentence_transformer(model_name: str):
    """Load and cache a SentenceTransformer, returning ``None`` on any failure."""
    with _MODEL_LOCK:
        if model_name in _MODEL_CACHE:
            return _MODEL_CACHE[model_name]
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            LOGGER.info("sentence-transformers not installed; using LSA fallback.")
            _MODEL_CACHE[model_name] = None
            return None
        try:
            LOGGER.info("Loading semantic model '%s' (first run downloads ~90 MB)", model_name)
            model = SentenceTransformer(model_name, device="cpu")
        except Exception as exc:
            LOGGER.warning("Could not load '%s' (%s); using LSA fallback.", model_name, exc)
            model = None
        _MODEL_CACHE[model_name] = model
        return model


def load_semantic_model(model_name: str = SEMANTIC_MODEL_NAME):
    """Public loader used by the UI to warm the cache. May return ``None``."""
    return _try_load_sentence_transformer(model_name)


def semantic_backend_available(backend: str = SEMANTIC_BACKEND) -> str:
    """Return the backend that would actually be used: ``"st"`` or ``"lsa"``."""
    if backend == "lsa":
        return "lsa"
    if backend == "st":
        return "st"
    try:
        import sentence_transformers  # noqa: F401

        return "st"
    except ImportError:
        return "lsa"


def _embedding_similarity(model, resume_text: str, job_text: str) -> float:
    """Chunked, symmetric best-match similarity using sentence embeddings."""
    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_text)
    if not resume_chunks or not job_chunks:
        return 0.0

    vectors = model.encode(
        resume_chunks + job_chunks,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )
    left = vectors[: len(resume_chunks)]
    right = vectors[len(resume_chunks) :]
    matrix = np.clip(left @ right.T, -1.0, 1.0)

    # Symmetric coverage: how well each job chunk is covered by the resume, and
    # vice versa. Averaging both directions avoids rewarding padded resumes.
    job_coverage = float(matrix.max(axis=0).mean())
    resume_coverage = float(matrix.max(axis=1).mean())
    document = float(matrix.mean())
    score = 0.5 * job_coverage + 0.25 * resume_coverage + 0.25 * document
    return _clamp(score)


def _lsa_similarity(resume_text: str, job_text: str) -> float:
    """Dependency-free semantic fallback: TF-IDF + SVD over document chunks."""
    resume_chunks = chunk_text(resume_text)
    job_chunks = chunk_text(job_text)
    if not resume_chunks or not job_chunks:
        return 0.0

    corpus = resume_chunks + job_chunks
    processed = [_prepare(c) or c.lower() for c in corpus]
    processed = [p for p in processed if p.strip()]
    if len(processed) < 2:
        return calculate_tfidf_similarity(resume_text, job_text)

    try:
        word_vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            token_pattern=r"(?u)\b[\w+#./-]{1,}\b",
        )
        word_matrix = word_vectorizer.fit_transform(processed)
        char_vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
            min_df=1,
        )
        char_matrix = char_vectorizer.fit_transform(processed)
    except ValueError:
        return calculate_tfidf_similarity(resume_text, job_text)

    from scipy.sparse import hstack  # scipy ships with scikit-learn

    combined = hstack([word_matrix, char_matrix]).tocsr()
    n_components = int(min(128, max(2, min(combined.shape) - 1)))
    try:
        svd = TruncatedSVD(n_components=n_components, random_state=42)
        reduced = svd.fit_transform(combined)
    except Exception:  # pragma: no cover - degenerate input
        reduced = combined.toarray()

    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    reduced = reduced / norms

    split = len(resume_chunks)
    left, right = reduced[:split], reduced[split:]
    if left.size == 0 or right.size == 0:
        return 0.0
    matrix = np.clip(left @ right.T, -1.0, 1.0)

    job_coverage = float(matrix.max(axis=0).mean())
    resume_coverage = float(matrix.max(axis=1).mean())
    document = float(matrix.mean())
    raw = 0.5 * job_coverage + 0.25 * resume_coverage + 0.25 * document
    # Stretch the narrow LSA range onto a transformer-like scale (see config).
    return _clamp(max(raw, 0.0) ** LSA_CALIBRATION_EXPONENT)


def calculate_semantic_similarity(
    resume_text: str,
    job_text: str,
    *,
    backend: str = SEMANTIC_BACKEND,
    model_name: str = SEMANTIC_MODEL_NAME,
    return_result: bool = False,
) -> float | SimilarityResult:
    """Semantic similarity between a resume and a job description.

    Args:
        resume_text: Resume text.
        job_text: Job description text.
        backend: ``"auto"``, ``"st"`` (sentence-transformers) or ``"lsa"``.
        model_name: Sentence-transformer model id.
        return_result: Return a :class:`SimilarityResult` instead of a float, so
            the caller can report which backend was used.

    Returns:
        A score in ``[0, 1]`` (or a :class:`SimilarityResult`).
    """
    if not (resume_text or "").strip() or not (job_text or "").strip():
        result = SimilarityResult(0.0, "semantic", "none", "One of the documents is empty")
        return result if return_result else 0.0

    model = None
    if backend in ("auto", "st"):
        model = _try_load_sentence_transformer(model_name)
        if model is None and backend == "st":
            raise RuntimeError(
                f"sentence-transformers backend requested but '{model_name}' could not "
                "be loaded. Install sentence-transformers or set RJM_SEMANTIC_BACKEND=lsa."
            )

    if model is not None:
        try:
            score = _embedding_similarity(model, resume_text, job_text)
            result = SimilarityResult(score, "semantic", "sentence-transformers", model_name)
            return result if return_result else score
        except Exception:  # pragma: no cover - runtime safety net
            LOGGER.exception("Embedding similarity failed; falling back to LSA.")

    score = _lsa_similarity(resume_text, job_text)
    result = SimilarityResult(score, "semantic", "lsa", "TF-IDF + SVD latent semantic fallback")
    return result if return_result else score


def compare_methods(resume_text: str, job_text: str, resume_skills=None, job_skills=None) -> dict:
    """Run every similarity method on one pair (used by the evaluation script)."""
    semantic = calculate_semantic_similarity(resume_text, job_text, return_result=True)
    assert isinstance(semantic, SimilarityResult)
    output = {
        "tfidf": calculate_tfidf_similarity(resume_text, job_text),
        "semantic": semantic.score,
        "semantic_backend": semantic.backend,
    }
    if resume_skills is not None and job_skills is not None:
        output["keyword"] = calculate_keyword_overlap(resume_skills, job_skills)
    return output


__all__ = [
    "SimilarityResult",
    "calculate_keyword_overlap",
    "calculate_semantic_similarity",
    "calculate_tfidf_similarity",
    "chunk_text",
    "compare_methods",
    "load_semantic_model",
    "semantic_backend_available",
]
