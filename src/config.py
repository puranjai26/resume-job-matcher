"""Central configuration for the resume-job matcher.

Every tunable number in the project lives here so that it can be changed in one
place. The scoring weights and classification thresholds are **project
assumptions**, not empirically optimal values -- see ``ScoringWeights``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"

SKILLS_CSV: Path = DATA_DIR / "skills.csv"
ALIASES_CSV: Path = DATA_DIR / "aliases.csv"
LABELS_CSV: Path = DATA_DIR / "labels.csv"
SAMPLE_RESUMES_DIR: Path = DATA_DIR / "sample_resumes"
SAMPLE_JOBS_DIR: Path = DATA_DIR / "sample_jobs"

# --------------------------------------------------------------------------- #
# Document handling limits
# --------------------------------------------------------------------------- #

#: Reject uploads larger than this. Keeps free-tier hosting memory bounded.
MAX_FILE_SIZE_MB: float = 5.0

#: Characters beyond this are ignored during analysis (very long resumes).
MAX_TEXT_CHARS: int = 120_000

#: Below this many characters we treat extraction as failed (e.g. scanned PDF).
MIN_USABLE_TEXT_CHARS: int = 30

SUPPORTED_EXTENSIONS: tuple[str, ...] = (".pdf", ".docx", ".txt", ".md")

# --------------------------------------------------------------------------- #
# Semantic model
# --------------------------------------------------------------------------- #

#: Small, CPU-friendly sentence embedding model (~90 MB).
SEMANTIC_MODEL_NAME: str = os.environ.get(
    "RJM_SEMANTIC_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

#: "auto"  -> use sentence-transformers if importable, otherwise fall back to LSA
#: "st"    -> force sentence-transformers (raises if unavailable)
#: "lsa"   -> force the dependency-free latent-semantic fallback
SEMANTIC_BACKEND: str = os.environ.get("RJM_SEMANTIC_BACKEND", "auto")

#: Texts are chunked before embedding so that a long resume is not squashed
#: into a single averaged vector.
SEMANTIC_CHUNK_WORDS: int = 60

#: Raw LSA cosine values live on a much narrower scale than transformer cosine
#: values (a clearly relevant pair scores ~0.15 with LSA but ~0.55 with
#: MiniLM). Applying ``score ** LSA_CALIBRATION_EXPONENT`` stretches the LSA
#: range so that the same final-score thresholds remain meaningful whichever
#: backend is active. This is a monotonic re-scaling for comparability, not a
#: claim of equivalence: rankings are unchanged, only the spread is.
LSA_CALIBRATION_EXPONENT: float = 0.45


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ScoringWeights:
    """Weights of the final score.

    .. warning::
       These weights are **configurable project assumptions**. They were chosen
       so that concrete skill evidence dominates the score and free-text
       similarity plays a supporting role. They have *not* been shown to be
       scientifically optimal, and no claim of optimality is made. Change them
       via :func:`ScoringWeights.replace` or by editing this file.

    If a component cannot be computed (for example a resume that states no
    experience at all), its weight is redistributed proportionally over the
    remaining components rather than being counted as zero. That keeps the
    score honest instead of penalising missing information.
    """

    required_skills: float = 0.40
    preferred_skills: float = 0.15
    semantic_similarity: float = 0.25
    tfidf_similarity: float = 0.10
    experience: float = 0.05
    education: float = 0.05

    def as_dict(self) -> dict[str, float]:
        return {
            "required_skills": self.required_skills,
            "preferred_skills": self.preferred_skills,
            "semantic_similarity": self.semantic_similarity,
            "tfidf_similarity": self.tfidf_similarity,
            "experience": self.experience,
            "education": self.education,
        }

    def replace(self, **kwargs: float) -> "ScoringWeights":
        """Return a copy with some weights overridden."""
        current = self.as_dict()
        unknown = set(kwargs) - set(current)
        if unknown:
            raise ValueError(f"Unknown weight(s): {sorted(unknown)}")
        current.update(kwargs)
        return ScoringWeights(**current)

    def total(self) -> float:
        return sum(self.as_dict().values())


@dataclass(frozen=True)
class MatchThresholds:
    """Score bands used to turn a 0-100 score into a label.

    Also configurable; validated against ``data/labels.csv`` by
    ``python -m src.evaluation`` rather than assumed to be universally correct.
    """

    weak_max: float = 39.0
    moderate_max: float = 69.0
    strong_max: float = 84.0

    def classify(self, score_percent: float) -> str:
        if score_percent <= self.weak_max:
            return "Weak Match"
        if score_percent <= self.moderate_max:
            return "Moderate Match"
        if score_percent <= self.strong_max:
            return "Strong Match"
        return "Excellent Match"


@dataclass(frozen=True)
class Settings:
    """Bundle passed through the pipeline."""

    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: MatchThresholds = field(default_factory=MatchThresholds)
    semantic_backend: str = SEMANTIC_BACKEND
    semantic_model_name: str = SEMANTIC_MODEL_NAME
    #: Fuzzy skill match cut-off (0-1). Used only for close spelling variants.
    fuzzy_threshold: float = 0.88
    #: Treat a preferred-skill section as present only if it has >= this many skills.
    min_section_skills: int = 1


DEFAULT_SETTINGS = Settings()

MATCH_LABELS: tuple[str, ...] = (
    "Weak Match",
    "Moderate Match",
    "Strong Match",
    "Excellent Match",
)
