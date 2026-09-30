"""Text preprocessing.

Two rules drive this module:

1. **The original text is never destroyed.** :class:`ProcessedText` carries both
   ``original_text`` and ``processed_text`` because skill extraction, section
   detection and experience parsing all need the original casing and layout.
2. **Nothing here requires a network download.** spaCy and NLTK are used when
   they are available; otherwise a built-in stop-word list and a rule-based
   lemmatiser take over, so the app still runs on a bare install.
"""

from __future__ import annotations

import functools
import logging
import re
import unicodedata
from dataclasses import dataclass, field

LOGGER = logging.getLogger(__name__)

# Characters that are part of technology names and must survive tokenisation:
# c++, c#, node.js, ci/cd, scikit-learn, .net, a/b
_TOKEN_RE = re.compile(r"[a-z0-9](?:[a-z0-9+#./_-]*[a-z0-9+#])?", re.IGNORECASE)

_FALLBACK_STOPWORDS = frozenset("""
    a about above after again against all am an and any are aren't as at be because been
    before being below between both but by can cannot could couldn't did didn't do does
    doesn't doing don't down during each few for from further had hadn't has hasn't have
    haven't having he her here hers herself him himself his how i if in into is isn't it
    its itself just me more most my myself no nor not of off on once only or other ought
    our ours ourselves out over own same shan't she should shouldn't so some such than
    that the their theirs them themselves then there these they this those through to too
    under until up very was wasn't we were weren't what when where which while who whom
    why with won't would wouldn't you your yours yourself yourselves also across upon
    """.split())

# Words that look like stop-words to a generic list but carry meaning here.
_PROTECTED_WORDS = frozenset({"it", "r", "c", "go", "no", "not", "own"})

_IRREGULAR_LEMMAS = {
    "built": "build",
    "wrote": "write",
    "led": "lead",
    "ran": "run",
    "began": "begin",
    "chose": "choose",
    "taught": "teach",
    "brought": "bring",
    "made": "make",
    "grew": "grow",
    "drove": "drive",
    "held": "hold",
    "data": "data",
    "analyses": "analysis",
    "analytics": "analytics",
    "apis": "api",
    "indices": "index",
    "criteria": "criterion",
}


@dataclass
class ProcessedText:
    """Result of the preprocessing pipeline."""

    original_text: str
    processed_text: str
    tokens: list[str] = field(default_factory=list)
    lemmas: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.original_text.strip()

    @property
    def word_count(self) -> int:
        return len(self.tokens)

    def __repr__(self) -> str:  # pragma: no cover - debugging nicety
        return (
            f"ProcessedText(words={self.word_count}, "
            f"sentences={len(self.sentences)}, chars={len(self.original_text)})"
        )


# --------------------------------------------------------------------------- #
# Optional NLP backends
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def get_spacy_nlp():
    """Load a small spaCy pipeline once, or return ``None`` if unavailable.

    The parser and NER components are kept (they power sentence segmentation and
    organisation/date detection) but the pipeline is only ever used on text that
    has already been truncated, so it stays fast on free hosting.
    """
    try:
        import spacy
    except ImportError:
        LOGGER.info("spaCy not installed - using rule-based fallbacks.")
        return None
    try:
        return spacy.load("en_core_web_sm", exclude=["ner"])
    except OSError:
        LOGGER.info(
            "spaCy model 'en_core_web_sm' not found - using rule-based fallbacks. "
            "Install it with: python -m spacy download en_core_web_sm"
        )
        return None
    except Exception:  # pragma: no cover - defensive
        LOGGER.exception("spaCy failed to load; continuing without it.")
        return None


@functools.lru_cache(maxsize=1)
def get_spacy_ner():
    """Load a spaCy pipeline *with* NER, or ``None``."""
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load("en_core_web_sm")
    except Exception:
        return None


@functools.lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Return a stop-word set from NLTK if downloaded, else the built-in list."""
    try:
        from nltk.corpus import stopwords  # type: ignore

        words = frozenset(stopwords.words("english"))
        if words:
            return frozenset(words) - _PROTECTED_WORDS
    except Exception:
        LOGGER.debug("NLTK stopwords unavailable - using built-in list.")
    return frozenset(_FALLBACK_STOPWORDS) - _PROTECTED_WORDS


# --------------------------------------------------------------------------- #
# Core functions
# --------------------------------------------------------------------------- #


def normalize_unicode(text: str) -> str:
    """Fold accents and compatibility characters into plain ASCII-ish text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    return "".join(
        ch for ch in unicodedata.normalize("NFKD", text) if not unicodedata.combining(ch)
    )


def clean_text(text: str, *, lowercase: bool = True) -> str:
    """Collapse whitespace, drop control characters, optionally lowercase.

    URLs, e-mail addresses and phone numbers are replaced with a space: they add
    noise to TF-IDF vectors and are irrelevant to skill matching.
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b", " ", text)
    text = re.sub(r"(\+\d{1,3}[\s-]?)?\(?\d{3,5}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b", " ", text)
    text = re.sub(r"[^\S\n]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text.lower() if lowercase else text


def tokenize(text: str) -> list[str]:
    """Split text into lowercase tokens, keeping technology punctuation intact."""
    if not text:
        return []
    return [m.group(0).lower() for m in _TOKEN_RE.finditer(text)]


def remove_stopwords(tokens: list[str]) -> list[str]:
    stop = get_stopwords()
    return [t for t in tokens if t not in stop]


def simple_lemmatize(token: str) -> str:
    """Rule-based lemmatiser used when spaCy is not installed.

    Deliberately conservative: it only strips endings that are safe for the
    vocabulary in resumes, and never shortens tokens below four characters.
    """
    if token in _IRREGULAR_LEMMAS:
        return _IRREGULAR_LEMMAS[token]
    if len(token) <= 3 or not token.isalpha():
        return token
    for suffix, replacement in (
        ("ies", "y"),
        ("sses", "ss"),
        ("ches", "ch"),
        ("shes", "sh"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("s", ""),
    ):
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            stem = token[: len(token) - len(suffix)] + replacement
            if suffix in ("ing", "ed") and len(stem) >= 2 and stem[-1] == stem[-2]:
                stem = stem[:-1]  # running -> run, planned -> plan
            return stem
    return token


def lemmatize(tokens: list[str]) -> list[str]:
    """Lemmatise tokens with spaCy when available, else with rules."""
    nlp = get_spacy_nlp()
    if nlp is None:
        return [simple_lemmatize(t) for t in tokens]
    doc = nlp.tokenizer(" ".join(tokens))
    try:
        for name, proc in nlp.pipeline:
            if name in ("tagger", "attribute_ruler", "lemmatizer"):
                doc = proc(doc)
    except Exception:  # pragma: no cover - defensive
        return [simple_lemmatize(t) for t in tokens]
    return [t.lemma_.lower() for t in doc if not t.is_space]


def split_sentences(text: str) -> list[str]:
    """Split text into sentences.

    Resumes are full of bullet lines without terminal punctuation, so each
    non-empty line is treated as at least one unit before sentence splitting.
    """
    if not text:
        return []
    sentences: list[str] = []
    for line in text.splitlines():
        line = line.strip(" \t-*\u2022")
        if not line:
            continue
        if len(line) < 200 and line.count(".") <= 1:
            sentences.append(line)
            continue
        parts = re.split(r"(?<=[.!?;])\s+(?=[A-Z(])", line)
        sentences.extend(p.strip() for p in parts if p.strip())
    return [s for s in sentences if len(s) > 1]


def preprocess_text(
    text: str,
    *,
    remove_stop: bool = True,
    do_lemmatize: bool = True,
) -> ProcessedText:
    """Run the full preprocessing pipeline.

    Args:
        text: Raw extracted document text.
        remove_stop: Drop English stop-words from the token stream.
        do_lemmatize: Reduce tokens to their base form.

    Returns:
        A :class:`ProcessedText` holding both the untouched original and the
        processed representation.
    """
    original = text or ""
    if not original.strip():
        return ProcessedText(original_text=original, processed_text="")

    cleaned = clean_text(original, lowercase=True)
    tokens = tokenize(cleaned)
    working = remove_stopwords(tokens) if remove_stop else list(tokens)
    lemmas = lemmatize(working) if do_lemmatize else list(working)
    lemmas = [t for t in lemmas if t.strip()]

    return ProcessedText(
        original_text=original,
        processed_text=" ".join(lemmas),
        tokens=tokens,
        lemmas=lemmas,
        sentences=split_sentences(original),
    )


__all__ = [
    "ProcessedText",
    "clean_text",
    "get_spacy_ner",
    "get_spacy_nlp",
    "get_stopwords",
    "lemmatize",
    "normalize_unicode",
    "preprocess_text",
    "remove_stopwords",
    "simple_lemmatize",
    "split_sentences",
    "tokenize",
]
