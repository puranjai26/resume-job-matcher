"""Evaluation: does semantic NLP beat keyword matching on the labelled set?

Run it with::

    python -m src.evaluation

Four methods are compared on ``data/labels.csv``:

1. ``keyword``  -- Jaccard overlap of extracted skill sets (exact matching)
2. ``tfidf``    -- cosine similarity of TF-IDF vectors
3. ``semantic`` -- sentence embeddings (or the LSA fallback)
4. ``pipeline`` -- the full weighted score used by the application

Because the four methods produce scores on different scales, a single fixed
cut-off would be unfair. Each method is therefore evaluated at the threshold
that maximises its own F1 (reported alongside that threshold), and ranking
metrics -- which are threshold-free -- are reported as well.

The label set is small (25 hand-labelled pairs over 5 resumes and 5 jobs), so
these numbers demonstrate the method, they do not establish general accuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path

from .config import (
    DEFAULT_SETTINGS,
    LABELS_CSV,
    REPORTS_DIR,
    SAMPLE_JOBS_DIR,
    SAMPLE_RESUMES_DIR,
    Settings,
)
from .parser import extract_text
from .pipeline import match_resume_to_job
from .similarity import calculate_keyword_overlap, semantic_backend_available
from .skill_extractor import get_extractor

LOGGER = logging.getLogger(__name__)

METHODS = ("keyword", "tfidf", "semantic", "pipeline")
RELEVANT_AT = 2  # graded relevance >= 2 counts as a true match


@dataclass
class PairScore:
    """Scores and label for one resume-job pair."""

    resume_id: str
    job_id: str
    relevance: int
    label: str
    scores: dict[str, float]

    @property
    def is_relevant(self) -> bool:
        return self.relevance >= RELEVANT_AT


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #


def _find_document(directory: Path, doc_id: str) -> Path:
    matches = sorted(directory.glob(f"{doc_id}*"))
    if not matches:
        raise FileNotFoundError(f"No document starting with '{doc_id}' in {directory}")
    return matches[0]


def load_documents(
    resumes_dir: Path = SAMPLE_RESUMES_DIR, jobs_dir: Path = SAMPLE_JOBS_DIR
) -> tuple[dict[str, str], dict[str, str]]:
    """Load every sample resume and job, keyed by their ``R00x`` / ``J00x`` id."""
    resumes: dict[str, str] = {}
    for path in sorted(resumes_dir.glob("*")):
        if path.suffix.lower() in (".txt", ".pdf", ".docx", ".md"):
            resumes[path.stem.split("_")[0]] = extract_text(path)
    jobs: dict[str, str] = {}
    for path in sorted(jobs_dir.glob("*")):
        if path.suffix.lower() in (".txt", ".pdf", ".docx", ".md"):
            jobs[path.stem.split("_")[0]] = extract_text(path)
    return resumes, jobs


def load_labels(path: Path = LABELS_CSV) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Label file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as fh:
        return [row for row in csv.DictReader(fh) if row.get("resume_id")]


# --------------------------------------------------------------------------- #
# Metrics (implemented directly so the maths stays visible)
# --------------------------------------------------------------------------- #


def confusion(y_true: list[bool], y_pred: list[bool]) -> tuple[int, int, int, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred) if t and p)
    fp = sum(1 for t, p in zip(y_true, y_pred) if not t and p)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t and not p)
    tn = sum(1 for t, p in zip(y_true, y_pred) if not t and not p)
    return tp, fp, fn, tn


def prf(y_true: list[bool], y_pred: list[bool]) -> dict[str, float]:
    """Accuracy, precision, recall and F1 for a binary prediction."""
    tp, fp, fn, tn = confusion(y_true, y_pred)
    total = tp + fp + fn + tn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def best_threshold(scores: list[float], y_true: list[bool]) -> tuple[float, dict[str, float]]:
    """Pick the cut-off that maximises F1 for this method."""
    candidates = sorted({round(s, 4) for s in scores} | {0.0, 1.0})
    best_value, best_metrics = 0.5, prf(y_true, [s >= 0.5 for s in scores])
    best_f1 = -1.0
    for candidate in candidates:
        metrics = prf(y_true, [s >= candidate for s in scores])
        if metrics["f1"] > best_f1 or (
            metrics["f1"] == best_f1 and metrics["accuracy"] > best_metrics["accuracy"]
        ):
            best_f1, best_value, best_metrics = metrics["f1"], candidate, metrics
    return best_value, best_metrics


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation with average ranks for ties."""
    if len(x) < 3:
        return 0.0

    def rank(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        ranks = [0.0] * len(values)
        index = 0
        while index < len(order):
            stop = index
            while stop + 1 < len(order) and values[order[stop + 1]] == values[order[index]]:
                stop += 1
            average = (index + stop) / 2 + 1
            for position in range(index, stop + 1):
                ranks[order[position]] = average
            index = stop + 1
        return ranks

    rx, ry = rank(x), rank(y)
    mean_x, mean_y = sum(rx) / len(rx), sum(ry) / len(ry)
    numerator = sum((a - mean_x) * (b - mean_y) for a, b in zip(rx, ry))
    denominator = math.sqrt(sum((a - mean_x) ** 2 for a in rx) * sum((b - mean_y) ** 2 for b in ry))
    return numerator / denominator if denominator else 0.0


def dcg(relevances: list[int]) -> float:
    return sum(rel / math.log2(position + 2) for position, rel in enumerate(relevances))


def ndcg_at_k(ranked_relevances: list[int], k: int) -> float:
    ideal = sorted(ranked_relevances, reverse=True)[:k]
    ideal_dcg = dcg(ideal)
    return dcg(ranked_relevances[:k]) / ideal_dcg if ideal_dcg else 0.0


def ranking_metrics(pairs: list[PairScore], method: str, k: int = 3) -> dict[str, float]:
    """Precision@1, MRR and NDCG@k, averaged over jobs (each job ranks resumes)."""
    by_job: dict[str, list[PairScore]] = {}
    for pair in pairs:
        by_job.setdefault(pair.job_id, []).append(pair)

    precisions, reciprocal_ranks, ndcgs = [], [], []
    for candidates in by_job.values():
        if not any(c.is_relevant for c in candidates):
            continue
        ranked = sorted(candidates, key=lambda c: c.scores[method], reverse=True)
        precisions.append(1.0 if ranked[0].is_relevant else 0.0)
        rank_of_first = next((index + 1 for index, c in enumerate(ranked) if c.is_relevant), 0)
        reciprocal_ranks.append(1.0 / rank_of_first if rank_of_first else 0.0)
        ndcgs.append(ndcg_at_k([c.relevance for c in ranked], k))

    count = len(precisions) or 1
    return {
        "precision_at_1": sum(precisions) / count,
        "mrr": sum(reciprocal_ranks) / count,
        f"ndcg_at_{k}": sum(ndcgs) / count,
        "queries": len(precisions),
    }


# --------------------------------------------------------------------------- #
# Scoring the labelled pairs
# --------------------------------------------------------------------------- #


def score_pairs(
    labels: list[dict[str, str]],
    resumes: dict[str, str],
    jobs: dict[str, str],
    settings: Settings | None = None,
) -> list[PairScore]:
    """Compute every method's score for every labelled pair."""
    settings = settings or DEFAULT_SETTINGS
    extractor = get_extractor()
    scored: list[PairScore] = []

    for row in labels:
        resume_id, job_id = row["resume_id"].strip(), row["job_id"].strip()
        if resume_id not in resumes or job_id not in jobs:
            LOGGER.warning("Skipping unknown pair %s/%s", resume_id, job_id)
            continue

        resume_text, job_text = resumes[resume_id], jobs[job_id]
        report = match_resume_to_job(resume_text, job_text, settings=settings, extractor=extractor)
        keyword = calculate_keyword_overlap(report.resume_skills.skills, report.job.all_skills)
        scored.append(
            PairScore(
                resume_id=resume_id,
                job_id=job_id,
                relevance=int(row.get("relevance") or 0),
                label=row.get("label", ""),
                scores={
                    "keyword": keyword,
                    "tfidf": report.tfidf_score,
                    "semantic": report.semantic.score,
                    "pipeline": report.overall_score / 100.0,
                },
            )
        )
    return scored


def evaluate(pairs: list[PairScore]) -> dict:
    """Compute all metrics for all methods."""
    y_true = [p.is_relevant for p in pairs]
    graded = [float(p.relevance) for p in pairs]

    results: dict[str, dict] = {}
    for method in METHODS:
        scores = [p.scores[method] for p in pairs]
        threshold, metrics = best_threshold(scores, y_true)
        results[method] = {
            "best_threshold": threshold,
            **{k: round(v, 4) if isinstance(v, float) else v for k, v in metrics.items()},
            "spearman": round(spearman(scores, graded), 4),
            **{
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in ranking_metrics(pairs, method).items()
            },
            "mean_relevant": round(
                sum(s for s, t in zip(scores, y_true) if t) / max(1, sum(y_true)), 4
            ),
            "mean_irrelevant": round(
                sum(s for s, t in zip(scores, y_true) if not t) / max(1, len(y_true) - sum(y_true)),
                4,
            ),
        }
        results[method]["separation"] = round(
            results[method]["mean_relevant"] - results[method]["mean_irrelevant"], 4
        )
    return results


def band_agreement(pairs: list[PairScore], settings: Settings | None = None) -> dict:
    """How often the configured score bands reproduce the human label."""
    settings = settings or DEFAULT_SETTINGS
    mapping = {"Weak Match": 0, "Moderate Match": 1, "Strong Match": 2, "Excellent Match": 2}
    exact, adjacent = 0, 0
    rows = []
    for pair in pairs:
        predicted_label = settings.thresholds.classify(pair.scores["pipeline"] * 100)
        predicted = mapping[predicted_label]
        if predicted == pair.relevance:
            exact += 1
        if abs(predicted - pair.relevance) <= 1:
            adjacent += 1
        rows.append(
            {
                "pair": f"{pair.resume_id}-{pair.job_id}",
                "score": round(pair.scores["pipeline"] * 100, 1),
                "predicted": predicted_label,
                "actual": pair.label,
            }
        )
    total = len(pairs) or 1
    return {
        "exact_agreement": round(exact / total, 4),
        "within_one_band": round(adjacent / total, 4),
        "rows": rows,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def format_report(results: dict, bands: dict, pairs: list[PairScore], backend: str) -> str:
    """Render the evaluation report as Markdown."""
    relevant = sum(1 for p in pairs if p.is_relevant)
    lines = [
        "# Evaluation Report",
        "",
        f"- Labelled pairs: **{len(pairs)}** "
        f"({relevant} relevant, {len(pairs) - relevant} not relevant)",
        f"- Semantic backend used: **{backend}**",
        "- Relevance rule: a pair counts as a true match when graded relevance is 2.",
        "",
        "## Method comparison",
        "",
        "Each method is scored at its own best-F1 threshold, because the four methods",
        "produce values on different scales.",
        "",
        "| Method | Threshold | Accuracy | Precision | Recall | F1 | Spearman | Separation |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for method in METHODS:
        r = results[method]
        lines.append(
            f"| {method} | {r['best_threshold']:.2f} | {r['accuracy']:.2f} | "
            f"{r['precision']:.2f} | {r['recall']:.2f} | {r['f1']:.2f} | "
            f"{r['spearman']:.2f} | {r['separation']:.2f} |"
        )

    lines += [
        "",
        "`Separation` is the mean score on relevant pairs minus the mean on irrelevant",
        "pairs: how far apart the method pushes good and bad matches.",
        "",
        "## Ranking quality",
        "",
        "For each job, all resumes are ranked by score. Threshold-free, so it compares",
        "the methods on equal footing.",
        "",
        "| Method | Precision@1 | MRR | NDCG@3 |",
        "|---|---|---|---|",
    ]
    for method in METHODS:
        r = results[method]
        lines.append(
            f"| {method} | {r['precision_at_1']:.2f} | {r['mrr']:.2f} | {r['ndcg_at_3']:.2f} |"
        )

    lines += [
        "",
        "## Score band agreement",
        "",
        f"- Exact agreement with the human label: **{bands['exact_agreement']:.0%}**",
        f"- Within one band: **{bands['within_one_band']:.0%}**",
        "",
        "| Pair | Score | Predicted | Human label |",
        "|---|---|---|---|",
    ]
    for row in bands["rows"]:
        lines.append(
            f"| {row['pair']} | {row['score']:.1f} | {row['predicted']} | {row['actual']} |"
        )

    lines += [
        "",
        "## How to read this",
        "",
        "Two caveats come first. The dataset is small (a few dozen pairs) and was",
        "labelled by the project author, so it carries that author's judgement. And",
        "each method's threshold is chosen on the same pairs it is then scored on,",
        "with no held-out split, which makes every number here optimistic. Treat the",
        "absolute values as illustrative rather than as a general accuracy claim. What",
        "the table is meant to show is the *relative* behaviour of the methods: whether semantic",
        "similarity separates good from bad matches more cleanly than exact keyword",
        "overlap, and whether combining the signals beats either one alone.",
        "",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: ``python -m src.evaluation``."""
    parser = argparse.ArgumentParser(description="Evaluate resume-job matching methods.")
    parser.add_argument("--labels", type=Path, default=LABELS_CSV)
    parser.add_argument("--resumes", type=Path, default=SAMPLE_RESUMES_DIR)
    parser.add_argument("--jobs", type=Path, default=SAMPLE_JOBS_DIR)
    parser.add_argument("--output", type=Path, default=REPORTS_DIR / "evaluation_report.md")
    parser.add_argument("--json", type=Path, default=None, help="Also write raw metrics as JSON")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(levelname)s: %(message)s",
    )

    labels = load_labels(args.labels)
    resumes, jobs = load_documents(args.resumes, args.jobs)
    print(f"Loaded {len(resumes)} resumes, {len(jobs)} jobs, {len(labels)} labelled pairs.")
    print("Scoring pairs (the first run may download the semantic model)...")

    pairs = score_pairs(labels, resumes, jobs)
    if not pairs:
        print("No pairs could be scored. Check that data/ ids match labels.csv.")
        return 1

    results = evaluate(pairs)
    bands = band_agreement(pairs)
    backend = semantic_backend_available()
    report = format_report(results, bands, pairs, backend)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print()
    print(f"{'method':<10}{'F1':>8}{'Acc':>8}{'P@1':>8}{'NDCG@3':>9}{'Separation':>12}")
    for method in METHODS:
        r = results[method]
        print(
            f"{method:<10}{r['f1']:>8.2f}{r['accuracy']:>8.2f}"
            f"{r['precision_at_1']:>8.2f}{r['ndcg_at_3']:>9.2f}{r['separation']:>12.2f}"
        )
    print(f"\nReport written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
