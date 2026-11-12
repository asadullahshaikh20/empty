"""Convert model probabilities into scores, labels, confidences, AI-Readiness and verdict.

Rules come from SRS section 4.4 (Tables 25 and 25c) and SDS 5.1.2. The SRS and SDS
define the verdict differently: the SRS uses per-head thresholds, the SDS uses
AI-Readiness thresholds. This module applies the stricter of the two, with
precedence Reject > Review > Pass:

    Reject  if suitability <= 40, or any head < 34, or readiness < 50
    Review  else if suitability <= 70, or any head <= 66, or readiness < 75
    Pass    otherwise
"""

from __future__ import annotations

from typing import Mapping, Sequence

from . import config


def probs_to_score(probs: Sequence[float]) -> float:
    """Expected score over the three classes (good=100, mid=50, bad=0), range 0-100."""
    if len(probs) != config.NUM_CLASSES:
        raise ValueError(f"expected {config.NUM_CLASSES} probabilities, got {len(probs)}")
    score = sum(float(p) * s for p, s in zip(probs, config.CLASS_SCORE))
    return round(min(100.0, max(0.0, score)), 2)


def score_label(head: str, score: float) -> str:
    s = int(round(score))
    for lo, hi, label in config.SCORE_LABELS[head]:
        if lo <= s <= hi:
            return label
    raise ValueError(f"score {score} out of range for {head}")


def confidence_level(confidence: float) -> str:
    for threshold, name in config.CONFIDENCE_LEVELS:
        if confidence >= threshold:
            return name
    return config.CONFIDENCE_LEVELS[-1][1]


def ai_readiness(scores: Mapping[str, float], weights: Mapping[str, float] | None = None) -> float:
    weights = weights or config.READINESS_WEIGHTS
    total_w = sum(weights.values())
    value = sum(float(scores[h]) * w for h, w in weights.items()) / total_w
    return round(min(100.0, max(0.0, value)), 2)


def verdict(scores: Mapping[str, float], readiness: float, thresholds: Mapping[str, float] | None = None) -> str:
    t = dict(config.VERDICT_THRESHOLDS)
    if thresholds:
        t.update(thresholds)
    rounded = {h: int(round(float(scores[h]))) for h in config.HEADS}
    suit = rounded["diagnostic_suitability"]
    ready = int(round(float(readiness)))
    if (
        suit <= t["suitability_reject_max"]
        or any(v < t["head_reject_below"] for v in rounded.values())
        or ready < t["readiness_reject_below"]
    ):
        return "Reject"
    if (
        suit <= t["suitability_review_max"]
        or any(v <= t["head_review_max"] for v in rounded.values())
        or ready < t["readiness_pass_min"]
    ):
        return "Review"
    return "Pass"


def summarize(head_probs: Mapping[str, Sequence[float]]) -> dict:
    """Build the full assessment summary from per-head softmax probabilities."""
    missing = [h for h in config.HEADS if h not in head_probs]
    if missing:
        raise ValueError(f"missing heads: {missing}")

    heads: dict[str, dict] = {}
    scores: dict[str, float] = {}
    warnings: list[str] = []
    for h in config.HEADS:
        probs = [float(p) for p in head_probs[h]]
        score = probs_to_score(probs)
        conf = round(max(probs), 4)
        scores[h] = score
        heads[h] = {
            "score": score,
            "label": score_label(h, score),
            "confidence": conf,
            "confidence_level": confidence_level(conf),
            "predicted_class": config.HEAD_CLASSES[h][probs.index(max(probs))],
            "probabilities": [round(p, 4) for p in probs],
        }
        if conf < config.LOW_CONFIDENCE_THRESHOLD:
            warnings.append(
                f"Low confidence on {config.HEAD_DISPLAY[h]} ({conf:.2f}); manual review recommended."
            )

    out_of_distribution = all(heads[h]["confidence"] < config.OOD_CONFIDENCE_THRESHOLD for h in config.HEADS)
    if out_of_distribution:
        warnings.append(
            "All heads have confidence below 0.5: the image may be out of distribution "
            "(unseen modality or view). Manual quality assessment is recommended."
        )

    readiness = ai_readiness(scores)
    return {
        "heads": heads,
        "ai_readiness_score": readiness,
        "verdict": verdict(scores, readiness),
        "artifact_present": int(round(scores["artifact"])) <= config.VERDICT_THRESHOLDS["head_review_max"],
        "out_of_distribution": out_of_distribution,
        "warnings": warnings,
    }
