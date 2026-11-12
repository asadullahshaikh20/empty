import pytest

from iqa_ml import config
from iqa_ml.scoring import ai_readiness, confidence_level, probs_to_score, score_label, summarize, verdict

ALL = config.HEADS


def scores(default=100.0, **override):
    s = {h: default for h in ALL}
    s.update(override)
    return s


def test_probs_to_score():
    assert probs_to_score([1, 0, 0]) == 100
    assert probs_to_score([0, 1, 0]) == 50
    assert probs_to_score([0, 0, 1]) == 0
    assert probs_to_score([0.5, 0.5, 0]) == 75
    with pytest.raises(ValueError):
        probs_to_score([1, 0])


@pytest.mark.parametrize(
    "head, score, label",
    [
        ("technical_quality", 33, "Poor quality"),
        ("technical_quality", 34, "Acceptable quality"),
        ("technical_quality", 66, "Acceptable quality"),
        ("technical_quality", 67, "High quality"),
        ("technical_quality", 33.4, "Poor quality"),
        ("technical_quality", 33.6, "Acceptable quality"),
        ("anatomical_completeness", 0, "Incomplete"),
        ("artifact", 50, "Minor artifact"),
        ("orientation", 100, "Correctly oriented"),
        ("diagnostic_suitability", 40, "Reject"),
        ("diagnostic_suitability", 41, "Review"),
        ("diagnostic_suitability", 70, "Review"),
        ("diagnostic_suitability", 71, "Pass"),
    ],
)
def test_score_labels(head, score, label):
    assert score_label(head, score) == label


@pytest.mark.parametrize("conf, level", [(0.95, "High"), (0.9, "High"), (0.89, "Medium"), (0.7, "Medium"),
                                         (0.69, "Low"), (0.5, "Low"), (0.49, "Very low"), (0.0, "Very low")])
def test_confidence_levels(conf, level):
    assert confidence_level(conf) == level


def test_readiness_weights():
    assert ai_readiness(scores(100)) == 100
    assert ai_readiness(scores(0)) == 0
    assert ai_readiness(scores(100, diagnostic_suitability=0)) == 60
    assert abs(sum(config.READINESS_WEIGHTS.values()) - 1.0) < 1e-9


@pytest.mark.parametrize(
    "s, readiness, expected",
    [
        (scores(100), 100, "Pass"),
        (scores(67, diagnostic_suitability=71), 75, "Pass"),  # all boundaries just passing
        (scores(100, diagnostic_suitability=70), 88, "Review"),  # suitability 70
        (scores(100, diagnostic_suitability=41), 76, "Review"),
        (scores(100, diagnostic_suitability=40), 76, "Reject"),  # suitability 40
        (scores(100, technical_quality=66), 93, "Review"),  # a head at 66
        (scores(100, technical_quality=34), 87, "Review"),  # a head at 34
        (scores(100, orientation=33), 93, "Reject"),  # a head at 33
        (scores(100), 74, "Review"),  # readiness 74
        (scores(100), 50, "Review"),  # readiness 50
        (scores(100), 49, "Reject"),  # readiness 49
        (scores(100, artifact=66.6), 95, "Pass"),  # rounds to 67
        (scores(100, artifact=66.4), 95, "Review"),  # rounds to 66
    ],
)
def test_verdict_boundaries(s, readiness, expected):
    assert verdict(s, readiness) == expected


def test_verdict_threshold_override():
    assert verdict(scores(100), 74, {"readiness_pass_min": 70}) == "Pass"


def test_summarize_confident_pass():
    out = summarize({h: [0.98, 0.01, 0.01] for h in ALL})
    assert out["verdict"] == "Pass"
    assert out["ai_readiness_score"] == pytest.approx(98.5)
    assert out["warnings"] == [] and not out["out_of_distribution"] and not out["artifact_present"]
    h = out["heads"]["technical_quality"]
    assert h["label"] == "High quality" and h["confidence_level"] == "High" and h["predicted_class"] == "high"


def test_summarize_reject_and_artifact_flag():
    probs = {h: [0.9, 0.05, 0.05] for h in ALL}
    probs["artifact"] = [0.05, 0.1, 0.85]
    out = summarize(probs)
    assert out["verdict"] == "Reject"
    assert out["artifact_present"] is True
    assert out["heads"]["artifact"]["label"] == "Artifact present"


def test_summarize_low_confidence_and_ood():
    out = summarize({h: [0.4, 0.35, 0.25] for h in ALL})
    assert out["out_of_distribution"] is True
    assert len(out["warnings"]) == len(ALL) + 1
    one_low = {h: [0.95, 0.03, 0.02] for h in ALL}
    one_low["orientation"] = [0.55, 0.40, 0.05]
    out = summarize(one_low)
    assert out["out_of_distribution"] is False and len(out["warnings"]) == 1


def test_summarize_missing_head():
    with pytest.raises(ValueError):
        summarize({"technical_quality": [1, 0, 0]})
