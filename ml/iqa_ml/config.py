"""Single source of truth for heads, class names, thresholds and model settings.

Values follow SRS section 4.4 (Tables 25, 25b, 25c) and SDS section 5.1.2.
"""

from __future__ import annotations

# ---------------------------------------------------------------- image input
IMAGE_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
MIN_IMAGE_SIDE = 128  # FR-IMG-12
THUMBNAIL_SIZE = 256  # FR-IMG-09

# ---------------------------------------------------------------- modalities
MODALITIES = ("xray", "ct", "mri", "retinal")
MODALITY_DISPLAY = {"xray": "X-ray", "ct": "CT", "mri": "MRI", "retinal": "Retinal"}

# DICOM (0008,0060) Modality tag -> our modality
DICOM_MODALITY_MAP = {
    "CR": "xray",
    "DX": "xray",
    "RG": "xray",
    "MG": "xray",
    "CT": "ct",
    "MR": "mri",
    "OP": "retinal",
    "OPT": "retinal",
    "XC": "retinal",
}

# ---------------------------------------------------------------- quality heads
# Every head is a 3-class classifier. Class index 0 is always the GOOD state,
# 1 the intermediate state and 2 the BAD state ("severity" 0/1/2).
HEADS = (
    "technical_quality",
    "anatomical_completeness",
    "artifact",
    "orientation",
    "diagnostic_suitability",
)

HEAD_DISPLAY = {
    "technical_quality": "Technical Quality",
    "anatomical_completeness": "Anatomical Completeness",
    "artifact": "Artifact Detection",
    "orientation": "Orientation Correctness",
    "diagnostic_suitability": "Diagnostic Suitability",
}

# Class names, index = severity (0 good .. 2 bad)
HEAD_CLASSES = {
    "technical_quality": ("high", "acceptable", "poor"),
    "anatomical_completeness": ("complete", "partial", "incomplete"),
    "artifact": ("none", "minor", "present"),
    "orientation": ("correct", "slight", "significant"),
    "diagnostic_suitability": ("pass", "review", "reject"),
}
NUM_CLASSES = 3

# Score -> label ranges (SRS Table 25). Each tuple: (low, high, label), inclusive bounds
# on the integer-rounded score.
SCORE_LABELS = {
    "technical_quality": ((0, 33, "Poor quality"), (34, 66, "Acceptable quality"), (67, 100, "High quality")),
    "anatomical_completeness": ((0, 33, "Incomplete"), (34, 66, "Partially complete"), (67, 100, "Complete")),
    "artifact": ((0, 33, "Artifact present"), (34, 66, "Minor artifact"), (67, 100, "No artifact")),
    "orientation": (
        (0, 33, "Significantly misoriented"),
        (34, 66, "Slightly misoriented"),
        (67, 100, "Correctly oriented"),
    ),
    "diagnostic_suitability": ((0, 40, "Reject"), (41, 70, "Review"), (71, 100, "Pass")),
}

# Score assigned to each class when converting softmax output to a 0-100 score:
# score = sum_k p_k * CLASS_SCORE[k]
CLASS_SCORE = (100.0, 50.0, 0.0)
assert len(CLASS_SCORE) == NUM_CLASSES

# ---------------------------------------------------------------- AI-Readiness / verdict
READINESS_WEIGHTS = {
    "diagnostic_suitability": 0.40,
    "technical_quality": 0.20,
    "anatomical_completeness": 0.15,
    "artifact": 0.15,
    "orientation": 0.10,
}

VERDICT_THRESHOLDS = {
    # SRS 4.4
    "head_reject_below": 34,  # any head score < 34 -> Reject
    "head_review_max": 66,  # any head score <= 66 -> at least Review
    "suitability_reject_max": 40,  # suitability <= 40 -> Reject
    "suitability_review_max": 70,  # suitability <= 70 -> at least Review
    # SDS 5.1.2 classify()
    "readiness_reject_below": 50,  # readiness < 50 -> Reject
    "readiness_pass_min": 75,  # readiness < 75 -> at least Review
}

VERDICTS = ("Pass", "Review", "Reject")

# ---------------------------------------------------------------- confidence (SRS Table 25c)
CONFIDENCE_LEVELS = ((0.90, "High"), (0.70, "Medium"), (0.50, "Low"), (0.0, "Very low"))
LOW_CONFIDENCE_THRESHOLD = 0.6
OOD_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------- model (SRS Table 25b)
EFFNET_FEATURES = 1792  # EfficientNet-B4 pooled feature size
VIT_FEATURES = 768  # ViT-B/16 CLS token size
FUSED_FEATURES = 512
HEAD_HIDDEN = 256
HEAD_DROPOUT = 0.3
