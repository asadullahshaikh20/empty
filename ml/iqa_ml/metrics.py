"""Per-head evaluation metrics: accuracy, macro precision/recall/F1, one-vs-rest AUC, confusion matrix."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(y_true, probs, num_classes: int) -> dict:
    y_true = np.asarray(y_true, dtype=np.int64)
    probs = np.asarray(probs, dtype=np.float64)
    if y_true.size == 0:
        raise ValueError("no samples")
    y_pred = probs.argmax(axis=1)
    labels = list(range(num_classes))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    present = np.unique(y_true)
    if len(present) < 2:
        auc = None  # AUC undefined with a single class present
    else:
        # restrict to classes present in y_true; renormalise their probabilities
        sub = probs[:, present]
        sub = sub / np.clip(sub.sum(axis=1, keepdims=True), 1e-12, None)
        if len(present) == 2:
            auc = float(roc_auc_score((y_true == present[1]).astype(int), sub[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, sub, multi_class="ovr", average="macro", labels=list(present)))
    return {
        "n": int(y_true.size),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }
