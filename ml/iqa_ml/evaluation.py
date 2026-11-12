"""Run a model over a DataLoader and compute losses and per-head metrics."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from . import config
from .metrics import classification_metrics

MODALITY_LOSS_WEIGHT = 0.5


def compute_loss(
    outputs: dict[str, torch.Tensor],
    y: torch.Tensor,
    m: torch.Tensor,
    criteria: dict[str, nn.Module],
) -> torch.Tensor:
    """Mean of the five weighted cross-entropy head losses + 0.5 x modality cross-entropy."""
    quality = sum(criteria[h](outputs[h].float(), y[:, i]) for i, h in enumerate(config.HEADS)) / len(config.HEADS)
    return quality + MODALITY_LOSS_WEIGHT * criteria["modality"](outputs["modality"].float(), m)


def make_criteria(class_weights: dict[str, list[float]] | None, device: torch.device) -> dict[str, nn.Module]:
    criteria: dict[str, nn.Module] = {}
    for h in config.HEADS:
        w = None if class_weights is None else torch.tensor(class_weights[h], dtype=torch.float32, device=device)
        criteria[h] = nn.CrossEntropyLoss(weight=w)
    criteria["modality"] = nn.CrossEntropyLoss()
    return criteria


@torch.inference_mode()
def evaluate(model: nn.Module, loader, device: torch.device, criteria: dict[str, nn.Module] | None = None) -> dict:
    model.eval()
    probs = {k: [] for k in (*config.HEADS, "modality")}
    targets = {k: [] for k in (*config.HEADS, "modality")}
    total_loss, n = 0.0, 0
    for x, y, m in loader:
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model(x)
        if criteria is not None:
            total_loss += float(compute_loss(out, y, m, criteria)) * x.shape[0]
        n += x.shape[0]
        for i, h in enumerate(config.HEADS):
            probs[h].append(torch.softmax(out[h].float(), 1).cpu().numpy())
            targets[h].append(y[:, i].cpu().numpy())
        probs["modality"].append(torch.softmax(out["modality"].float(), 1).cpu().numpy())
        targets["modality"].append(m.cpu().numpy())

    per_head = {}
    for h in config.HEADS:
        per_head[h] = classification_metrics(np.concatenate(targets[h]), np.concatenate(probs[h]), config.NUM_CLASSES)
    modality = classification_metrics(
        np.concatenate(targets["modality"]), np.concatenate(probs["modality"]), len(config.MODALITIES)
    )
    aucs = [per_head[h]["auc"] for h in config.HEADS if per_head[h]["auc"] is not None]
    return {
        "n": n,
        "loss": total_loss / n if criteria is not None and n else None,
        "heads": per_head,
        "modality": modality,
        "mean_accuracy": float(np.mean([per_head[h]["accuracy"] for h in config.HEADS])),
        "mean_f1": float(np.mean([per_head[h]["f1"] for h in config.HEADS])),
        "mean_auc": float(np.mean(aucs)) if aucs else None,
    }


def metrics_markdown(metrics: dict) -> str:
    lines = [
        "| Head | Accuracy | Precision | Recall | F1 | AUC |",
        "|---|---|---|---|---|---|",
    ]
    for h in config.HEADS:
        r = metrics["heads"][h]
        auc = "n/a" if r["auc"] is None else f"{r['auc']:.3f}"
        lines.append(
            f"| {config.HEAD_DISPLAY[h]} | {r['accuracy']:.3f} | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} | {auc} |"
        )
    mod = metrics["modality"]
    lines.append(f"| Modality (aux) | {mod['accuracy']:.3f} | {mod['precision']:.3f} | {mod['recall']:.3f} | {mod['f1']:.3f} | "
                 f"{'n/a' if mod['auc'] is None else format(mod['auc'], '.3f')} |")
    mean_auc = "n/a" if metrics["mean_auc"] is None else f"{metrics['mean_auc']:.3f}"
    lines.append(f"\nMean accuracy (5 heads): **{metrics['mean_accuracy']:.3f}**, mean F1: {metrics['mean_f1']:.3f}, "
                 f"mean AUC: {mean_auc}, samples: {metrics['n']}")
    return "\n".join(lines)
