"""Checkpoint save/load with metadata, so a checkpoint is self-describing."""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any

import torch

from . import config
from .model import QualityAssessmentModel

CHECKPOINT_FORMAT = "iqa-quality-v1"


def save_checkpoint(path: str | Path, model: QualityAssessmentModel, version: str, **extra: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "version": version,
        "created_at": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "image_size": config.IMAGE_SIZE,
        "heads": list(config.HEADS),
        "head_classes": {h: list(c) for h, c in config.HEAD_CLASSES.items()},
        "modalities": list(config.MODALITIES),
        "state_dict": model.state_dict(),
        **extra,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)  # atomic on the same filesystem


def read_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a {CHECKPOINT_FORMAT} checkpoint")
    if list(payload["heads"]) != list(config.HEADS) or list(payload["modalities"]) != list(config.MODALITIES):
        raise ValueError("checkpoint heads/modalities do not match this code version")
    if int(payload["image_size"]) != config.IMAGE_SIZE:
        raise ValueError("checkpoint image size does not match this code version")
    return payload


def load_model(path: str | Path, device: str | torch.device = "cpu") -> tuple[QualityAssessmentModel, dict]:
    payload = read_checkpoint(path, map_location="cpu")
    model = QualityAssessmentModel(pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    meta = {k: v for k, v in payload.items() if k not in ("state_dict", "optimizer", "scheduler", "scaler")}
    return model, meta
