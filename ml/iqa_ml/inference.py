"""Single-image quality assessment (used by the FastAPI backend)."""

from __future__ import annotations

import time
from pathlib import Path

import torch

from . import config, scoring
from .checkpoint import load_model
from .model import QualityAssessmentModel
from .preprocessing import ImagePreprocessor, LoadedImage, PathOrBytes


def select_device(preferred: str | None = None) -> torch.device:
    if preferred and preferred != "auto":
        return torch.device(preferred)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")  # FR-QA-06


class QualityAssessor:
    """Holds one loaded model in memory and assesses images (FR-QA-01, FR-QA-18)."""

    def __init__(self, model: QualityAssessmentModel, version: str, device: torch.device | str = "cpu"):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        self.version = version
        self.preprocessor = ImagePreprocessor()

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str | None = None) -> "QualityAssessor":
        dev = select_device(device)
        model, meta = load_model(path, device=dev)
        return cls(model, version=str(meta["version"]), device=dev)

    @torch.inference_mode()
    def predict_probs(self, tensor: torch.Tensor) -> dict[str, list[float]]:
        """tensor: 3 x H x W normalised -> softmax probabilities per head (+ modality)."""
        out = self.model(tensor.unsqueeze(0).to(self.device))
        return {k: torch.softmax(v.float(), dim=1)[0].cpu().tolist() for k, v in out.items()}

    def assess_loaded(self, loaded: LoadedImage, modality: str | None = None) -> dict:
        if modality is not None and modality not in config.MODALITIES:
            raise ValueError(f"unknown modality {modality!r}")
        start = time.perf_counter()
        probs = self.predict_probs(self.preprocessor.preprocess(loaded.pixels))
        result = scoring.summarize({h: probs[h] for h in config.HEADS})

        mod_probs = probs["modality"]
        best = max(range(len(mod_probs)), key=mod_probs.__getitem__)
        predicted = {"modality": config.MODALITIES[best], "confidence": round(float(mod_probs[best]), 4)}
        if modality:
            resolved, source = modality, "user"
        elif loaded.metadata_modality:
            resolved, source = loaded.metadata_modality, "dicom"
        else:
            resolved, source = predicted["modality"], "model"

        result.update(
            {
                "modality": resolved,
                "modality_source": source,
                "modality_prediction": predicted,
                "model_version": self.version,  # FR-QA-17
                "inference_ms": int(round((time.perf_counter() - start) * 1000)),
            }
        )
        return result

    def assess(self, source: PathOrBytes, modality: str | None = None) -> dict:
        """Load (PNG/JPEG/DICOM) and assess. Raises ImageLoadError for unreadable input."""
        return self.assess_loaded(self.preprocessor.load(source), modality=modality)
