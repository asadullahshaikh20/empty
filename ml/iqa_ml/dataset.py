"""Dataset that degrades clean images on the fly and returns quality labels.

Manifest CSV columns: path (relative to the manifest's folder), modality, source, split.
Validation/test samples are deterministic: sample i always receives the same degradation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from . import config
from .degradation import SyntheticDegradationPipeline
from .preprocessing import ImagePreprocessor

MANIFEST_COLUMNS = ("path", "modality", "source", "split")


def read_manifest(manifest_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(manifest_path)
    missing = [c for c in MANIFEST_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"manifest is missing columns {missing}")
    bad = set(df["modality"]) - set(config.MODALITIES)
    if bad:
        raise ValueError(f"manifest has unknown modalities {sorted(bad)}")
    return df


def light_augment(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Train-only augmentation: scale 0.95-1.05 and rotation <= 3 degrees.

    No flips and no larger rotations: those would change the orientation label.
    """
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), rng.uniform(-3.0, 3.0), rng.uniform(0.95, 1.05))
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


class QualityDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        split: str,
        train: bool = False,
        seed: int = 0,
        max_samples: int | None = None,
        pipeline: SyntheticDegradationPipeline | None = None,
    ):
        self.root = Path(manifest_path).parent
        df = read_manifest(manifest_path)
        df = df[df["split"] == split].reset_index(drop=True)
        if max_samples is not None and len(df) > max_samples:
            # keep modality balance when subsampling
            parts = [
                g.sample(n=max(1, round(max_samples * len(g) / len(df))), random_state=seed)
                for _, g in df.groupby("modality")
            ]
            df = pd.concat(parts).reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f"no samples for split {split!r} in {manifest_path}")
        self.df = df
        self.train = train
        self.seed = seed
        self.pipeline = pipeline or SyntheticDegradationPipeline()
        self.preprocessor = ImagePreprocessor()
        self._counter = 0

    def __len__(self) -> int:
        return len(self.df)

    def _rng(self, index: int) -> np.random.Generator:
        if self.train:
            self._counter += 1  # different degradation each time an index is revisited
            return np.random.default_rng([self.seed, torch.initial_seed() % (2**32), index, self._counter])
        return np.random.default_rng([self.seed, index])

    def load_clean(self, index: int) -> np.ndarray:
        row = self.df.iloc[index]
        loaded = self.preprocessor.load(self.root / row["path"], validate_size=False)
        return self.preprocessor.resize(loaded.pixels)

    def sample(self, index: int):
        """Return (degraded uint8 image, labels dict, applied list, modality)."""
        rng = self._rng(index)
        img = self.load_clean(index)
        if self.train:
            img = light_augment(img, rng)
        result = self.pipeline(img, rng)
        return result.image, result.labels, result.applied, self.df.iloc[index]["modality"]

    def __getitem__(self, index: int):
        img, labels, _, modality = self.sample(index)
        x = self.preprocessor.to_tensor(img)
        y = torch.tensor([labels[h] for h in config.HEADS], dtype=torch.long)
        m = torch.tensor(config.MODALITIES.index(modality), dtype=torch.long)
        return x, y, m


def estimate_class_weights(
    pipeline: SyntheticDegradationPipeline | None = None, n: int = 50_000, seed: int = 0
) -> dict[str, list[float]]:
    """Inverse-frequency class weights per head, estimated from the label sampler (no images needed)."""
    pipeline = pipeline or SyntheticDegradationPipeline()
    rng = np.random.default_rng(seed)
    counts = {h: np.zeros(config.NUM_CLASSES) for h in config.HEADS}
    for _ in range(n):
        labels = pipeline.full_labels(pipeline.sample_severities(rng))
        for h in config.HEADS:
            counts[h][labels[h]] += 1
    weights = {}
    for h, c in counts.items():
        c = np.maximum(c, 1.0)
        w = c.sum() / (config.NUM_CLASSES * c)
        weights[h] = (w / w.mean()).round(4).tolist()
    return weights
