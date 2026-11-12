import io
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from PIL import Image

ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))


def synthetic_scan(seed: int = 0, size: int = 224, color: bool = False) -> np.ndarray:
    """Textured, anatomy-like test image (ellipses + gradient + fine texture), H x W x 3 uint8."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    img = 40 + 60 * (yy / size)
    for _ in range(6):
        cy, cx = rng.uniform(0.25, 0.75, 2) * size
        ry, rx = rng.uniform(0.08, 0.25, 2) * size
        mask = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2 <= 1
        img[mask] += rng.uniform(40, 90)
    img += 12 * np.sin(xx / 3.0) * np.cos(yy / 4.0)  # fine texture (sensitive to blur)
    img += rng.uniform(-10, 10, size=img.shape)  # pixel-level texture
    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack([img] * 3, axis=-1)
    if color:
        rgb[..., 0] = np.clip(rgb[..., 0].astype(int) + 40, 0, 255)
    return rgb


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def scan():
    return synthetic_scan(0)


@pytest.fixture
def tiny_manifest(tmp_path):
    """A manifest with 4 modalities x (6 train, 3 val, 3 test) synthetic images."""
    rows = []
    i = 0
    for modality in ("xray", "ct", "mri", "retinal"):
        for split, n in (("train", 6), ("val", 3), ("test", 3)):
            for _ in range(n):
                rel = Path("images") / modality / f"img_{i:04d}.png"
                (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(tmp_path / rel), cv2.cvtColor(synthetic_scan(i, color=modality == "retinal"), cv2.COLOR_RGB2BGR))
                rows.append({"path": rel.as_posix(), "modality": modality, "source": "synthetic", "split": split})
                i += 1
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame(rows).to_csv(manifest, index=False)
    return manifest
