"""Synthetic degradation pipeline (Proposal Phase 2, SDS SyntheticDegradationPipeline).

Takes a clean H x W x 3 uint8 image and applies controlled, labelled degradations.
Every quality head except diagnostic suitability gets a severity in {0, 1, 2}
(0 = good, 1 = intermediate, 2 = bad). Diagnostic suitability is derived:

    suitability = max(severity of the four other heads)

so an image with any severe defect is "reject", any minor defect "review", and a
fully clean image "pass", matching the verdict logic of SRS section 4.4.

Sampling is balanced over the suitability label: by default one third of samples are
clean, one third have at most minor defects, one third have at least one severe defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEGRADED_HEADS = ("technical_quality", "anatomical_completeness", "artifact", "orientation")

# Parameter ranges per severity (index 1 = minor, 2 = severe). Calibrated for 224x224 input.
TECH_PARAMS = {
    "gaussian_blur": {1: (1.0, 1.8), 2: (2.6, 4.5)},  # sigma (px)
    "motion_blur": {1: (5, 9), 2: (13, 21)},  # kernel length (px)
    "gaussian_noise": {1: (8.0, 15.0), 2: (24.0, 40.0)},  # std (intensity units)
    "speckle_noise": {1: (0.08, 0.15), 2: (0.25, 0.40)},  # relative std
    "low_contrast": {1: (0.45, 0.60), 2: (0.15, 0.30)},  # contrast factor
    "underexposure": {1: (1.8, 2.5), 2: (3.5, 5.0)},  # gamma (>1 darkens)
    "overexposure": {1: (0.45, 0.60), 2: (0.20, 0.30)},  # gamma (<1 brightens)
}
ANATOMY_FRACTION = {1: (0.15, 0.30), 2: (0.35, 0.55)}  # fraction of image shifted out of view
RING_COUNT = {1: (3, 6), 2: (6, 12)}  # number of rings per severity
ARTIFACT_PARAMS = {
    "ghosting": {1: (0.15, 0.25), 2: (0.35, 0.50)},  # ghost blend weight
    "ring": {1: (25.0, 40.0), 2: (50.0, 80.0)},  # thin-ring amplitude
    "banding": {1: (12.0, 20.0), 2: (30.0, 50.0)},  # stripe amplitude
}
ORIENT_SLIGHT_DEG = (8.0, 20.0)
ORIENT_SIGNIFICANT_DEG = (25.0, 180.0)


@dataclass
class DegradationResult:
    image: np.ndarray
    labels: dict[str, int]  # all five heads, severity 0..2
    applied: list[str] = field(default_factory=list)


# ------------------------------------------------------------------ primitives
def _as_float(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)


def motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    length = max(3, int(length) | 1)  # odd
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2 - 0.5, length / 2 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    total = kernel.sum()
    if total <= 0:
        kernel[length // 2, :] = 1.0
        total = kernel.sum()
    kernel /= total
    return cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_REFLECT)


def gaussian_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=img.shape[:2]).astype(np.float32)[..., None]
    return _to_uint8(_as_float(img) + noise)


def speckle_noise(img: np.ndarray, std: float, rng: np.random.Generator) -> np.ndarray:
    noise = rng.normal(0.0, std, size=img.shape[:2]).astype(np.float32)[..., None]
    return _to_uint8(_as_float(img) * (1.0 + noise))


def adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    f = _as_float(img)
    return _to_uint8((f - f.mean()) * factor + f.mean())


def adjust_gamma(img: np.ndarray, gamma: float) -> np.ndarray:
    lut = (np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255.0)
    return cv2.LUT(img, _to_uint8(lut))


def shift_out(img: np.ndarray, fraction: float, rng: np.random.Generator) -> np.ndarray:
    """Translate the image so `fraction` of it leaves the frame (black fill)."""
    h, w = img.shape[:2]
    if rng.random() < 0.5:  # one axis
        if rng.random() < 0.5:
            dx, dy = fraction * w * rng.choice([-1, 1]), 0.0
        else:
            dx, dy = 0.0, fraction * h * rng.choice([-1, 1])
    else:  # split between both axes so that the removed area is ~fraction
        a = rng.uniform(0.3, 0.7)
        # removed area = 1 - (1 - fx)(1 - fy); choose fx then solve fy
        fx = fraction * a
        fy = min(1.0, max(0.0, 1 - (1 - fraction) / max(1e-6, 1 - fx)))
        dx = fx * w * rng.choice([-1, 1])
        dy = fy * h * rng.choice([-1, 1])
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def ghosting(img: np.ndarray, weight: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    shift = int(rng.integers(max(4, w // 28), max(6, w // 9)))
    if rng.random() < 0.5:
        m = np.float32([[1, 0, shift * rng.choice([-1, 1])], [0, 1, 0]])
    else:
        m = np.float32([[1, 0, 0], [0, 1, shift * rng.choice([-1, 1])]])
    ghost = cv2.warpAffine(img, m, (w, h), borderMode=cv2.BORDER_WRAP)
    return _to_uint8((1 - weight) * _as_float(img) + weight * _as_float(ghost))


def ring_artifact(img: np.ndarray, amplitude: float, n_rings: int, rng: np.random.Generator) -> np.ndarray:
    """Sparse thin concentric rings (as produced by miscalibrated CT detector elements)."""
    h, w = img.shape[:2]
    cy = h / 2 + rng.uniform(-0.05, 0.05) * h
    cx = w / 2 + rng.uniform(-0.05, 0.05) * w
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    pattern = np.zeros((h, w), dtype=np.float32)
    for _ in range(n_rings):
        radius = rng.uniform(0.04, 0.45) * min(h, w)
        width = rng.uniform(1.0, 2.5)
        sign = rng.choice([-1.0, 1.0])
        pattern += sign * amplitude * np.exp(-0.5 * ((r - radius) / width) ** 2)
    return _to_uint8(_as_float(img) + pattern[..., None])


def banding(img: np.ndarray, amplitude: float, rng: np.random.Generator) -> np.ndarray:
    h, w = img.shape[:2]
    period = rng.uniform(4.0, 12.0)
    phase = rng.uniform(0, 2 * np.pi)
    if rng.random() < 0.5:
        line = amplitude * np.sin(2 * np.pi * np.arange(h, dtype=np.float32) / period + phase)
        pattern = np.repeat(line[:, None], w, axis=1)
    else:
        line = amplitude * np.sin(2 * np.pi * np.arange(w, dtype=np.float32) / period + phase)
        pattern = np.repeat(line[None, :], h, axis=0)
    return _to_uint8(_as_float(img) + pattern[..., None])


# ------------------------------------------------------------------ pipeline
class SyntheticDegradationPipeline:
    """Apply labelled synthetic degradations. Call with an RNG for reproducibility."""

    def __init__(self, p_clean: float = 1 / 3, p_minor: float = 1 / 3, p_severe: float = 1 / 3):
        total = p_clean + p_minor + p_severe
        if total <= 0 or min(p_clean, p_minor, p_severe) < 0:
            raise ValueError("probabilities must be non-negative and not all zero")
        self.p = np.array([p_clean, p_minor, p_severe], dtype=np.float64) / total

    # -- label sampling -------------------------------------------------------
    def sample_severities(self, rng: np.random.Generator) -> dict[str, int]:
        target = int(rng.choice(3, p=self.p))
        sev = {h: 0 for h in DEGRADED_HEADS}
        if target == 1:
            for h in DEGRADED_HEADS:
                sev[h] = int(rng.random() < 0.5)
            if max(sev.values()) == 0:
                sev[DEGRADED_HEADS[int(rng.integers(len(DEGRADED_HEADS)))]] = 1
        elif target == 2:
            for h in DEGRADED_HEADS:
                sev[h] = int(rng.choice(3, p=[0.45, 0.25, 0.30]))
            if max(sev.values()) < 2:
                sev[DEGRADED_HEADS[int(rng.integers(len(DEGRADED_HEADS)))]] = 2
        return sev

    @staticmethod
    def full_labels(severities: dict[str, int]) -> dict[str, int]:
        labels = {h: int(severities.get(h, 0)) for h in DEGRADED_HEADS}
        labels["diagnostic_suitability"] = max(labels.values())
        return labels

    # -- application ---------------------------------------------------------
    def _technical(self, img, sev, rng, applied):
        names = list(TECH_PARAMS)
        n = 1 if sev == 1 else int(rng.integers(1, 3))
        for name in rng.choice(names, size=n, replace=False):
            lo, hi = TECH_PARAMS[name][sev]
            if name == "gaussian_blur":
                img = gaussian_blur(img, rng.uniform(lo, hi))
            elif name == "motion_blur":
                img = motion_blur(img, int(rng.integers(lo, hi + 1)), rng.uniform(0, 180))
            elif name == "gaussian_noise":
                img = gaussian_noise(img, rng.uniform(lo, hi), rng)
            elif name == "speckle_noise":
                img = speckle_noise(img, rng.uniform(lo, hi), rng)
            elif name == "low_contrast":
                img = adjust_contrast(img, rng.uniform(lo, hi))
            elif name == "underexposure":
                img = adjust_gamma(img, rng.uniform(lo, hi))
            elif name == "overexposure":
                img = adjust_gamma(img, rng.uniform(lo, hi))
            applied.append(f"technical:{name}:{sev}")
        return img

    def _anatomical(self, img, sev, rng, applied):
        lo, hi = ANATOMY_FRACTION[sev]
        img = shift_out(img, rng.uniform(lo, hi), rng)
        applied.append(f"anatomical:shift_out:{sev}")
        return img

    def _artifact(self, img, sev, rng, applied):
        name = str(rng.choice(list(ARTIFACT_PARAMS)))
        lo, hi = ARTIFACT_PARAMS[name][sev]
        if name == "ghosting":
            img = ghosting(img, rng.uniform(lo, hi), rng)
        elif name == "ring":
            img = ring_artifact(img, rng.uniform(lo, hi), int(rng.integers(RING_COUNT[sev][0], RING_COUNT[sev][1] + 1)), rng)
        else:
            img = banding(img, rng.uniform(lo, hi), rng)
        applied.append(f"artifact:{name}:{sev}")
        return img

    def _orientation(self, img, sev, rng, applied):
        if sev == 1:
            angle = rng.uniform(*ORIENT_SLIGHT_DEG) * rng.choice([-1, 1])
            applied.append("orientation:rotate:1")
            return rotate(img, angle)
        mode = int(rng.integers(3))
        if mode == 0:
            angle = rng.uniform(*ORIENT_SIGNIFICANT_DEG) * rng.choice([-1, 1])
            applied.append("orientation:rotate:2")
            return rotate(img, angle)
        if mode == 1:
            k = int(rng.integers(1, 4))
            applied.append(f"orientation:rot90x{k}:2")
            return np.ascontiguousarray(np.rot90(img, k))
        applied.append("orientation:vflip:2")
        return np.ascontiguousarray(img[::-1])

    def apply(self, image: np.ndarray, severities: dict[str, int], rng: np.random.Generator) -> DegradationResult:
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be H x W x 3 uint8")
        if image.shape[0] != image.shape[1]:
            raise ValueError("image must be square (resize before degrading)")
        for h, s in severities.items():
            if h not in DEGRADED_HEADS or s not in (0, 1, 2):
                raise ValueError(f"invalid severity {h}={s}")
        img = image.copy()
        applied: list[str] = []
        # Order: positioning first, then acquisition artefacts, then technical degradation.
        if severities.get("anatomical_completeness", 0):
            img = self._anatomical(img, severities["anatomical_completeness"], rng, applied)
        if severities.get("orientation", 0):
            img = self._orientation(img, severities["orientation"], rng, applied)
        if severities.get("artifact", 0):
            img = self._artifact(img, severities["artifact"], rng, applied)
        if severities.get("technical_quality", 0):
            img = self._technical(img, severities["technical_quality"], rng, applied)
        return DegradationResult(image=np.ascontiguousarray(img), labels=self.full_labels(severities), applied=applied)

    def __call__(self, image: np.ndarray, rng: np.random.Generator) -> DegradationResult:
        return self.apply(image, self.sample_severities(rng), rng)
