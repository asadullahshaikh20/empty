import cv2
import numpy as np
import pytest

from conftest import synthetic_scan
from iqa_ml import degradation as D
from iqa_ml.degradation import DEGRADED_HEADS, SyntheticDegradationPipeline


def sharpness(img):
    return float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())


@pytest.fixture
def pipe():
    return SyntheticDegradationPipeline()


def test_label_sampler_is_balanced_and_consistent(pipe):
    rng = np.random.default_rng(0)
    counts = np.zeros(3)
    for _ in range(6000):
        sev = pipe.sample_severities(rng)
        labels = pipe.full_labels(sev)
        assert set(sev) == set(DEGRADED_HEADS)
        assert all(v in (0, 1, 2) for v in sev.values())
        assert labels["diagnostic_suitability"] == max(sev.values())
        counts[labels["diagnostic_suitability"]] += 1
    frac = counts / counts.sum()
    assert np.allclose(frac, [1 / 3, 1 / 3, 1 / 3], atol=0.03)


def test_clean_sample_is_unchanged(pipe, scan):
    res = pipe.apply(scan, {h: 0 for h in DEGRADED_HEADS}, np.random.default_rng(1))
    assert np.array_equal(res.image, scan)
    assert res.applied == []
    assert res.labels == {**{h: 0 for h in DEGRADED_HEADS}, "diagnostic_suitability": 0}


@pytest.mark.parametrize("head", DEGRADED_HEADS)
@pytest.mark.parametrize("sev", [1, 2])
def test_each_degradation_changes_image_and_labels(pipe, scan, head, sev):
    for seed in range(8):
        sevs = {h: 0 for h in DEGRADED_HEADS}
        sevs[head] = sev
        res = pipe.apply(scan, sevs, np.random.default_rng(seed))
        assert res.image.shape == scan.shape and res.image.dtype == np.uint8
        diff = np.abs(res.image.astype(int) - scan.astype(int)).mean()
        assert diff > 1.0, f"{head} sev {sev} seed {seed}: image barely changed ({diff:.2f})"
        assert res.labels[head] == sev
        assert res.labels["diagnostic_suitability"] == sev
        assert all(a.startswith(head.split("_")[0]) for a in res.applied)


def test_blur_severity_ordering(scan):
    s0 = sharpness(scan)
    s1 = sharpness(D.gaussian_blur(scan, 1.4))
    s2 = sharpness(D.gaussian_blur(scan, 3.5))
    assert s0 > s1 > s2


def test_ring_artifact_is_sparse_and_ordered(scan):
    minor = [np.abs(D.ring_artifact(scan, 30, 4, np.random.default_rng(s)).astype(int) - scan).mean() for s in range(10)]
    severe = [np.abs(D.ring_artifact(scan, 65, 9, np.random.default_rng(s)).astype(int) - scan).mean() for s in range(10)]
    assert min(minor) > 0.5 and np.mean(severe) > 2 * np.mean(minor)
    # thin rings: most pixels are untouched (unlike a full-image sinusoid)
    changed = (np.abs(D.ring_artifact(scan, 30, 4, np.random.default_rng(0)).astype(int) - scan) > 3).mean()
    assert changed < 0.35


def test_noise_severity_ordering(scan):
    rng = np.random.default_rng(0)
    d1 = np.abs(D.gaussian_noise(scan, 10, rng).astype(int) - scan).mean()
    d2 = np.abs(D.gaussian_noise(scan, 30, rng).astype(int) - scan).mean()
    assert d2 > d1 > 0


@pytest.mark.parametrize("fraction", [0.2, 0.45])
def test_shift_out_removes_expected_fraction(fraction):
    white = np.full((224, 224, 3), 255, np.uint8)
    for seed in range(10):
        out = D.shift_out(white, fraction, np.random.default_rng(seed))
        removed = (out[..., 0] < 128).mean()
        assert abs(removed - fraction) < 0.03, (seed, removed)


def test_motion_blur_kernel_valid(scan):
    out = D.motion_blur(scan, 9, 37.0)
    assert out.shape == scan.shape
    assert abs(float(out.mean()) - float(scan.mean())) < 3  # kernel sums to 1


def test_deterministic_with_seed(pipe, scan):
    a = pipe(scan, np.random.default_rng([5, 9]))
    b = pipe(scan, np.random.default_rng([5, 9]))
    assert np.array_equal(a.image, b.image) and a.labels == b.labels and a.applied == b.applied
    different = [pipe(scan, np.random.default_rng([6, i])).labels for i in range(20)]
    assert len({tuple(sorted(lab.items())) for lab in different}) > 1


def test_input_validation(pipe, scan):
    with pytest.raises(ValueError):
        pipe.apply(scan.astype(np.float32), {}, np.random.default_rng())
    with pytest.raises(ValueError):
        pipe.apply(scan[:200], {}, np.random.default_rng())
    with pytest.raises(ValueError):
        pipe.apply(scan, {"technical_quality": 3}, np.random.default_rng())
    with pytest.raises(ValueError):
        SyntheticDegradationPipeline(0, 0, 0)


def test_input_not_modified(pipe):
    img = synthetic_scan(3)
    copy = img.copy()
    for seed in range(10):
        pipe(img, np.random.default_rng(seed))
    assert np.array_equal(img, copy)
