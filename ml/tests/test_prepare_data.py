"""prepare_data.py: folder and IXI sources (no network: the IXI tar is built locally)."""

import io
import sys
import tarfile

import cv2
import numpy as np
import pandas as pd
import pytest

from conftest import ML_ROOT, synthetic_scan

sys.path.insert(0, str(ML_ROOT / "scripts"))
import prepare_data  # noqa: E402


def test_pad_to_square():
    out = prepare_data.pad_to_square(np.ones((10, 4, 3), np.uint8))
    assert out.shape == (10, 10, 3) and out[:, :3].sum() == 0 and out[:, 3:7].min() == 1 and out[:, 7:].sum() == 0


def test_folder_source(tmp_path):
    raw = tmp_path / "raw"
    for modality, n in (("xray", 10), ("retinal", 7)):
        (raw / modality / "sub").mkdir(parents=True)
        for i in range(n):
            cv2.imwrite(str(raw / modality / "sub" / f"{i}.png"), synthetic_scan(i, size=300))
    (raw / "xray" / "notes.txt").write_text("ignored")
    cv2.imwrite(str(raw / "xray" / "small.png"), synthetic_scan(0, size=64))  # rejected: < 128 px
    out = tmp_path / "prepared"
    prepare_data.main(["--source", "folder", "--raw", str(raw), "--out", str(out)])
    df = pd.read_csv(out / "manifest.csv")
    assert set(df["modality"]) == {"xray", "retinal"}
    assert (df["modality"] == "retinal").sum() == 7 and (df["modality"] == "xray").sum() == 10
    assert set(df["split"]) <= {"train", "val", "test"} and (df["split"] == "train").sum() >= 10
    img = cv2.imread(str(out / df.iloc[0]["path"]))
    assert img.shape == (224, 224, 3)

    # --append adds without duplicating
    prepare_data.main(["--source", "folder", "--raw", str(raw), "--out", str(out), "--append"])
    assert len(pd.read_csv(out / "manifest.csv")) == 17


def _nifti_bytes(seed: int) -> bytes:
    import nibabel as nib

    rng = np.random.default_rng(seed)
    vol = rng.uniform(0, 1000, size=(40, 48, 30)).astype(np.float32)
    affine = np.diag([1.2, 0.9, 0.9, 1.0])  # anisotropic voxels, like IXI T1
    img = nib.Nifti1Image(vol, affine)
    img.header.set_zooms((1.2, 0.9, 0.9))
    path = io.BytesIO()
    fh = nib.FileHolder(fileobj=path)
    img.to_file_map({"header": fh, "image": fh})
    return path.getvalue()


def test_axial_slices(tmp_path):
    import gzip

    p = tmp_path / "v.nii.gz"
    p.write_bytes(gzip.compress(_nifti_bytes(0)))
    slices = prepare_data._axial_slices(p, 3)
    assert len(slices) == 3
    for s in slices:
        assert s.dtype == np.uint8 and s.max() == 255 and s.min() == 0
        # x spacing 1.2 resampled to 0.9: 40 * 1.2 / 0.9 = 53 columns after the 90 degree rotation
        assert s.shape == (48, 53)


def test_ixi_source_from_local_tar(tmp_path):
    import gzip

    cache = tmp_path / "cache"
    cache.mkdir()
    with tarfile.open(cache / "IXI-T1.tar", "w") as tar:
        for i in range(10):
            data = gzip.compress(_nifti_bytes(i))
            info = tarfile.TarInfo(f"IXI{i:03d}-Guys-0000-T1.nii.gz")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    out = tmp_path / "prepared"
    prepare_data.main(["--source", "ixi", "--out", str(out), "--cache", str(cache), "--max-per-modality", "30"])
    df = pd.read_csv(out / "manifest.csv")
    assert len(df) == 30 and set(df["modality"]) == {"mri"}
    # all slices of one subject are in the same split (no subject leakage)
    df["subject"] = df["path"].str.extract(r"ixi_(IXI\d+)")
    assert df.groupby("subject")["split"].nunique().max() == 1
    assert sorted(df.groupby("split")["subject"].nunique().to_dict().items()) == [("test", 2), ("train", 6), ("val", 2)]


def test_folder_source_without_images(tmp_path):
    (tmp_path / "raw").mkdir()
    with pytest.raises(RuntimeError):
        prepare_data.main(["--source", "folder", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "o")])
