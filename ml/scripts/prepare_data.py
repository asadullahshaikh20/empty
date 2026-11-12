"""Build the clean-image dataset + manifest used for training.

Sources
  medmnist  MedMNIST+ (CC BY 4.0): ChestMNIST -> xray (from NIH ChestX-ray14),
            OrganAMNIST -> ct (from LiTS abdominal CT), RetinaMNIST -> retinal (from DeepDRiD).
            Official MedMNIST train/val/test splits are kept (avoids patient leakage).
  ixi       IXI T1 brain MRI (CC BY-SA 3.0): NIfTI volumes -> axial PNG slices, split by subject.
  folder    Your own images in <raw>/<modality>/** (e.g. ChestX-ray14, EyePACS, BraTS slices).

Every image is padded to square and stored as 224x224 PNG under <out>/images/<modality>/.
The manifest <out>/manifest.csv has columns path, modality, source, split. Use --append
to add a source to an existing manifest.

Examples
  python scripts/prepare_data.py --source medmnist --out data/prepared --max-per-modality 6000
  python scripts/prepare_data.py --source ixi --out data/prepared --append --max-per-modality 6000
  python scripts/prepare_data.py --source folder --raw data/raw --out data/prepared --append
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iqa_ml import config  # noqa: E402
from iqa_ml.preprocessing import ImageLoadError, ImagePreprocessor, to_rgb_uint8  # noqa: E402

MEDMNIST_SOURCES = {"xray": "ChestMNIST", "ct": "OrganAMNIST", "retinal": "RetinaMNIST"}
IXI_T1_URL = "https://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar"
RAW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".dicom"}


def pad_to_square(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if h == w:
        return img
    side = max(h, w)
    out = np.zeros((side, side) + img.shape[2:], dtype=img.dtype)
    y, x = (side - h) // 2, (side - w) // 2
    out[y : y + h, x : x + w] = img
    return out


def save_png(img: np.ndarray, path: Path) -> None:
    img = pad_to_square(to_rgb_uint8(img))
    size = config.IMAGE_SIZE
    if img.shape[0] != size:
        interp = cv2.INTER_AREA if img.shape[0] > size else cv2.INTER_LINEAR
        img = cv2.resize(img, (size, size), interpolation=interp)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR)):
        raise OSError(f"failed to write {path}")


def split_counts(n_total: int) -> dict[str, int]:
    n_val = int(round(n_total * 0.15))
    n_test = int(round(n_total * 0.15))
    return {"train": n_total - n_val - n_test, "val": n_val, "test": n_test}


# ----------------------------------------------------------------------------- medmnist
def prepare_medmnist(out: Path, cache: Path, max_per_modality: int, size: int, seed: int,
                     modalities: list[str] | None = None) -> list[dict]:
    import medmnist

    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    caps = split_counts(max_per_modality)
    for modality, cls_name in MEDMNIST_SOURCES.items():
        if modalities and modality not in modalities:
            continue
        cls = getattr(medmnist, cls_name)
        for split in ("train", "val", "test"):
            ds = cls(split=split, download=True, size=size, root=str(cache))
            imgs = ds.imgs
            n = min(len(imgs), caps[split])
            idx = np.sort(rng.choice(len(imgs), size=n, replace=False))
            for i in tqdm(idx, desc=f"{cls_name}/{split}", leave=False):
                rel = Path("images") / modality / f"medmnist_{cls_name.lower()}_{split}_{i:06d}.png"
                save_png(imgs[i], out / rel)
                rows.append({"path": rel.as_posix(), "modality": modality, "source": cls_name, "split": split})
            print(f"{cls_name:12s} {split:5s}: {n} images")
    return rows


# ----------------------------------------------------------------------------- IXI
def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"Downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:  # noqa: S310 (fixed https/http URL)
        total = int(resp.headers.get("Content-Length", 0))
        with tqdm(total=total, unit="B", unit_scale=True) as bar:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
    tmp.replace(dest)


def _axial_slices(nifti_path: Path, n_slices: int) -> list[np.ndarray]:
    import nibabel as nib

    img = nib.as_closest_canonical(nib.load(str(nifti_path)))  # RAS orientation
    vol = np.asarray(img.dataobj, dtype=np.float32)
    zooms = img.header.get_zooms()[:3]
    nz = vol.shape[2]
    centre = nz // 2
    offsets = np.linspace(-0.15, 0.15, n_slices) if n_slices > 1 else [0.0]
    slices = []
    for off in offsets:
        k = int(np.clip(centre + off * nz, 0, nz - 1))
        sl = vol[:, :, k]
        # resample to isotropic in-plane spacing, then rotate so anterior is up
        sx, sy = float(zooms[0]), float(zooms[1])
        if abs(sx - sy) > 1e-3:
            target = min(sx, sy)
            sl = cv2.resize(
                sl, (int(round(sl.shape[1] * sy / target)), int(round(sl.shape[0] * sx / target))),
                interpolation=cv2.INTER_LINEAR,
            )
        sl = np.rot90(sl)
        lo, hi = np.percentile(sl, [0.5, 99.5])
        sl = np.clip((sl - lo) / max(hi - lo, 1e-6), 0, 1) * 255
        slices.append(sl.astype(np.uint8))
    return slices


def prepare_ixi(out: Path, cache: Path, max_per_modality: int, seed: int, slices_per_volume: int = 3) -> list[dict]:
    tar_path = cache / "IXI-T1.tar"
    _download(IXI_T1_URL, tar_path)
    extract_dir = cache / "IXI-T1"
    if not extract_dir.exists():
        print("Extracting IXI-T1.tar ...")
        with tarfile.open(tar_path) as tar:
            if hasattr(tarfile, "data_filter"):  # Python 3.12+, 3.11.4+, 3.10.12+
                tar.extractall(extract_dir, filter="data")
            else:
                tar.extractall(extract_dir)  # noqa: S202 - older Python without extraction filters
    volumes = sorted(extract_dir.rglob("*.nii.gz"))
    if not volumes:
        raise RuntimeError(f"no NIfTI volumes found in {extract_dir}")
    rng = np.random.default_rng(seed)
    n_vol = min(len(volumes), max(1, max_per_modality // slices_per_volume))
    chosen = [volumes[i] for i in sorted(rng.choice(len(volumes), size=n_vol, replace=False))]
    order = rng.permutation(len(chosen))  # split by subject
    counts = split_counts(len(chosen))
    split_of = {}
    for rank, i in enumerate(order):
        split_of[i] = "train" if rank < counts["train"] else ("val" if rank < counts["train"] + counts["val"] else "test")
    rows = []
    for i, vol_path in enumerate(tqdm(chosen, desc="IXI-T1")):
        try:
            slices = _axial_slices(vol_path, slices_per_volume)
        except Exception as exc:  # corrupt volume: skip, report
            print(f"skip {vol_path.name}: {exc}")
            continue
        stem = vol_path.name.replace(".nii.gz", "")
        for j, sl in enumerate(slices):
            rel = Path("images") / "mri" / f"ixi_{stem}_{j}.png"
            save_png(sl, out / rel)
            rows.append({"path": rel.as_posix(), "modality": "mri", "source": "IXI-T1", "split": split_of[i]})
    print(f"IXI-T1: {len(rows)} slices from {len(chosen)} subjects")
    return rows


# ----------------------------------------------------------------------------- folder
def prepare_folder(out: Path, raw: Path, max_per_modality: int, seed: int) -> list[dict]:
    pre = ImagePreprocessor()
    rng = np.random.default_rng(seed)
    rows = []
    for modality in config.MODALITIES:
        folder = raw / modality
        if not folder.is_dir():
            continue
        files = sorted(p for p in folder.rglob("*") if p.suffix.lower() in RAW_EXTENSIONS)
        if not files:
            continue
        n = min(len(files), max_per_modality)
        files = [files[i] for i in sorted(rng.choice(len(files), size=n, replace=False))]
        order = rng.permutation(n)
        counts = split_counts(n)
        kept = 0
        for rank, i in enumerate(tqdm(order, desc=f"folder/{modality}")):
            split = "train" if rank < counts["train"] else ("val" if rank < counts["train"] + counts["val"] else "test")
            try:
                loaded = pre.load(files[i], validate_size=True)
            except ImageLoadError as exc:
                print(f"skip {files[i]}: {exc}")
                continue
            rel = Path("images") / modality / f"folder_{modality}_{i:06d}.png"
            save_png(loaded.pixels, out / rel)
            rows.append({"path": rel.as_posix(), "modality": modality, "source": f"folder:{folder.name}", "split": split})
            kept += 1
        print(f"folder {modality}: {kept} images")
    if not rows:
        raise RuntimeError(f"no images found under {raw}/<modality>/ (modalities: {config.MODALITIES})")
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=["medmnist", "ixi", "folder"], required=True)
    ap.add_argument("--out", type=Path, required=True, help="output folder (images + manifest.csv)")
    ap.add_argument("--cache", type=Path, default=None, help="download cache (default <out>/../cache)")
    ap.add_argument("--raw", type=Path, help="folder source: root containing <modality>/ subfolders")
    ap.add_argument("--max-per-modality", type=int, default=6000)
    ap.add_argument("--medmnist-size", type=int, default=224, choices=[64, 128, 224])
    ap.add_argument("--modalities", nargs="+", choices=list(MEDMNIST_SOURCES), help="medmnist: only these modalities")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--append", action="store_true", help="add to an existing manifest")
    args = ap.parse_args(argv)

    out: Path = args.out
    cache: Path = args.cache or out.parent / "cache"
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    if args.source == "medmnist":
        rows = prepare_medmnist(out, cache, args.max_per_modality, args.medmnist_size, args.seed, args.modalities)
    elif args.source == "ixi":
        rows = prepare_ixi(out, cache, args.max_per_modality, args.seed)
    else:
        if args.raw is None:
            ap.error("--raw is required for --source folder")
        rows = prepare_folder(out, args.raw, args.max_per_modality, args.seed)

    manifest = out / "manifest.csv"
    new = pd.DataFrame(rows, columns=["path", "modality", "source", "split"])
    if args.append and manifest.exists():
        new = pd.concat([pd.read_csv(manifest), new]).drop_duplicates(subset="path", keep="last")
    new.to_csv(manifest, index=False)
    print(f"\nManifest: {manifest}")
    print(new.groupby(["modality", "split"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
