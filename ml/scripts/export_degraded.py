"""Export degraded samples + labels for inspection (Phase 2 deliverable).

Writes <out>/<index>_<suitability>.png and <out>/labels.csv with one row per image:
source path, modality, the five head labels (severity 0/1/2 and class name), and the
degradations applied.

Example
  python scripts/export_degraded.py --manifest data/prepared/manifest.csv --out data/degraded_samples --n 200
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iqa_ml import config  # noqa: E402
from iqa_ml.dataset import QualityDataset  # noqa: E402


def main(argv=None) -> Path:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args(argv)

    ds = QualityDataset(args.manifest, args.split, seed=args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    labels_path = args.out / "labels.csv"
    fields = ["file", "source_path", "modality"]
    for h in config.HEADS:
        fields += [f"{h}_severity", f"{h}_class"]
    fields.append("applied")
    with open(labels_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(min(args.n, len(ds))):
            img, labels, applied, modality = ds.sample(i)
            suit = config.HEAD_CLASSES["diagnostic_suitability"][labels["diagnostic_suitability"]]
            name = f"{i:05d}_{suit}.png"
            cv2.imwrite(str(args.out / name), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            row = {"file": name, "source_path": ds.df.iloc[i]["path"], "modality": modality, "applied": ";".join(applied)}
            for h in config.HEADS:
                row[f"{h}_severity"] = labels[h]
                row[f"{h}_class"] = config.HEAD_CLASSES[h][labels[h]]
            writer.writerow(row)
    print(f"wrote {min(args.n, len(ds))} samples to {args.out} (labels: {labels_path})")
    return labels_path


if __name__ == "__main__":
    main()
