"""Evaluate a checkpoint on a manifest split and write JSON + Markdown metrics.

Example
  python scripts/evaluate.py --checkpoint runs/v1/best.pt --manifest data/prepared/manifest.csv --split test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iqa_ml.checkpoint import load_model  # noqa: E402
from iqa_ml.dataset import QualityDataset  # noqa: E402
from iqa_ml.evaluation import evaluate, metrics_markdown  # noqa: E402
from iqa_ml.inference import select_device  # noqa: E402


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=44, help="degradation seed (train.py uses 42+2 for test)")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", type=Path, default=None, help="output prefix (default: next to checkpoint)")
    args = ap.parse_args(argv)

    device = select_device(args.device)
    model, meta = load_model(args.checkpoint, device=device)
    ds = QualityDataset(args.manifest, args.split, seed=args.seed, max_samples=args.max_samples)
    dl = DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)
    metrics = evaluate(model, dl, device)
    metrics["model_version"] = meta["version"]
    metrics["split"] = args.split

    prefix = args.out or args.checkpoint.with_name(f"eval_{args.split}")
    Path(str(prefix) + ".json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    Path(str(prefix) + ".md").write_text(f"Model {meta['version']} - split {args.split}\n\n" + metrics_markdown(metrics) + "\n", encoding="utf-8")
    print(metrics_markdown(metrics))
    return metrics


if __name__ == "__main__":
    main()
