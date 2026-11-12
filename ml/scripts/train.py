"""Train the hybrid EfficientNet-B4 + ViT-B/16 quality model (SRS Table 25b).

Defaults: AdamW lr 1e-4 wd 1e-4, cosine LR schedule, weighted cross-entropy per head,
batch 32, FP16 mixed precision on CUDA, 50 epochs, early stopping (patience 10) on
validation loss, checkpoint every 5 epochs, resumable.

Outputs in --out-dir:  best.pt (best validation loss, with test metrics),
last.pt (resume state), history.csv, test_metrics.json, test_metrics.md

Examples
  # Colab GPU
  python scripts/train.py --manifest data/prepared/manifest.csv --out-dir runs/v1 --version v1.0.0
  # CPU smoke test (proves the pipeline only; accuracy is meaningless)
  python scripts/train.py --manifest data/prepared/manifest.csv --out-dir runs/smoke --epochs 1 \
      --batch-size 8 --freeze-backbone --max-train-samples 96 --max-eval-samples 48 --num-workers 0
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iqa_ml.checkpoint import read_checkpoint, save_checkpoint  # noqa: E402
from iqa_ml.dataset import QualityDataset, estimate_class_weights  # noqa: E402
from iqa_ml.evaluation import compute_loss, evaluate, make_criteria, metrics_markdown  # noqa: E402
from iqa_ml.inference import select_device  # noqa: E402
from iqa_ml.model import QualityAssessmentModel  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--version", default=None, help="model version string (default: v<timestamp>)")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--checkpoint-every", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--no-amp", action="store_true", help="disable FP16 mixed precision on CUDA")
    ap.add_argument("--freeze-backbone", action="store_true", help="train fusion + heads only")
    ap.add_argument("--no-pretrained", action="store_true", help="random init (tests only)")
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-eval-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true", help="resume from <out-dir>/last.pt")
    return ap.parse_args(argv)


def _set_train_mode(model: QualityAssessmentModel, frozen: bool) -> None:
    model.train()
    if frozen:  # keep frozen backbone BatchNorm statistics fixed
        model.cnn_features.eval()
        model.vit.eval()


def main(argv=None) -> dict:
    args = parse_args(argv)
    set_seed(args.seed)
    device = select_device(args.device)
    use_amp = device.type == "cuda" and not args.no_amp
    version = args.version or dt.datetime.now(dt.timezone.utc).strftime("v%Y%m%d-%H%MZ")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"device={device} amp={use_amp} version={version}")

    train_ds = QualityDataset(args.manifest, "train", train=True, seed=args.seed, max_samples=args.max_train_samples)
    val_ds = QualityDataset(args.manifest, "val", seed=args.seed + 1, max_samples=args.max_eval_samples)
    test_ds = QualityDataset(args.manifest, "test", seed=args.seed + 2, max_samples=args.max_eval_samples)
    print(f"samples: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)}")

    loader_kw = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda",
                     persistent_workers=args.num_workers > 0)
    train_dl = DataLoader(train_ds, shuffle=True, drop_last=len(train_ds) > args.batch_size, **loader_kw)
    val_dl = DataLoader(val_ds, shuffle=False, **loader_kw)
    test_dl = DataLoader(test_ds, shuffle=False, **loader_kw)

    model = QualityAssessmentModel(pretrained=not args.no_pretrained).to(device)
    if args.freeze_backbone:
        model.set_backbone_trainable(False)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    class_weights = estimate_class_weights(seed=args.seed)
    criteria = make_criteria(class_weights, device)
    train_config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    train_config["class_weights"] = class_weights

    start_epoch, best_val, bad_epochs, history = 1, float("inf"), 0, []
    last_path, best_path = args.out_dir / "last.pt", args.out_dir / "best.pt"
    if args.resume:
        if not last_path.exists():
            raise FileNotFoundError(f"--resume given but {last_path} does not exist")
        state = read_checkpoint(last_path, map_location="cpu")  # optimizer.load_state_dict moves tensors to the right device
        model.load_state_dict(state["state_dict"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_val, bad_epochs, history = state["best_val_loss"], state["bad_epochs"], state["history"]
        version = state["version"]
        print(f"resumed from epoch {state['epoch']} (best val loss {best_val:.4f})")

    for epoch in range(start_epoch, args.epochs + 1):
        _set_train_mode(model, args.freeze_backbone)
        t0, run_loss, seen = time.time(), 0.0, 0
        for x, y, m in train_dl:
            x, y, m = x.to(device, non_blocking=True), y.to(device, non_blocking=True), m.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = compute_loss(out, y, m, criteria)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            run_loss += loss.item() * x.shape[0]
            seen += x.shape[0]
        scheduler.step()

        val = evaluate(model, val_dl, device, criteria)
        row = {
            "epoch": epoch,
            "train_loss": round(run_loss / max(seen, 1), 5),
            "val_loss": round(val["loss"], 5),
            "val_mean_accuracy": round(val["mean_accuracy"], 4),
            "val_mean_f1": round(val["mean_f1"], 4),
            "lr": scheduler.get_last_lr()[0],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        print(json.dumps(row))

        improved = val["loss"] < best_val
        if improved:
            best_val, bad_epochs = val["loss"], 0
            save_checkpoint(best_path, model, version, epoch=epoch, val_metrics=val, train_config=train_config)
        else:
            bad_epochs += 1

        stop = bad_epochs >= args.patience
        if epoch % args.checkpoint_every == 0 or epoch == args.epochs or stop:
            save_checkpoint(
                last_path, model, version, epoch=epoch, optimizer=optimizer.state_dict(),
                scheduler=scheduler.state_dict(), scaler=scaler.state_dict(), best_val_loss=best_val,
                bad_epochs=bad_epochs, history=history, train_config=train_config,
            )
        if stop:
            print(f"early stopping at epoch {epoch} (no val improvement for {args.patience} epochs)")
            break

    if history:
        with open(args.out_dir / "history.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)
    if not best_path.exists():
        raise RuntimeError("no best checkpoint was saved (validation loss never finite?)")

    # Final test evaluation with the best checkpoint; metrics stored inside best.pt
    best = read_checkpoint(best_path, map_location="cpu")
    model.load_state_dict(best["state_dict"])
    test = evaluate(model, test_dl, device, criteria)
    save_checkpoint(
        best_path, model, version, epoch=best["epoch"], val_metrics=best["val_metrics"], test_metrics=test,
        accuracy=test["mean_accuracy"], train_config=train_config,
    )
    (args.out_dir / "test_metrics.json").write_text(json.dumps(test, indent=2), encoding="utf-8")
    (args.out_dir / "test_metrics.md").write_text(metrics_markdown(test) + "\n", encoding="utf-8")
    print("\nTest metrics (best checkpoint):\n" + metrics_markdown(test))
    print(f"\nBest checkpoint: {best_path}")
    return test


if __name__ == "__main__":
    main()
