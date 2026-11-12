"""Register a trained checkpoint in the model registry and optionally make it the active model.

Usage (from backend/):
  python scripts/register_model.py --checkpoint models/v1.0.0.pt --activate
  python scripts/register_model.py --activate-version v1.0.0      # switch to an already registered version
  python scripts/register_model.py --list

The version and test accuracy are read from the checkpoint metadata. The running API picks up
a newly activated version on the next assessment request (no restart needed).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import select, update  # noqa: E402

from app.config import BACKEND_DIR  # noqa: E402
from app.db import SessionLocal  # noqa: E402
from app.models import ModelRegistry  # noqa: E402


def _stored_path(path: Path) -> str:
    """Store paths relative to backend/ when possible so the project folder can be moved."""
    path = path.resolve()
    try:
        return path.relative_to(BACKEND_DIR).as_posix()
    except ValueError:
        return str(path)


def activate(db, record: ModelRegistry) -> None:
    db.execute(update(ModelRegistry).where(ModelRegistry.id != record.id).values(is_active=False))
    db.flush()  # deactivate others before activating (single-active unique index)
    record.is_active = True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--activate-version")
    group.add_argument("--list", action="store_true")
    ap.add_argument("--activate", action="store_true", help="make the registered checkpoint active")
    ap.add_argument("--version", help="override the version stored in the checkpoint")
    args = ap.parse_args(argv)

    with SessionLocal() as db:
        if args.list:
            for m in db.scalars(select(ModelRegistry).order_by(ModelRegistry.deployed_at)):
                acc = "n/a" if m.accuracy is None else f"{m.accuracy:.3f}"
                print(f"{'*' if m.is_active else ' '} {m.version:20s} acc={acc:6s} {m.checkpoint_path}")
            return 0

        if args.activate_version:
            record = db.scalar(select(ModelRegistry).where(ModelRegistry.version == args.activate_version))
            if record is None:
                print(f"error: version {args.activate_version} is not registered", file=sys.stderr)
                return 1
            activate(db, record)
            db.commit()
            print(f"Active model: {record.version}")
            return 0

        from iqa_ml.checkpoint import read_checkpoint

        path = args.checkpoint
        if not path.is_file():
            print(f"error: {path} does not exist", file=sys.stderr)
            return 1
        meta = read_checkpoint(path)  # validates format/heads
        version = args.version or str(meta["version"])
        if db.scalar(select(ModelRegistry.id).where(ModelRegistry.version == version)):
            print(f"error: version {version} is already registered (use --version to rename)", file=sys.stderr)
            return 1
        test = meta.get("test_metrics") or {}
        record = ModelRegistry(
            version=version,
            checkpoint_path=_stored_path(path),
            accuracy=meta.get("accuracy"),
            metrics={
                "mean_accuracy": test.get("mean_accuracy"),
                "mean_f1": test.get("mean_f1"),
                "mean_auc": test.get("mean_auc"),
                "n_test": test.get("n"),
                "epoch": meta.get("epoch"),
            },
        )
        db.add(record)
        db.flush()
        if args.activate:
            activate(db, record)
        db.commit()
        print(f"Registered model {version}{' (active)' if args.activate else ''} -> {record.checkpoint_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
