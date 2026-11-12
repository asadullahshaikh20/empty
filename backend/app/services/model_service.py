"""Keeps the active quality model in memory and reloads it when the active version changes.

FR-QA-01 (load versioned checkpoint), FR-QA-06 (GPU/CPU), FR-QA-18 (cache, reload on change).
"""

from __future__ import annotations

import logging
import threading
import uuid
from pathlib import Path
from typing import Protocol

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import BACKEND_DIR, get_settings
from ..models import ModelRegistry

log = logging.getLogger(__name__)


class ModelUnavailable(Exception):
    """No active model registered, or the active checkpoint could not be loaded."""


class Assessor(Protocol):
    version: str

    def assess_loaded(self, loaded, modality: str | None = None) -> dict: ...


def resolve_checkpoint(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (BACKEND_DIR / p).resolve()


class ModelService:
    def __init__(self, device: str | None = None):
        self.device = device or get_settings().model_device
        self._lock = threading.Lock()
        self._assessor: Assessor | None = None
        self._model_id: uuid.UUID | None = None

    @property
    def loaded_version(self) -> str | None:
        return self._assessor.version if self._assessor is not None else None

    def _load(self, record: ModelRegistry) -> Assessor:
        from iqa_ml.inference import QualityAssessor  # heavy import (torch) only when needed

        path = resolve_checkpoint(record.checkpoint_path)
        if not path.is_file():
            raise ModelUnavailable(f"Checkpoint for model {record.version} not found at {path}")
        try:
            assessor = QualityAssessor.from_checkpoint(path, device=self.device)
        except Exception as exc:
            log.exception("Failed to load model %s", record.version)
            raise ModelUnavailable(f"Failed to load model {record.version}: {exc}") from exc
        # Registry version is authoritative for traceability
        assessor.version = record.version
        return assessor

    def get(self, db: Session) -> tuple[Assessor, ModelRegistry]:
        record = db.scalar(select(ModelRegistry).where(ModelRegistry.is_active.is_(True)))
        if record is None:
            raise ModelUnavailable("No active model is registered. Run scripts/register_model.py --activate.")
        with self._lock:
            if self._assessor is None or self._model_id != record.id:
                log.info("Loading model %s from %s", record.version, record.checkpoint_path)
                self._assessor = self._load(record)
                self._model_id = record.id
            return self._assessor, record

    def warm_up(self, db: Session) -> None:
        try:
            self.get(db)
        except ModelUnavailable as exc:
            log.warning("Model not loaded at startup: %s", exc)
