"""Filesystem storage for uploaded originals, thumbnails (FR-IMG-09) and PNG previews."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

from PIL import Image as PILImage

from iqa_ml.preprocessing import LoadedImage, make_thumbnail

from ..config import get_settings

EXTENSIONS = {"PNG": ".png", "JPEG": ".jpg", "DICOM": ".dcm"}
MEDIA_TYPES = {"PNG": "image/png", "JPEG": "image/jpeg", "DICOM": "application/dicom"}
SUBDIRS = ("uploads", "thumbnails", "previews")


@dataclass
class StoredPaths:
    file_path: str
    thumbnail_path: str
    preview_path: str


def storage_root() -> Path:
    return get_settings().storage_dir


def ensure_dirs() -> None:
    for sub in SUBDIRS:
        (storage_root() / sub).mkdir(parents=True, exist_ok=True)


def resolve(relative: str) -> Path:
    """Absolute path of a stored file; refuses paths that escape the storage root."""
    root = storage_root().resolve()
    path = (root / relative).resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("path outside storage root")
    return path


def save_image(image_id: uuid.UUID, data: bytes, loaded: LoadedImage) -> StoredPaths:
    ensure_dirs()
    rel = StoredPaths(
        file_path=f"uploads/{image_id}{EXTENSIONS[loaded.file_format]}",  # unique filename (FR-IMG-03)
        thumbnail_path=f"thumbnails/{image_id}.png",
        preview_path=f"previews/{image_id}.png",
    )
    written: list[Path] = []
    try:
        original = resolve(rel.file_path)
        original.write_bytes(data)
        written.append(original)

        thumb = resolve(rel.thumbnail_path)
        make_thumbnail(loaded.pixels, thumb)
        written.append(thumb)

        preview = resolve(rel.preview_path)
        img = PILImage.fromarray(loaded.pixels)
        side = get_settings().preview_max_side
        img.thumbnail((side, side), PILImage.Resampling.LANCZOS)
        img.save(preview, format="PNG")
        written.append(preview)
    except Exception:
        for p in written:
            p.unlink(missing_ok=True)
        raise
    return rel


def delete_files(*relative_paths: str) -> None:
    for rel in relative_paths:
        try:
            resolve(rel).unlink(missing_ok=True)
        except (ValueError, OSError):
            pass
