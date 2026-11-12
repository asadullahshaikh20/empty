"""Image ingestion (UC-02, FR-IMG-01..12)."""

from __future__ import annotations

import math
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Response, UploadFile, status
from fastapi.responses import FileResponse
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from iqa_ml.preprocessing import ImageLoadError, ImagePreprocessor

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import MODALITIES, Image, User
from ..schemas import ImageOut, Page
from ..services import storage

router = APIRouter(prefix="/images", tags=["images"])
_preprocessor = ImagePreprocessor()
CHUNK = 1024 * 1024


def get_owned_image(db: Session, image_id: uuid.UUID, user: User) -> Image:
    """Return the image if the user owns it (or is admin); 404 otherwise so ids of others are not revealed."""
    image = db.get(Image, image_id)
    if image is None or (image.user_id != user.id and user.role != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image not found")
    return image


def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := upload.file.read(CHUNK):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(
                413,
                detail=f"File exceeds the {get_settings().max_upload_mb} MB limit",
            )
    return bytes(buf)


@router.post("/upload", response_model=ImageOut, status_code=status.HTTP_201_CREATED)
def upload_image(
    file: UploadFile = File(...),
    modality: Optional[str] = Form(None, description="xray | ct | mri | retinal; empty or 'auto' to detect"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Image:
    if modality is not None:
        modality = modality.strip().lower()
        if modality in ("", "auto"):
            modality = None
        elif modality not in MODALITIES:
            raise HTTPException(422, detail=f"Unknown modality '{modality}'")

    data = _read_limited(file, get_settings().max_upload_bytes)
    try:
        loaded = _preprocessor.load(data)  # validates type by content, readability and size (FR-IMG-02, -12)
    except ImageLoadError as exc:
        raise HTTPException(422, detail=str(exc)) from exc

    if modality is not None:
        modality_source = "user"
    elif loaded.metadata_modality is not None:
        modality, modality_source = loaded.metadata_modality, "dicom"
    else:
        modality_source = None  # filled in by the model at assessment time

    image_id = uuid.uuid4()
    paths = storage.save_image(image_id, data, loaded)
    filename = (file.filename or "upload").replace("\\", "/").rsplit("/", 1)[-1][:255] or "upload"
    image = Image(
        id=image_id,
        user_id=user.id,
        original_filename=filename,
        file_path=paths.file_path,
        thumbnail_path=paths.thumbnail_path,
        preview_path=paths.preview_path,
        file_format=loaded.file_format,
        modality=modality,
        modality_source=modality_source,
        size_bytes=len(data),
        width=loaded.width,
        height=loaded.height,
    )
    db.add(image)
    try:
        db.commit()
    except Exception:
        db.rollback()
        storage.delete_files(paths.file_path, paths.thumbnail_path, paths.preview_path)
        raise
    db.refresh(image)
    return image


@router.get("", response_model=Page[ImageOut])
def list_images(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    total = db.scalar(select(func.count()).select_from(Image).where(Image.user_id == user.id)) or 0
    items = db.scalars(
        select(Image)
        .where(Image.user_id == user.id)
        .order_by(Image.upload_timestamp.desc(), Image.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    return {"items": items, "total": total, "page": page, "page_size": page_size,
            "pages": math.ceil(total / page_size) if total else 0}


@router.get("/{image_id}", response_model=ImageOut)
def get_image(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Image:
    return get_owned_image(db, image_id, user)


def _file_response(rel: str, media_type: str, filename: str | None = None) -> FileResponse:
    path = storage.resolve(rel)
    if not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Stored file is missing")
    return FileResponse(path, media_type=media_type, filename=filename, headers={"Cache-Control": "private, max-age=3600"})


@router.get("/{image_id}/file")
def download_original(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    image = get_owned_image(db, image_id, user)
    return _file_response(image.file_path, storage.MEDIA_TYPES[image.file_format], image.original_filename)


@router.get("/{image_id}/thumbnail")
def thumbnail(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return _file_response(get_owned_image(db, image_id, user).thumbnail_path, "image/png")


@router.get("/{image_id}/preview")
def preview(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    """PNG rendering of the image (also for DICOM), for display in the browser."""
    return _file_response(get_owned_image(db, image_id, user).preview_path, "image/png")


@router.delete("/{image_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_image(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> Response:
    """FR-IMG-08: delete an own image together with its assessment records."""
    image = get_owned_image(db, image_id, user)
    paths = (image.file_path, image.thumbnail_path, image.preview_path)
    db.delete(image)
    db.commit()
    storage.delete_files(*paths)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
