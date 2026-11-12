"""Quality assessment (UC-03, FR-QA-*) and assessment history (UC-07, FR-RPT-05/06)."""

from __future__ import annotations

import datetime as dt
import logging
import math
import uuid
from typing import Literal, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from iqa_ml import config as mlconfig
from iqa_ml.preprocessing import ImageLoadError, ImagePreprocessor

from ..db import get_db
from ..deps import get_current_user
from ..models import AssessmentResult, Image, ModelRegistry, User, utcnow
from ..schemas import AssessmentOut, AssessmentStatus, AssessmentSummary, ImageOut, Page
from ..services import storage
from ..services.model_service import ModelService, ModelUnavailable
from .images import get_owned_image

log = logging.getLogger(__name__)
router = APIRouter(tags=["assessments"])
_preprocessor = ImagePreprocessor()

SCORE_COLUMNS = {h: f"{h}_score" for h in mlconfig.HEADS}
CONFIDENCE_COLUMNS = {h: f"{h}_confidence" for h in mlconfig.HEADS}


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service


def to_out(result: AssessmentResult) -> AssessmentOut:
    details = result.details or {}
    heads = []
    if result.status == "completed":
        for h in mlconfig.HEADS:
            d = details.get("heads", {}).get(h, {})
            heads.append({
                "key": h,
                "name": mlconfig.HEAD_DISPLAY[h],
                "score": getattr(result, SCORE_COLUMNS[h]),
                "label": d.get("label", ""),
                "confidence": getattr(result, CONFIDENCE_COLUMNS[h]),
                "confidence_level": d.get("confidence_level", ""),
                "predicted_class": d.get("predicted_class", ""),
                "probabilities": d.get("probabilities", []),
            })
    return AssessmentOut(
        id=result.id,
        image_id=result.image_id,
        status=result.status,
        error_code=result.error_code,
        error_message=result.error_message,
        model_version=result.model.version,
        heads=heads,
        ai_readiness_score=result.ai_readiness_score,
        verdict=result.verdict,
        artifact_present=result.artifact_present,
        out_of_distribution=result.out_of_distribution,
        warnings=details.get("warnings", []),
        modality=result.image.modality,
        modality_source=result.image.modality_source,
        modality_prediction=details.get("modality_prediction"),
        inference_ms=result.inference_ms,
        created_at=result.created_at,
        completed_at=result.completed_at,
        image=ImageOut.model_validate(result.image),
    )


def _get_owned_result(db: Session, result_id: uuid.UUID, user: User) -> AssessmentResult:
    result = db.scalar(
        select(AssessmentResult)
        .options(joinedload(AssessmentResult.image), joinedload(AssessmentResult.model))
        .where(AssessmentResult.id == result_id)
    )
    if result is None or (result.image.user_id != user.id and user.role != "admin"):
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Assessment not found")
    return result


@router.post("/assess/{image_id}", response_model=AssessmentOut)
def assess_image(
    image_id: uuid.UUID,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
    service: ModelService = Depends(get_model_service),
) -> AssessmentOut:
    image = get_owned_image(db, image_id, user)
    try:
        assessor, model_record = service.get(db)
    except ModelUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc

    result = AssessmentResult(image_id=image.id, model_id=model_record.id, status="processing")
    db.add(result)
    db.commit()  # status visible to GET /assess/{id}/status while inference runs (FR-QA-08)

    try:
        loaded = _preprocessor.load(storage.resolve(image.file_path))
        user_modality = image.modality if image.modality_source == "user" else None
        out = assessor.assess_loaded(loaded, modality=user_modality)
    except ImageLoadError as exc:
        _mark_error(db, result, exc.code, str(exc))
        raise HTTPException(
            422,
            detail={"message": str(exc), "code": exc.code, "assessment_id": str(result.id)},
        ) from exc
    except Exception as exc:  # FR-QA-16: log, store reason, no stack trace to client
        log.exception("Inference failed for image %s", image.id)
        _mark_error(db, result, "inference_error", f"{type(exc).__name__}: {exc}"[:500])
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Quality assessment failed", "code": "inference_error", "assessment_id": str(result.id)},
        ) from exc

    for h in mlconfig.HEADS:
        setattr(result, SCORE_COLUMNS[h], out["heads"][h]["score"])
        setattr(result, CONFIDENCE_COLUMNS[h], out["heads"][h]["confidence"])
    result.ai_readiness_score = out["ai_readiness_score"]
    result.verdict = out["verdict"]
    result.artifact_present = bool(out["artifact_present"])
    result.out_of_distribution = bool(out["out_of_distribution"])
    result.inference_ms = int(out["inference_ms"])
    result.details = {
        "heads": {h: {k: v for k, v in out["heads"][h].items() if k not in ("score", "confidence")}
                  for h in mlconfig.HEADS},
        "warnings": list(out["warnings"]),
        "modality_prediction": out.get("modality_prediction"),
        "model_version": out.get("model_version"),
    }
    result.status = "completed"
    result.completed_at = utcnow()
    if image.modality is None or image.modality_source == "model":
        image.modality = out["modality"]
        image.modality_source = out["modality_source"]
    db.commit()
    return to_out(result)


def _mark_error(db: Session, result: AssessmentResult, code: str, message: str) -> None:
    result.status = "error"
    result.error_code = code
    result.error_message = message[:500]
    result.completed_at = utcnow()
    db.commit()


@router.get("/assess/{result_id}/status", response_model=AssessmentStatus)
def assessment_status(result_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    r = _get_owned_result(db, result_id, user)
    return AssessmentStatus(id=r.id, status=r.status, error_code=r.error_code, error_message=r.error_message)


@router.get("/assessments/stats")
def assessment_stats(db: Session = Depends(get_db), user: User = Depends(get_current_user)) -> dict:
    """Completed assessments of the user, counted per verdict (history KPI cards)."""
    rows = db.execute(
        select(AssessmentResult.verdict, func.count())
        .join(Image, AssessmentResult.image_id == Image.id)
        .where(Image.user_id == user.id, AssessmentResult.status == "completed")
        .group_by(AssessmentResult.verdict)
    ).all()
    counts = {"Pass": 0, "Review": 0, "Reject": 0}
    for verdict, n in rows:
        if verdict in counts:
            counts[verdict] = n
    return {"total": sum(counts.values()), **counts}


@router.get("/assessments/{result_id}", response_model=AssessmentOut)
def get_assessment(result_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    return to_out(_get_owned_result(db, result_id, user))


@router.get("/images/{image_id}/assessments/latest", response_model=AssessmentOut)
def latest_for_image(image_id: uuid.UUID, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    image = get_owned_image(db, image_id, user)
    result_id = db.scalar(
        select(AssessmentResult.id)
        .where(AssessmentResult.image_id == image.id)
        .order_by(AssessmentResult.created_at.desc())
        .limit(1)
    )
    if result_id is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Image has not been assessed yet")
    return to_out(_get_owned_result(db, result_id, user))


@router.get("/assessments", response_model=Page[AssessmentSummary])
def history(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    verdict: Optional[Literal["Pass", "Review", "Reject"]] = None,
    modality: Optional[Literal["xray", "ct", "mri", "retinal"]] = None,
    status_filter: Optional[Literal["processing", "completed", "error"]] = Query(None, alias="status"),
    date_from: Optional[dt.date] = None,
    date_to: Optional[dt.date] = None,
    q: Optional[str] = Query(None, max_length=100, description="search in the file name"),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
) -> dict:
    """Paginated history of the user's assessments, newest first, with filters (FR-RPT-05/06)."""
    if date_from and date_to and date_from > date_to:
        raise HTTPException(422, detail="date_from must not be after date_to")
    conditions = [Image.user_id == user.id]
    if verdict:
        conditions.append(AssessmentResult.verdict == verdict)
    if modality:
        conditions.append(Image.modality == modality)
    if status_filter:
        conditions.append(AssessmentResult.status == status_filter)
    if q and q.strip():
        conditions.append(func.lower(Image.original_filename).contains(q.strip().lower(), autoescape=True))
    if date_from:
        conditions.append(AssessmentResult.created_at >= dt.datetime.combine(date_from, dt.time.min, dt.timezone.utc))
    if date_to:
        end = dt.datetime.combine(date_to + dt.timedelta(days=1), dt.time.min, dt.timezone.utc)
        conditions.append(AssessmentResult.created_at < end)

    base = select(AssessmentResult).join(Image, AssessmentResult.image_id == Image.id).where(*conditions)
    total = db.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = db.execute(
        select(AssessmentResult, Image, ModelRegistry.version)
        .join(Image, AssessmentResult.image_id == Image.id)
        .join(ModelRegistry, AssessmentResult.model_id == ModelRegistry.id)
        .where(*conditions)
        .order_by(AssessmentResult.created_at.desc(), AssessmentResult.id)
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()
    items = [
        AssessmentSummary(
            id=r.id, image_id=img.id, original_filename=img.original_filename, modality=img.modality,
            status=r.status, ai_readiness_score=r.ai_readiness_score, verdict=r.verdict, model_version=version,
            created_at=r.created_at, upload_timestamp=img.upload_timestamp,
        )
        for r, img, version in rows
    ]
    return {"items": items, "total": total, "page": page, "page_size": page_size,
            "pages": math.ceil(total / page_size) if total else 0}
