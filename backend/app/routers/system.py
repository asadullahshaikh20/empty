"""Health check and (admin-only) model registry listing."""

from __future__ import annotations

import datetime as dt
import uuid
from typing import Optional

from fastapi import APIRouter, Depends, Request
from fastapi.security import HTTPAuthorizationCredentials
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from ..db import get_db
from ..deps import bearer, require_roles
from ..models import ModelRegistry, User
from ..security import ACCESS, TokenError, decode_token

router = APIRouter(tags=["system"])


def _is_admin(creds: HTTPAuthorizationCredentials | None = Depends(bearer), db: Session = Depends(get_db)) -> bool:
    """True only for a valid admin access token; never raises (health stays public)."""
    if creds is None:
        return False
    try:
        payload = decode_token(creds.credentials, ACCESS)
    except TokenError:
        return False
    user = db.get(User, payload["sub"])
    return bool(user and user.is_active and user.role == "admin")


@router.get("/health")
def health(request: Request, db: Session = Depends(get_db), is_admin: bool = Depends(_is_admin)) -> dict:
    """Public: overall status only. Model details are shown to administrators only."""
    try:
        db.execute(text("SELECT 1"))
        database = "ok"
    except Exception:  # pragma: no cover - depends on DB outage
        database = "unavailable"
    active = db.scalar(select(ModelRegistry.version).where(ModelRegistry.is_active.is_(True))) if database == "ok" else None
    service = request.app.state.model_service
    body = {"status": "ok" if database == "ok" else "degraded", "database": database}
    if is_admin:
        body.update(
            active_model_version=active,
            model_loaded=service.loaded_version is not None,
            loaded_model_version=service.loaded_version,
        )
    return body


class ModelOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    version: str
    checkpoint_path: str
    accuracy: Optional[float]
    is_active: bool
    deployed_at: dt.datetime


@router.get("/admin/models", response_model=list[ModelOut])
def list_models(db: Session = Depends(get_db), _admin=Depends(require_roles("admin"))):
    return db.scalars(select(ModelRegistry).order_by(ModelRegistry.deployed_at.desc())).all()
