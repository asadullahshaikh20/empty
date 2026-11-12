"""ORM models (SDS section 6). UUID primary keys; timestamps are timezone-aware UTC."""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Uuid,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .db import Base

JSONType = JSON().with_variant(JSONB(), "postgresql")

ROLES = ("radiologist", "researcher", "admin")
ASSESSMENT_STATUSES = ("processing", "completed", "error")
VERDICTS = ("Pass", "Review", "Reject")
MODALITIES = ("xray", "ct", "mri", "retinal")
MODALITY_SOURCES = ("user", "dicom", "model")
FILE_FORMATS = ("DICOM", "PNG", "JPEG")


def _in(column: str, values: tuple[str, ...]) -> str:
    return f"{column} IN (" + ", ".join(f"'{v}'" for v in values) + ")"


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


class User(Base):
    __tablename__ = "users"
    __table_args__ = (CheckConstraint(_in("role", ROLES), name="ck_users_role"),)

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120))
    username: Mapped[str] = mapped_column(String(50), unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)  # stored lower-case
    password_hash: Mapped[str] = mapped_column(String(255))
    role: Mapped[str] = mapped_column(String(20), default="radiologist", server_default="radiologist")
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default=text("true"))
    failed_login_attempts: Mapped[int] = mapped_column(Integer, default=0, server_default="0")
    locked_until: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())

    images: Mapped[list[Image]] = relationship(back_populates="user", cascade="all, delete-orphan", passive_deletes=True)


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)  # = JWT "jti"
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())


class Image(Base):
    __tablename__ = "images"
    __table_args__ = (
        CheckConstraint(_in("file_format", FILE_FORMATS), name="ck_images_file_format"),
        CheckConstraint(f"modality IS NULL OR {_in('modality', MODALITIES)}", name="ck_images_modality"),
        CheckConstraint(
            f"modality_source IS NULL OR {_in('modality_source', MODALITY_SOURCES)}", name="ck_images_modality_source"
        ),
        Index("ix_images_user_uploaded", "user_id", "upload_timestamp"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"))
    original_filename: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str] = mapped_column(String(500))  # relative to STORAGE_DIR
    thumbnail_path: Mapped[str] = mapped_column(String(500))
    preview_path: Mapped[str] = mapped_column(String(500))
    file_format: Mapped[str] = mapped_column(String(10))
    modality: Mapped[str | None] = mapped_column(String(20), nullable=True)
    modality_source: Mapped[str | None] = mapped_column(String(10), nullable=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    upload_timestamp: Mapped[dt.datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, server_default=func.now()
    )

    user: Mapped[User] = relationship(back_populates="images")
    assessments: Mapped[list[AssessmentResult]] = relationship(
        back_populates="image", cascade="all, delete-orphan", passive_deletes=True,
        order_by="AssessmentResult.created_at.desc()",
    )


class ModelRegistry(Base):
    __tablename__ = "model_registry"
    __table_args__ = (
        # at most one active model version (SDS 4.1.6)
        Index("uq_model_registry_single_active", "is_active", unique=True, postgresql_where=text("is_active"),
              sqlite_where=text("is_active")),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    version: Mapped[str] = mapped_column(String(50), unique=True)
    checkpoint_path: Mapped[str] = mapped_column(String(500))
    accuracy: Mapped[float | None] = mapped_column(Float, nullable=True)
    metrics: Mapped[dict | None] = mapped_column(JSONType, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, server_default=text("false"))
    deployed_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())


class AssessmentResult(Base):
    __tablename__ = "assessment_results"
    __table_args__ = (
        CheckConstraint(_in("status", ASSESSMENT_STATUSES), name="ck_assessment_status"),
        CheckConstraint(f"verdict IS NULL OR {_in('verdict', VERDICTS)}", name="ck_assessment_verdict"),
        Index("ix_assessment_image_created", "image_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    image_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("images.id", ondelete="CASCADE"))
    # RESTRICT: a model version that produced results cannot be deleted (traceability, FR-QA-17)
    model_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("model_registry.id", ondelete="RESTRICT"), index=True)
    status: Mapped[str] = mapped_column(String(12), default="processing")
    error_code: Mapped[str | None] = mapped_column(String(50), nullable=True)
    error_message: Mapped[str | None] = mapped_column(String(500), nullable=True)

    technical_quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    anatomical_completeness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    orientation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    diagnostic_suitability_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    technical_quality_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    anatomical_completeness_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    artifact_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    orientation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    diagnostic_suitability_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_readiness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String(10), nullable=True)
    artifact_present: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    out_of_distribution: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    details: Mapped[dict | None] = mapped_column(JSONType, nullable=True)  # labels, probabilities, warnings
    inference_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=utcnow, server_default=func.now())
    completed_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    image: Mapped[Image] = relationship(back_populates="assessments")
    model: Mapped[ModelRegistry] = relationship()
