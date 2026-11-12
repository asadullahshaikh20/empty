"""Request/response schemas (Pydantic v2)."""

from __future__ import annotations

import datetime as dt
import re
import uuid
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

T = TypeVar("T")

Role = Literal["radiologist", "researcher", "admin"]
Modality = Literal["xray", "ct", "mri", "retinal"]
VerdictT = Literal["Pass", "Review", "Reject"]

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,50}$")


def validate_password_policy(password: str) -> str:
    """FR-AUTH-02: >= 8 characters, at least one uppercase letter and one digit (<= 72 bytes for bcrypt)."""
    problems = []
    if len(password) < 8:
        problems.append("at least 8 characters")
    if not any(c.isupper() for c in password):
        problems.append("one uppercase letter")
    if not any(c.isdigit() for c in password):
        problems.append("one digit")
    if problems:
        raise ValueError("Password must contain " + ", ".join(problems))
    if len(password.encode("utf-8")) > 72:
        raise ValueError("Password must be at most 72 bytes long")
    return password


# ------------------------------------------------------------------ auth
class RegisterRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    email: EmailStr
    username: str
    password: str
    # Administrator accounts cannot self-register; they are created with scripts/create_admin.py
    role: Literal["radiologist", "researcher"] = "radiologist"

    @field_validator("name")
    @classmethod
    def _name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name is required")
        return v

    @field_validator("username")
    @classmethod
    def _username(cls, v: str) -> str:
        v = v.strip()
        if not USERNAME_RE.fullmatch(v):
            raise ValueError("Username must be 3-50 characters: letters, digits, '_', '.' or '-'")
        return v

    @field_validator("password")
    @classmethod
    def _password(cls, v: str) -> str:
        return validate_password_policy(v)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=200)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    username: str
    email: str
    role: Role
    is_active: bool
    created_at: dt.datetime


class TokenResponse(BaseModel):
    access_token: str
    token_type: Literal["bearer"] = "bearer"
    expires_in: int
    user: UserOut


# ------------------------------------------------------------------ pagination
class Page(BaseModel, Generic[T]):
    items: list[T]
    total: int
    page: int
    page_size: int
    pages: int


# ------------------------------------------------------------------ images
class ImageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    original_filename: str
    file_format: str
    modality: Optional[Modality]
    modality_source: Optional[str]
    size_bytes: int
    width: int
    height: int
    upload_timestamp: dt.datetime


# ------------------------------------------------------------------ assessments
class HeadResult(BaseModel):
    key: str
    name: str
    score: float
    label: str
    confidence: float
    confidence_level: str
    predicted_class: str
    probabilities: list[float]


class ModalityPrediction(BaseModel):
    modality: Modality
    confidence: float


class AssessmentOut(BaseModel):
    id: uuid.UUID
    image_id: uuid.UUID
    status: Literal["processing", "completed", "error"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    model_version: str
    heads: list[HeadResult] = []
    ai_readiness_score: Optional[float] = None
    verdict: Optional[VerdictT] = None
    artifact_present: Optional[bool] = None
    out_of_distribution: Optional[bool] = None
    warnings: list[str] = []
    modality: Optional[Modality] = None
    modality_source: Optional[str] = None
    modality_prediction: Optional[ModalityPrediction] = None
    inference_ms: Optional[int] = None
    created_at: dt.datetime
    completed_at: Optional[dt.datetime] = None
    image: ImageOut


class AssessmentStatus(BaseModel):
    id: uuid.UUID
    status: Literal["processing", "completed", "error"]
    error_code: Optional[str] = None
    error_message: Optional[str] = None


class AssessmentSummary(BaseModel):
    """Row of the history table (FR-RPT-05)."""

    id: uuid.UUID
    image_id: uuid.UUID
    original_filename: str
    modality: Optional[Modality]
    status: str
    ai_readiness_score: Optional[float]
    verdict: Optional[VerdictT]
    model_version: str
    created_at: dt.datetime
    upload_timestamp: dt.datetime
