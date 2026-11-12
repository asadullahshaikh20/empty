"""Authentication (UC-01, FR-AUTH-01..09): register, login, refresh, logout, me."""

from __future__ import annotations

import datetime as dt
import math
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from ..config import get_settings
from ..db import get_db
from ..deps import get_current_user
from ..models import RefreshToken, User, utcnow
from ..schemas import LoginRequest, RegisterRequest, TokenResponse, UserOut
from ..security import (
    DUMMY_HASH,
    REFRESH,
    TokenError,
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)

router = APIRouter(prefix="/auth", tags=["auth"])

INVALID_CREDENTIALS = "Invalid email or password"


def _issue_tokens(db: Session, user: User, response: Response) -> TokenResponse:
    s = get_settings()
    access, expires_in = create_access_token(user.id, user.role)
    refresh, jti, expires_at = create_refresh_token(user.id)
    # housekeeping: drop this user's expired refresh tokens (revoked, unexpired ones are kept for reuse detection)
    db.execute(delete(RefreshToken).where(RefreshToken.user_id == user.id, RefreshToken.expires_at < utcnow()))
    db.add(RefreshToken(id=jti, user_id=user.id, expires_at=expires_at))
    db.commit()
    response.set_cookie(
        key=s.refresh_cookie_name,
        value=refresh,
        max_age=s.refresh_token_days * 24 * 3600,
        httponly=True,
        secure=s.cookie_secure,
        samesite="strict",
        path="/",
    )
    return TokenResponse(access_token=access, expires_in=expires_in, user=UserOut.model_validate(user))


def _clear_cookie(response: Response) -> None:
    s = get_settings()
    response.delete_cookie(s.refresh_cookie_name, path="/", httponly=True, secure=s.cookie_secure, samesite="strict")


@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    email = body.email.lower()
    if db.scalar(select(User.id).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email already registered")
    if db.scalar(select(User.id).where(func.lower(User.username) == body.username.lower())):
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Username already taken")
    user = User(
        name=body.name, email=email, username=body.username, password_hash=hash_password(body.password), role=body.role
    )
    db.add(user)
    try:
        db.commit()
    except IntegrityError as exc:  # concurrent registration with the same email/username
        db.rollback()
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Email or username already registered") from exc
    db.refresh(user)
    return _issue_tokens(db, user, response)


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, response: Response, db: Session = Depends(get_db)) -> TokenResponse:
    s = get_settings()
    user = db.scalar(select(User).where(User.email == body.email.lower()))
    if user is None:
        verify_password(body.password, DUMMY_HASH)  # equalise timing
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=INVALID_CREDENTIALS)

    now = utcnow()
    if user.locked_until is not None and user.locked_until > now:
        minutes = max(1, math.ceil((user.locked_until - now).total_seconds() / 60))
        raise HTTPException(
            status.HTTP_423_LOCKED,
            detail=f"Account temporarily locked after repeated failed logins. Try again in {minutes} minute(s).",
        )

    if not verify_password(body.password, user.password_hash):
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= s.login_max_failures:  # FR-AUTH-09
            user.locked_until = now + dt.timedelta(minutes=s.lockout_minutes)
            user.failed_login_attempts = 0
        db.commit()
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=INVALID_CREDENTIALS)

    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Account is deactivated")

    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()
    return _issue_tokens(db, user, response)


def _refresh_failure(detail: str) -> JSONResponse:
    resp = JSONResponse(status_code=status.HTTP_401_UNAUTHORIZED, content={"detail": detail})
    _clear_cookie(resp)
    return resp


@router.post("/refresh", response_model=TokenResponse, responses={401: {"description": "Invalid refresh token"}})
def refresh(request: Request, response: Response, db: Session = Depends(get_db)):
    """FR-AUTH-06: rotate the refresh token (cookie) and issue a new access token."""
    s = get_settings()
    token = request.cookies.get(s.refresh_cookie_name)
    if not token:
        return _refresh_failure("No refresh token")
    try:
        payload = decode_token(token, REFRESH)
        jti = uuid.UUID(payload["jti"])
    except (TokenError, ValueError):
        return _refresh_failure("Invalid or expired refresh token")

    record = db.get(RefreshToken, jti)
    now = utcnow()
    if record is None or record.user_id != payload["sub"]:
        return _refresh_failure("Invalid refresh token")
    if record.revoked_at is not None:
        # Re-use of a rotated token: assume theft and revoke every session of this user.
        db.execute(
            update(RefreshToken)
            .where(RefreshToken.user_id == record.user_id, RefreshToken.revoked_at.is_(None))
            .values(revoked_at=now)
        )
        db.commit()
        return _refresh_failure("Refresh token has been revoked")
    if record.expires_at <= now:
        return _refresh_failure("Refresh token has expired")
    user = db.get(User, record.user_id)
    if user is None or not user.is_active:
        return _refresh_failure("User not found or deactivated")

    record.revoked_at = now
    db.commit()
    return _issue_tokens(db, user, response)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
def logout(request: Request, db: Session = Depends(get_db)) -> Response:
    """FR-AUTH-06: revoke the refresh token on logout. Always succeeds."""
    s = get_settings()
    token = request.cookies.get(s.refresh_cookie_name)
    if token:
        try:
            payload = decode_token(token, REFRESH)
            record = db.get(RefreshToken, uuid.UUID(payload["jti"]))
            if record is not None and record.revoked_at is None:
                record.revoked_at = utcnow()
                db.commit()
        except (TokenError, ValueError):
            pass
    response = Response(status_code=status.HTTP_204_NO_CONTENT)
    _clear_cookie(response)
    return response


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)) -> User:
    return user
