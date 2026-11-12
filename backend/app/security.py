"""Password hashing (bcrypt, FR-AUTH-03) and JWT access/refresh tokens (FR-AUTH-04)."""

from __future__ import annotations

import datetime as dt
import uuid

import bcrypt
import jwt

from .config import get_settings

ACCESS = "access"
REFRESH = "refresh"
BCRYPT_MAX_BYTES = 72


class TokenError(Exception):
    pass


def hash_password(password: str) -> str:
    raw = password.encode("utf-8")
    if len(raw) > BCRYPT_MAX_BYTES:
        raise ValueError("password longer than 72 bytes")
    return bcrypt.hashpw(raw, bcrypt.gensalt(rounds=12)).decode("ascii")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("ascii"))
    except ValueError:  # malformed hash or >72-byte password
        return False


# Used to spend the same time on unknown e-mails as on wrong passwords (no user enumeration by timing)
DUMMY_HASH = bcrypt.hashpw(b"dummy-password-for-timing", bcrypt.gensalt(rounds=12)).decode("ascii")


def _now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def create_access_token(user_id: uuid.UUID, role: str) -> tuple[str, int]:
    s = get_settings()
    now = _now()
    expires = now + dt.timedelta(minutes=s.access_token_minutes)
    payload = {"sub": str(user_id), "role": role, "type": ACCESS, "iat": now, "exp": expires, "jti": uuid.uuid4().hex}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm), s.access_token_minutes * 60


def create_refresh_token(user_id: uuid.UUID) -> tuple[str, uuid.UUID, dt.datetime]:
    s = get_settings()
    now = _now()
    jti = uuid.uuid4()
    expires = now + dt.timedelta(days=s.refresh_token_days)
    payload = {"sub": str(user_id), "type": REFRESH, "iat": now, "exp": expires, "jti": str(jti)}
    return jwt.encode(payload, s.jwt_secret, algorithm=s.jwt_algorithm), jti, expires


def decode_token(token: str, expected_type: str) -> dict:
    s = get_settings()
    try:
        payload = jwt.decode(
            token, s.jwt_secret, algorithms=[s.jwt_algorithm], options={"require": ["exp", "sub", "type", "jti"]}
        )
    except jwt.ExpiredSignatureError as exc:
        raise TokenError("Token has expired") from exc
    except jwt.InvalidTokenError as exc:
        raise TokenError("Invalid token") from exc
    if payload.get("type") != expected_type:
        raise TokenError("Invalid token type")
    try:
        payload["sub"] = uuid.UUID(payload["sub"])
    except (ValueError, TypeError) as exc:
        raise TokenError("Invalid token subject") from exc
    return payload
