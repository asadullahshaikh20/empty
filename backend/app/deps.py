from __future__ import annotations

from collections.abc import Callable

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from .db import get_db
from .models import User
from .security import ACCESS, TokenError, decode_token

bearer = HTTPBearer(auto_error=False)


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(status.HTTP_401_UNAUTHORIZED, detail=detail, headers={"WWW-Authenticate": "Bearer"})


def get_current_user(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer),
    db: Session = Depends(get_db),
) -> User:
    """FR-AUTH-05: every protected endpoint requires a valid, unexpired access token."""
    if creds is None or creds.scheme.lower() != "bearer":
        raise _unauthorized("Not authenticated")
    try:
        payload = decode_token(creds.credentials, ACCESS)
    except TokenError as exc:
        raise _unauthorized(str(exc)) from exc
    user = db.get(User, payload["sub"])
    if user is None or not user.is_active:
        raise _unauthorized("User not found or deactivated")
    return user


def require_roles(*roles: str) -> Callable[[User], User]:
    """FR-AUTH-07 / FR-ADM-06: role-based access control, 403 otherwise."""

    def checker(user: User = Depends(get_current_user)) -> User:
        if user.role not in roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user

    return checker
