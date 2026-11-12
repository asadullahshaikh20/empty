import datetime as dt
import uuid

import jwt
import pytest

from app.config import get_settings
from app.db import SessionLocal
from app.models import RefreshToken, User
from conftest import PASSWORD, auth_header, make_admin, register


def test_register_returns_tokens_and_cookie(client):
    body = register(client, email="Bob@Example.com", username="bob")
    assert body["token_type"] == "bearer" and body["expires_in"] == 30 * 60
    assert body["user"]["email"] == "bob@example.com" and body["user"]["role"] == "radiologist"
    assert "password" not in str(body["user"]).lower()
    assert get_settings().refresh_cookie_name in client.cookies
    with SessionLocal() as db:
        u = db.query(User).one()
        assert u.password_hash.startswith("$2") and PASSWORD not in u.password_hash  # FR-AUTH-03


def test_register_researcher_role(client):
    assert register(client, role="researcher")["user"]["role"] == "researcher"


@pytest.mark.parametrize(
    "field, value",
    [
        ("role", "admin"),  # admins cannot self-register
        ("password", "short1A"),
        ("password", "alllowercase1"),
        ("password", "NoDigitsHere"),
        ("password", "A1" + "x" * 80),  # > 72 bytes
        ("email", "not-an-email"),
        ("username", "ab"),
        ("username", "bad name"),
        ("name", "   "),
    ],
)
def test_register_validation(client, field, value):
    payload = {"name": "Al", "email": "al@example.com", "username": "al_1", "password": PASSWORD, "role": "radiologist"}
    payload[field] = value
    assert client.post("/auth/register", json=payload).status_code == 422


def test_duplicate_email_and_username(client):
    register(client)
    r = client.post("/auth/register", json={"name": "X", "email": "ALICE@example.com", "username": "other",
                                            "password": PASSWORD})
    assert r.status_code == 409 and r.json()["detail"] == "Email already registered"  # TC-02
    r = client.post("/auth/register", json={"name": "X", "email": "new@example.com", "username": "ALICE",
                                            "password": PASSWORD})
    assert r.status_code == 409 and r.json()["detail"] == "Username already taken"


def test_login_success_and_me(client):
    register(client)
    client.cookies.clear()
    r = client.post("/auth/login", json={"email": "Alice@Example.com", "password": PASSWORD})  # TC-03
    assert r.status_code == 200
    token = r.json()["access_token"]
    me = client.get("/auth/me", headers=auth_header(token))
    assert me.status_code == 200 and me.json()["username"] == "alice"


def test_login_wrong_password_and_unknown_user(client):
    register(client)
    for email, pw in (("alice@example.com", "Wrong1234"), ("nobody@example.com", PASSWORD)):
        r = client.post("/auth/login", json={"email": email, "password": pw})
        assert r.status_code == 401 and r.json()["detail"] == "Invalid email or password"  # TC-04


def test_lockout_after_five_failures(client):
    register(client)
    for _ in range(5):
        assert client.post("/auth/login", json={"email": "alice@example.com", "password": "Wrong1234"}).status_code == 401
    r = client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD})
    assert r.status_code == 423 and "locked" in r.json()["detail"]
    with SessionLocal() as db:  # lock expires
        u = db.query(User).one()
        assert u.locked_until is not None
        u.locked_until = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
        db.commit()
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 200


def test_failed_counter_resets_on_success(client):
    register(client)
    for _ in range(4):
        client.post("/auth/login", json={"email": "alice@example.com", "password": "Wrong1234"})
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 200
    for _ in range(4):
        client.post("/auth/login", json={"email": "alice@example.com", "password": "Wrong1234"})
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 200


def test_deactivated_user(client):
    token = register(client)["access_token"]
    with SessionLocal() as db:
        db.query(User).update({"is_active": False})
        db.commit()
    assert client.post("/auth/login", json={"email": "alice@example.com", "password": PASSWORD}).status_code == 403
    assert client.get("/auth/me", headers=auth_header(token)).status_code == 401


@pytest.mark.parametrize("header", [None, "Bearer garbage", "Basic abc", "Bearer "])
def test_protected_endpoint_requires_valid_token(client, header):
    headers = {"Authorization": header} if header else {}
    assert client.get("/auth/me", headers=headers).status_code == 401
    assert client.get("/assessments", headers=headers).status_code == 401
    assert client.post("/images/upload", headers=headers).status_code == 401


def test_expired_and_wrong_type_tokens(client):
    body = register(client)
    s = get_settings()
    uid = body["user"]["id"]
    expired = jwt.encode({"sub": uid, "type": "access", "jti": "x", "exp": dt.datetime.now(dt.timezone.utc)
                          - dt.timedelta(minutes=1)}, s.jwt_secret, algorithm="HS256")
    r = client.get("/auth/me", headers=auth_header(expired))
    assert r.status_code == 401 and r.json()["detail"] == "Token has expired"
    refresh_as_access = client.cookies.get(s.refresh_cookie_name)
    assert client.get("/auth/me", headers=auth_header(refresh_as_access)).status_code == 401
    forged = jwt.encode({"sub": uid, "type": "access", "jti": "x", "exp": dt.datetime.now(dt.timezone.utc)
                         + dt.timedelta(minutes=5)}, "another-secret-" + "y" * 40, algorithm="HS256")
    assert client.get("/auth/me", headers=auth_header(forged)).status_code == 401


def test_refresh_rotation_and_reuse_detection(client):
    register(client)
    name = get_settings().refresh_cookie_name
    first = client.cookies.get(name)
    r = client.post("/auth/refresh")
    assert r.status_code == 200 and r.json()["access_token"]
    second = client.cookies.get(name)
    assert second and second != first
    # replaying the rotated (revoked) token revokes every session of the user
    client.cookies.clear()
    client.cookies.set(name, first)
    assert client.post("/auth/refresh").status_code == 401
    client.cookies.clear()
    client.cookies.set(name, second)
    assert client.post("/auth/refresh").status_code == 401
    with SessionLocal() as db:
        assert all(t.revoked_at is not None for t in db.query(RefreshToken).all())


def test_refresh_without_cookie_and_logout(client):
    register(client)
    name = get_settings().refresh_cookie_name
    token = client.cookies.get(name)
    r = client.post("/auth/logout")
    assert r.status_code == 204
    assert name not in client.cookies
    client.cookies.clear()
    client.cookies.set(name, token)
    assert client.post("/auth/refresh").status_code == 401  # revoked on logout
    client.cookies.clear()
    assert client.post("/auth/refresh").status_code == 401
    assert client.post("/auth/logout").status_code == 204  # idempotent


def test_expired_refresh_token(client):
    register(client)
    with SessionLocal() as db:
        db.query(RefreshToken).update({"expires_at": dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)})
        db.commit()
    assert client.post("/auth/refresh").status_code == 401


def test_admin_rbac(client, active_model):
    token = register(client)["access_token"]
    r = client.get("/admin/models", headers=auth_header(token))
    assert r.status_code == 403 and r.json()["detail"] == "Insufficient permissions"  # TC-14
    make_admin()
    client.cookies.clear()
    admin_token = client.post("/auth/login", json={"email": "admin@example.com", "password": PASSWORD}).json()[
        "access_token"
    ]
    r = client.get("/admin/models", headers=auth_header(admin_token))
    assert r.status_code == 200 and r.json()[0]["version"] == "v-test-1"


def test_token_for_deleted_user(client):
    token = register(client)["access_token"]
    with SessionLocal() as db:
        db.query(User).delete()
        db.commit()
    assert client.get("/auth/me", headers=auth_header(token)).status_code == 401
    assert uuid.UUID(jwt.decode(token, options={"verify_signature": False})["sub"])
