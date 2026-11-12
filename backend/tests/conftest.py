"""Backend test fixtures.

Tests run against a separate PostgreSQL database (TEST_DATABASE_URL, from the environment or
backend/.env). Its schema is rebuilt with Alembic at the start of the session and all tables
are truncated before every test. For safety, the database name must contain 'test'.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))


def _test_db_url() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    env_file = BACKEND_DIR / ".env"
    if not url and env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("TEST_DATABASE_URL="):
                url = line.split("=", 1)[1].strip()
    if not url:
        pytest.exit("TEST_DATABASE_URL is not set (environment or backend/.env)", returncode=2)
    if "test" not in url.rsplit("/", 1)[-1]:
        pytest.exit(f"Refusing to run: test database name must contain 'test' ({url.rsplit('/', 1)[-1]})", returncode=2)
    return url


TEST_DB_URL = _test_db_url()
_STORAGE = tempfile.mkdtemp(prefix="iqa_test_storage_")
os.environ.update(
    {
        "DATABASE_URL": TEST_DB_URL,
        "JWT_SECRET": "test-secret-" + "x" * 40,
        "STORAGE_DIR": _STORAGE,
        "LOAD_MODEL_ON_STARTUP": "false",
        "MODEL_DEVICE": "cpu",
        "COOKIE_SECURE": "false",
    }
)

from alembic import command  # noqa: E402
from alembic.config import Config  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
from sqlalchemy import text  # noqa: E402

from app.db import SessionLocal, engine  # noqa: E402
from app.main import create_app  # noqa: E402
from app.models import ModelRegistry, User  # noqa: E402
from app.security import hash_password  # noqa: E402
from app.services.model_service import ModelService  # noqa: E402
from iqa_ml import scoring  # noqa: E402
from iqa_ml import config as mlconfig  # noqa: E402

PASSWORD = "Secret123"


def alembic_config() -> Config:
    cfg = Config(str(BACKEND_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    return cfg


@pytest.fixture(scope="session", autouse=True)
def migrated_database():
    os.environ["ALEMBIC_DATABASE_URL"] = TEST_DB_URL
    cfg = alembic_config()
    with engine.begin() as conn:  # start from a clean schema even if a previous run crashed
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    command.upgrade(cfg, "head")
    yield
    engine.dispose()


@pytest.fixture(autouse=True)
def clean_tables(migrated_database):
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE assessment_results, images, model_registry, refresh_tokens, users CASCADE"))
    yield


# ------------------------------------------------------------------ fake model
class FakeAssessor:
    """Deterministic stand-in for QualityAssessor."""

    def __init__(self, version: str, probs: dict | None = None, fail: Exception | None = None):
        self.version = version
        self.probs = probs or {h: [0.96, 0.03, 0.01] for h in mlconfig.HEADS}
        self.fail = fail
        self.calls = 0

    def assess_loaded(self, loaded, modality=None):
        self.calls += 1
        if self.fail:
            raise self.fail
        out = scoring.summarize(self.probs)
        predicted = {"modality": "xray", "confidence": 0.9}
        if modality:
            resolved, src = modality, "user"
        elif loaded.metadata_modality:
            resolved, src = loaded.metadata_modality, "dicom"
        else:
            resolved, src = "xray", "model"
        out.update(modality=resolved, modality_source=src, modality_prediction=predicted,
                   model_version=self.version, inference_ms=12)
        return out


class FakeModelService(ModelService):
    def __init__(self):
        super().__init__(device="cpu")
        self.next_probs: dict | None = None
        self.next_fail: Exception | None = None
        self.loads: list[str] = []

    def _load(self, record):
        self.loads.append(record.version)
        return FakeAssessor(record.version)

    def get(self, db):
        assessor, record = super().get(db)
        assessor.probs = self.next_probs or {h: [0.96, 0.03, 0.01] for h in mlconfig.HEADS}
        assessor.fail = self.next_fail
        return assessor, record


@pytest.fixture
def model_service():
    return FakeModelService()


@pytest.fixture
def client(model_service):
    app = create_app(model_service=model_service, load_model=False)
    with TestClient(app) as c:
        yield c


@pytest.fixture
def active_model():
    with SessionLocal() as db:
        m = ModelRegistry(version="v-test-1", checkpoint_path="models/fake.pt", accuracy=0.9, is_active=True)
        db.add(m)
        db.commit()
        return m


# ------------------------------------------------------------------ helpers
def register(client: TestClient, email="alice@example.com", username="alice", role="radiologist", password=PASSWORD):
    r = client.post("/auth/register", json={"name": username.title(), "email": email, "username": username,
                                            "password": password, "role": role})
    assert r.status_code == 201, r.text
    return r.json()


def auth_header(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def user_token(client):
    return register(client)["access_token"]


def make_admin(email="admin@example.com", username="admin") -> uuid.UUID:
    with SessionLocal() as db:
        u = User(name="Admin", email=email, username=username, password_hash=hash_password(PASSWORD), role="admin")
        db.add(u)
        db.commit()
        return u.id


def png_bytes(size=(256, 256), seed=0, mode="L") -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0]), dtype=np.uint8)
    img = PILImage.fromarray(arr, mode="L")
    if mode != "L":
        img = img.convert(mode)
    buf = io.BytesIO()
    img.save(buf, format="PNG" if mode != "JPEG" else "JPEG")
    return buf.getvalue()


def jpeg_bytes(size=(256, 256), seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    PILImage.fromarray(arr).save(buf, format="JPEG")
    return buf.getvalue()


def dicom_bytes(modality="CT", size=160) -> bytes:
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = SecondaryCaptureImageStorage
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = modality
    ds.Rows = ds.Columns = size
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 12, 11, 0
    ds.PixelData = (np.arange(size * size).reshape(size, size) % 4096).astype(np.uint16).tobytes()
    buf = io.BytesIO()
    ds.save_as(buf, enforce_file_format=True)
    return buf.getvalue()


def upload(client, token, data=None, filename="scan.png", modality=None):
    files = {"file": (filename, data if data is not None else png_bytes(), "application/octet-stream")}
    form = {"modality": modality} if modality is not None else None
    return client.post("/images/upload", files=files, data=form, headers=auth_header(token))
