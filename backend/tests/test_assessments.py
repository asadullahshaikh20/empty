import datetime as dt

import pytest
import torch

from app.db import SessionLocal
from app.models import AssessmentResult, Image, ModelRegistry
from app.services.model_service import ModelService
from conftest import PASSWORD, auth_header, dicom_bytes, make_admin, png_bytes, register, upload
from iqa_ml import config as mlconfig


def test_assess_pass(client, user_token, active_model):
    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    r = client.post(f"/assess/{image_id}", headers=h)  # TC-07
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "completed" and body["verdict"] == "Pass"
    assert body["ai_readiness_score"] >= 75
    assert body["model_version"] == "v-test-1"  # FR-QA-17
    assert [x["key"] for x in body["heads"]] == list(mlconfig.HEADS)
    for head in body["heads"]:
        assert 0 <= head["score"] <= 100 and 0 <= head["confidence"] <= 1 and head["label"]
    assert body["warnings"] == [] and body["out_of_distribution"] is False
    assert body["modality"] == "xray" and body["modality_source"] == "model"
    assert body["image"]["id"] == image_id
    # modality detected by the model is stored on the image
    assert client.get(f"/images/{image_id}", headers=h).json()["modality"] == "xray"


def test_assess_reject_with_warnings(client, user_token, active_model, model_service):
    model_service.next_probs = {h: [0.02, 0.08, 0.9] for h in mlconfig.HEADS}
    model_service.next_probs["orientation"] = [0.4, 0.35, 0.25]
    image_id = upload(client, user_token).json()["id"]
    body = client.post(f"/assess/{image_id}", headers=auth_header(user_token)).json()  # TC-08
    assert body["verdict"] == "Reject" and body["ai_readiness_score"] < 50
    assert body["artifact_present"] is True
    assert any("Orientation" in w for w in body["warnings"])


def test_user_and_dicom_modality_are_kept(client, user_token, active_model):
    h = auth_header(user_token)
    a = upload(client, user_token, modality="ct").json()["id"]
    assert client.post(f"/assess/{a}", headers=h).json()["modality"] == "ct"
    b = upload(client, user_token, dicom_bytes("MR"), filename="x.dcm").json()["id"]
    body = client.post(f"/assess/{b}", headers=h).json()
    assert body["modality"] == "mri" and body["modality_source"] == "dicom"


def test_no_active_model_returns_503(client, user_token):
    image_id = upload(client, user_token).json()["id"]
    r = client.post(f"/assess/{image_id}", headers=auth_header(user_token))
    assert r.status_code == 503 and "No active model" in r.json()["detail"]


def test_inference_error_is_recorded(client, user_token, active_model, model_service):
    model_service.next_fail = RuntimeError("CUDA out of memory")
    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    r = client.post(f"/assess/{image_id}", headers=h)
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert detail["code"] == "inference_error" and "Traceback" not in r.text
    status = client.get(f"/assess/{detail['assessment_id']}/status", headers=h).json()  # FR-QA-08 / FR-QA-16
    assert status["status"] == "error" and status["error_code"] == "inference_error"
    assert "out of memory" in status["error_message"]
    full = client.get(f"/assessments/{detail['assessment_id']}", headers=h).json()
    assert full["status"] == "error" and full["heads"] == [] and full["verdict"] is None


def test_missing_stored_file_is_recorded(client, user_token, active_model):
    from app.services import storage

    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    with SessionLocal() as db:
        storage.resolve(db.get(Image, image_id).file_path).unlink()
    r = client.post(f"/assess/{image_id}", headers=h)
    assert r.status_code == 500
    assert client.get(f"/assess/{r.json()['detail']['assessment_id']}/status", headers=h).json()["status"] == "error"


def test_status_get_and_latest(client, user_token, active_model):
    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    assert client.get(f"/images/{image_id}/assessments/latest", headers=h).status_code == 404
    first = client.post(f"/assess/{image_id}", headers=h).json()
    second = client.post(f"/assess/{image_id}", headers=h).json()
    assert client.get(f"/assess/{first['id']}/status", headers=h).json()["status"] == "completed"
    assert client.get(f"/assessments/{first['id']}", headers=h).json()["id"] == first["id"]
    assert client.get(f"/images/{image_id}/assessments/latest", headers=h).json()["id"] == second["id"]


def test_assessment_ownership(client, user_token, active_model):
    image_id = upload(client, user_token).json()["id"]
    result_id = client.post(f"/assess/{image_id}", headers=auth_header(user_token)).json()["id"]
    other = auth_header(register(client, email="eve@example.com", username="eve")["access_token"])
    assert client.post(f"/assess/{image_id}", headers=other).status_code == 404
    assert client.get(f"/assessments/{result_id}", headers=other).status_code == 404
    assert client.get(f"/assess/{result_id}/status", headers=other).status_code == 404
    assert client.get("/assessments", headers=other).json()["total"] == 0


def test_history_pagination_and_filters(client, user_token, active_model, model_service):
    h = auth_header(user_token)
    ids = []
    for i in range(25):  # TC-12
        model_service.next_probs = None if i % 2 == 0 else {k: [0.02, 0.08, 0.9] for k in mlconfig.HEADS}
        image_id = upload(client, user_token, png_bytes(seed=i), modality="ct" if i < 5 else None).json()["id"]
        ids.append(client.post(f"/assess/{image_id}", headers=h).json()["id"])
    page2 = client.get("/assessments?page=2&page_size=10", headers=h).json()
    assert page2["total"] == 25 and page2["pages"] == 3 and page2["page"] == 2
    newest_first = list(reversed(ids))
    assert [x["id"] for x in page2["items"]] == newest_first[10:20]  # assessments 11-20, newest first
    assert len(client.get("/assessments?page=3&page_size=10", headers=h).json()["items"]) == 5
    assert client.get("/assessments?page=4&page_size=10", headers=h).json()["items"] == []

    assert client.get("/assessments?verdict=Pass", headers=h).json()["total"] == 13
    assert client.get("/assessments?verdict=Reject", headers=h).json()["total"] == 12
    assert client.get("/assessments?modality=ct", headers=h).json()["total"] == 5
    assert client.get("/assessments?modality=ct&verdict=Pass", headers=h).json()["total"] == 3
    today = dt.datetime.now(dt.timezone.utc).date()
    assert client.get(f"/assessments?date_from={today}&date_to={today}", headers=h).json()["total"] == 25
    tomorrow = today + dt.timedelta(days=1)
    assert client.get(f"/assessments?date_from={tomorrow}", headers=h).json()["total"] == 0
    assert client.get(f"/assessments?date_from={tomorrow}&date_to={today}", headers=h).status_code == 422
    assert client.get("/assessments?q=SCAN", headers=h).json()["total"] == 25  # case-insensitive file-name search
    assert client.get("/assessments?q=nomatch", headers=h).json()["total"] == 0
    assert client.get("/assessments?q=%25", headers=h).json()["total"] == 0  # LIKE wildcards are escaped
    assert client.get("/assessments?verdict=Maybe", headers=h).status_code == 422
    assert client.get("/assessments?page=0", headers=h).status_code == 422
    assert client.get("/assessments?page_size=101", headers=h).status_code == 422


def test_model_reload_on_version_change(client, user_token, active_model, model_service):
    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    assert client.post(f"/assess/{image_id}", headers=h).json()["model_version"] == "v-test-1"
    assert client.post(f"/assess/{image_id}", headers=h).json()["model_version"] == "v-test-1"
    assert model_service.loads == ["v-test-1"]  # cached, not reloaded (FR-QA-18)
    with SessionLocal() as db:
        db.query(ModelRegistry).update({"is_active": False})
        db.add(ModelRegistry(version="v-test-2", checkpoint_path="models/fake2.pt", is_active=True))
        db.commit()
    assert client.post(f"/assess/{image_id}", headers=h).json()["model_version"] == "v-test-2"
    assert model_service.loads == ["v-test-1", "v-test-2"]


def test_single_active_model_constraint(active_model):
    from sqlalchemy.exc import IntegrityError

    with SessionLocal() as db:
        db.add(ModelRegistry(version="v-other", checkpoint_path="x.pt", is_active=True))
        with pytest.raises(IntegrityError):
            db.commit()


def test_health(client, active_model):
    assert client.get("/health").json() == {"status": "ok", "database": "ok"}  # no model details for the public
    assert client.get("/health", headers={"Authorization": "Bearer junk"}).json() == {"status": "ok", "database": "ok"}
    user = register(client)["access_token"]
    assert "active_model_version" not in client.get("/health", headers=auth_header(user)).json()
    make_admin()
    admin = client.post("/auth/login", json={"email": "admin@example.com", "password": PASSWORD}).json()["access_token"]
    body = client.get("/health", headers=auth_header(admin)).json()
    assert body["active_model_version"] == "v-test-1" and body["model_loaded"] is False


def test_real_model_end_to_end(user_token, tmp_path):
    """Real ModelService + real QualityAssessor (random weights) through the API."""
    from fastapi.testclient import TestClient

    from app.main import create_app
    from iqa_ml.checkpoint import save_checkpoint
    from iqa_ml.model import QualityAssessmentModel

    torch.manual_seed(0)
    ckpt = tmp_path / "real.pt"
    save_checkpoint(ckpt, QualityAssessmentModel(pretrained=False), "v-random")
    with SessionLocal() as db:
        db.add(ModelRegistry(version="v-random", checkpoint_path=str(ckpt), is_active=True))
        db.commit()
    app = create_app(model_service=ModelService(device="cpu"), load_model=True)
    with TestClient(app) as c:
        make_admin()
        admin = c.post("/auth/login", json={"email": "admin@example.com", "password": PASSWORD}).json()["access_token"]
        assert c.get("/health", headers=auth_header(admin)).json()["loaded_model_version"] == "v-random"
        h = auth_header(user_token)
        image_id = upload(c, user_token).json()["id"]
        body = c.post(f"/assess/{image_id}", headers=h).json()
        assert body["status"] == "completed" and body["verdict"] in ("Pass", "Review", "Reject")
        assert body["model_version"] == "v-random" and body["inference_ms"] < 10_000  # FR-QA-07 (CPU)
    with SessionLocal() as db:
        r = db.query(AssessmentResult).one()
        assert r.details["heads"]["technical_quality"]["probabilities"]


def test_missing_checkpoint_gives_503(user_token):
    from fastapi.testclient import TestClient

    from app.main import create_app

    with SessionLocal() as db:
        db.add(ModelRegistry(version="v-missing", checkpoint_path="models/does-not-exist.pt", is_active=True))
        db.commit()
    with TestClient(create_app(model_service=ModelService(device="cpu"), load_model=True)) as c:
        image_id = upload(c, user_token).json()["id"]
        r = c.post(f"/assess/{image_id}", headers=auth_header(user_token))
        assert r.status_code == 503 and "not found" in r.json()["detail"]


def test_stats_counts_per_verdict(client, user_token, active_model, model_service):
    h = auth_header(user_token)
    assert client.get("/assessments/stats", headers=h).json() == {"total": 0, "Pass": 0, "Review": 0, "Reject": 0}
    for i in range(3):
        model_service.next_probs = None if i < 2 else {k: [0.02, 0.08, 0.9] for k in mlconfig.HEADS}
        image_id = upload(client, user_token, png_bytes(seed=i)).json()["id"]
        client.post(f"/assess/{image_id}", headers=h)
    assert client.get("/assessments/stats", headers=h).json() == {"total": 3, "Pass": 2, "Review": 0, "Reject": 1}
    other = auth_header(register(client, email="eve@example.com", username="eve")["access_token"])
    assert client.get("/assessments/stats", headers=other).json()["total"] == 0
    assert client.get("/assessments/stats").status_code == 401
