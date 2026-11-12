from app.config import get_settings
from app.db import SessionLocal
from app.models import AssessmentResult, Image
from app.services import storage
from conftest import auth_header, dicom_bytes, jpeg_bytes, png_bytes, register, upload


def test_upload_png(client, user_token):
    r = upload(client, user_token, png_bytes((300, 200)), filename="chest.png")  # TC-05
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["file_format"] == "PNG" and body["width"] == 300 and body["height"] == 200
    assert body["modality"] is None and body["modality_source"] is None
    assert body["original_filename"] == "chest.png"
    with SessionLocal() as db:
        img = db.get(Image, body["id"])
        for rel in (img.file_path, img.thumbnail_path, img.preview_path):
            assert storage.resolve(rel).is_file()
        assert img.file_path.startswith("uploads/") and body["id"] in img.file_path  # unique filename


def test_upload_jpeg_with_modality(client, user_token):
    r = upload(client, user_token, jpeg_bytes(), filename="eye.jpg", modality="retinal")
    assert r.status_code == 201
    assert r.json()["file_format"] == "JPEG"
    assert r.json()["modality"] == "retinal" and r.json()["modality_source"] == "user"


def test_upload_dicom_detects_modality(client, user_token):
    r = upload(client, user_token, dicom_bytes("MR"), filename="brain.dcm")
    assert r.status_code == 201
    assert r.json()["file_format"] == "DICOM"
    assert r.json()["modality"] == "mri" and r.json()["modality_source"] == "dicom"
    # browser preview is a PNG even for DICOM
    p = client.get(f"/images/{r.json()['id']}/preview", headers=auth_header(user_token))
    assert p.status_code == 200 and p.headers["content-type"] == "image/png" and p.content[:4] == b"\x89PNG"


def test_upload_auto_modality_values(client, user_token):
    for value in ("", "auto", "AUTO"):
        r = upload(client, user_token, modality=value)
        assert r.status_code == 201 and r.json()["modality"] is None


def test_upload_rejections(client, user_token):
    cases = [
        (b"hello, I am a text file", "notes.txt"),  # TC-06 (server side)
        (b"\x89PNG\r\n\x1a\n" + b"broken" * 100, "broken.png"),
        (png_bytes((100, 300)), "tiny.png"),  # < 128 px
        (b"", "empty.png"),
    ]
    for data, name in cases:
        r = upload(client, user_token, data, filename=name)
        assert r.status_code == 422, (name, r.text)
    assert upload(client, user_token, modality="ultrasound").status_code == 422
    with SessionLocal() as db:
        assert db.query(Image).count() == 0


def test_upload_too_large(client, user_token):
    s = get_settings()
    old = s.max_upload_mb
    s.max_upload_mb = 1
    try:
        r = upload(client, user_token, b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 1))
        assert r.status_code == 413
    finally:
        s.max_upload_mb = old


def test_upload_without_file(client, user_token):
    assert client.post("/images/upload", headers=auth_header(user_token)).status_code == 422


def test_get_list_files_and_ownership(client, user_token):
    ids = [upload(client, user_token, png_bytes(seed=i)).json()["id"] for i in range(3)]
    h = auth_header(user_token)
    page = client.get("/images?page=1&page_size=2", headers=h).json()
    assert page["total"] == 3 and page["pages"] == 2 and len(page["items"]) == 2
    assert page["items"][0]["id"] == ids[-1]  # newest first
    assert client.get(f"/images/{ids[0]}", headers=h).status_code == 200
    f = client.get(f"/images/{ids[0]}/file", headers=h)
    assert f.status_code == 200 and f.content == png_bytes(seed=0)
    t = client.get(f"/images/{ids[0]}/thumbnail", headers=h)
    assert t.status_code == 200 and t.content[:4] == b"\x89PNG"

    other = register(client, email="eve@example.com", username="eve")["access_token"]
    for path in (f"/images/{ids[0]}", f"/images/{ids[0]}/file", f"/images/{ids[0]}/thumbnail"):
        assert client.get(path, headers=auth_header(other)).status_code == 404
    assert client.delete(f"/images/{ids[0]}", headers=auth_header(other)).status_code == 404
    assert client.get("/images", headers=auth_header(other)).json()["total"] == 0
    assert client.get("/images/not-a-uuid", headers=h).status_code == 422


def test_delete_image_cascades(client, user_token, active_model):
    h = auth_header(user_token)
    image_id = upload(client, user_token).json()["id"]
    assert client.post(f"/assess/{image_id}", headers=h).status_code == 200
    with SessionLocal() as db:
        img = db.get(Image, image_id)
        paths = [img.file_path, img.thumbnail_path, img.preview_path]
    assert client.delete(f"/images/{image_id}", headers=h).status_code == 204  # FR-IMG-08
    assert client.get(f"/images/{image_id}", headers=h).status_code == 404
    with SessionLocal() as db:
        assert db.query(AssessmentResult).count() == 0
    assert not any(storage.resolve(p).exists() for p in paths)
