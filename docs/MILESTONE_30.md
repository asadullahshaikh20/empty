# FYP 23F-34: 30% Milestone, Implementation Status

**Project:** AI-Based Medical Image Quality and Reliability Assessment
**Milestone scope:** an end-to-end working slice. Login → upload → preprocessing and modality → hybrid
EfficientNet-B4 + ViT-B/16 model with five quality heads → AI-Readiness Score and Pass/Review/Reject verdict →
dashboard results and history. Also delivered: the synthetic degradation pipeline (Phase 2) and the training/evaluation
pipeline with a Google Colab notebook (Phase 3).

**Planned for later milestones:** Grad-CAM explainability, the enhancement pipeline, PDF/JSON report export,
admin UI and audit log, benchmarking UI, password-reset e-mail, Docker/Nginx deployment.

## 1. Requirement traceability

| Requirement | Status | Where | Verified by |
|---|---|---|---|
| FR-AUTH-01 Register (name, email, username, password, role) | Done (Admin is created by script, see D2) | `backend/app/routers/auth.py` | `test_auth.py::test_register_*` |
| FR-AUTH-02 Email / username / password policy | Done | `backend/app/schemas.py` | `test_register_validation` (9 cases) |
| FR-AUTH-03 bcrypt hashing | Done | `backend/app/security.py` | `test_register_returns_tokens_and_cookie` |
| FR-AUTH-04 JWT access 30 min / refresh 7 days | Done | `security.py`, `auth.py` | `test_register_returns_tokens_and_cookie` |
| FR-AUTH-05 401 on missing/invalid/expired token | Done | `backend/app/deps.py` | `test_protected_endpoint_requires_valid_token`, `test_expired_and_wrong_type_tokens` |
| FR-AUTH-06 Refresh + revoke on logout | Done (rotation and reuse detection) | `auth.py` | `test_refresh_rotation_and_reuse_detection`, `test_refresh_without_cookie_and_logout` |
| FR-AUTH-07 / FR-ADM-06 RBAC, 403 for non-admins | Done | `deps.py`, `GET /admin/models` | `test_admin_rbac` (SDS TC-14) |
| FR-AUTH-09 Lock 15 min after 5 failures | Done | `auth.py` | `test_lockout_after_five_failures` |
| FR-IMG-01 DICOM/PNG/JPEG ≤ 50 MB | Done | `backend/app/routers/images.py` | `test_upload_png/jpeg/dicom`, `test_upload_too_large` |
| FR-IMG-02 Reject corrupt / non-image files | Done (type detected from content, not extension) | `ml/iqa_ml/preprocessing.py` | `test_upload_rejections`, `test_rejects_bad_files` |
| FR-IMG-03 Unique stored filename | Done | `backend/app/services/storage.py` | `test_upload_png` |
| FR-IMG-04 Metadata in DB | Done | `images` table | `test_upload_png` |
| FR-IMG-05 Resize 224×224, normalise, tensor | Done | `preprocessing.py` | `test_preprocess_tensor` |
| FR-IMG-06 Modality detection | Done: user choice > DICOM tag > model modality head | `inference.py`, `model.py` | `test_user_and_dicom_modality_are_kept`, `test_dicom_modality_map` |
| FR-IMG-07 Upload progress bar | Done | `frontend/src/pages/UploadPage.jsx` | manual |
| FR-IMG-08 Delete own image + records | Done (API) | `DELETE /images/{id}` | `test_delete_image_cascades` |
| FR-IMG-09 256 px thumbnail | Done | `storage.py` | `test_thumbnail`, `test_get_list_files_and_ownership` |
| FR-IMG-10 Drag and drop | Done | `UploadPage.jsx` | manual |
| FR-IMG-12 Reject < 128×128 | Done | `preprocessing.py` | `test_minimum_size`, `test_upload_rejections` |
| FR-QA-01 Load versioned checkpoint | Done | `backend/app/services/model_service.py` | `test_real_model_end_to_end` |
| FR-QA-02 Five scores 0–100 + AI-Readiness | Done | `ml/iqa_ml/scoring.py` | `test_scoring.py`, `test_assess_pass` |
| FR-QA-03 Confidence from softmax | Done | `scoring.py` | `test_summarize_*` |
| FR-QA-04 Verdict from thresholds | Done (see D1) | `scoring.py` | `test_verdict_boundaries` (13 boundary cases) |
| FR-QA-05 Store result in DB | Done | `assessment_results` table | `test_real_model_end_to_end` |
| FR-QA-06 GPU with CPU fallback | Done | `inference.select_device` | CPU tested; GPU on Colab |
| FR-QA-07 < 10 s per image on CPU | Done | | `test_real_model_end_to_end` (asserts < 10 s); see §4 |
| FR-QA-08 Status endpoint | Done | `GET /assess/{id}/status` | `test_inference_error_is_recorded` |
| FR-QA-11..15 Head label semantics | Done | `config.py` (SRS Table 25) | `test_score_labels` |
| FR-QA-16 Log errors, store reason code | Done | `assessments.py` | `test_inference_error_is_recorded`, `test_missing_stored_file_is_recorded` |
| FR-QA-17 Model version with each result | Done | | `test_assess_pass` |
| FR-QA-18 Cache model, reload on version change | Done | `model_service.py` | `test_model_reload_on_version_change` |
| FR-QA-19 DICOM window/level | Done | `preprocessing.py` (VOI LUT, MONOCHROME1) | `test_dicom_monochrome1_window_and_modality` |
| FR-RPT-04 Score gauges | Done | `ReadinessGauge.jsx`, radar chart, per-head bars | `components.test.jsx` |
| FR-RPT-05 Paginated history, newest first | Done | `GET /assessments`, `HistoryPage.jsx` | `test_history_pagination_and_filters` (SDS TC-12) |
| FR-RPT-06 Filter by date, modality, verdict | Done | same | same |
| NFR-SAF-01 Disclaimer | Done | footer on every screen, API description | manual |
| SRS Table 25b training configuration | Done | `ml/scripts/train.py` | `test_train_evaluate_export_scripts` |
| SRS Table 25c confidence levels, OOD warning | Done | `scoring.py` | `test_confidence_levels`, `test_summarize_low_confidence_and_ood` |
| Phase 2 synthetic degradation (blur, noise, exposure, rotation, artifacts) | Done | `ml/iqa_ml/degradation.py` | `test_degradation.py` |

## 2. Design decisions (where SRS and SDS differ or are silent)

| # | Decision | Reason |
|---|---|---|
| D1 | **Verdict = stricter of the SRS and SDS rules**, precedence Reject > Review > Pass: Reject if suitability ≤ 40, any head < 34, or readiness < 50; else Review if suitability ≤ 70, any head ≤ 66, or readiness < 75; else Pass. | SRS §4.4 defines the verdict by head scores; SDS §5.1.2 and test cases TC-07/TC-08 define it by AI-Readiness (≥ 75 / < 50). The SRS rules also overlap (suitability 80 with a head at 50 matches both "Pass" and "Review"). The combined rule satisfies every stated condition. Thresholds are configuration values (`ml/iqa_ml/config.py`). |
| D2 | Self-registration allows Radiologist or Researcher only. Administrators are created with `backend/scripts/create_admin.py`. | Letting anyone register as Administrator would bypass RBAC. SDS §9.2 screen 1 also lists only these two roles. |
| D3 | AI-Readiness = 0.40·suitability + 0.20·technical + 0.15·anatomical + 0.15·artifact + 0.10·orientation. | The SDS asks for a "weighted aggregate" without giving weights. Diagnostic suitability is the overall head, so it gets the largest weight. |
| D4 | Each head is a 3-class classifier. Score = 100·P(good) + 50·P(intermediate) + 0·P(bad). The label comes from the SRS Table 25 score ranges. Confidence = max softmax probability. | Gives the 0–100 score and softmax confidence required by FR-QA-02/03 from one classifier. |
| D5 | Training augmentation uses scale 0.95–1.05 and rotation ≤ 3°, **without random flips**. | SRS Table 25b lists flips and 5° rotation, but flips and rotations are themselves *orientation* labels. Augmenting with them would corrupt those labels. |
| D6 | The orientation degradation uses rotations and vertical flips, never horizontal flips. | A mirrored retina (left vs right eye) cannot be detected from pixels, so it would be label noise. |
| D7 | Image → Assessment is one-to-many. The UI shows the latest. | The SRS data model has "many"; the SDS ERD has 0..1. One-to-many is a superset and allows re-assessment after a model update. |
| D8 | `assessment_results.model_id` is `ON DELETE RESTRICT` (all other FKs CASCADE as in the SDS). | Deleting a model version must not silently delete the results it produced (traceability, FR-QA-17). |
| D9 | The modality classifier is an extra lightweight head on the shared fused features, trained jointly. | FR-IMG-06 allows "a lightweight classifier or metadata". This adds no extra model to deploy. |
| D10 | A partial unique index guarantees at most one active model in `model_registry`. | SDS §4.1.6: "Only one record has isActive = true at any given time", enforced by the database. |
| D11 | Access token in memory and refresh token in an httpOnly, SameSite=Strict cookie, rotated on every refresh. Re-use of a rotated token revokes all of that user's sessions. | SDS §3.1 keeps the session in memory. The cookie lets a page reload keep the user logged in without exposing the token to JavaScript. |
| D12 | Initial training data: MedMNIST+ at 224 px (ChestMNIST from NIH ChestX-ray14, OrganAMNIST from LiTS CT, RetinaMNIST from DeepDRiD) and IXI T1 MRI. Full ChestX-ray14, EyePACS and BraTS can be added with `prepare_data.py --source folder`. | These download without registration on Colab. MedMNIST's official splits are kept, and IXI is split by subject, so no patient appears in more than one split. EyePACS (Kaggle) and BraTS need accounts. |

## 3. Synthetic degradation labels

Each training image gets a target suitability, sampled ⅓ clean, ⅓ minor, ⅓ severe so the classes are balanced. Per-head
severities are then sampled consistently, and suitability = max(severity of the other four heads).

| Head | Severity 1 (intermediate) | Severity 2 (bad) |
|---|---|---|
| Technical quality | one of: Gaussian blur σ 1.0–1.8, motion blur 5–9 px, Gaussian noise σ 8–15, speckle 8–15 %, contrast ×0.45–0.60, gamma 1.8–2.5 / 0.45–0.60 | one or two of the same, σ 2.6–4.5, 13–21 px, σ 24–40, 25–40 %, ×0.15–0.30, gamma 3.5–5 / 0.2–0.3 |
| Anatomical completeness | 15–30 % of the anatomy shifted out of the field of view | 35–55 % |
| Artifact | ghosting (weight 0.15–0.25), 3–6 thin rings (amplitude 25–40) or banding (12–20) | ghosting 0.35–0.50, 6–12 rings (50–80) or banding (30–50) |
| Orientation | rotation 8–20° | rotation 25–180°, 90/180/270° or vertical flip |

Validation and test images get a fixed degradation per image (seeded), so metrics are reproducible.
`ml/scripts/export_degraded.py` writes labelled examples for inspection.

## 4. Verification results (this milestone)

| Suite | Result |
|---|---|
| ML package (`ml/tests`): preprocessing, degradation, scoring, model, checkpoint, dataset, metrics, data preparation, train/evaluate/export scripts | **94 passed** |
| Backend API (`backend/tests`), against PostgreSQL 16 | **51 passed** (auth, RBAC, lockout, token rotation, uploads incl. DICOM, ownership, assessment, error handling, history pagination/filters/search, verdict statistics, admin-only health details, model reload, real-model end-to-end) |
| Database migration: `alembic upgrade head` → `downgrade base` → `upgrade head`; `alembic check` | no drift between migration and ORM models |
| Frontend: `npm run lint` (0 warnings), `npm test`, `npm run build` | lint clean, **14 tests passed**, production build OK |
| End-to-end over HTTP through the dashboard proxy (register → upload → assess → history) | **21/21 checks passed** with the real servers and a registered model; CPU inference 0.3–0.4 s per image (≈3 s on the first request, which loads the model). All four screens were rendered in headless Edge and inspected. |

**About accuracy:** the model checkpoint used for the local demo is a *CPU smoke model*. It was trained for 2 epochs on
96 retinal images with frozen backbones, only to prove the pipeline end to end, so **its scores are not meaningful**.
The real model is trained with `ml/notebooks/train_colab.ipynb` on a Colab GPU. Its per-head accuracy, F1 and AUC are
written to `test_metrics.md` and should be reported against the OBJ-3 target (≥ 85 % mean accuracy, AUC > 0.85).

## 5. Changes after the internal code review

| Area | Change |
|---|---|
| Security | Refresh-token cookie is `Secure` by default (`COOKIE_SECURE=false` only for local http). Checkpoints are loaded with `torch.load(weights_only=True)`, so a model file cannot execute code. Expired refresh tokens are deleted at each login. `/health` returns only `status`/`database` unless the caller is an administrator. Login redirects are limited to in-app paths. `create_admin.py` confirms the password and validates the email and username. |
| Robustness | Model loading at startup runs in a worker thread. Path-traversal check uses `Path.is_relative_to`. DICOM windowing failures are logged. Tar extraction works on Python versions without extraction filters. The IXI download uses HTTPS. |
| Training | Loss computed inside the autocast block, GradScaler created for the actual device, checkpoints read to CPU before resuming, UTC model version names. |
| API | `GET /assessments/stats` (completed assessments per verdict) and a `q` file-name search on `GET /assessments` (case-insensitive, wildcards escaped). |
| Dashboard | Redesign: sticky header with user menu, modality cards, image preview with progress steps, results page with gauge, per-head cards, zoomable image viewer and warnings panel, history page with KPI cards, search, filter chips, keyboard-accessible rows and numbered pagination. |

Review items not applied, with reasons: the deactivated-account check stays after the password check (otherwise it reveals account status without the password); the duplicate-email message stays as required by SDS TC-02; role selection at registration stays as required by FR-AUTH-01.

