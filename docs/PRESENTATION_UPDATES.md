# Slide text to align `project_documents/23F-34_Presentation.pptx` with the 30% implementation

Some statements in the mid-progress deck describe work that is not in the current code. Evaluators will see the
demo, so each slide should match what runs. Replace the text below; slides not listed (1, 2, 4, 6, 7, 8) can stay.
Fill the `[ ]` placeholders after the Colab training run (`test_metrics.md`).

## Slide 3 — Work done and achievements so far

| Box | Current text (remove) | Replace with |
|---|---|---|
| Dataset pipeline | "ChestX-ray14, CheXpert, EyePACS, BraTS" | Public data: MedMNIST+ chest X-ray (from NIH ChestX-ray14), abdominal CT (LiTS), retinal fundus (DeepDRiD) and IXI brain MRI. Loader for full ChestX-ray14 / EyePACS / BraTS folders. Synthetic degradation pipeline: blur, noise, contrast/exposure, field-of-view cut-off, motion/ring/banding artifacts, rotation/flip — labelled per quality head. |
| Model architecture | "Baseline accuracy: 83% avg (target 85%)" | EfficientNet-B4 + ViT-B/16 hybrid with 5 quality heads and a modality head, implemented and tested. Training on Colab GPU: mean accuracy [ ]%, mean AUC [ ] (test split). |
| Grad-CAM integration | whole box | Move to slide 9 (Phase 4). Not part of the 30% build. |
| FastAPI backend | "Docker containerization in progress" | Done: JWT auth with refresh-token rotation and lockout, role-based access, DICOM/PNG/JPEG upload, quality assessment endpoint, history with filters and search, PostgreSQL 16 with Alembic migrations. 51 automated API tests. |
| (new box) Web dashboard | — | React dashboard: sign-in/registration, upload with modality selection, results (AI-Readiness gauge, 5 head scores, Pass/Review/Reject verdict, warnings), assessment history. |

## Slide 5 — Problems and challenges

| Box | Change |
|---|---|
| Model accuracy gap | Replace "Current avg accuracy: 83%" with: "No local GPU; training moved to Google Colab with resumable checkpoints." |
| DICOM format handling | Replace "Tested on 3 DICOM datasets — 94% parse success" with: "Window/level (VOI LUT) and MONOCHROME1 handled; compressed DICOM transfer syntaxes not yet supported." |
| React-FastAPI integration | Replace with: "Solved with a same-origin dev proxy and an httpOnly refresh-token cookie; integration verified end to end." |
| Synthetic degradation bias, GPU constraints, timeline | Can stay. |

## Slide 9 — Next steps

- **Phase 4:** Grad-CAM heatmaps; enhancement pipeline (denoising, CLAHE, deblurring) with PSNR/SSIM and re-assessment.
- **Phase 5:** PDF/JSON reports, admin screens (users, model versions, audit log), Docker + Nginx deployment.
- **Phase 6:** benchmarking against baselines, usability study (SUS ≥ 68), final report.
