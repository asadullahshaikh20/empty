"""Generates train_colab.ipynb (kept as code so the notebook stays reviewable). Run: python build_notebook.py"""

import json
from pathlib import Path

CELLS = [
    ("markdown", """# Training the hybrid EfficientNet-B4 + ViT-B/16 quality model (Google Colab)

FYP 23F-34 · AI-Based Medical Image Quality and Reliability Assessment

**Before you start:** `Runtime → Change runtime type → GPU` (T4 or better).

Steps: 1) get the code, 2) prepare the data, 3) inspect degraded samples, 4) train (resumable), 5) evaluate, 6) download `best.pt`.

All outputs (dataset cache, checkpoints) are written to Google Drive under `MyDrive/iqa/` so an interrupted session can resume."""),
    ("code", """!nvidia-smi -L || echo "No GPU - select Runtime > Change runtime type > GPU" """),
    ("code", """from google.colab import drive
drive.mount('/content/drive')
DRIVE = '/content/drive/MyDrive/iqa'
!mkdir -p "$DRIVE\""""),
    ("markdown", """## 1. Get the code
Zip the project's `ml/` folder on your PC (right-click → *Compress*), name it `ml.zip` and upload it to `MyDrive/iqa/`.
(Alternatively, if the project is on GitHub, replace this cell with `!git clone <repo-url>` and set `ML_DIR`.)"""),
    ("code", """import os
ML_DIR = '/content/ml'
!rm -rf /content/ml && mkdir -p /content/ml && unzip -q "$DRIVE/ml.zip" -d /content/ml_unzip
# the zip may contain the files directly or inside an ml/ folder
src = '/content/ml_unzip/ml' if os.path.isdir('/content/ml_unzip/ml') else '/content/ml_unzip'
!cp -r "$src"/. /content/ml/ && rm -rf /content/ml_unzip
%cd /content/ml
!pip install -q -e ".[data]"
!ls"""),
    ("markdown", """## 2. Prepare the data
* `medmnist`: ChestMNIST (X-ray), OrganAMNIST (CT), RetinaMNIST (retinal) at 224×224 – official splits kept.
* `ixi`: IXI-T1 brain MRI (~4.5 GB download, extracted to axial slices, split by subject).

The prepared dataset is stored on Drive; re-running skips nothing, so only run this once (set `PREPARE = False` afterwards)."""),
    ("code", """PREPARE = True
MAX_PER_MODALITY = 6000
DATA = f'{DRIVE}/prepared'
if PREPARE:
    !python scripts/prepare_data.py --source medmnist --out "$DATA" --cache "$DRIVE/cache" --max-per-modality $MAX_PER_MODALITY
    !python scripts/prepare_data.py --source ixi --out "$DATA" --cache "$DRIVE/cache" --max-per-modality $MAX_PER_MODALITY --append
# Copy to the local disk: much faster than reading thousands of small files from Drive during training
!rm -rf /content/prepared && cp -r "$DATA" /content/prepared
!head -3 /content/prepared/manifest.csv"""),
    ("markdown", "## 3. Inspect synthetic degradations (Phase 2 deliverable)"),
    ("code", """!python scripts/export_degraded.py --manifest /content/prepared/manifest.csv --out /content/degraded_samples --n 24
import pandas as pd, matplotlib.pyplot as plt, matplotlib.image as mpimg
df = pd.read_csv('/content/degraded_samples/labels.csv')
fig, axes = plt.subplots(3, 8, figsize=(20, 8))
for ax, (_, r) in zip(axes.flat, df.iterrows()):
    ax.imshow(mpimg.imread(f"/content/degraded_samples/{r.file}")); ax.axis('off')
    ax.set_title(f"{r.modality} | {r.diagnostic_suitability_class}\\nT{r.technical_quality_severity} A{r.anatomical_completeness_severity} "
                 f"R{r.artifact_severity} O{r.orientation_severity}", fontsize=8)
plt.tight_layout(); plt.show()
df.head()"""),
    ("markdown", """## 4. Train
Defaults follow SRS Table 25b (AdamW 1e-4, cosine LR, batch 32, FP16, 50 epochs, early stopping patience 10, checkpoint every 5 epochs).
If the Colab session disconnects, re-run the cells above (with `PREPARE = False`) and this cell: it resumes from `last.pt` on Drive.
Reduce `BATCH` to 16 if you get a CUDA out-of-memory error."""),
    ("code", """VERSION = 'v1.0.0'
BATCH = 32
RUN = f'{DRIVE}/runs/{VERSION}'
resume = '--resume' if os.path.exists(f'{RUN}/last.pt') else ''
!python scripts/train.py --manifest /content/prepared/manifest.csv --out-dir "$RUN" --version $VERSION --batch-size $BATCH --num-workers 2 $resume"""),
    ("markdown", "## 5. Evaluate on the test split"),
    ("code", """!python scripts/evaluate.py --checkpoint "$RUN/best.pt" --manifest /content/prepared/manifest.csv --split test
from IPython.display import Markdown, display
display(Markdown(open(f'{RUN}/eval_test.md').read()))
pd.read_csv(f'{RUN}/history.csv').plot(x='epoch', y=['train_loss', 'val_loss'], figsize=(8, 4), grid=True)"""),
    ("markdown", """## 6. Download the checkpoint
Put the file in the project as `backend/models/v1.0.0.pt`, then from `backend/` run
`python scripts/register_model.py --checkpoint models/v1.0.0.pt --activate`."""),
    ("code", """import shutil
from google.colab import files
out = f'/content/{VERSION}.pt'
shutil.copy(f'{RUN}/best.pt', out)
files.download(out)"""),
]


def main():
    nb = {
        "cells": [
            {
                "cell_type": kind,
                "metadata": {},
                "source": src.strip("\n").splitlines(keepends=True),
                **({"execution_count": None, "outputs": []} if kind == "code" else {}),
            }
            for kind, src in CELLS
        ],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 0,
    }
    out = Path(__file__).with_name("train_colab.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
