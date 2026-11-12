"""Model, checkpoint, inference, dataset, metrics and end-to-end script tests (CPU, random weights)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from conftest import ML_ROOT, png_bytes, synthetic_scan
from iqa_ml import config
from iqa_ml.checkpoint import load_model, read_checkpoint, save_checkpoint
from iqa_ml.dataset import QualityDataset, estimate_class_weights, read_manifest
from iqa_ml.inference import QualityAssessor
from iqa_ml.metrics import classification_metrics
from iqa_ml.model import QualityAssessmentModel

sys.path.insert(0, str(ML_ROOT / "scripts"))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return QualityAssessmentModel(pretrained=False).eval()


def test_forward_shapes(model):
    with torch.inference_mode():
        out = model(torch.randn(2, 3, 224, 224))
    assert set(out) == {*config.HEADS, "modality"}
    for h in config.HEADS:
        assert out[h].shape == (2, 3)
    assert out["modality"].shape == (2, 4)


def test_forward_rejects_wrong_shape(model):
    with pytest.raises(ValueError):
        model(torch.randn(1, 3, 128, 128))


def test_freeze_backbone(model):
    model.set_backbone_trainable(False)
    assert not any(p.requires_grad for p in model.backbone_parameters())
    assert all(p.requires_grad for p in model.heads.parameters())
    model.set_backbone_trainable(True)
    assert all(p.requires_grad for p in model.backbone_parameters())


def test_checkpoint_roundtrip_and_assessor(model, tmp_path):
    path = tmp_path / "m.pt"
    save_checkpoint(path, model, "v-test", accuracy=0.5)
    meta = read_checkpoint(path)
    assert meta["version"] == "v-test" and meta["accuracy"] == 0.5
    loaded, meta = load_model(path)
    x = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        assert torch.allclose(loaded(x)["technical_quality"], model(x)["technical_quality"], atol=1e-5)

    assessor = QualityAssessor.from_checkpoint(path, device="cpu")
    res = assessor.assess(png_bytes(synthetic_scan(1)))
    assert res["model_version"] == "v-test"
    assert res["verdict"] in config.VERDICTS
    assert 0 <= res["ai_readiness_score"] <= 100
    assert set(res["heads"]) == set(config.HEADS)
    assert res["modality_source"] == "model" and res["modality"] in config.MODALITIES
    assert res["inference_ms"] >= 0
    res_user = assessor.assess(png_bytes(synthetic_scan(1)), modality="ct")
    assert res_user["modality"] == "ct" and res_user["modality_source"] == "user"
    # identical input -> identical output (eval mode, no dropout)
    assert res_user["heads"] == res["heads"]
    with pytest.raises(ValueError):
        assessor.assess(png_bytes(synthetic_scan(1)), modality="ultrasound")


def test_bad_checkpoint(tmp_path):
    path = tmp_path / "bad.pt"
    torch.save({"hello": 1}, path)
    with pytest.raises(ValueError):
        read_checkpoint(path)


def test_dataset(tiny_manifest):
    df = read_manifest(tiny_manifest)
    assert len(df) == 48
    ds = QualityDataset(tiny_manifest, "val", seed=3)
    assert len(ds) == 12
    x, y, m = ds[0]
    assert x.shape == (3, 224, 224) and y.shape == (5,) and y.dtype == torch.long
    assert int(m) in range(4) and all(0 <= int(v) <= 2 for v in y)
    assert int(y[4]) == int(y[:4].max())  # suitability = max severity
    x2, y2, _ = ds[0]
    assert torch.equal(x, x2) and torch.equal(y, y2)  # val is deterministic

    train = QualityDataset(tiny_manifest, "train", train=True, seed=3)
    labels = {tuple(train[0][1].tolist()) for _ in range(12)}
    assert len(labels) > 1  # train re-samples degradations

    sub = QualityDataset(tiny_manifest, "train", max_samples=8)
    assert len(sub) == 8 and set(sub.df["modality"]) == set(config.MODALITIES)


def test_manifest_validation(tmp_path):
    p = tmp_path / "m.csv"
    pd.DataFrame([{"path": "a.png", "modality": "ultrasound", "source": "x", "split": "train"}]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_manifest(p)
    pd.DataFrame([{"path": "a.png"}]).to_csv(p, index=False)
    with pytest.raises(ValueError):
        read_manifest(p)


def test_class_weights():
    w = estimate_class_weights(n=3000)
    assert set(w) == set(config.HEADS)
    for h in config.HEADS:
        assert len(w[h]) == 3 and abs(np.mean(w[h]) - 1) < 1e-3 and min(w[h]) > 0


def test_metrics():
    y = [0, 1, 2, 2]
    probs = [[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.6, 0.2, 0.2]]
    m = classification_metrics(y, probs, 3)
    assert m["accuracy"] == 0.75 and m["n"] == 4
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [1, 0, 1]]
    assert 0 <= m["auc"] <= 1
    assert classification_metrics([1, 1], [[0.1, 0.9, 0], [0.2, 0.8, 0]], 3)["auc"] is None
    assert classification_metrics([0, 2], [[0.9, 0, 0.1], [0.2, 0, 0.8]], 3)["auc"] == 1.0


def test_train_evaluate_export_scripts(tiny_manifest, tmp_path):
    import evaluate as eval_script
    import export_degraded
    import train as train_script

    out = tmp_path / "run"
    test = train_script.main([
        "--manifest", str(tiny_manifest), "--out-dir", str(out), "--version", "v-smoke",
        "--epochs", "2", "--batch-size", "4", "--num-workers", "0", "--no-pretrained",
        "--freeze-backbone", "--checkpoint-every", "1",
    ])
    assert (out / "best.pt").exists() and (out / "last.pt").exists()
    assert (out / "history.csv").exists() and (out / "test_metrics.md").exists()
    meta = read_checkpoint(out / "best.pt")
    assert meta["version"] == "v-smoke" and "test_metrics" in meta
    assert meta["accuracy"] == pytest.approx(test["mean_accuracy"])
    assert json.loads((out / "test_metrics.json").read_text())["n"] == 12

    # resume continues from epoch 2 to 3
    train_script.main([
        "--manifest", str(tiny_manifest), "--out-dir", str(out), "--epochs", "3", "--batch-size", "4",
        "--num-workers", "0", "--no-pretrained", "--freeze-backbone", "--checkpoint-every", "1", "--resume",
    ])
    assert read_checkpoint(out / "last.pt")["epoch"] == 3

    metrics = eval_script.main(["--checkpoint", str(out / "best.pt"), "--manifest", str(tiny_manifest),
                                "--num-workers", "0", "--batch-size", "4"])
    assert metrics["model_version"] == "v-smoke" and (out / "eval_test.json").exists()

    labels = export_degraded.main(["--manifest", str(tiny_manifest), "--out", str(tmp_path / "deg"), "--n", "5"])
    df = pd.read_csv(labels)
    assert len(df) == 5 and (tmp_path / "deg" / df.iloc[0]["file"]).exists()
