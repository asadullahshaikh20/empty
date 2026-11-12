"""Hybrid EfficientNet-B4 + Vision Transformer quality model (SDS 5.1.2, SRS Table 25b).

    image (3x224x224)
      |-- EfficientNet-B4 features -> global avg pool -> 1792-d   (local texture / blur / noise)
      |-- ViT-B/16 encoder -> CLS token              -> 768-d    (global layout / orientation)
      concat (2560-d) -> Linear 512 -> LayerNorm -> GELU          (fused feature vector)
      |-- 5 quality heads: MLP 512 -> 256 -> 256 -> 3 (dropout 0.3)
      |-- modality head:   MLP 512 -> 128 -> 4  (lightweight modality classifier, FR-IMG-06)
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

from . import config


def _quality_head() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(config.FUSED_FEATURES, config.HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(config.HEAD_DROPOUT),
        nn.Linear(config.HEAD_HIDDEN, config.HEAD_HIDDEN),
        nn.ReLU(inplace=True),
        nn.Dropout(config.HEAD_DROPOUT),
        nn.Linear(config.HEAD_HIDDEN, config.NUM_CLASSES),
    )


class QualityAssessmentModel(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        eff_weights = models.EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        vit_weights = models.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None

        effnet = models.efficientnet_b4(weights=eff_weights)
        self.cnn_features = effnet.features  # last block = target layer for Grad-CAM (later phase)
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        vit = models.vit_b_16(weights=vit_weights)
        vit.heads = nn.Identity()  # forward() now returns the 768-d CLS embedding
        self.vit = vit

        self.fusion = nn.Sequential(
            nn.Linear(config.EFFNET_FEATURES + config.VIT_FEATURES, config.FUSED_FEATURES),
            nn.LayerNorm(config.FUSED_FEATURES),
            nn.GELU(),
        )
        self.heads = nn.ModuleDict({h: _quality_head() for h in config.HEADS})
        self.modality_head = nn.Sequential(
            nn.Linear(config.FUSED_FEATURES, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, len(config.MODALITIES)),
        )

    def backbone_parameters(self):
        yield from self.cnn_features.parameters()
        yield from self.vit.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for p in self.backbone_parameters():
            p.requires_grad = trainable

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 3 or x.shape[2] != config.IMAGE_SIZE or x.shape[3] != config.IMAGE_SIZE:
            raise ValueError(
                f"expected input of shape (B, 3, {config.IMAGE_SIZE}, {config.IMAGE_SIZE}), got {tuple(x.shape)}"
            )
        cnn = self.cnn_pool(self.cnn_features(x)).flatten(1)
        vit = self.vit(x)
        fused = self.fusion(torch.cat([cnn, vit], dim=1))
        out = {h: head(fused) for h, head in self.heads.items()}
        out["modality"] = self.modality_head(fused)
        return out
