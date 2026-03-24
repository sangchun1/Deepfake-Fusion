from __future__ import annotations

from typing import Any, Mapping, Optional

import timm
import torch
import torch.nn as nn


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """dict / attribute 접근을 모두 지원하는 config getter."""
    current = cfg
    for key in keys:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return current


def _create_timm_vit(
    model_name: str,
    pretrained: bool = True,
    in_channels: int = 3,
    img_size: Optional[int] = None,
    global_pool: Optional[str] = "token",
    drop_path_rate: float = 0.0,
) -> nn.Module:
    """
    timm ViT 생성.
    timm 버전/모델별로 일부 kwargs를 지원하지 않을 수 있어서 fallback 처리 포함.
    """
    create_kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "in_chans": in_channels,
        "num_classes": 0,  # feature extractor로 사용, classifier head는 별도 구성
    }

    if img_size is not None:
        create_kwargs["img_size"] = img_size
    if global_pool is not None:
        create_kwargs["global_pool"] = global_pool
    if drop_path_rate > 0.0:
        create_kwargs["drop_path_rate"] = drop_path_rate

    try:
        return timm.create_model(model_name, **create_kwargs)
    except TypeError:
        # 일부 timm 모델/버전 호환 fallback
        fallback_kwargs = {
            "pretrained": pretrained,
            "in_chans": in_channels,
            "num_classes": 0,
        }
        return timm.create_model(model_name, **fallback_kwargs)


class ViTClassifier(nn.Module):
    """timm ViT 기반 이미지 분류기."""

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        dropout: float = 0.0,
        freeze_backbone: bool = False,
        img_size: Optional[int] = 224,
        global_pool: str = "token",
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()

        if not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got: {num_classes}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got: {in_channels}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got: {dropout}")
        if drop_path_rate < 0.0:
            raise ValueError(f"drop_path_rate must be >= 0, got: {drop_path_rate}")

        backbone = _create_timm_vit(
            model_name=model_name,
            pretrained=pretrained,
            in_channels=in_channels,
            img_size=img_size,
            global_pool=global_pool,
            drop_path_rate=drop_path_rate,
        )

        feature_dim = int(getattr(backbone, "num_features", 0))
        if feature_dim <= 0:
            raise ValueError(
                f"Could not determine feature dimension from model: {model_name}"
            )

        self.model_name = model_name
        self.feature_dim = feature_dim
        self.backbone = backbone

        if dropout > 0.0:
            self.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(self.feature_dim, num_classes),
            )
        else:
            self.head = nn.Linear(self.feature_dim, num_classes)

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        """backbone 파라미터만 freeze."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """backbone 전체 unfreeze."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        classifier head 직전 feature 추출.
        반환 shape: [B, feature_dim]
        """
        features = self.backbone.forward_features(x)

        # timm VisionTransformer 계열은 forward_head(..., pre_logits=True) 지원
        if hasattr(self.backbone, "forward_head"):
            try:
                features = self.backbone.forward_head(features, pre_logits=True)
            except TypeError:
                features = self.backbone.forward_head(features)

        # 모델/버전에 따라 tuple/list 반환 가능성 방어
        if isinstance(features, (tuple, list)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor from feature extractor, got: {type(features)}"
            )

        # fallback: token sequence면 CLS 또는 average token pooling
        if features.ndim == 3:
            # [B, N, C]
            global_pool = getattr(self.backbone, "global_pool", "token")
            if global_pool == "avg":
                num_prefix_tokens = int(
                    getattr(self.backbone, "num_prefix_tokens", 1)
                )
                features = features[:, num_prefix_tokens:, :].mean(dim=1)
            else:
                features = features[:, 0, :]
        elif features.ndim == 4:
            # [B, C, H, W] 형태 fallback
            features = features.mean(dim=(2, 3))
        elif features.ndim != 2:
            raise ValueError(
                f"Unsupported feature shape from ViT backbone: {tuple(features.shape)}"
            )

        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.extract_features(x)
        logits = self.head(features)
        return logits


def build_vit(model_cfg: Any) -> ViTClassifier:
    """
    config 기반 ViT 생성 함수.

    기대하는 config 예시:
      cfg.model.name
      cfg.model.backbone.name
      cfg.model.backbone.pretrained
      cfg.model.backbone.in_channels
      cfg.model.backbone.freeze
      cfg.model.backbone.img_size
      cfg.model.backbone.global_pool
      cfg.model.backbone.drop_path_rate
      cfg.model.head.num_classes
      cfg.model.head.dropout
    """
    model_name = str(
        _cfg_get(model_cfg, "backbone", "name", default="vit_small_patch16_224")
    )
    pretrained = bool(_cfg_get(model_cfg, "backbone", "pretrained", default=True))
    in_channels = int(_cfg_get(model_cfg, "backbone", "in_channels", default=3))
    freeze_backbone = bool(_cfg_get(model_cfg, "backbone", "freeze", default=False))
    img_size = _cfg_get(model_cfg, "backbone", "img_size", default=224)
    global_pool = str(_cfg_get(model_cfg, "backbone", "global_pool", default="token"))
    drop_path_rate = float(
        _cfg_get(model_cfg, "backbone", "drop_path_rate", default=0.0)
    )

    num_classes = int(_cfg_get(model_cfg, "head", "num_classes", default=2))
    dropout = float(_cfg_get(model_cfg, "head", "dropout", default=0.0))

    model = ViTClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
        img_size=img_size,
        global_pool=global_pool,
        drop_path_rate=drop_path_rate,
    )
    return model