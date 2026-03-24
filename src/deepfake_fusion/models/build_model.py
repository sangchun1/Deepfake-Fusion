from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from src.deepfake_fusion.models.cnn.resnet18 import build_resnet18
from src.deepfake_fusion.models.transformer.vit import build_vit


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """
    dict / attribute 접근을 모두 지원하는 config getter.
    """
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


def get_model_name(model_cfg: Any) -> str:
    """
    모델 이름 추출.

    우선순위:
    1) model_cfg.name
    2) model_cfg.backbone.name
    """
    name = _cfg_get(model_cfg, "name", default=None)
    if name is None:
        name = _cfg_get(model_cfg, "backbone", "name", default=None)

    if name is None:
        raise ValueError("Could not determine model name from config.")

    return str(name).strip().lower()


def build_model(model_cfg: Any) -> nn.Module:
    """
    config를 바탕으로 모델 생성.

    현재 지원:
    - resnet18
    - vit

    예:
        model = build_model(cfg.model)
    """
    model_name = get_model_name(model_cfg)

    if model_name == "resnet18":
        return build_resnet18(model_cfg)
    if model_name == "vit":
        return build_vit(model_cfg)

    raise ValueError(
        f"Unsupported model: {model_name}. "
        "Currently supported models: ['resnet18', 'vit']"
    )


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """
    모델 파라미터 수 계산.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_model_summary(model: nn.Module) -> dict:
    """
    로깅용 간단 모델 요약 정보.
    """
    return {
        "model_class": model.__class__.__name__,
        "total_params": count_parameters(model, trainable_only=False),
        "trainable_params": count_parameters(model, trainable_only=True),
    }