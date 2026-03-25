from __future__ import annotations

from typing import Any, Mapping, Optional

import torch
import torch.nn as nn

from src.deepfake_fusion.models.cnn.resnet18 import build_resnet18
from src.deepfake_fusion.models.fusion.fusion_block import build_fusion_block
from src.deepfake_fusion.models.spectral.spai import build_spai
from src.deepfake_fusion.models.transformer.vit import build_vit


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


def _normalize_name(name: Optional[str]) -> str:
    return str(name or "").strip().lower()


def _make_activation(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    raise ValueError(
        f"Unsupported activation: {name}. "
        "Choose from ['relu', 'gelu', 'silu']."
    )


class ProjectionHead(nn.Module):
    """
    branch feature -> shared fusion dimension projection.

    기본:
      LN -> Linear -> Act -> Dropout

    hidden_dim이 주어지면:
      LN -> Linear(in, hidden) -> Act -> Dropout -> Linear(hidden, out) -> Act -> Dropout
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if in_dim <= 0:
            raise ValueError(f"in_dim must be > 0, got: {in_dim}")
        if out_dim <= 0:
            raise ValueError(f"out_dim must be > 0, got: {out_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got: {dropout}")

        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)

        act1 = _make_activation(activation)

        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(self.in_dim))

        if hidden_dim is None:
            layers.extend(
                [
                    nn.Linear(self.in_dim, self.out_dim),
                    act1,
                    nn.Dropout(p=dropout),
                ]
            )
        else:
            hidden_dim = int(hidden_dim)
            if hidden_dim <= 0:
                raise ValueError(f"hidden_dim must be > 0, got: {hidden_dim}")

            act2 = _make_activation(activation)
            layers.extend(
                [
                    nn.Linear(self.in_dim, hidden_dim),
                    act1,
                    nn.Dropout(p=dropout),
                    nn.Linear(hidden_dim, self.out_dim),
                    act2,
                    nn.Dropout(p=dropout),
                ]
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _build_classifier_head(
    in_dim: int,
    num_classes: int = 2,
    hidden_dim: Optional[int] = None,
    dropout: float = 0.1,
    activation: str = "gelu",
) -> nn.Module:
    if in_dim <= 0:
        raise ValueError(f"in_dim must be > 0, got: {in_dim}")
    if num_classes < 2:
        raise ValueError(f"num_classes must be >= 2, got: {num_classes}")
    if dropout < 0.0:
        raise ValueError(f"dropout must be >= 0, got: {dropout}")

    if hidden_dim is None:
        if dropout > 0.0:
            return nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_dim, num_classes),
            )
        return nn.Linear(in_dim, num_classes)

    hidden_dim = int(hidden_dim)
    if hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be > 0, got: {hidden_dim}")

    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        _make_activation(activation),
        nn.Dropout(p=dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def _get_branch_name(branch_cfg: Any) -> str:
    name = _cfg_get(branch_cfg, "name", default=None)
    if name is None:
        name = _cfg_get(branch_cfg, "backbone", "name", default=None)
    if name is None:
        raise ValueError("Could not determine branch name from config.")
    return str(name).strip().lower()


def _build_spatial_branch(spatial_cfg: Any) -> nn.Module:
    spatial_name = _get_branch_name(spatial_cfg)

    if spatial_name == "resnet18":
        return build_resnet18(spatial_cfg)
    if spatial_name == "vit":
        return build_vit(spatial_cfg)

    raise ValueError(
        f"Unsupported spatial branch: {spatial_name}. "
        "Currently supported: ['resnet18', 'vit']"
    )


def _build_spectral_branch(spectral_cfg: Any) -> nn.Module:
    spectral_name = _get_branch_name(spectral_cfg)

    if spectral_name == "spai":
        return build_spai(spectral_cfg)

    raise ValueError(
        f"Unsupported spectral branch: {spectral_name}. "
        "Currently supported: ['spai']"
    )


class FusionClassifier(nn.Module):
    """
    Late fusion v1.

    흐름:
      x
        -> spatial_branch.extract_features(x)               -> f_spa
        -> spectral_branch.extract_spectral_features(x)     -> f_spec
        -> proj_spa(f_spa), proj_spec(f_spec)               -> p_spa, p_spec
        -> fusion_block(p_spa, p_spec)                      -> z
        -> classifier(z)                                    -> logits

    기본 설계:
      - spatial branch: ResNet18Classifier / ViTClassifier 재사용
      - spectral branch: SPAIClassifier 재사용
      - fusion block: gated late fusion
    """

    def __init__(
        self,
        spatial_cfg: Any,
        spectral_cfg: Any,
        projection_cfg: Any = None,
        fusion_cfg: Any = None,
        head_cfg: Any = None,
    ) -> None:
        super().__init__()

        self.spatial_name = _get_branch_name(spatial_cfg)
        self.spectral_name = _get_branch_name(spectral_cfg)

        self.spatial_branch = _build_spatial_branch(spatial_cfg)
        self.spectral_branch = _build_spectral_branch(spectral_cfg)

        if not hasattr(self.spatial_branch, "extract_features"):
            raise AttributeError(
                f"Spatial branch '{self.spatial_name}' must implement extract_features()."
            )

        self.spectral_feature_mode = str(
            _cfg_get(
                spectral_cfg,
                "aggregation",
                "spectral_feature_mode",
                default="aggregated_context",
            )
        )

        spatial_feature_dim = int(
            getattr(self.spatial_branch, "feature_dim", 0)
        )
        if spatial_feature_dim <= 0:
            raise ValueError(
                f"Could not determine spatial feature dim from '{self.spatial_name}'."
            )

        if hasattr(self.spectral_branch, "get_spectral_feature_dim"):
            spectral_feature_dim = int(
                self.spectral_branch.get_spectral_feature_dim(self.spectral_feature_mode)
            )
        else:
            spectral_feature_dim = int(
                getattr(self.spectral_branch, "feature_dim", 0)
            )

        if spectral_feature_dim <= 0:
            raise ValueError(
                f"Could not determine spectral feature dim from '{self.spectral_name}'."
            )

        projection_out_dim = _cfg_get(projection_cfg, "out_dim", default=None)
        if projection_out_dim is None:
            projection_out_dim = min(spatial_feature_dim, spectral_feature_dim)
        projection_out_dim = int(projection_out_dim)

        projection_hidden_dim = _cfg_get(projection_cfg, "hidden_dim", default=None)
        projection_dropout = float(_cfg_get(projection_cfg, "dropout", default=0.1))
        projection_activation = str(
            _cfg_get(projection_cfg, "activation", default="gelu")
        )
        projection_use_layernorm = bool(
            _cfg_get(projection_cfg, "use_layernorm", default=True)
        )

        self.spatial_feature_dim = spatial_feature_dim
        self.spectral_feature_dim = spectral_feature_dim
        self.projection_dim = projection_out_dim

        self.proj_spa = ProjectionHead(
            in_dim=self.spatial_feature_dim,
            out_dim=self.projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
            activation=projection_activation,
            use_layernorm=projection_use_layernorm,
        )
        self.proj_spec = ProjectionHead(
            in_dim=self.spectral_feature_dim,
            out_dim=self.projection_dim,
            hidden_dim=projection_hidden_dim,
            dropout=projection_dropout,
            activation=projection_activation,
            use_layernorm=projection_use_layernorm,
        )

        self.fusion_block = build_fusion_block(
            block_cfg=fusion_cfg,
            feature_dim=self.projection_dim,
        )

        self.feature_dim = int(self.fusion_block.get_output_dim())
        self.fused_feature_dim = self.feature_dim

        num_classes = int(_cfg_get(head_cfg, "num_classes", default=2))
        head_hidden_dim = _cfg_get(head_cfg, "hidden_dim", default=None)
        head_dropout = float(_cfg_get(head_cfg, "dropout", default=0.1))
        head_activation = str(_cfg_get(head_cfg, "activation", default="gelu"))

        self.classifier = _build_classifier_head(
            in_dim=self.fused_feature_dim,
            num_classes=num_classes,
            hidden_dim=head_hidden_dim,
            dropout=head_dropout,
            activation=head_activation,
        )

    def freeze_spatial_backbone(self) -> None:
        if hasattr(self.spatial_branch, "freeze_backbone"):
            self.spatial_branch.freeze_backbone()
            return
        raise AttributeError(
            f"Spatial branch '{self.spatial_name}' does not implement freeze_backbone()."
        )

    def unfreeze_spatial_backbone(self) -> None:
        if hasattr(self.spatial_branch, "unfreeze_backbone"):
            self.spatial_branch.unfreeze_backbone()
            return
        raise AttributeError(
            f"Spatial branch '{self.spatial_name}' does not implement unfreeze_backbone()."
        )

    def freeze_spectral_backbone(self) -> None:
        if hasattr(self.spectral_branch, "freeze_backbone"):
            self.spectral_branch.freeze_backbone()
            return
        raise AttributeError(
            f"Spectral branch '{self.spectral_name}' does not implement freeze_backbone()."
        )

    def unfreeze_spectral_backbone(self) -> None:
        if hasattr(self.spectral_branch, "unfreeze_backbone"):
            self.spectral_branch.unfreeze_backbone()
            return
        raise AttributeError(
            f"Spectral branch '{self.spectral_name}' does not implement unfreeze_backbone()."
        )

    def _extract_spatial_feature(self, x: torch.Tensor) -> torch.Tensor:
        feature = self.spatial_branch.extract_features(x)
        if not isinstance(feature, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor from spatial branch, got: {type(feature)}"
            )
        if feature.ndim != 2:
            raise ValueError(
                "Spatial feature must have shape [B, C], "
                f"got: {tuple(feature.shape)}"
            )
        return feature

    def _extract_spectral_feature(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if hasattr(self.spectral_branch, "extract_spectral_features"):
            return self.spectral_branch.extract_spectral_features(
                x,
                mode=self.spectral_feature_mode,
                return_dict=return_dict,
            )

        # fallback: 구형 구현과의 호환
        spectral_feature = self.spectral_branch.extract_features(x, return_dict=False)
        if not return_dict:
            return spectral_feature
        return {"spectral_feature": spectral_feature}

    def extract_features(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        fusion classifier head 직전 fused feature 추출.

        return_dict=False:
          - fused feature z [B, D]

        return_dict=True:
          - spatial/spectral/fusion intermediate dict 반환
        """
        spatial_feature = self._extract_spatial_feature(x)

        spectral_out = self._extract_spectral_feature(x, return_dict=True)
        spectral_feature = spectral_out["spectral_feature"]

        if spectral_feature.ndim != 2:
            raise ValueError(
                "Spectral feature must have shape [B, C], "
                f"got: {tuple(spectral_feature.shape)}"
            )

        projected_spatial = self.proj_spa(spatial_feature)
        projected_spectral = self.proj_spec(spectral_feature)

        fusion_out = self.fusion_block(
            projected_spatial,
            projected_spectral,
            return_dict=True,
        )
        fused_feature = fusion_out["fused"]

        if not return_dict:
            return fused_feature

        output: dict[str, torch.Tensor] = {
            "spatial_feature": spatial_feature,
            "spectral_feature": spectral_feature,
            "projected_spatial": projected_spatial,
            "projected_spectral": projected_spectral,
            "fused_feature": fused_feature,
            "gate": fusion_out["gate"],
            "gate_logits": fusion_out["gate_logits"],
            "interaction": fusion_out["interaction"],
            "abs_diff": fusion_out["abs_diff"],
            "elementwise_prod": fusion_out["elementwise_prod"],
        }

        for key in [
            "x_low",
            "x_high",
            "orig_global",
            "low_global",
            "high_global",
            "orig_context",
            "low_context",
            "high_context",
            "global_stack",
            "context_stack",
            "aggregated_context",
            "branch_weights",
            "similarity_stats",
        ]:
            if key in spectral_out and isinstance(spectral_out[key], torch.Tensor):
                output[key] = spectral_out[key]

        return output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused_feature = self.extract_features(x, return_dict=False)
        logits = self.classifier(fused_feature)
        return logits


def build_fusion(model_cfg: Any) -> FusionClassifier:
    """
    config 기반 fusion model 생성 함수.

    기대하는 config 예시:
      cfg.model.name = "fusion"

      cfg.model.spatial.*
      cfg.model.spectral.*
      cfg.model.projection.*
      cfg.model.fusion.*
      cfg.model.head.*
    """
    spatial_cfg = _cfg_get(model_cfg, "spatial", default=None)
    spectral_cfg = _cfg_get(model_cfg, "spectral", default=None)
    projection_cfg = _cfg_get(model_cfg, "projection", default={})
    fusion_cfg = _cfg_get(model_cfg, "fusion", default={})
    head_cfg = _cfg_get(model_cfg, "head", default={})

    if spatial_cfg is None:
        raise ValueError("cfg.model.spatial must be provided for fusion model.")
    if spectral_cfg is None:
        raise ValueError("cfg.model.spectral must be provided for fusion model.")

    return FusionClassifier(
        spatial_cfg=spatial_cfg,
        spectral_cfg=spectral_cfg,
        projection_cfg=projection_cfg,
        fusion_cfg=fusion_cfg,
        head_cfg=head_cfg,
    )