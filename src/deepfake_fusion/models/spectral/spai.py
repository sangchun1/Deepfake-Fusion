from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .frequency_encoder import (
    build_frequency_encoder,
)


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
    vit.py와 동일한 fallback 패턴 유지.
    """
    create_kwargs: dict[str, Any] = {
        "pretrained": pretrained,
        "in_chans": in_channels,
        "num_classes": 0,
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
        fallback_kwargs = {
            "pretrained": pretrained,
            "in_chans": in_channels,
            "num_classes": 0,
        }
        return timm.create_model(model_name, **fallback_kwargs)


class SPAIClassifier(nn.Module):
    """
    SPAI-inspired frequency-only detector (v1).

    흐름:
      x
        -> FrequencyEncoder -> x_low, x_high
        -> shared ViT backbone G for {x, x_low, x_high}
        -> branch global/context feature 추출
        -> similarity statistics (SRS-inspired)
        -> branch context gating (SCV/SCA-inspired lightweight aggregation)
        -> 3-layer MLP
        -> binary logits

    추가:
      - extract_features(): 기존 frequency-only head 직전 fused feature 반환
      - extract_spectral_features(): fusion용 spectral embedding 반환
      - forward_spectral_features(): fusion model에서 호출하기 쉬운 alias
    """

    _SPECTRAL_FEATURE_MODES = {
        "aggregated_context",
        "context_avg",
        "global_avg",
        "orig_context",
        "low_context",
        "high_context",
        "orig_global",
        "low_global",
        "high_global",
        "fused",
    }

    def __init__(
        self,
        model_name: str = "vit_small_patch16_224",
        num_classes: int = 2,
        pretrained: bool = True,
        in_channels: int = 3,
        freeze_backbone: bool = False,
        img_size: Optional[int] = 224,
        global_pool: str = "token",
        drop_path_rate: float = 0.0,
        mlp_dropout: float = 0.1,
        mlp_hidden_dim: Optional[int] = None,
        mlp_hidden_dim2: Optional[int] = None,
        selected_blocks: Optional[Sequence[int]] = None,
        num_selected_blocks: int = 0,
        token_pool: str = "attention",
        feature_pool: str = "cls",
        explain_branch: str = "original",
        frequency_cfg: Any = None,
        spectral_feature_mode: str = "aggregated_context",
    ) -> None:
        super().__init__()

        if not model_name:
            raise ValueError("model_name must be a non-empty string.")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got: {num_classes}")
        if in_channels < 1:
            raise ValueError(f"in_channels must be >= 1, got: {in_channels}")
        if mlp_dropout < 0.0:
            raise ValueError(f"mlp_dropout must be >= 0, got: {mlp_dropout}")
        if drop_path_rate < 0.0:
            raise ValueError(f"drop_path_rate must be >= 0, got: {drop_path_rate}")

        token_pool = str(token_pool).strip().lower()
        if token_pool not in {"attention", "avg", "cls"}:
            raise ValueError(
                f"Unsupported token_pool: {token_pool}. "
                "Choose from ['attention', 'avg', 'cls']."
            )

        feature_pool = str(feature_pool).strip().lower()
        if feature_pool not in {"cls", "avg"}:
            raise ValueError(
                f"Unsupported feature_pool: {feature_pool}. "
                "Choose from ['cls', 'avg']."
            )

        explain_branch = str(explain_branch).strip().lower()
        if explain_branch not in {"original", "low", "high"}:
            raise ValueError(
                f"Unsupported explain_branch: {explain_branch}. "
                "Choose from ['original', 'low', 'high']."
            )

        spectral_feature_mode = self._normalize_spectral_feature_mode(
            spectral_feature_mode
        )

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
        self.frequency_encoder = build_frequency_encoder(frequency_cfg)
        self.token_pool_mode = token_pool
        self.feature_pool_mode = feature_pool
        self.explain_branch = explain_branch
        self.spectral_feature_mode = spectral_feature_mode

        # attention_rollout.py 수정 시 branch별 hook filtering에 활용 가능하도록 유지
        self._active_branch_name: str | None = None

        self.num_prefix_tokens = int(getattr(self.backbone, "num_prefix_tokens", 1))
        self.selected_blocks = (
            [int(idx) for idx in selected_blocks] if selected_blocks is not None else []
        )
        self.num_selected_blocks = int(num_selected_blocks)

        # patch-level attention pooling (SCA-inspired lightweight v1)
        self.token_attention = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
        )

        # branch-level context gating (SCV/SCA-inspired lightweight v1)
        self.branch_gate = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, 1),
        )

        # similarity statistics: cosine + l1_mean for 3 pairs = 6 dims
        self.similarity_dim = 6

        # final feature:
        #   [orig_global, low_global, high_global] -> 3C
        #   aggregated_context -> C
        #   similarity_stats -> 6
        self.fused_dim = (4 * self.feature_dim) + self.similarity_dim
        self.spectral_feature_dim = self._infer_spectral_feature_dim(
            self.spectral_feature_mode
        )

        hidden_dim1 = (
            int(mlp_hidden_dim)
            if mlp_hidden_dim is not None
            else int(2 * self.feature_dim)
        )
        hidden_dim2 = (
            int(mlp_hidden_dim2)
            if mlp_hidden_dim2 is not None
            else int(self.feature_dim)
        )

        self.head = nn.Sequential(
            nn.Linear(self.fused_dim, hidden_dim1),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.GELU(),
            nn.Dropout(p=mlp_dropout),
            nn.Linear(hidden_dim2, num_classes),
        )

        if freeze_backbone:
            self.freeze_backbone()

    @classmethod
    def _normalize_spectral_feature_mode(cls, mode: Optional[str]) -> str:
        mode = str(mode or "aggregated_context").strip().lower()
        if mode not in cls._SPECTRAL_FEATURE_MODES:
            raise ValueError(
                f"Unsupported spectral_feature_mode: {mode}. "
                f"Choose from {sorted(cls._SPECTRAL_FEATURE_MODES)}."
            )
        return mode

    def _infer_spectral_feature_dim(self, mode: Optional[str] = None) -> int:
        mode = self._normalize_spectral_feature_mode(mode or self.spectral_feature_mode)
        if mode == "fused":
            return self.fused_dim
        return self.feature_dim

    def get_spectral_feature_dim(self, mode: Optional[str] = None) -> int:
        """fusion model에서 projection layer 차원 결정 시 사용."""
        return self._infer_spectral_feature_dim(mode)

    def freeze_backbone(self) -> None:
        """ViT backbone만 freeze."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """ViT backbone 전체 unfreeze."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _resolve_selected_block_indices(self) -> list[int]:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None:
            return []

        num_blocks = len(blocks)
        indices: list[int] = []

        if self.selected_blocks:
            indices = [int(idx) for idx in self.selected_blocks]
        elif self.num_selected_blocks > 0:
            start = max(0, num_blocks - int(self.num_selected_blocks))
            indices = list(range(start, num_blocks))

        resolved: list[int] = []
        for idx in indices:
            if idx < 0:
                idx = num_blocks + idx
            if 0 <= idx < num_blocks:
                resolved.append(int(idx))

        # 중복 제거 + 순서 정렬
        resolved = sorted(set(resolved))
        return resolved

    def _pool_feature(self, x: torch.Tensor, mode: Optional[str] = None) -> torch.Tensor:
        """
        feature pooling helper.

        Supported:
          - [B, N, C] token sequence
          - [B, C] already pooled
          - [B, C, H, W] CNN-like fallback
        """
        mode = (mode or self.feature_pool_mode).strip().lower()

        if x.ndim == 2:
            return x

        if x.ndim == 4:
            return x.mean(dim=(2, 3))

        if x.ndim != 3:
            raise ValueError(f"Unsupported feature shape for pooling: {tuple(x.shape)}")

        if mode == "cls":
            return x[:, 0, :]

        # avg mode: patch tokens만 평균
        patch_tokens = x[:, self.num_prefix_tokens :, :]
        if patch_tokens.numel() == 0:
            return x.mean(dim=1)
        return patch_tokens.mean(dim=1)

    def _token_attention_pool(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        patch-level token aggregation.
        token_pool_mode:
          - attention: learned attention pooling
          - avg: patch average
          - cls: cls token
        """
        if tokens.ndim == 2:
            return tokens

        if tokens.ndim == 4:
            # [B, C, H, W] -> [B, HW, C]
            tokens = tokens.flatten(2).transpose(1, 2)

        if tokens.ndim != 3:
            raise ValueError(
                f"Unsupported token tensor shape for token pooling: {tuple(tokens.shape)}"
            )

        if self.token_pool_mode == "cls":
            return tokens[:, 0, :]

        patch_tokens = tokens[:, self.num_prefix_tokens :, :]
        if patch_tokens.numel() == 0:
            patch_tokens = tokens

        if self.token_pool_mode == "avg":
            return patch_tokens.mean(dim=1)

        scores = self.token_attention(patch_tokens)  # [B, N, 1]
        weights = torch.softmax(scores, dim=1)
        pooled = (weights * patch_tokens).sum(dim=1)
        return pooled

    def _extract_global_from_backbone_output(self, features: torch.Tensor) -> torch.Tensor:
        """
        vit.py의 extract_features 흐름과 최대한 유사하게 global feature 추출.
        """
        if hasattr(self.backbone, "forward_head"):
            try:
                pooled = self.backbone.forward_head(features, pre_logits=True)
            except TypeError:
                pooled = self.backbone.forward_head(features)

            if isinstance(pooled, (tuple, list)):
                pooled = pooled[0]

            if isinstance(pooled, torch.Tensor):
                if pooled.ndim == 2:
                    return pooled
                if pooled.ndim == 3:
                    return self._pool_feature(pooled)
                if pooled.ndim == 4:
                    return pooled.mean(dim=(2, 3))

        return self._pool_feature(features)

    def _forward_backbone_with_hooks(
        self,
        x: torch.Tensor,
        branch_name: str,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        backbone forward + selected block outputs 수집.
        """
        block_indices = self._resolve_selected_block_indices()
        captured: list[torch.Tensor | None] = [None for _ in block_indices]
        handles = []

        def _make_hook(slot_idx: int):
            def _hook(module, inputs, output):
                captured[slot_idx] = output

            return _hook

        blocks = getattr(self.backbone, "blocks", None)
        if blocks is not None:
            for slot_idx, block_idx in enumerate(block_indices):
                handles.append(blocks[block_idx].register_forward_hook(_make_hook(slot_idx)))

        self._active_branch_name = branch_name
        try:
            features = self.backbone.forward_features(x)
        finally:
            self._active_branch_name = None
            for handle in handles:
                handle.remove()

        selected_outputs = [out for out in captured if out is not None]
        return features, selected_outputs

    def _extract_branch_outputs(
        self,
        x: torch.Tensor,
        branch_name: str,
    ) -> dict[str, torch.Tensor]:
        """
        single branch (original / low / high)에서 feature 추출.
        """
        features, selected_outputs = self._forward_backbone_with_hooks(
            x=x,
            branch_name=branch_name,
        )

        if isinstance(features, (tuple, list)):
            features = features[0]

        if not isinstance(features, torch.Tensor):
            raise TypeError(
                f"Expected torch.Tensor from backbone.forward_features, got {type(features)}"
            )

        global_feature = self._extract_global_from_backbone_output(features)
        token_context = self._token_attention_pool(features)

        if len(selected_outputs) > 0:
            block_pooled = [self._pool_feature(out) for out in selected_outputs]
            block_context = torch.stack(block_pooled, dim=1).mean(dim=1)
        else:
            block_context = global_feature

        branch_context = 0.5 * (block_context + token_context)

        return {
            "global": global_feature,
            "context": branch_context,
        }

    def _compute_similarity_stats(self, branch_features: torch.Tensor) -> torch.Tensor:
        """
        branch_features: [B, 3, C]
        returns: [B, 6]
          [cos(o,l), cos(o,h), cos(l,h), l1(o,l), l1(o,h), l1(l,h)]
        """
        if branch_features.ndim != 3 or branch_features.size(1) != 3:
            raise ValueError(
                "branch_features must have shape [B, 3, C], "
                f"got {tuple(branch_features.shape)}"
            )

        orig = branch_features[:, 0, :]
        low = branch_features[:, 1, :]
        high = branch_features[:, 2, :]

        cos_ol = F.cosine_similarity(orig, low, dim=-1, eps=1e-8).unsqueeze(1)
        cos_oh = F.cosine_similarity(orig, high, dim=-1, eps=1e-8).unsqueeze(1)
        cos_lh = F.cosine_similarity(low, high, dim=-1, eps=1e-8).unsqueeze(1)

        l1_ol = (orig - low).abs().mean(dim=-1, keepdim=True)
        l1_oh = (orig - high).abs().mean(dim=-1, keepdim=True)
        l1_lh = (low - high).abs().mean(dim=-1, keepdim=True)

        return torch.cat([cos_ol, cos_oh, cos_lh, l1_ol, l1_oh, l1_lh], dim=1)

    def _extract_multibranch_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        original / low / high branch 전체를 한 번에 처리하고
        frequency-only / fusion 양쪽에서 재사용할 intermediate dict를 반환.
        """
        x_low, x_high = self.frequency_encoder(x)

        orig_out = self._extract_branch_outputs(x, branch_name="original")
        low_out = self._extract_branch_outputs(x_low, branch_name="low")
        high_out = self._extract_branch_outputs(x_high, branch_name="high")

        global_stack = torch.stack(
            [orig_out["global"], low_out["global"], high_out["global"]],
            dim=1,
        )  # [B, 3, C]

        context_stack = torch.stack(
            [orig_out["context"], low_out["context"], high_out["context"]],
            dim=1,
        )  # [B, 3, C]

        similarity_input = 0.5 * (global_stack + context_stack)
        similarity_stats = self._compute_similarity_stats(similarity_input)  # [B, 6]

        branch_scores = self.branch_gate(context_stack)  # [B, 3, 1]
        branch_weights = torch.softmax(branch_scores, dim=1)
        aggregated_context = (branch_weights * context_stack).sum(dim=1)  # [B, C]

        fused = torch.cat(
            [
                global_stack.reshape(global_stack.size(0), -1),  # [B, 3C]
                aggregated_context,                              # [B, C]
                similarity_stats,                                # [B, 6]
            ],
            dim=1,
        )  # [B, 4C + 6]

        return {
            "fused": fused,
            "x_low": x_low,
            "x_high": x_high,
            "orig_global": orig_out["global"],
            "low_global": low_out["global"],
            "high_global": high_out["global"],
            "orig_context": orig_out["context"],
            "low_context": low_out["context"],
            "high_context": high_out["context"],
            "global_stack": global_stack,
            "context_stack": context_stack,
            "aggregated_context": aggregated_context,
            "branch_weights": branch_weights.squeeze(-1),
            "similarity_stats": similarity_stats,
        }

    def _select_spectral_feature(
        self,
        features_dict: Mapping[str, torch.Tensor],
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        fusion branch에서 사용할 spectral embedding 선택.

        기본값:
          aggregated_context [B, C]

        지원 모드:
          - aggregated_context
          - context_avg
          - global_avg
          - orig_context / low_context / high_context
          - orig_global / low_global / high_global
          - fused
        """
        mode = self._normalize_spectral_feature_mode(mode or self.spectral_feature_mode)

        if mode == "aggregated_context":
            return features_dict["aggregated_context"]
        if mode == "context_avg":
            return features_dict["context_stack"].mean(dim=1)
        if mode == "global_avg":
            return features_dict["global_stack"].mean(dim=1)
        if mode == "orig_context":
            return features_dict["orig_context"]
        if mode == "low_context":
            return features_dict["low_context"]
        if mode == "high_context":
            return features_dict["high_context"]
        if mode == "orig_global":
            return features_dict["orig_global"]
        if mode == "low_global":
            return features_dict["low_global"]
        if mode == "high_global":
            return features_dict["high_global"]
        if mode == "fused":
            return features_dict["fused"]

        raise RuntimeError(f"Unhandled spectral feature mode: {mode}")

    def extract_features(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        classifier head 직전 fused feature 추출.

        주의:
          - 기존 SPAI frequency-only 분류 경로용 API
          - 반환 차원: [B, 4C + 6]

        return_dict=True면 branch별 중간 결과도 함께 반환.
        """
        features_dict = self._extract_multibranch_features(x)

        if not return_dict:
            return features_dict["fused"]

        return features_dict

    def extract_spectral_features(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """
        fusion용 spectral embedding 추출.

        기본:
          mode=None -> self.spectral_feature_mode 사용
          기본 모드는 aggregated_context

        return:
          - return_dict=False: selected spectral feature tensor
          - return_dict=True : intermediate dict + spectral_feature
        """
        features_dict = self._extract_multibranch_features(x)
        spectral_feature = self._select_spectral_feature(features_dict, mode=mode)

        if not return_dict:
            return spectral_feature

        output = dict(features_dict)
        output["spectral_feature"] = spectral_feature
        return output

    def forward_spectral_features(
        self,
        x: torch.Tensor,
        mode: Optional[str] = None,
    ) -> torch.Tensor:
        """
        fusion model에서 쓰기 쉬운 alias.
        """
        return self.extract_spectral_features(x, mode=mode, return_dict=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fused = self.extract_features(x, return_dict=False)
        logits = self.head(fused)
        return logits


def build_spai(model_cfg: Any) -> SPAIClassifier:
    """
    config 기반 SPAI 생성 함수.

    기대하는 config 예시:
      cfg.model.name
      cfg.model.backbone.name
      cfg.model.backbone.pretrained
      cfg.model.backbone.in_channels
      cfg.model.backbone.freeze
      cfg.model.backbone.img_size
      cfg.model.backbone.global_pool
      cfg.model.backbone.drop_path_rate

      cfg.model.frequency.*

      cfg.model.aggregation.selected_blocks
      cfg.model.aggregation.num_selected_blocks
      cfg.model.aggregation.token_pool
      cfg.model.aggregation.feature_pool
      cfg.model.aggregation.spectral_feature_mode   # optional
      cfg.model.explain.rollout_branch

      cfg.model.head.num_classes
      cfg.model.head.dropout
      cfg.model.head.hidden_dim
      cfg.model.head.hidden_dim2
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
    mlp_dropout = float(_cfg_get(model_cfg, "head", "dropout", default=0.1))
    mlp_hidden_dim = _cfg_get(model_cfg, "head", "hidden_dim", default=None)
    mlp_hidden_dim2 = _cfg_get(model_cfg, "head", "hidden_dim2", default=None)

    selected_blocks = _cfg_get(model_cfg, "aggregation", "selected_blocks", default=None)
    num_selected_blocks = int(
        _cfg_get(model_cfg, "aggregation", "num_selected_blocks", default=0)
    )
    token_pool = str(
        _cfg_get(model_cfg, "aggregation", "token_pool", default="attention")
    )
    feature_pool = str(
        _cfg_get(model_cfg, "aggregation", "feature_pool", default="cls")
    )
    spectral_feature_mode = str(
        _cfg_get(
            model_cfg,
            "aggregation",
            "spectral_feature_mode",
            default="aggregated_context",
        )
    )

    explain_branch = str(
        _cfg_get(model_cfg, "explain", "rollout_branch", default="original")
    )

    frequency_cfg = _cfg_get(model_cfg, "frequency", default={})

    return SPAIClassifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=pretrained,
        in_channels=in_channels,
        freeze_backbone=freeze_backbone,
        img_size=img_size,
        global_pool=global_pool,
        drop_path_rate=drop_path_rate,
        mlp_dropout=mlp_dropout,
        mlp_hidden_dim=mlp_hidden_dim,
        mlp_hidden_dim2=mlp_hidden_dim2,
        selected_blocks=selected_blocks,
        num_selected_blocks=num_selected_blocks,
        token_pool=token_pool,
        feature_pool=feature_pool,
        explain_branch=explain_branch,
        frequency_cfg=frequency_cfg,
        spectral_feature_mode=spectral_feature_mode,
    )