from __future__ import annotations

from typing import Any, Mapping, Optional

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


class GatedLateFusionBlock(nn.Module):
    """
    Late fusion v1.

    입력:
      - p_spa:  [B, D]
      - p_spec: [B, D]

    내부:
      u = [p_spa ; p_spec ; |p_spa - p_spec| ; p_spa * p_spec]  -> [B, 4D]
      gate = sigmoid(MLP(u))                                     -> [B, D] or [B, 1]
      z = gate * p_spa + (1 - gate) * p_spec                    -> [B, D]

    출력:
      - return_dict=False: fused feature z
      - return_dict=True : intermediate dict
    """

    def __init__(
        self,
        feature_dim: int,
        gate_hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        gate_mode: str = "channel",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        if feature_dim <= 0:
            raise ValueError(f"feature_dim must be > 0, got: {feature_dim}")
        if dropout < 0.0:
            raise ValueError(f"dropout must be >= 0, got: {dropout}")

        gate_mode = str(gate_mode).strip().lower()
        if gate_mode not in {"channel", "scalar"}:
            raise ValueError(
                f"Unsupported gate_mode: {gate_mode}. "
                "Choose from ['channel', 'scalar']."
            )

        self.feature_dim = int(feature_dim)
        self.interaction_dim = 4 * self.feature_dim
        self.gate_mode = gate_mode
        self.output_dim = self.feature_dim

        hidden_dim = (
            int(gate_hidden_dim)
            if gate_hidden_dim is not None
            else max(self.feature_dim, self.interaction_dim // 2)
        )
        gate_out_dim = self.feature_dim if self.gate_mode == "channel" else 1

        layers: list[nn.Module] = []
        if use_layernorm:
            layers.append(nn.LayerNorm(self.interaction_dim))

        layers.extend(
            [
                nn.Linear(self.interaction_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(p=dropout),
                nn.Linear(hidden_dim, gate_out_dim),
            ]
        )

        self.gate_mlp = nn.Sequential(*layers)

    def get_output_dim(self) -> int:
        return self.output_dim

    def _validate_inputs(
        self,
        p_spa: torch.Tensor,
        p_spec: torch.Tensor,
    ) -> None:
        if not isinstance(p_spa, torch.Tensor):
            raise TypeError(f"p_spa must be torch.Tensor, got: {type(p_spa)}")
        if not isinstance(p_spec, torch.Tensor):
            raise TypeError(f"p_spec must be torch.Tensor, got: {type(p_spec)}")

        if p_spa.ndim != 2:
            raise ValueError(f"p_spa must have shape [B, D], got: {tuple(p_spa.shape)}")
        if p_spec.ndim != 2:
            raise ValueError(
                f"p_spec must have shape [B, D], got: {tuple(p_spec.shape)}"
            )
        if p_spa.shape != p_spec.shape:
            raise ValueError(
                "p_spa and p_spec must have the same shape, "
                f"got: {tuple(p_spa.shape)} vs {tuple(p_spec.shape)}"
            )
        if p_spa.size(-1) != self.feature_dim:
            raise ValueError(
                f"Expected feature dim {self.feature_dim}, got p_spa dim {p_spa.size(-1)}"
            )

    def _build_interaction(
        self,
        p_spa: torch.Tensor,
        p_spec: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        abs_diff = (p_spa - p_spec).abs()
        elementwise_prod = p_spa * p_spec
        interaction = torch.cat(
            [p_spa, p_spec, abs_diff, elementwise_prod],
            dim=-1,
        )

        return {
            "interaction": interaction,
            "abs_diff": abs_diff,
            "elementwise_prod": elementwise_prod,
        }

    def forward(
        self,
        p_spa: torch.Tensor,
        p_spec: torch.Tensor,
        return_dict: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._validate_inputs(p_spa, p_spec)

        interaction_dict = self._build_interaction(p_spa, p_spec)
        interaction = interaction_dict["interaction"]

        gate_logits = self.gate_mlp(interaction)
        gate = torch.sigmoid(gate_logits)

        if gate.size(-1) == 1:
            gate = gate.expand_as(p_spa)

        fused = gate * p_spa + (1.0 - gate) * p_spec

        if not return_dict:
            return fused

        return {
            "fused": fused,
            "gate": gate,
            "gate_logits": gate_logits,
            "interaction": interaction,
            "abs_diff": interaction_dict["abs_diff"],
            "elementwise_prod": interaction_dict["elementwise_prod"],
            "p_spa": p_spa,
            "p_spec": p_spec,
        }


def build_fusion_block(
    block_cfg: Any,
    feature_dim: int,
) -> GatedLateFusionBlock:
    """
    config 기반 fusion block 생성 helper.

    기대 예시:
      cfg.model.fusion.feature_dim          # 직접 쓰는 경우는 드묾
      cfg.model.fusion.gate_hidden_dim
      cfg.model.fusion.dropout
      cfg.model.fusion.gate_mode
      cfg.model.fusion.use_layernorm
    """
    gate_hidden_dim = _cfg_get(block_cfg, "gate_hidden_dim", default=None)
    dropout = float(_cfg_get(block_cfg, "dropout", default=0.1))
    gate_mode = str(_cfg_get(block_cfg, "gate_mode", default="channel"))
    use_layernorm = bool(_cfg_get(block_cfg, "use_layernorm", default=True))

    return GatedLateFusionBlock(
        feature_dim=feature_dim,
        gate_hidden_dim=gate_hidden_dim,
        dropout=dropout,
        gate_mode=gate_mode,
        use_layernorm=use_layernorm,
    )