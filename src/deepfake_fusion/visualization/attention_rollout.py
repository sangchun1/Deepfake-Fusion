from __future__ import annotations

from typing import Any, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn


def _get_vit_backbone(model: nn.Module) -> nn.Module:
    """
    ViT backbone 추론.
    - custom wrapper면 model.backbone 사용
    - 순수 timm ViT면 model 자체 사용
    """
    backbone = model.backbone if hasattr(model, "backbone") else model
    if not hasattr(backbone, "blocks"):
        raise ValueError(
            "AttentionRollout expects a ViT-like model with '.blocks'. "
            "If you use a wrapper model, make sure the ViT encoder is available as '.backbone'."
        )
    return backbone


def _get_num_prefix_tokens(backbone: nn.Module) -> int:
    return int(getattr(backbone, "num_prefix_tokens", 1))


def _normalize_map(cam: np.ndarray) -> np.ndarray:
    cam = np.asarray(cam, dtype=np.float32)
    cam = np.maximum(cam, 0.0)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)


def _infer_patch_grid(
    backbone: nn.Module,
    num_patch_tokens: int,
) -> tuple[int, int]:
    patch_embed = getattr(backbone, "patch_embed", None)
    grid_size = getattr(patch_embed, "grid_size", None)

    if isinstance(grid_size, Sequence) and len(grid_size) == 2:
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh * gw == num_patch_tokens:
            return gh, gw

    side = int(round(num_patch_tokens**0.5))
    if side * side != num_patch_tokens:
        raise ValueError(
            f"Could not infer patch grid from num_patch_tokens={num_patch_tokens}. "
            "Expected a square number or a valid backbone.patch_embed.grid_size."
        )
    return side, side


class AttentionRollout:
    """
    ViT attention rollout visualizer.

    Notes
    -----
    - timm VisionTransformer는 fused attention이 켜져 있으면 attention map hook이
      잡히지 않을 수 있어서, rollout 생성 시 해당 옵션을 임시로 꺼둔다.
    - rollout은 기본적으로 class-agnostic이다.
      target_class 인자는 결과 포맷 통일용으로만 유지한다.
    """

    def __init__(
        self,
        model: nn.Module,
        head_fusion: str = "mean",
        discard_ratio: float = 0.0,
        start_layer: int = 0,
    ) -> None:
        if head_fusion not in {"mean", "max", "min"}:
            raise ValueError(
                f"Unsupported head_fusion: {head_fusion}. "
                "Choose from ['mean', 'max', 'min']."
            )
        if not (0.0 <= discard_ratio < 1.0):
            raise ValueError(
                f"discard_ratio must be in [0, 1), got: {discard_ratio}"
            )
        if start_layer < 0:
            raise ValueError(f"start_layer must be >= 0, got: {start_layer}")

        self.model = model
        self.backbone = _get_vit_backbone(model)
        self.head_fusion = head_fusion
        self.discard_ratio = float(discard_ratio)
        self.start_layer = int(start_layer)

        self.attentions: List[torch.Tensor] = []
        self.handles: List[Any] = []
        self._original_fused_attn: List[tuple[nn.Module, bool]] = []

        self._disable_fused_attention()
        self._register_hooks()

    def _disable_fused_attention(self) -> None:
        for block in getattr(self.backbone, "blocks", []):
            attn = getattr(block, "attn", None)
            if attn is not None and hasattr(attn, "fused_attn"):
                original = bool(attn.fused_attn)
                self._original_fused_attn.append((attn, original))
                attn.fused_attn = False

    def _restore_fused_attention(self) -> None:
        for attn, original in self._original_fused_attn:
            attn.fused_attn = original
        self._original_fused_attn = []

    def _register_hooks(self) -> None:
        blocks = getattr(self.backbone, "blocks", None)
        if blocks is None or len(blocks) == 0:
            raise ValueError("Could not find transformer blocks for attention rollout.")

        for idx, block in enumerate(blocks):
            attn = getattr(block, "attn", None)
            attn_drop = getattr(attn, "attn_drop", None) if attn is not None else None

            if attn_drop is None:
                raise ValueError(
                    f"Could not find attention dropout module at block index {idx}. "
                    "This implementation expects timm-style ViT blocks with '.attn.attn_drop'."
                )

            def _make_hook():
                def _hook(module, inputs, output):
                    if isinstance(output, torch.Tensor):
                        self.attentions.append(output.detach())

                return _hook

            self.handles.append(attn_drop.register_forward_hook(_make_hook()))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._restore_fused_attention()

    def _fuse_heads(self, attn: torch.Tensor) -> torch.Tensor:
        """
        attn: [heads, tokens, tokens] or [1, heads, tokens, tokens]
        return: [tokens, tokens]
        """
        if attn.ndim == 4:
            if attn.size(0) != 1:
                raise ValueError(
                    "AttentionRollout currently supports batch size 1 only."
                )
            attn = attn[0]

        if attn.ndim != 3:
            raise ValueError(
                f"Expected attention shape [H, N, N], got {tuple(attn.shape)}"
            )

        if self.head_fusion == "mean":
            return attn.mean(dim=0)
        if self.head_fusion == "max":
            return attn.max(dim=0).values
        if self.head_fusion == "min":
            return attn.min(dim=0).values

        raise RuntimeError("Unreachable head_fusion branch.")

    def _discard_low_attention(self, attn: torch.Tensor) -> torch.Tensor:
        """
        작은 attention 값을 버려서 더 선명한 rollout 생성.
        diagonal(identity residual)은 이후 단계에서 다시 더한다.
        """
        if self.discard_ratio <= 0.0:
            return attn

        flat = attn.reshape(-1)
        num_discard = int(flat.numel() * self.discard_ratio)
        if num_discard <= 0:
            return attn

        _, indices = torch.topk(flat, k=num_discard, largest=False)
        attn = attn.clone()
        flat = attn.reshape(-1)
        flat[indices] = 0.0
        return attn

    def _compute_rollout(self) -> torch.Tensor:
        if len(self.attentions) == 0:
            raise RuntimeError(
                "No attention maps were captured. "
                "Check whether the model is a timm VisionTransformer and whether fused attention was disabled."
            )

        if self.start_layer >= len(self.attentions):
            raise ValueError(
                f"start_layer={self.start_layer} is out of range for "
                f"{len(self.attentions)} captured attention maps."
            )

        first_attn = self.attentions[self.start_layer]
        if first_attn.ndim != 4:
            raise ValueError(
                f"Unexpected attention tensor shape: {tuple(first_attn.shape)}"
            )

        num_tokens = int(first_attn.shape[-1])
        device = first_attn.device
        dtype = first_attn.dtype

        result = torch.eye(num_tokens, device=device, dtype=dtype)

        for attn in self.attentions[self.start_layer:]:
            fused = self._fuse_heads(attn)
            fused = self._discard_low_attention(fused)

            identity = torch.eye(
                fused.size(-1), device=fused.device, dtype=fused.dtype
            )
            fused = fused + identity
            fused = fused / fused.sum(dim=-1, keepdim=True).clamp_min(1e-12)

            result = fused @ result

        return result

    def _get_pred_class(self, logits: torch.Tensor) -> int:
        if logits.ndim == 1:
            return int((torch.sigmoid(logits)[0] >= 0.5).item())

        if logits.ndim == 2 and logits.size(0) != 1:
            raise ValueError(
                "AttentionRollout.generate currently supports batch size 1 only."
            )

        if logits.ndim == 2 and logits.size(1) == 1:
            return int((torch.sigmoid(logits[0, 0]) >= 0.5).item())

        if logits.ndim == 2:
            return int(torch.argmax(logits, dim=1).item())

        raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> dict:
        """
        input_tensor: [1, C, H, W]

        return:
            {
                "cam": np.ndarray [H, W] in [0, 1],
                "logits": torch.Tensor,
                "pred_class": int,
                "target_class": int,
            }

        주의:
        - attention rollout은 class-agnostic이므로 target_class는 map 자체를 바꾸지 않는다.
        - target_class는 explain 파이프라인과 포맷을 맞추기 위해서만 반환한다.
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                "AttentionRollout currently expects input shape [1, C, H, W], "
                f"got {tuple(input_tensor.shape)}"
            )

        self.attentions.clear()
        self.model.zero_grad(set_to_none=True)

        with torch.no_grad():
            logits = self.model(input_tensor)

        pred_class = self._get_pred_class(logits)
        if target_class is None:
            target_class = pred_class

        rollout = self._compute_rollout()

        num_prefix_tokens = _get_num_prefix_tokens(self.backbone)
        token_scores = rollout[0, num_prefix_tokens:]  # CLS -> patch tokens

        gh, gw = _infer_patch_grid(self.backbone, token_scores.numel())
        cam = token_scores.reshape(gh, gw).detach().cpu().numpy().astype(np.float32)

        h, w = int(input_tensor.shape[-2]), int(input_tensor.shape[-1])
        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        cam = _normalize_map(cam)

        return {
            "cam": cam,
            "logits": logits.detach(),
            "pred_class": int(pred_class),
            "target_class": int(target_class),
        }