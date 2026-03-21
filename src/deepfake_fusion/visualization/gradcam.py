from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn as nn


def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    """
    dotted path로 모듈 찾기.
    예:
        backbone.layer4.1
        layer4.1
    """
    current: Any = model
    for part in module_name.split("."):
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    if not isinstance(current, nn.Module):
        raise ValueError(f"Resolved object is not nn.Module: {module_name}")
    return current


def resolve_target_layer(model: nn.Module, target_layer: Optional[str] = None) -> nn.Module:
    """
    Grad-CAM target layer 자동 선택.
    기본:
    1) 사용자가 target_layer를 주면 그 레이어 사용
    2) ResNet 스타일이면 backbone.layer4[-1] 또는 layer4[-1]
    3) 마지막 Conv2d fallback
    """
    if target_layer is not None:
        return get_module_by_name(model, target_layer)

    if hasattr(model, "backbone") and hasattr(model.backbone, "layer4"):
        return model.backbone.layer4[-1]

    if hasattr(model, "layer4"):
        return model.layer4[-1]

    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Could not find a suitable Conv2d layer for Grad-CAM.")
    return conv_layers[-1]


def normalize_cam(cam: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam, 0)
    cam_min = float(cam.min())
    cam_max = float(cam.max())
    if cam_max - cam_min < 1e-12:
        return np.zeros_like(cam, dtype=np.float32)
    cam = (cam - cam_min) / (cam_max - cam_min)
    return cam.astype(np.float32)


def apply_colormap_to_cam(cam: np.ndarray) -> np.ndarray:
    """
    cam: [H, W] in [0, 1]
    return: RGB uint8 [H, W, 3]
    """
    cam_uint8 = np.uint8(np.clip(cam, 0, 1) * 255.0)
    heatmap_bgr = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)
    return heatmap_rgb


def overlay_cam_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    alpha: float = 0.35,
) -> np.ndarray:
    """
    image_rgb: uint8 [H, W, 3]
    cam: [H, W] in [0, 1]
    """
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must be uint8.")
    if cam.ndim != 2:
        raise ValueError("cam must be 2D.")

    h, w = image_rgb.shape[:2]
    cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap = apply_colormap_to_cam(cam_resized)

    overlay = cv2.addWeighted(image_rgb, 1.0 - alpha, heatmap, alpha, 0)
    return overlay


def denormalize_image_tensor(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """
    image_tensor: [C, H, W], normalized float tensor
    return: RGB uint8 [H, W, 3]
    """
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got: {tuple(image_tensor.shape)}")

    image = image_tensor.detach().cpu().float().clone()
    mean_t = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)

    image = image * std_t + mean_t
    image = image.clamp(0.0, 1.0)

    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return image


def make_gradcam_panel(
    original_rgb: np.ndarray,
    heatmap_rgb: np.ndarray,
    overlay_rgb: np.ndarray,
    text_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    원본 / heatmap / overlay를 가로로 붙이고,
    위쪽에 설명 텍스트를 얹은 panel 생성.
    """
    if original_rgb.shape != heatmap_rgb.shape or original_rgb.shape != overlay_rgb.shape:
        raise ValueError("All images must have the same shape.")

    panel = np.concatenate([original_rgb, heatmap_rgb, overlay_rgb], axis=1)

    if not text_lines:
        return panel

    text_lines = list(text_lines)
    header_h = 36 + 28 * len(text_lines)
    canvas = np.full(
        (panel.shape[0] + header_h, panel.shape[1], 3),
        255,
        dtype=np.uint8,
    )
    canvas[header_h:, :, :] = panel

    y = 30
    for line in text_lines:
        cv2.putText(
            canvas,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return canvas


def save_rgb_image(image_rgb: np.ndarray, save_path: str | Path) -> None:
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(str(save_path), image_bgr)
    if not success:
        raise RuntimeError(f"Failed to save image: {save_path}")


class GradCAM:
    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
    ) -> None:
        self.model = model
        self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self.handles = []

        self._register_hooks()

    def _register_hooks(self) -> None:
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.handles.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles = []

    def _get_target_score(
        self,
        logits: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        binary / multiclass logits 모두 지원.
        """
        if logits.ndim == 1:
            logit = logits[0]
            return logit if target_class == 1 else -logit

        if logits.ndim == 2 and logits.size(0) != 1:
            raise ValueError("Grad-CAM generate currently supports batch size 1 only.")

        if logits.ndim == 2 and logits.size(1) == 1:
            logit = logits[0, 0]
            return logit if target_class == 1 else -logit

        if logits.ndim == 2:
            return logits[0, target_class]

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
                "target_class": int,
            }
        """
        if input_tensor.ndim != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"Grad-CAM currently expects input shape [1, C, H, W], got {tuple(input_tensor.shape)}"
            )

        self.model.zero_grad(set_to_none=True)
        logits = self.model(input_tensor)

        if target_class is None:
            if logits.ndim == 1:
                target_class = int((torch.sigmoid(logits)[0] >= 0.5).item())
            elif logits.ndim == 2 and logits.size(1) == 1:
                target_class = int((torch.sigmoid(logits[0, 0]) >= 0.5).item())
            else:
                target_class = int(torch.argmax(logits, dim=1).item())

        score = self._get_target_score(logits, target_class)
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations/gradients for Grad-CAM.")

        activations = self.activations[0]   # [C, h, w]
        gradients = self.gradients[0]       # [C, h, w]

        weights = gradients.mean(dim=(1, 2), keepdim=True)   # [C, 1, 1]
        cam = (weights * activations).sum(dim=0)             # [h, w]
        cam = torch.relu(cam)

        cam_np = cam.detach().cpu().numpy().astype(np.float32)
        cam_np = cv2.resize(
            cam_np,
            (input_tensor.shape[-1], input_tensor.shape[-2]),
            interpolation=cv2.INTER_LINEAR,
        )
        cam_np = normalize_cam(cam_np)

        return {
            "cam": cam_np,
            "logits": logits.detach(),
            "target_class": int(target_class),
        }