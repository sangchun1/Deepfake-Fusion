from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
import torch

from src.deepfake_fusion.models.spectral.frequency_encoder import (
    fft2_image,
    log_magnitude_spectrum,
)


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _as_float_tensor(x: torch.Tensor) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(x)}")
    return x.detach().cpu().float()


def _minmax_normalize_np(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - x_min) / (x_max - x_min)


def _ensure_hwc3_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected [H, W, 3], got {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def _resize_to_match(image: np.ndarray, target_hw: tuple[int, int]) -> np.ndarray:
    h, w = target_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)


def save_rgb_image(image_rgb: np.ndarray, save_path: str | Path) -> None:
    save_path = _to_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    image_rgb = _ensure_hwc3_uint8(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    success = cv2.imwrite(str(save_path), image_bgr)
    if not success:
        raise RuntimeError(f"Failed to save image: {save_path}")


def save_json(data: dict[str, Any], save_path: str | Path, indent: int = 2) -> None:
    save_path = _to_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def denormalize_image_tensor(
    image_tensor: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> np.ndarray:
    """
    image_tensor: [C, H, W], normalized tensor
    return: RGB uint8 [H, W, 3]
    """
    image = _as_float_tensor(image_tensor)
    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {tuple(image.shape)}")

    mean_t = torch.tensor(mean, dtype=image.dtype).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=image.dtype).view(-1, 1, 1)

    if mean_t.size(0) != image.size(0) or std_t.size(0) != image.size(0):
        raise ValueError(
            f"mean/std channel size must match image channels. "
            f"Got image C={image.size(0)}, mean={len(mean)}, std={len(std)}"
        )

    image = image * std_t + mean_t
    image = image.clamp(0.0, 1.0)
    image = image.permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    return image


def tensor_abs_to_rgb_image(
    image_tensor: torch.Tensor,
    percentile: float = 99.5,
    channel_reduce: Optional[str] = None,
) -> np.ndarray:
    """
    고주파 residual 시각화용.
    기본은 abs 후 per-sample robust normalization.

    Args:
        image_tensor: [C, H, W]
        percentile: 상한 clipping percentile
        channel_reduce:
            - None: 채널별 유지 후 RGB처럼 표시
            - "mean": 채널 평균 후 grayscale 3채널
            - "max": 채널 최대 후 grayscale 3채널
    """
    image = _as_float_tensor(image_tensor)
    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {tuple(image.shape)}")

    image = image.abs()

    if channel_reduce is not None:
        channel_reduce = str(channel_reduce).strip().lower()
        if channel_reduce == "mean":
            image = image.mean(dim=0, keepdim=True)
        elif channel_reduce == "max":
            image = image.max(dim=0, keepdim=True).values
        else:
            raise ValueError(
                f"Unsupported channel_reduce: {channel_reduce}. "
                "Choose from [None, 'mean', 'max']."
            )

    image_np = image.permute(1, 2, 0).numpy()  # [H, W, C]
    high = float(np.percentile(image_np, percentile)) if image_np.size > 0 else 1.0
    if high <= 1e-8:
        image_np = np.zeros_like(image_np, dtype=np.float32)
    else:
        image_np = np.clip(image_np / high, 0.0, 1.0)

    if image_np.shape[2] == 1:
        image_np = np.repeat(image_np, 3, axis=2)
    elif image_np.shape[2] > 3:
        image_np = image_np[:, :, :3]

    return (image_np * 255.0).round().astype(np.uint8)


def spectrum_tensor_to_rgb_image(
    spectrum_tensor: torch.Tensor,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    log-magnitude tensor -> grayscale RGB uint8.

    Args:
        spectrum_tensor: [C, H, W] or [H, W]
    """
    spectrum = _as_float_tensor(spectrum_tensor)

    if spectrum.ndim == 3:
        spectrum = spectrum.mean(dim=0)  # channel average -> [H, W]
    elif spectrum.ndim != 2:
        raise ValueError(
            f"Expected spectrum shape [C, H, W] or [H, W], got {tuple(spectrum.shape)}"
        )

    spec_np = spectrum.numpy()
    lo = float(np.percentile(spec_np, percentile_low))
    hi = float(np.percentile(spec_np, percentile_high))

    if hi - lo < 1e-8:
        spec_np = np.zeros_like(spec_np, dtype=np.float32)
    else:
        spec_np = np.clip((spec_np - lo) / (hi - lo), 0.0, 1.0)

    spec_uint8 = (spec_np * 255.0).round().astype(np.uint8)
    return np.stack([spec_uint8, spec_uint8, spec_uint8], axis=-1)


def input_tensor_to_logmag_image(
    image_tensor: torch.Tensor,
    fft_norm: str = "ortho",
) -> np.ndarray:
    """
    입력 이미지 [C, H, W] -> log-magnitude spectrum RGB 이미지.
    """
    image = _as_float_tensor(image_tensor)
    if image.ndim != 3:
        raise ValueError(f"Expected [C, H, W], got {tuple(image.shape)}")

    spectrum = fft2_image(image.unsqueeze(0), norm=fft_norm, shift=True)[0]
    logmag = log_magnitude_spectrum(spectrum)
    return spectrum_tensor_to_rgb_image(logmag)


def mask_to_rgb_image(mask_tensor: torch.Tensor) -> np.ndarray:
    """
    mask: [1, 1, H, W] / [1, H, W] / [H, W] -> grayscale RGB uint8
    """
    mask = _as_float_tensor(mask_tensor)

    while mask.ndim > 2:
        mask = mask[0]

    if mask.ndim != 2:
        raise ValueError(f"Expected mask to reduce to [H, W], got {tuple(mask.shape)}")

    mask_np = _minmax_normalize_np(mask.numpy())
    mask_uint8 = (mask_np * 255.0).round().astype(np.uint8)
    return np.stack([mask_uint8, mask_uint8, mask_uint8], axis=-1)


def compute_spatial_energy(x: torch.Tensor) -> float:
    """
    간단한 spatial-domain energy = mean(x^2)
    """
    x = _as_float_tensor(x)
    return float(torch.mean(x * x).item())


def build_frequency_metrics(
    *,
    input_tensor: Optional[torch.Tensor] = None,
    x_low: Optional[torch.Tensor] = None,
    x_high: Optional[torch.Tensor] = None,
    explain_dict: Optional[Mapping[str, Any]] = None,
    true_label: Optional[int] = None,
    pred_label: Optional[int] = None,
    pred_prob: Optional[float] = None,
    source_filepath: Optional[str] = None,
    sample_index: Optional[int] = None,
    group: Optional[str] = None,
    frequency_cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """
    SPAI frequency explain용 sample metric dict 생성.
    explain_dict는 spai.extract_features(..., return_dict=True) 결과를 기대.
    """
    metrics: dict[str, Any] = {}

    if sample_index is not None:
        metrics["index"] = int(sample_index)
    if group is not None:
        metrics["group"] = str(group)
    if true_label is not None:
        metrics["true_label"] = int(true_label)
    if pred_label is not None:
        metrics["pred_label"] = int(pred_label)
    if pred_prob is not None:
        metrics["pred_prob"] = float(pred_prob)
    if source_filepath is not None:
        metrics["source_filepath"] = str(source_filepath)

    if frequency_cfg is not None:
        for key in ["mask_mode", "radius_ratio", "fft_norm", "high_from_residual"]:
            if key in frequency_cfg:
                value = frequency_cfg[key]
                if isinstance(value, (np.floating, np.integer)):
                    value = value.item()
                metrics[key] = value

    if input_tensor is not None:
        metrics["input_energy"] = compute_spatial_energy(input_tensor)
    if x_low is not None:
        metrics["low_energy"] = compute_spatial_energy(x_low)
    if x_high is not None:
        metrics["high_energy"] = compute_spatial_energy(x_high)

    if "low_energy" in metrics and "high_energy" in metrics:
        denom = metrics["low_energy"] + metrics["high_energy"]
        if denom > 1e-12:
            metrics["low_energy_ratio"] = metrics["low_energy"] / denom
            metrics["high_energy_ratio"] = metrics["high_energy"] / denom
        else:
            metrics["low_energy_ratio"] = 0.0
            metrics["high_energy_ratio"] = 0.0

    if explain_dict is None:
        return metrics

    branch_weights = explain_dict.get("branch_weights", None)
    if isinstance(branch_weights, torch.Tensor):
        bw = _as_float_tensor(branch_weights)
        if bw.ndim == 2 and bw.size(0) >= 1 and bw.size(1) >= 3:
            bw = bw[0]
        elif bw.ndim != 1 or bw.numel() < 3:
            bw = None
        if bw is not None:
            metrics["branch_weight_original"] = float(bw[0].item())
            metrics["branch_weight_low"] = float(bw[1].item())
            metrics["branch_weight_high"] = float(bw[2].item())

    similarity_stats = explain_dict.get("similarity_stats", None)
    if isinstance(similarity_stats, torch.Tensor):
        sim = _as_float_tensor(similarity_stats)
        if sim.ndim == 2 and sim.size(0) >= 1 and sim.size(1) >= 6:
            sim = sim[0]
        elif sim.ndim != 1 or sim.numel() < 6:
            sim = None
        if sim is not None:
            metrics["sim_cos_ol"] = float(sim[0].item())
            metrics["sim_cos_oh"] = float(sim[1].item())
            metrics["sim_cos_lh"] = float(sim[2].item())
            metrics["sim_l1_ol"] = float(sim[3].item())
            metrics["sim_l1_oh"] = float(sim[4].item())
            metrics["sim_l1_lh"] = float(sim[5].item())

    for key in [
        "orig_global",
        "low_global",
        "high_global",
        "aggregated_context",
    ]:
        tensor = explain_dict.get(key, None)
        if isinstance(tensor, torch.Tensor):
            t = _as_float_tensor(tensor)
            if t.ndim >= 2:
                t = t[0]
            metrics[f"{key}_norm"] = float(torch.norm(t).item())

    return metrics


def build_frequency_text_lines(
    metrics: Mapping[str, Any],
    max_lines: int = 8,
) -> list[str]:
    """
    panel header에 넣을 요약 텍스트 생성.
    """
    lines: list[str] = []

    if "index" in metrics:
        lines.append(f"idx={metrics['index']}")
    if "group" in metrics:
        lines.append(f"group={metrics['group']}")

    if "true_label" in metrics and "pred_label" in metrics:
        prob = metrics.get("pred_prob", None)
        if prob is None:
            lines.append(
                f"true={metrics['true_label']} pred={metrics['pred_label']}"
            )
        else:
            lines.append(
                f"true={metrics['true_label']} pred={metrics['pred_label']} prob={float(prob):.4f}"
            )

    if "low_energy_ratio" in metrics and "high_energy_ratio" in metrics:
        lines.append(
            "energy(low/high)="
            f"{float(metrics['low_energy_ratio']):.3f}/{float(metrics['high_energy_ratio']):.3f}"
        )

    has_bw = all(
        key in metrics
        for key in [
            "branch_weight_original",
            "branch_weight_low",
            "branch_weight_high",
        ]
    )
    if has_bw:
        lines.append(
            "branch_w(o/l/h)="
            f"{float(metrics['branch_weight_original']):.3f}/"
            f"{float(metrics['branch_weight_low']):.3f}/"
            f"{float(metrics['branch_weight_high']):.3f}"
        )

    has_sim = all(
        key in metrics for key in ["sim_cos_ol", "sim_cos_oh", "sim_cos_lh"]
    )
    if has_sim:
        lines.append(
            "sim_cos(ol/oh/lh)="
            f"{float(metrics['sim_cos_ol']):.3f}/"
            f"{float(metrics['sim_cos_oh']):.3f}/"
            f"{float(metrics['sim_cos_lh']):.3f}"
        )

    return lines[:max_lines]


def _draw_label(image: np.ndarray, label: str) -> np.ndarray:
    image = image.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 30), (255, 255, 255), thickness=-1)
    cv2.putText(
        image,
        label,
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )
    return image


def make_frequency_panel(
    input_rgb: np.ndarray,
    low_rgb: np.ndarray,
    high_abs_rgb: np.ndarray,
    input_logmag_rgb: np.ndarray,
    text_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    2x2 grid:
      [input_rgb, low_rgb]
      [high_abs_rgb, input_logmag_rgb]
    """
    input_rgb = _ensure_hwc3_uint8(input_rgb)
    low_rgb = _ensure_hwc3_uint8(low_rgb)
    high_abs_rgb = _ensure_hwc3_uint8(high_abs_rgb)
    input_logmag_rgb = _ensure_hwc3_uint8(input_logmag_rgb)

    base_h, base_w = input_rgb.shape[:2]
    low_rgb = _resize_to_match(low_rgb, (base_h, base_w))
    high_abs_rgb = _resize_to_match(high_abs_rgb, (base_h, base_w))
    input_logmag_rgb = _resize_to_match(input_logmag_rgb, (base_h, base_w))

    top_row = np.concatenate(
        [
            _draw_label(input_rgb, "input"),
            _draw_label(low_rgb, "low"),
        ],
        axis=1,
    )
    bottom_row = np.concatenate(
        [
            _draw_label(high_abs_rgb, "high_abs"),
            _draw_label(input_logmag_rgb, "input_logmag"),
        ],
        axis=1,
    )
    panel = np.concatenate([top_row, bottom_row], axis=0)

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
            str(line),
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        y += 28

    return canvas


def build_frequency_visuals(
    input_tensor: torch.Tensor,
    x_low: torch.Tensor,
    x_high: torch.Tensor,
    *,
    mean: Sequence[float],
    std: Sequence[float],
    fft_norm: str = "ortho",
    high_channel_reduce: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """
    샘플별 기본 frequency explain 이미지 4종 생성.

    Returns keys:
      - input_rgb
      - low_rgb
      - high_abs
      - input_logmag
    """
    if input_tensor.ndim != 3:
        raise ValueError(f"Expected input_tensor [C, H, W], got {tuple(input_tensor.shape)}")
    if x_low.ndim != 3:
        raise ValueError(f"Expected x_low [C, H, W], got {tuple(x_low.shape)}")
    if x_high.ndim != 3:
        raise ValueError(f"Expected x_high [C, H, W], got {tuple(x_high.shape)}")

    input_rgb = denormalize_image_tensor(input_tensor, mean=mean, std=std)
    low_rgb = denormalize_image_tensor(x_low, mean=mean, std=std)
    high_abs = tensor_abs_to_rgb_image(
        x_high,
        channel_reduce=high_channel_reduce,
    )
    input_logmag = input_tensor_to_logmag_image(
        input_tensor,
        fft_norm=fft_norm,
    )

    return {
        "input_rgb": input_rgb,
        "low_rgb": low_rgb,
        "high_abs": high_abs,
        "input_logmag": input_logmag,
    }


def save_frequency_sample_artifacts(
    save_dir: str | Path,
    visuals: Mapping[str, np.ndarray],
    metrics: Optional[Mapping[str, Any]] = None,
    *,
    save_individual_images: bool = True,
    save_panel: bool = True,
) -> dict[str, str]:
    """
    샘플 폴더에 frequency explain 결과 저장.

    저장 파일:
      - 00_input_rgb.png
      - 01_low_rgb.png
      - 02_high_abs.png
      - 03_input_logmag.png
      - panel.png
      - sample_metrics.json
    """
    save_dir = _to_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    saved: dict[str, str] = {}

    if save_individual_images:
        ordered = [
            ("00_input_rgb.png", visuals["input_rgb"]),
            ("01_low_rgb.png", visuals["low_rgb"]),
            ("02_high_abs.png", visuals["high_abs"]),
            ("03_input_logmag.png", visuals["input_logmag"]),
        ]
        for filename, image in ordered:
            path = save_dir / filename
            save_rgb_image(image, path)
            saved[filename] = str(path)

    if save_panel:
        text_lines = build_frequency_text_lines(metrics or {})
        panel = make_frequency_panel(
            input_rgb=visuals["input_rgb"],
            low_rgb=visuals["low_rgb"],
            high_abs_rgb=visuals["high_abs"],
            input_logmag_rgb=visuals["input_logmag"],
            text_lines=text_lines,
        )
        panel_path = save_dir / "panel.png"
        save_rgb_image(panel, panel_path)
        saved["panel.png"] = str(panel_path)

    if metrics is not None:
        metrics_path = save_dir / "sample_metrics.json"
        save_json(dict(metrics), metrics_path)
        saved["sample_metrics.json"] = str(metrics_path)

    return saved


def save_frequency_run_artifacts(
    save_dir: str | Path,
    low_mask: torch.Tensor,
    high_mask: torch.Tensor,
    mask_info: Optional[dict[str, Any]] = None,
) -> dict[str, str]:
    """
    run-level mask 이미지/정보 저장.

    저장 파일:
      - mask_low.png
      - mask_high.png
      - mask_info.json
    """
    save_dir = _to_path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    low_mask_rgb = mask_to_rgb_image(low_mask)
    high_mask_rgb = mask_to_rgb_image(high_mask)

    low_path = save_dir / "mask_low.png"
    high_path = save_dir / "mask_high.png"

    save_rgb_image(low_mask_rgb, low_path)
    save_rgb_image(high_mask_rgb, high_path)

    saved = {
        "mask_low.png": str(low_path),
        "mask_high.png": str(high_path),
    }

    if mask_info is not None:
        info_path = save_dir / "mask_info.json"
        save_json(mask_info, info_path)
        saved["mask_info.json"] = str(info_path)

    return saved