from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Type

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

from deepfake_fusion.datasets.semitruths_dataset import SemiTruthsDataset
from deepfake_fusion.models.build_model import build_model
from deepfake_fusion.transforms.image_aug import build_transforms_from_config
from deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
)
from deepfake_fusion.utils.seed import seed_everything
from deepfake_fusion.utils.semitruths_metadata import (
    add_default_analysis_bins,
    build_directional_edit_column,
    standardize_semitruths_metadata,
)
from deepfake_fusion.visualization.attention_rollout import AttentionRollout
from deepfake_fusion.visualization.frequency_visualize import (
    build_frequency_metrics,
    build_frequency_visuals,
    save_frequency_run_artifacts,
    save_frequency_sample_artifacts,
)
from deepfake_fusion.visualization.gradcam import (
    GradCAM,
    apply_colormap_to_cam,
    denormalize_image_tensor,
    make_gradcam_panel,
    overlay_cam_on_image,
    resolve_target_layer,
    save_rgb_image,
)


DATASET_REGISTRY: Dict[str, Type] = {
    "semitruths": SemiTruthsDataset,
    "semitruths_eval": SemiTruthsDataset,
    "SemiTruthsDataset": SemiTruthsDataset,
}


# -----------------------------------------------------------------------------
# argparse / config helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explanation visualization script for Semi-Truths Evalset."
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/semitruths_eval.yaml",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_resnet_openfake.yaml",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path. If omitted, output_dir/best.pth is used.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cuda / cpu / mps",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="auto",
        choices=["auto", "gradcam", "rollout", "frequency"],
        help="Explanation method. 'auto' uses frequency for SPAI, rollout for ViT, gradcam otherwise.",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default=None,
        help="Grad-CAM target layer. Example: backbone.layer4.1",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="설명 대상을 predicted class 기준으로 할지, true label 기준으로 할지 선택",
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=4,
        help="각 그룹(correct_real/correct_fake/wrong_real/wrong_fake)당 저장할 최대 개수",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="overlay에서 heatmap 비중",
    )
    parser.add_argument(
        "--head_fusion",
        type=str,
        default="mean",
        choices=["mean", "max", "min"],
        help="Attention rollout head fusion mode",
    )
    parser.add_argument(
        "--discard_ratio",
        type=float,
        default=0.0,
        help="Attention rollout discard ratio",
    )
    parser.add_argument(
        "--start_layer",
        type=int,
        default=0,
        help="Attention rollout start layer",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/explain_semitruths",
    )
    parser.add_argument(
        "--save_individual_frequency_images",
        action="store_true",
        help="frequency method에서 panel 외 개별 이미지도 함께 저장",
    )
    parser.add_argument(
        "--save_prepared_csv",
        action="store_true",
        help="표준화한 Semi-Truths metadata CSV를 함께 저장",
    )
    parser.add_argument(
        "--require_mask_for_fake",
        action="store_true",
        help="fake 샘플은 mask가 있는 경우만 저장",
    )
    parser.add_argument(
        "--cam_topk_fraction",
        type=float,
        default=0.2,
        help="CAM-mask overlap 계산 시 상위 활성 영역 비율 (0~1)",
    )
    parser.add_argument(
        "--mask_threshold",
        type=float,
        default=0.5,
        help="마스크를 binary로 만들 때 사용할 threshold",
    )
    return parser.parse_args()



def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
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



def _to_plain_python(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_plain_python(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain_python(v) for v in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return None if np.isnan(out) else out
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            return value
    if isinstance(value, float) and np.isnan(value):
        return None
    if pd.isna(value):
        return None
    return value



def resolve_device(device_name: str | None) -> torch.device:
    device_name = (device_name or "cuda").lower()

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if (
        device_name == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_plain_python(data), f, indent=2, ensure_ascii=False)



def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = True,
) -> Dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    return checkpoint



def get_prob_and_pred(logits: torch.Tensor) -> tuple[float, int]:
    if logits.ndim == 1:
        prob_pos = float(torch.sigmoid(logits)[0].item())
        pred = int(prob_pos >= 0.5)
        return prob_pos, pred

    if logits.ndim == 2 and logits.size(1) == 1:
        prob_pos = float(torch.sigmoid(logits[0, 0]).item())
        pred = int(prob_pos >= 0.5)
        return prob_pos, pred

    if logits.ndim == 2:
        probs = torch.softmax(logits, dim=1)[0]
        pred = int(torch.argmax(probs).item())
        prob = float(probs[pred].item())
        return prob, pred

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")



def categorize_case(true_label: int, pred_label: int) -> str:
    if true_label == 0 and pred_label == 0:
        return "correct_real"
    if true_label == 1 and pred_label == 1:
        return "correct_fake"
    if true_label == 0 and pred_label == 1:
        return "wrong_real"
    if true_label == 1 and pred_label == 0:
        return "wrong_fake"
    return "unknown"



def short_label_name(label: int) -> str:
    return "real" if label == 0 else "fake"



def is_fusion_model(model: torch.nn.Module) -> bool:
    return hasattr(model, "spatial_branch") and hasattr(model, "spectral_branch")



def resolve_frequency_explain_components(
    model: torch.nn.Module,
) -> tuple[torch.nn.Module, Any]:
    if hasattr(model, "frequency_encoder") and hasattr(model, "extract_features"):
        return model, model.frequency_encoder

    if (
        hasattr(model, "spectral_branch")
        and hasattr(model, "extract_features")
        and hasattr(model.spectral_branch, "frequency_encoder")
    ):
        return model, model.spectral_branch.frequency_encoder

    raise ValueError(
        "Frequency explanation requires either:\n"
        "1) a SPAI-like model with 'frequency_encoder' and "
        "'extract_features(return_dict=True)', or\n"
        "2) a fusion model with 'spectral_branch.frequency_encoder' and "
        "'extract_features(return_dict=True)'."
    )



def get_frequency_cfg_dict(cfg: Any) -> Dict[str, Any]:
    model_name = str(_cfg_get(cfg, "model", "name", default="")).lower()

    if model_name == "fusion":
        freq_cfg = _cfg_get(cfg, "model", "spectral", "frequency", default=None)
    else:
        freq_cfg = _cfg_get(cfg, "model", "frequency", default=None)

    if freq_cfg is None:
        return {}

    return {
        "mask_mode": _to_plain_python(_cfg_get(freq_cfg, "mask_mode", default="radial")),
        "radius_ratio": _to_plain_python(_cfg_get(freq_cfg, "radius_ratio", default=0.25)),
        "fft_norm": _to_plain_python(_cfg_get(freq_cfg, "fft_norm", default="ortho")),
        "high_from_residual": _to_plain_python(
            _cfg_get(freq_cfg, "high_from_residual", default=True)
        ),
        "clamp_output": _to_plain_python(_cfg_get(freq_cfg, "clamp_output", default=False)),
        "eps": _to_plain_python(_cfg_get(freq_cfg, "eps", default=1e-8)),
    }


# -----------------------------------------------------------------------------
# Semi-Truths metadata preparation
# -----------------------------------------------------------------------------


def build_column_override_map(cfg: Any) -> Dict[str, str]:
    metadata_cfg = _cfg_get(cfg, "data", "metadata", default=None)
    if metadata_cfg is None:
        return {}

    mapping = {
        "sample_id": _cfg_get(cfg, "data", "metadata", "image_id_col", default=None),
        "filepath": _cfg_get(cfg, "data", "metadata", "image_path_col", default=None),
        "label": _cfg_get(cfg, "data", "metadata", "label_col", default=None),
        "dataset": _cfg_get(cfg, "data", "metadata", "dataset_col", default=None),
        "entity": _cfg_get(cfg, "data", "metadata", "entity_col", default=None),
        "method": _cfg_get(cfg, "data", "metadata", "method_col", default=None),
        "diffusion_model": _cfg_get(cfg, "data", "metadata", "diffusion_model_col", default=None),
        "area_ratio": _cfg_get(cfg, "data", "metadata", "area_ratio_col", default=None),
        "semantic_mag": _cfg_get(cfg, "data", "metadata", "semantic_mag_col", default=None),
        "scene_complexity": _cfg_get(cfg, "data", "metadata", "scene_complexity_col", default=None),
        "scene_diversity": _cfg_get(cfg, "data", "metadata", "scene_diversity_col", default=None),
        "change_type": _cfg_get(cfg, "data", "metadata", "change_type_col", default=None),
        "mask_path": _cfg_get(cfg, "data", "metadata", "mask_path_col", default=None),
    }
    return {k: v for k, v in mapping.items() if v}



def prepare_metadata_dataframe(cfg: Any) -> pd.DataFrame:
    metadata_csv = resolve_path(cfg.data.paths.metadata_csv)
    root_dir = cfg.data.paths.root_dir
    include_real = bool(_cfg_get(cfg, "data", "metadata", "include_real", default=True))
    include_inpainting = bool(_cfg_get(cfg, "data", "metadata", "include_inpainting", default=True))
    include_p2p = bool(_cfg_get(cfg, "data", "metadata", "include_p2p", default=True))

    df = standardize_semitruths_metadata(
        metadata=metadata_csv,
        root_dir=root_dir,
        column_map=build_column_override_map(cfg),
        include_real=include_real,
        include_inpainting=include_inpainting,
        include_p2p=include_p2p,
        validate_paths=False,
    )

    grouping_cfg = _cfg_get(cfg, "data", "grouping", default=None)
    if grouping_cfg is not None:
        df = add_default_analysis_bins(
            df,
            area_ratio_bins=_cfg_get(cfg, "data", "grouping", "area_ratio_bins", default=None),
            area_ratio_labels=_cfg_get(
                cfg,
                "data",
                "grouping",
                "area_ratio_bin_labels",
                default=("small", "medium", "large"),
            ),
            semantic_mag_bins=_cfg_get(cfg, "data", "grouping", "semantic_mag_bins", default=None),
            semantic_mag_labels=_cfg_get(
                cfg,
                "data",
                "grouping",
                "semantic_mag_bin_labels",
                default=("small", "medium", "large"),
            ),
        )
    else:
        df = add_default_analysis_bins(df)

    df = build_directional_edit_column(df)
    return df.reset_index(drop=True)



def get_dataset_class(cfg: Any):
    dataset_key = None

    if getattr(cfg.data, "dataset_class", None) is not None:
        dataset_key = str(cfg.data.dataset_class)
    elif getattr(cfg.data, "name", None) is not None:
        dataset_key = str(cfg.data.name)

    if dataset_key is None:
        raise ValueError(
            "Could not determine dataset class. "
            "Set cfg.data.dataset_class or cfg.data.name in the data config."
        )

    if dataset_key not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(
            f"Unsupported dataset '{dataset_key}'. Supported values: {supported}"
        )

    return DATASET_REGISTRY[dataset_key]



def infer_explain_method(cfg, requested_method: str) -> str:
    if requested_method != "auto":
        return requested_method

    model_name = str(getattr(cfg.model, "name", "")).lower()
    backbone = getattr(cfg.model, "backbone", None)
    backbone_name = str(getattr(backbone, "name", "")).lower()

    if model_name == "spai":
        return "frequency"

    if model_name == "fusion":
        return "gradcam"

    if model_name == "vit" or backbone_name.startswith("vit"):
        return "rollout"

    return "gradcam"



def build_explainer(
    model: torch.nn.Module,
    method: str,
    args: argparse.Namespace,
):
    if method == "gradcam":
        target_layer = resolve_target_layer(model, args.target_layer)
        print(f"Using target layer for Grad-CAM: {target_layer}")
        explainer = GradCAM(model=model, target_layer=target_layer)
        return explainer, target_layer

    if method == "rollout":
        if is_fusion_model(model):
            raise ValueError(
                "Attention rollout is not supported for fusion v1. "
                "Use '--method gradcam' for the spatial branch or "
                "'--method frequency' for the spectral branch."
            )

        explainer = AttentionRollout(
            model=model,
            head_fusion=args.head_fusion,
            discard_ratio=float(args.discard_ratio),
            start_layer=int(args.start_layer),
        )
        print(
            "Using Attention Rollout "
            f"(head_fusion={args.head_fusion}, "
            f"discard_ratio={args.discard_ratio}, "
            f"start_layer={args.start_layer})"
        )
        return explainer, None

    if method == "frequency":
        resolve_frequency_explain_components(model)
        if is_fusion_model(model):
            print("Using frequency explanation for the spectral branch inside fusion model.")
        else:
            print("Using frequency-only explanation for SPAI.")
        return None, None

    raise ValueError(f"Unsupported explanation method: {method}")


# -----------------------------------------------------------------------------
# mask helpers / overlap metrics
# -----------------------------------------------------------------------------


def normalize_map(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D map, got shape={x.shape}")
    x = np.maximum(x, 0.0)
    x_min = float(x.min())
    x_max = float(x.max())
    if x_max - x_min < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - x_min) / (x_max - x_min)).astype(np.float32)



def grayscale_from_rgb(image_rgb: np.ndarray) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim == 2:
        gray = image_rgb.astype(np.float32)
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        gray = cv2.cvtColor(image_rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image shape: {image_rgb.shape}")
    return normalize_map(gray)



def resolve_existing_path(path_value: Any, root_dir: Optional[str | Path] = None) -> Optional[Path]:
    if path_value is None:
        return None
    if isinstance(path_value, float) and np.isnan(path_value):
        return None

    path = Path(str(path_value))
    candidates: List[Path] = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(path)
        if root_dir is not None:
            candidates.append(Path(root_dir) / path)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None



def infer_mask_path_from_sample(sample: Mapping[str, Any], root_dir: Optional[str | Path] = None) -> Optional[Path]:
    for key in ("absolute_mask_path", "mask_path"):
        mask_path = resolve_existing_path(sample.get(key, None), root_dir=root_dir)
        if mask_path is not None:
            return mask_path

    filepath = sample.get("absolute_filepath", None) or sample.get("filepath", None)
    filepath_resolved = resolve_existing_path(filepath, root_dir=root_dir)
    if filepath_resolved is None:
        return None

    filepath_posix = filepath_resolved.as_posix()
    candidates: List[Path] = []

    if "/p2p/" in filepath_posix:
        candidates.append(Path(filepath_posix.replace("/p2p/", "/p2p_masks/", 1)))

    if "/original/images/" in filepath_posix:
        candidates.append(Path(filepath_posix.replace("/original/images/", "/original/masks/", 1)))

    if "/inpainting/" in filepath_posix:
        candidates.append(Path(filepath_posix.replace("/inpainting/", "/original/masks/", 1)))
        if filepath_resolved.parent.parent.name == "inpainting":
            candidates.append(filepath_resolved.parents[2] / "original" / "masks" / filepath_resolved.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None



def load_binary_mask(
    mask_path: str | Path,
    output_size_hw: Tuple[int, int],
    threshold: float = 0.5,
) -> np.ndarray:
    mask_path = Path(mask_path)
    if not mask_path.exists():
        raise FileNotFoundError(f"Mask file not found: {mask_path}")

    mask = Image.open(mask_path).convert("L")
    width = int(output_size_hw[1])
    height = int(output_size_hw[0])
    mask = mask.resize((width, height), resample=Image.Resampling.NEAREST)
    mask_np = np.asarray(mask, dtype=np.float32)
    if mask_np.max() > 1.0:
        mask_np /= 255.0
    return (mask_np >= float(threshold)).astype(np.uint8)



def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    mask = (mask.astype(np.uint8) > 0).astype(np.uint8)
    rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb[mask > 0] = np.array([255, 255, 255], dtype=np.uint8)
    return rgb



def overlay_mask_on_image(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.30,
    color: Tuple[int, int, int] = (0, 255, 0),
    contour_thickness: int = 2,
) -> np.ndarray:
    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    mask = (np.asarray(mask) > 0).astype(np.uint8)
    overlay = image_rgb.copy()

    tint = np.zeros_like(image_rgb, dtype=np.uint8)
    tint[:, :] = np.array(color, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = cv2.addWeighted(
        image_rgb[mask_bool], 1.0 - alpha, tint[mask_bool], alpha, 0
    )

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, color, contour_thickness)
    return overlay



def overlay_cam_and_mask_on_image(
    image_rgb: np.ndarray,
    cam: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.35,
    contour_color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    overlay = overlay_cam_on_image(image_rgb=image_rgb, cam=cam, alpha=alpha)
    return overlay_mask_on_image(
        image_rgb=overlay,
        mask=mask,
        alpha=0.18,
        color=contour_color,
        contour_thickness=2,
    )



def make_2x2_panel(
    top_left: np.ndarray,
    top_right: np.ndarray,
    bottom_left: np.ndarray,
    bottom_right: np.ndarray,
    labels: Sequence[str],
    text_lines: Optional[Sequence[str]] = None,
) -> np.ndarray:
    def _with_label(img: np.ndarray, label: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (out.shape[1], 30), (255, 255, 255), thickness=-1)
        cv2.putText(
            out,
            label,
            (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        return out

    imgs = [top_left, top_right, bottom_left, bottom_right]
    imgs = [_with_label(np.asarray(img, dtype=np.uint8), label) for img, label in zip(imgs, labels)]

    top = np.concatenate([imgs[0], imgs[1]], axis=1)
    bottom = np.concatenate([imgs[2], imgs[3]], axis=1)
    panel = np.concatenate([top, bottom], axis=0)

    if not text_lines:
        return panel

    text_lines = list(text_lines)
    header_h = 36 + 28 * len(text_lines)
    canvas = np.full((panel.shape[0] + header_h, panel.shape[1], 3), 255, dtype=np.uint8)
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



def compute_map_mask_metrics(
    score_map: np.ndarray,
    mask: np.ndarray,
    topk_fraction: float = 0.2,
    prefix: str = "cam",
) -> Dict[str, Optional[float]]:
    score_map = normalize_map(score_map)
    mask = (np.asarray(mask) > 0).astype(np.uint8)

    metrics: Dict[str, Optional[float]] = {
        f"{prefix}_mask_area_ratio": float(mask.mean()),
    }

    mask_bool = mask.astype(bool)
    if mask_bool.sum() == 0:
        metrics.update(
            {
                f"{prefix}_mean_in_mask": None,
                f"{prefix}_mean_out_mask": None,
                f"{prefix}_mass_in_mask": None,
                f"{prefix}_topk_precision": None,
                f"{prefix}_topk_recall": None,
                f"{prefix}_topk_iou": None,
                f"{prefix}_peak_in_mask": None,
            }
        )
        return metrics

    inv_mask = ~mask_bool
    total_mass = float(score_map.sum())
    metrics[f"{prefix}_mean_in_mask"] = float(score_map[mask_bool].mean())
    metrics[f"{prefix}_mean_out_mask"] = float(score_map[inv_mask].mean()) if inv_mask.any() else 0.0
    metrics[f"{prefix}_mass_in_mask"] = (
        float(score_map[mask_bool].sum()) / total_mass if total_mass > 1e-12 else None
    )

    peak_idx = int(np.argmax(score_map.reshape(-1)))
    peak_row, peak_col = np.unravel_index(peak_idx, score_map.shape)
    metrics[f"{prefix}_peak_in_mask"] = float(mask_bool[peak_row, peak_col])

    k = max(1, int(round(float(topk_fraction) * score_map.size)))
    flat = score_map.reshape(-1)
    top_indices = np.argpartition(flat, -k)[-k:]
    top_mask = np.zeros_like(flat, dtype=np.uint8)
    top_mask[top_indices] = 1
    top_mask = top_mask.reshape(score_map.shape).astype(bool)

    inter = int(np.logical_and(top_mask, mask_bool).sum())
    union = int(np.logical_or(top_mask, mask_bool).sum())
    topk_count = int(top_mask.sum())
    mask_count = int(mask_bool.sum())

    metrics[f"{prefix}_topk_precision"] = inter / topk_count if topk_count > 0 else None
    metrics[f"{prefix}_topk_recall"] = inter / mask_count if mask_count > 0 else None
    metrics[f"{prefix}_topk_iou"] = inter / union if union > 0 else None
    return metrics



def extract_sample_metadata(sample: Mapping[str, Any]) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {}
    for key, value in sample.items():
        if key in {"image", "label"}:
            continue
        metadata[key] = _to_plain_python(value)
    return metadata


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    seed = int(cfg.train.experiment.seed)
    seed_everything(seed)

    device = resolve_device(args.device or cfg.train.experiment.device)
    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / "best.pth")
    )
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    method = infer_explain_method(cfg, args.method)
    dataset_name = getattr(cfg.data, "name", "unknown")
    save_root = ensure_dir(
        resolve_path(args.save_dir)
        / method
        / dataset_name
        / args.split
        / checkpoint_path.stem
    )

    prepared_df = prepare_metadata_dataframe(cfg)
    prepared_csv_path = save_root / f"prepared_{args.split}.csv"
    prepared_df.to_csv(prepared_csv_path, index=False)
    if not args.save_prepared_csv:
        # dataset 입력용으로는 사용하되, 요약에는 경로를 남기고 옵션 설명만 다르게 한다.
        pass

    transforms = build_transforms_from_config(cfg)
    dataset_cls = get_dataset_class(cfg)
    dataset = dataset_cls(
        csv_path=prepared_csv_path,
        root_dir=cfg.data.paths.root_dir,
        transform=transforms[args.split],
        validate_files=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("No samples found after preparing Semi-Truths metadata.")

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {dataset_name}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"size: {len(dataset)}")
    if hasattr(dataset, "class_counts"):
        print(f"class counts: {dataset.class_counts}")

    model = build_model(cfg.model)
    checkpoint = load_checkpoint_to_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=bool(cfg.model.checkpoint.strict),
    )
    model = model.to(device)
    model.eval()

    explainer, resolved_target_layer = build_explainer(model, method, args)

    mean = list(cfg.data.image.mean)
    std = list(cfg.data.image.std)
    target_groups = ["correct_real", "correct_fake", "wrong_real", "wrong_fake"]
    saved_counts = defaultdict(int)
    records: List[Dict[str, Any]] = []

    run_artifacts: Dict[str, Any] = {}
    frequency_cfg = get_frequency_cfg_dict(cfg)
    frequency_run_saved = False

    try:
        for idx in range(len(dataset)):
            if all(saved_counts[g] >= args.max_per_group for g in target_groups):
                break

            sample = dataset[idx]
            image_tensor = sample["image"]
            label_tensor = sample["label"]
            filepath = sample.get("filepath", "")
            metadata = extract_sample_metadata(sample)

            x = image_tensor.unsqueeze(0).to(device)
            true_label = int(label_tensor.item())

            if method == "frequency":
                with torch.no_grad():
                    logits = model(x)
                    pred_prob, pred_label = get_prob_and_pred(logits)
                    group = categorize_case(true_label, pred_label)
                    if saved_counts[group] >= args.max_per_group:
                        continue

                    feature_model, frequency_encoder = resolve_frequency_explain_components(model)
                    explain_out = feature_model.extract_features(x, return_dict=True)
                    split_dict = frequency_encoder.split_spectrum(x)

                target_class = pred_label if args.target_type == "pred" else true_label

                if not frequency_run_saved:
                    mask_info = {
                        **frequency_cfg,
                        "image_size": list(image_tensor.shape[-2:]),
                        "dataset": dataset_name,
                        "split": args.split,
                    }
                    run_artifacts = save_frequency_run_artifacts(
                        save_dir=save_root,
                        low_mask=split_dict["low_mask"],
                        high_mask=split_dict["high_mask"],
                        mask_info=mask_info,
                    )
                    frequency_run_saved = True

                x_low = explain_out["x_low"][0].detach().cpu()
                x_high = explain_out["x_high"][0].detach().cpu()

                visuals = build_frequency_visuals(
                    input_tensor=image_tensor,
                    x_low=x_low,
                    x_high=x_high,
                    mean=mean,
                    std=std,
                    fft_norm=str(frequency_cfg.get("fft_norm", "ortho")),
                    high_channel_reduce="mean",
                )

                metrics = build_frequency_metrics(
                    input_tensor=image_tensor,
                    x_low=x_low,
                    x_high=x_high,
                    explain_dict=explain_out,
                    true_label=true_label,
                    pred_label=pred_label,
                    pred_prob=pred_prob,
                    source_filepath=filepath,
                    sample_index=idx,
                    group=group,
                    frequency_cfg=frequency_cfg,
                )

                mask_path = infer_mask_path_from_sample(sample, root_dir=cfg.data.paths.root_dir)
                mask_binary: Optional[np.ndarray] = None
                extra_saved_files: Dict[str, str] = {}
                if mask_path is not None:
                    mask_binary = load_binary_mask(
                        mask_path=mask_path,
                        output_size_hw=(image_tensor.shape[-2], image_tensor.shape[-1]),
                        threshold=float(args.mask_threshold),
                    )
                if args.require_mask_for_fake and true_label == 1 and mask_binary is None:
                    continue

                if mask_binary is not None:
                    high_abs_map = grayscale_from_rgb(visuals["high_abs"])
                    metrics.update(
                        compute_map_mask_metrics(
                            high_abs_map,
                            mask_binary,
                            topk_fraction=float(args.cam_topk_fraction),
                            prefix="high_abs",
                        )
                    )
                    mask_rgb = mask_to_rgb(mask_binary)
                    mask_overlay_rgb = overlay_mask_on_image(visuals["input_rgb"], mask_binary)
                    extra_panel = make_2x2_panel(
                        top_left=visuals["input_rgb"],
                        top_right=mask_overlay_rgb,
                        bottom_left=visuals["high_abs"],
                        bottom_right=visuals["input_logmag"],
                        labels=["Original", "Mask Overlay", "High Abs", "Input LogMag"],
                        text_lines=[
                            f"group={group} | idx={idx} | method={method}",
                            f"true={short_label_name(true_label)}({true_label}) | pred={short_label_name(pred_label)}({pred_label}) | prob={pred_prob:.4f}",
                            f"subset={metadata.get('subset', 'unknown')} | data_method={metadata.get('method', 'unknown')}",
                            f"high_abs_mass_in_mask={metrics.get('high_abs_mass_in_mask', None)}",
                        ],
                    )
                else:
                    mask_rgb = None
                    mask_overlay_rgb = None
                    extra_panel = None

                sample_name = (
                    f"{idx:05d}"
                    f"_true-{short_label_name(true_label)}"
                    f"_pred-{short_label_name(pred_label)}"
                    f"_prob-{pred_prob:.4f}"
                )
                sample_dir = ensure_dir(save_root / group / sample_name)

                saved_files = save_frequency_sample_artifacts(
                    save_dir=sample_dir,
                    visuals=visuals,
                    metrics=metrics,
                    save_individual_images=bool(args.save_individual_frequency_images),
                    save_panel=True,
                )

                if mask_rgb is not None:
                    mask_path_out = sample_dir / "04_mask.png"
                    save_rgb_image(mask_rgb, mask_path_out)
                    extra_saved_files["04_mask.png"] = mask_path_out.as_posix()
                if mask_overlay_rgb is not None:
                    mask_overlay_out = sample_dir / "05_mask_overlay.png"
                    save_rgb_image(mask_overlay_rgb, mask_overlay_out)
                    extra_saved_files["05_mask_overlay.png"] = mask_overlay_out.as_posix()
                if extra_panel is not None:
                    panel_out = sample_dir / "panel_semitruths.png"
                    save_rgb_image(extra_panel, panel_out)
                    extra_saved_files["panel_semitruths.png"] = panel_out.as_posix()

                record = {
                    "index": idx,
                    "group": group,
                    "method": method,
                    "true_label": true_label,
                    "pred_label": pred_label,
                    "pred_prob": pred_prob,
                    "target_type": args.target_type,
                    "target_class": target_class,
                    "source_filepath": filepath,
                    "saved_dir": sample_dir.as_posix(),
                    "saved_files": {**saved_files, **extra_saved_files},
                    "metrics": metrics,
                    "metadata": metadata,
                    "mask_path": mask_path.as_posix() if mask_path is not None else None,
                }
                records.append(record)
                saved_counts[group] += 1

                print(
                    f"[Saved] method={method} | group={group} | idx={idx} | "
                    f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | {sample_dir}"
                )
                continue

            if method == "rollout":
                result = explainer.generate(x, target_class=None)
                logits = result["logits"]
                pred_prob, pred_label = get_prob_and_pred(logits)
                target_class = pred_label if args.target_type == "pred" else true_label
            else:
                with torch.no_grad():
                    logits = model(x)
                pred_prob, pred_label = get_prob_and_pred(logits)
                target_class = pred_label if args.target_type == "pred" else true_label
                result = explainer.generate(x, target_class=target_class)

            group = categorize_case(true_label, pred_label)
            if saved_counts[group] >= args.max_per_group:
                continue

            mask_path = infer_mask_path_from_sample(sample, root_dir=cfg.data.paths.root_dir)
            mask_binary: Optional[np.ndarray] = None
            if mask_path is not None:
                mask_binary = load_binary_mask(
                    mask_path=mask_path,
                    output_size_hw=(image_tensor.shape[-2], image_tensor.shape[-1]),
                    threshold=float(args.mask_threshold),
                )
            if args.require_mask_for_fake and true_label == 1 and mask_binary is None:
                continue

            input_rgb = denormalize_image_tensor(image_tensor, mean=mean, std=std)
            cam = result["cam"]
            heatmap_rgb = apply_colormap_to_cam(cam)
            overlay_rgb = overlay_cam_on_image(input_rgb, cam, alpha=args.alpha)

            text_lines = [
                (
                    f"dataset={dataset_name} | split={args.split} | "
                    f"group={group} | idx={idx} | method={method}"
                ),
                (
                    f"true={short_label_name(true_label)}({true_label}) | "
                    f"pred={short_label_name(pred_label)}({pred_label}) | "
                    f"prob={pred_prob:.4f}"
                ),
                f"subset={metadata.get('subset', 'unknown')} | data_method={metadata.get('method', 'unknown')}",
                (
                    f"dataset={metadata.get('dataset', 'unknown')} | "
                    f"diffusion={metadata.get('diffusion_model', 'unknown')}"
                ),
                f"path={Path(filepath).name}",
            ]

            overlap_metrics: Dict[str, Optional[float]] = {}
            extra_saved_files: Dict[str, str] = {}
            if mask_binary is not None:
                overlap_metrics = compute_map_mask_metrics(
                    cam,
                    mask_binary,
                    topk_fraction=float(args.cam_topk_fraction),
                    prefix="cam",
                )
                text_lines.append(
                    "cam_mass_in_mask="
                    f"{overlap_metrics.get('cam_mass_in_mask', None)} | "
                    f"cam_topk_iou={overlap_metrics.get('cam_topk_iou', None)}"
                )
                mask_overlay_rgb = overlay_mask_on_image(input_rgb, mask_binary)
                cam_mask_overlay_rgb = overlay_cam_and_mask_on_image(
                    input_rgb,
                    cam,
                    mask_binary,
                    alpha=args.alpha,
                )
                mask_rgb = mask_to_rgb(mask_binary)
                panel = make_2x2_panel(
                    top_left=input_rgb,
                    top_right=mask_overlay_rgb,
                    bottom_left=heatmap_rgb,
                    bottom_right=cam_mask_overlay_rgb,
                    labels=["Original", "Mask Overlay", "Heatmap", "CAM + Mask"],
                    text_lines=text_lines,
                )
            else:
                mask_rgb = None
                mask_overlay_rgb = None
                cam_mask_overlay_rgb = None
                panel = make_gradcam_panel(
                    original_rgb=input_rgb,
                    heatmap_rgb=heatmap_rgb,
                    overlay_rgb=overlay_rgb,
                    text_lines=text_lines,
                )

            filename = (
                f"{idx:05d}"
                f"_true-{short_label_name(true_label)}"
                f"_pred-{short_label_name(pred_label)}"
                f"_prob-{pred_prob:.4f}.png"
            )
            sample_dir = ensure_dir(save_root / group)
            save_path = sample_dir / filename
            save_rgb_image(panel, save_path)

            stem_dir = ensure_dir(sample_dir / filename.replace(".png", ""))
            original_out = stem_dir / "00_original.png"
            heatmap_out = stem_dir / "01_heatmap.png"
            overlay_out = stem_dir / "02_overlay.png"
            save_rgb_image(input_rgb, original_out)
            save_rgb_image(heatmap_rgb, heatmap_out)
            save_rgb_image(overlay_rgb, overlay_out)
            extra_saved_files.update(
                {
                    "00_original.png": original_out.as_posix(),
                    "01_heatmap.png": heatmap_out.as_posix(),
                    "02_overlay.png": overlay_out.as_posix(),
                }
            )

            if mask_rgb is not None:
                mask_out = stem_dir / "03_mask.png"
                mask_overlay_out = stem_dir / "04_mask_overlay.png"
                cam_mask_overlay_out = stem_dir / "05_cam_mask_overlay.png"
                save_rgb_image(mask_rgb, mask_out)
                save_rgb_image(mask_overlay_rgb, mask_overlay_out)
                save_rgb_image(cam_mask_overlay_rgb, cam_mask_overlay_out)
                extra_saved_files.update(
                    {
                        "03_mask.png": mask_out.as_posix(),
                        "04_mask_overlay.png": mask_overlay_out.as_posix(),
                        "05_cam_mask_overlay.png": cam_mask_overlay_out.as_posix(),
                    }
                )

            record = {
                "index": idx,
                "group": group,
                "method": method,
                "true_label": true_label,
                "pred_label": pred_label,
                "pred_prob": pred_prob,
                "target_type": args.target_type,
                "target_class": target_class,
                "source_filepath": filepath,
                "saved_path": save_path.as_posix(),
                "saved_files": extra_saved_files,
                "metrics": overlap_metrics,
                "metadata": metadata,
                "mask_path": mask_path.as_posix() if mask_path is not None else None,
            }
            records.append(record)
            saved_counts[group] += 1

            print(
                f"[Saved] method={method} | group={group} | idx={idx} | "
                f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | {save_path}"
            )

    finally:
        if explainer is not None and hasattr(explainer, "remove_hooks"):
            explainer.remove_hooks()

    summary = {
        "dataset_name": dataset_name,
        "dataset_class": dataset_cls.__name__,
        "checkpoint": checkpoint_path.as_posix(),
        "split": args.split,
        "method": method,
        "target_type": args.target_type,
        "prepared_csv": prepared_csv_path.as_posix(),
        "saved_prepared_csv": bool(args.save_prepared_csv),
        "target_layer": (
            args.target_layer
            if method == "gradcam" and args.target_layer is not None
            else (
                str(resolved_target_layer)
                if method == "gradcam" and resolved_target_layer is not None
                else None
            )
        ),
        "rollout": {
            "head_fusion": args.head_fusion,
            "discard_ratio": args.discard_ratio,
            "start_layer": args.start_layer,
        }
        if method == "rollout"
        else None,
        "frequency": {
            "config": frequency_cfg,
            "run_artifacts": run_artifacts,
            "save_individual_frequency_images": bool(args.save_individual_frequency_images),
        }
        if method == "frequency"
        else None,
        "checkpoint_info": {
            "epoch": checkpoint.get("epoch", None) if isinstance(checkpoint, dict) else None,
            "best_epoch": checkpoint.get("best_epoch", None) if isinstance(checkpoint, dict) else None,
            "best_score": checkpoint.get("best_score", None) if isinstance(checkpoint, dict) else None,
        },
        "resolved_save_dir": save_root.as_posix(),
        "saved_counts": dict(saved_counts),
        "dataset_size": len(dataset),
        "records": records,
    }

    save_json(summary, save_root / "summary.json")

    print("=" * 80)
    print("Explanation Finished")
    print("=" * 80)
    print(f"method: {method}")
    print(f"dataset: {dataset_name}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"save_dir: {save_root}")
    print(f"saved: {dict(saved_counts)}")


if __name__ == "__main__":
    main()
