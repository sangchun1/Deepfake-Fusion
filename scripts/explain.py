from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Type

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from src.deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from src.deepfake_fusion.datasets.genimage_dataset import GenImageDataset
from src.deepfake_fusion.models.build_model import build_model
from src.deepfake_fusion.transforms.robustness import (
    build_clean_eval_transform,
    build_corrupted_eval_transform,
    get_corruption_params,
)
from src.deepfake_fusion.utils.config import (
    load_experiment_config,
    load_yaml,
    pretty_print_config,
    resolve_path,
)
from src.deepfake_fusion.utils.seed import seed_everything
from src.deepfake_fusion.visualization.attention_rollout import AttentionRollout
from src.deepfake_fusion.visualization.frequency_visualize import (
    build_frequency_metrics,
    build_frequency_visuals,
    save_frequency_run_artifacts,
    save_frequency_sample_artifacts,
)
from src.deepfake_fusion.visualization.gradcam import (
    GradCAM,
    apply_colormap_to_cam,
    denormalize_image_tensor,
    make_gradcam_panel,
    overlay_cam_on_image,
    resolve_target_layer,
    save_rgb_image,
)

DATASET_REGISTRY: Dict[str, Type] = {
    "cifake": CIFAKEDataset,
    "CIFAKEDataset": CIFAKEDataset,
    "face130k": FACE130KDataset,
    "FACE130KDataset": FACE130KDataset,
    "genimage": GenImageDataset,
    "GenImageDataset": GenImageDataset,
    # OpenFake를 GenImageDataset 로더로 처리
    "openfake": GenImageDataset,
    "OpenFakeDataset": GenImageDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Explanation visualization script.")
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/cifake.yaml",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_resnet_cifake.yaml",
    )
    parser.add_argument(
        "--robustness_config",
        type=str,
        default="configs/train/robustness.yaml",
        help="Path to robustness config YAML. Used only when corruption != clean.",
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
        choices=["train", "val", "test"],
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
        default="outputs/explain",
    )
    parser.add_argument(
        "--save_individual_frequency_images",
        action="store_true",
        help="frequency method에서 panel 외 개별 이미지도 함께 저장",
    )
    parser.add_argument(
        "--corruption",
        type=str,
        default="clean",
        help="Corruption name for robustness explain. Example: clean, jpeg, gaussian_blur, gaussian_noise, resize_down_up",
    )
    parser.add_argument(
        "--severity",
        type=int,
        default=1,
        help="Corruption severity level (1-based). Ignored when corruption=clean.",
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
    if hasattr(value, "item") and callable(getattr(value, "item", None)):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _normalize_corruption_name(name: Optional[str]) -> str:
    if name is None:
        return "clean"

    name = str(name).strip().lower()
    aliases = {
        "clean": "clean",
        "none": "clean",
        "jpeg": "jpeg",
        "jpg": "jpeg",
        "resize": "resize_down_up",
        "resize_down_up": "resize_down_up",
        "down_up_resize": "resize_down_up",
        "blur": "gaussian_blur",
        "gaussian_blur": "gaussian_blur",
        "noise": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "brightness_contrast": "brightness_contrast",
        "color": "brightness_contrast",
    }
    return aliases.get(name, name)


def is_clean_corruption(name: Optional[str]) -> bool:
    return _normalize_corruption_name(name) == "clean"


def build_condition_name(corruption_name: str, severity: int) -> str:
    normalized = _normalize_corruption_name(corruption_name)
    if normalized == "clean":
        return "clean"
    return f"{normalized}_s{int(severity)}"


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
        json.dump(data, f, indent=2, ensure_ascii=False)


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
    """
    binary / multiclass logits 지원.

    반환:
        prob: predicted class confidence 또는 positive probability
        pred: predicted class index
    """
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
    """
    frequency 설명에 필요한 (feature_model, frequency_encoder) 반환.

    - SPAI:
        feature_model = model
        frequency_encoder = model.frequency_encoder

    - Fusion:
        feature_model = model
        frequency_encoder = model.spectral_branch.frequency_encoder
    """
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


def get_dataset_class(cfg):
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


def get_split_csv_path(cfg, split: str) -> str:
    if split == "train":
        return cfg.data.paths.train_csv
    if split == "val":
        return cfg.data.paths.val_csv
    if split == "test":
        return cfg.data.paths.test_csv
    raise ValueError(f"Unsupported split: {split}")


def resolve_robustness_cfg(args: argparse.Namespace) -> Optional[Any]:
    corruption_name = _normalize_corruption_name(args.corruption)
    if corruption_name == "clean":
        return None

    robustness_path = resolve_path(args.robustness_config)
    if not robustness_path.exists():
        raise FileNotFoundError(
            f"Robustness config not found: {robustness_path}"
        )
    return load_yaml(robustness_path)


def resolve_corruption_params(
    corruption_name: str,
    severity: int,
    robustness_cfg: Optional[Any],
) -> Dict[str, Any]:
    normalized = _normalize_corruption_name(corruption_name)
    if normalized == "clean":
        return {"name": "clean"}

    if robustness_cfg is None:
        raise ValueError("robustness_cfg is required when corruption != clean")

    return get_corruption_params(
        robustness_cfg=robustness_cfg,
        corruption_name=normalized,
        severity=int(severity),
    )


def build_dataset(
    cfg,
    split: str,
    corruption_name: str = "clean",
    severity: int = 1,
    robustness_cfg: Optional[Any] = None,
):
    """
    explain 시각화용 dataset 생성.

    - clean: deterministic eval transform
    - corrupted: corruption 적용 후 eval transform
    """
    normalized_corruption = _normalize_corruption_name(corruption_name)

    if normalized_corruption == "clean":
        transform = build_clean_eval_transform(cfg.data)
    else:
        transform = build_corrupted_eval_transform(
            data_cfg=cfg.data,
            corruption_name=normalized_corruption,
            severity=int(severity),
            robustness_cfg=robustness_cfg,
        )

    csv_path = get_split_csv_path(cfg, split)
    csv_path_resolved = resolve_path(csv_path)
    if not csv_path_resolved.exists():
        raise FileNotFoundError(f"{split} split CSV not found: {csv_path_resolved}")

    dataset_cls = get_dataset_class(cfg)
    dataset = dataset_cls(
        csv_path=csv_path,
        root_dir=cfg.data.paths.root_dir,
        transform=transform,
    )
    return dataset_cls, dataset


def infer_explain_method(cfg, requested_method: str) -> str:
    if requested_method != "auto":
        return requested_method

    model_name = str(getattr(cfg.model, "name", "")).lower()
    backbone = getattr(cfg.model, "backbone", None)
    backbone_name = str(getattr(backbone, "name", "")).lower()

    if model_name == "spai":
        return "frequency"

    if model_name == "fusion":
        # v1 기본값: fusion은 spatial branch 기준 Grad-CAM
        # spectral branch를 보고 싶으면 --method frequency 사용
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


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )
    robustness_cfg = resolve_robustness_cfg(args)
    normalized_corruption = _normalize_corruption_name(args.corruption)
    condition_name = build_condition_name(normalized_corruption, args.severity)
    corruption_params = resolve_corruption_params(
        corruption_name=normalized_corruption,
        severity=args.severity,
        robustness_cfg=robustness_cfg,
    )

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    if robustness_cfg is not None:
        print("=" * 80)
        print("Robustness Config")
        print("=" * 80)
        print(pretty_print_config(robustness_cfg))

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

    dataset_cls, dataset = build_dataset(
        cfg=cfg,
        split=args.split,
        corruption_name=normalized_corruption,
        severity=args.severity,
        robustness_cfg=robustness_cfg,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in split: {args.split}")

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"size: {len(dataset)}")
    print(f"corruption: {normalized_corruption}")
    print(f"condition: {condition_name}")
    print(f"corruption params: {corruption_params}")
    if hasattr(dataset, "class_counts"):
        print(f"class counts: {dataset.class_counts}")

    model = build_model(cfg.model)
    load_checkpoint_to_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=bool(cfg.model.checkpoint.strict),
    )
    model = model.to(device)
    model.eval()

    method = infer_explain_method(cfg, args.method)
    explainer, resolved_target_layer = build_explainer(model, method, args)

    mean = list(cfg.data.image.mean)
    std = list(cfg.data.image.std)
    dataset_name = getattr(cfg.data, "name", "unknown")
    save_root = ensure_dir(
        resolve_path(args.save_dir)
        / method
        / dataset_name
        / args.split
        / checkpoint_path.stem
        / condition_name
    )

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
                        "corruption": normalized_corruption,
                        "severity": 0 if normalized_corruption == "clean" else int(args.severity),
                        "condition": condition_name,
                        "corruption_params": corruption_params,
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
                metrics["corruption"] = normalized_corruption
                metrics["severity"] = 0 if normalized_corruption == "clean" else int(args.severity)
                metrics["condition"] = condition_name
                metrics["corruption_params"] = corruption_params

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
                    "corruption": normalized_corruption,
                    "severity": 0 if normalized_corruption == "clean" else int(args.severity),
                    "condition": condition_name,
                    "corruption_params": corruption_params,
                    "saved_dir": sample_dir.as_posix(),
                    "saved_files": saved_files,
                    "metrics": metrics,
                }
                records.append(record)
                saved_counts[group] += 1

                print(
                    f"[Saved] method={method} | condition={condition_name} | "
                    f"group={group} | idx={idx} | "
                    f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | "
                    f"{sample_dir}"
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
                f"target_type={args.target_type} | target_class={target_class}",
                f"condition={condition_name}",
                f"path={Path(filepath).name}",
            ]

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
            save_path = save_root / group / filename
            save_rgb_image(panel, save_path)

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
                "corruption": normalized_corruption,
                "severity": 0 if normalized_corruption == "clean" else int(args.severity),
                "condition": condition_name,
                "corruption_params": corruption_params,
                "saved_path": save_path.as_posix(),
            }
            records.append(record)
            saved_counts[group] += 1

            print(
                f"[Saved] method={method} | condition={condition_name} | "
                f"group={group} | idx={idx} | "
                f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | "
                f"{save_path}"
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
        "corruption": normalized_corruption,
        "severity": 0 if normalized_corruption == "clean" else int(args.severity),
        "condition": condition_name,
        "corruption_params": corruption_params,
        "robustness_config": (
            resolve_path(args.robustness_config).as_posix()
            if robustness_cfg is not None
            else None
        ),
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
        "resolved_save_dir": save_root.as_posix(),
        "saved_counts": dict(saved_counts),
        "records": records,
    }

    save_json(summary, save_root / "summary.json")

    print("=" * 80)
    print("Explanation Finished")
    print("=" * 80)
    print(f"method: {method}")
    print(f"dataset: {dataset_name}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"condition: {condition_name}")
    print(f"save_dir: {save_root}")
    print(f"saved: {dict(saved_counts)}")


if __name__ == "__main__":
    main()