from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Type

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from src.deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from src.deepfake_fusion.datasets.genimage_dataset import GenImageDataset
from src.deepfake_fusion.models.build_model import build_model
from src.deepfake_fusion.transforms.image_aug import build_image_transform
from src.deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
)
from src.deepfake_fusion.utils.seed import seed_everything
from src.deepfake_fusion.visualization.attention_rollout import AttentionRollout
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
        choices=["auto", "gradcam", "rollout"],
        help="Explanation method. 'auto' uses rollout for ViT, gradcam otherwise.",
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
    return parser.parse_args()


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


def get_dataset_class(cfg):
    dataset_key = None

    if getattr(cfg.data, "dataset_class", None) is not None:
        dataset_key = str(cfg.data.dataset_class)
    elif getattr(cfg.data, "name", None) is not None:
        dataset_key = str(cfg.data.name)

    if dataset_key is None:
        raise ValueError(
            "Could not determine dataset class. Set cfg.data.dataset_class "
            "or cfg.data.name in the data config."
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


def build_dataset(cfg, split: str):
    """
    시각화는 랜덤 augment 없이 deterministic transform 사용.
    train split을 보더라도 eval-style transform을 사용한다.
    """
    transform = build_image_transform(
        data_cfg=cfg.data,
        split="test",
        aug_cfg=None,
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

    raise ValueError(f"Unsupported explanation method: {method}")


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

    dataset_cls, dataset = build_dataset(cfg, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in split: {args.split}")

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"size: {len(dataset)}")
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
    )

    target_groups = ["correct_real", "correct_fake", "wrong_real", "wrong_fake"]
    saved_counts = defaultdict(int)
    records: List[Dict[str, Any]] = []

    try:
        for idx in range(len(dataset)):
            if all(saved_counts[g] >= args.max_per_group for g in target_groups):
                break

            sample = dataset[idx]
            image_tensor = sample["image"]
            label_tensor = sample["label"]
            filepath = sample["filepath"]

            x = image_tensor.unsqueeze(0).to(device)
            true_label = int(label_tensor.item())

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
                "saved_path": save_path.as_posix(),
            }
            records.append(record)
            saved_counts[group] += 1

            print(
                f"[Saved] method={method} | group={group} | idx={idx} | "
                f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | "
                f"{save_path}"
            )

    finally:
        if explainer is not None:
            explainer.remove_hooks()

    summary = {
        "dataset_name": dataset_name,
        "dataset_class": dataset_cls.__name__,
        "checkpoint": checkpoint_path.as_posix(),
        "split": args.split,
        "method": method,
        "target_type": args.target_type,
        "target_layer": (
            args.target_layer if method == "gradcam" and args.target_layer is not None
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
        "resolved_save_dir": save_root.as_posix(),
        "saved_counts": dict(saved_counts),
        "records": records,
    }
    save_json(summary, save_root / "summary.json")

    print("=" * 80)
    print("Explanation Finished")
    print("=" * 80)
    print(f"method:     {method}")
    print(f"dataset:    {dataset_name}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"save_dir:   {save_root}")
    print(f"saved:      {dict(saved_counts)}")


if __name__ == "__main__":
    main()