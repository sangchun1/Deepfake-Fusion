from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from src.deepfake_fusion.models.build_model import build_model
from src.deepfake_fusion.transforms.image_aug import build_image_transform
from src.deepfake_fusion.utils.config import load_experiment_config, pretty_print_config, resolve_path
from src.deepfake_fusion.utils.seed import seed_everything
from src.deepfake_fusion.visualization.gradcam import (
    GradCAM,
    apply_colormap_to_cam,
    denormalize_image_tensor,
    make_gradcam_panel,
    overlay_cam_on_image,
    resolve_target_layer,
    save_rgb_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grad-CAM visualization script.")

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
        default="configs/train/spatial.yaml",
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
        "--target_layer",
        type=str,
        default=None,
        help="Example: backbone.layer4.1",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Grad-CAM을 predicted class 기준으로 할지, true label 기준으로 할지 선택",
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
        "--save_dir",
        type=str,
        default="outputs/gradcam",
    )
    return parser.parse_args()


def resolve_device(device_name: str | None) -> torch.device:
    device_name = (device_name or "cuda").lower()

    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if device_name == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
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


def build_dataset(cfg, split: str) -> CIFAKEDataset:
    """
    시각화는 랜덤 augment 없이 deterministic transform 사용.
    train split을 보더라도 eval-style transform을 사용한다.
    """
    transform = build_image_transform(
        data_cfg=cfg.data,
        split="test",
        aug_cfg=None,
    )

    if split == "train":
        csv_path = cfg.data.paths.train_csv
    elif split == "val":
        csv_path = cfg.data.paths.val_csv
    elif split == "test":
        csv_path = cfg.data.paths.test_csv
    else:
        raise ValueError(f"Unsupported split: {split}")

    return CIFAKEDataset(
        csv_path=csv_path,
        root_dir=cfg.data.paths.root_dir,
        transform=transform,
    )


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

    dataset = build_dataset(cfg, args.split)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples found in split: {args.split}")

    model = build_model(cfg.model)
    load_checkpoint_to_model(
        model=model,
        checkpoint_path=checkpoint_path,
        device=device,
        strict=bool(cfg.model.checkpoint.strict),
    )
    model = model.to(device)
    model.eval()

    target_layer = resolve_target_layer(model, args.target_layer)
    print(f"Using target layer for Grad-CAM: {target_layer}")

    gradcam = GradCAM(model=model, target_layer=target_layer)

    mean = list(cfg.data.image.mean)
    std = list(cfg.data.image.std)

    save_root = ensure_dir(
        resolve_path(args.save_dir) / args.split / checkpoint_path.stem
    )

    target_groups = ["correct_real", "correct_fake", "wrong_real", "wrong_fake"]
    saved_counts = defaultdict(int)
    records: List[Dict[str, Any]] = []

    try:
        for idx in range(len(dataset)):
            if all(saved_counts[g] >= args.max_per_group for g in target_groups):
                break

            sample = dataset[idx]
            image_tensor = sample["image"]                 # [C, H, W]
            label_tensor = sample["label"]
            filepath = sample["filepath"]

            x = image_tensor.unsqueeze(0).to(device)
            true_label = int(label_tensor.item())

            with torch.no_grad():
                logits = model(x)
                pred_prob, pred_label = get_prob_and_pred(logits)

            group = categorize_case(true_label, pred_label)
            if group not in target_groups:
                continue
            if saved_counts[group] >= args.max_per_group:
                continue

            target_class = pred_label if args.target_type == "pred" else true_label
            result = gradcam.generate(x, target_class=target_class)

            input_rgb = denormalize_image_tensor(image_tensor, mean=mean, std=std)
            cam = result["cam"]
            heatmap_rgb = apply_colormap_to_cam(cam)
            overlay_rgb = overlay_cam_on_image(input_rgb, cam, alpha=args.alpha)

            text_lines = [
                f"split={args.split} | group={group} | idx={idx}",
                f"true={short_label_name(true_label)}({true_label}) | pred={short_label_name(pred_label)}({pred_label}) | prob={pred_prob:.4f}",
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
                f"[Saved] group={group} | idx={idx} | "
                f"true={true_label} pred={pred_label} prob={pred_prob:.4f} | "
                f"{save_path}"
            )

    finally:
        gradcam.remove_hooks()

    summary = {
        "checkpoint": checkpoint_path.as_posix(),
        "split": args.split,
        "target_type": args.target_type,
        "target_layer": args.target_layer if args.target_layer is not None else "auto",
        "resolved_save_dir": save_root.as_posix(),
        "saved_counts": dict(saved_counts),
        "records": records,
    }
    save_json(summary, save_root / "summary.json")

    print("=" * 80)
    print("Grad-CAM Finished")
    print("=" * 80)
    print(f"checkpoint: {checkpoint_path}")
    print(f"save_dir:   {save_root}")
    print(f"saved:      {dict(saved_counts)}")


if __name__ == "__main__":
    main()