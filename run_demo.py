from __future__ import annotations

import argparse
import json
# import sys
from pathlib import Path
from typing import Any, Mapping

import matplotlib.pyplot as plt
import torch
from PIL import Image

# PROJECT_ROOT = Path(__file__).resolve().parent
# SRC_ROOT = PROJECT_ROOT / "src"
# for p in (PROJECT_ROOT, SRC_ROOT):
#     if str(p) not in sys.path:
#         sys.path.insert(0, str(p))

from deepfake_fusion.models.build_model import build_model
from deepfake_fusion.transforms.image_aug import build_eval_transform
from deepfake_fusion.utils.config import load_experiment_config, resolve_path
from deepfake_fusion.visualization.gradcam import (
    GradCAM,
    apply_colormap_to_cam,
    denormalize_image_tensor,
    make_gradcam_panel,
    overlay_cam_on_image,
    resolve_target_layer,
    save_rgb_image,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fusion demo: predict real/fake and save Grad-CAM panel"
    )

    parser.add_argument("--image_dir", type=str, default="data/demo")
    parser.add_argument("--save_dir", type=str, default="outputs/demo")

    parser.add_argument("--data_config", type=str, default="configs/data/genimage.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model/fusion.yaml")
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/fusion_resnet_spai_genimage.yaml",
    )

    # 기본은 train config의 experiment.output_dir/best.pth 사용
    parser.add_argument("--checkpoint", type=str, default=None)

    parser.add_argument("--device", type=str, default=None, help="cuda / cpu / mps")
    parser.add_argument(
        "--target_layer",
        type=str,
        default="spatial_branch.backbone.layer4.1",
    )
    parser.add_argument("--alpha", type=float, default=0.35)
    parser.add_argument("--show", action="store_true")
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


def resolve_device(device_name: str | None, cfg: Any | None = None) -> torch.device:
    requested = device_name or _cfg_get(cfg, "experiment", "device", default="cuda")
    name = str(requested).lower()

    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")

    if (
        name == "mps"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_checkpoint_path(cfg: Any, checkpoint_arg: str | None) -> Path:
    if checkpoint_arg is not None:
        ckpt = resolve_path(checkpoint_arg)
        if not ckpt.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
        return ckpt

    output_dir = _cfg_get(cfg, "experiment", "output_dir", default=None)
    if output_dir is None:
        raise ValueError(
            "--checkpoint is required because cfg.experiment.output_dir is missing."
        )

    ckpt = resolve_path(output_dir) / "best.pth"
    if not ckpt.exists():
        raise FileNotFoundError(
            f"Default checkpoint not found: {ckpt}\n"
            "Train the fusion model first or pass --checkpoint explicitly."
        )
    return ckpt


def load_checkpoint_to_model(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
    strict: bool = True,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
    else:
        model.load_state_dict(checkpoint, strict=strict)

    return checkpoint


def get_probs_and_pred(logits: torch.Tensor) -> tuple[float, float, int]:
    """
    returns: (real_prob, fake_prob, pred_label)
    assumes class 0=real, 1=fake
    """
    if logits.ndim == 1:
        fake_prob = float(torch.sigmoid(logits[0]).item())
        real_prob = 1.0 - fake_prob
        pred = int(fake_prob >= 0.5)
        return real_prob, fake_prob, pred

    if logits.ndim == 2 and logits.size(1) == 1:
        fake_prob = float(torch.sigmoid(logits[0, 0]).item())
        real_prob = 1.0 - fake_prob
        pred = int(fake_prob >= 0.5)
        return real_prob, fake_prob, pred

    if logits.ndim == 2 and logits.size(1) >= 2:
        probs = torch.softmax(logits, dim=1)[0]
        real_prob = float(probs[0].item())
        fake_prob = float(probs[1].item())
        pred = int(torch.argmax(probs).item())
        return real_prob, fake_prob, pred

    raise ValueError(f"Unsupported logits shape: {tuple(logits.shape)}")


def short_label_name(label: int) -> str:
    return "real" if int(label) == 0 else "fake"


def list_images(image_dir: Path) -> list[Path]:
    return sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def build_demo_transform(cfg: Any):
    input_size = _cfg_get(cfg, "data", "image", "input_size", default=224)
    mean = _cfg_get(cfg, "data", "image", "mean", default=(0.485, 0.456, 0.406))
    std = _cfg_get(cfg, "data", "image", "std", default=(0.229, 0.224, 0.225))

    transform = build_eval_transform(
        input_size=input_size,
        mean=mean,
        std=std,
    )
    return transform, tuple(mean), tuple(std)


def save_json(obj: dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = parse_args()

    image_dir = resolve_path(args.image_dir)
    save_dir = ensure_dir(resolve_path(args.save_dir))

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    image_paths = list_images(image_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No images found in: {image_dir}\n"
            f"Supported extensions: {sorted(IMAGE_EXTS)}"
        )

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )

    device = resolve_device(args.device, cfg)
    checkpoint_path = resolve_checkpoint_path(cfg, args.checkpoint)

    model = build_model(cfg.model).to(device)
    load_checkpoint_to_model(model, checkpoint_path, device=device, strict=True)
    model.eval()

    transform, mean, std = build_demo_transform(cfg)
    target_layer = resolve_target_layer(model, args.target_layer)
    cam_explainer = GradCAM(model=model, target_layer=target_layer)

    model_name = str(_cfg_get(cfg, "model", "name", default="unknown")).lower()

    print("=" * 80)
    print("Demo configuration")
    print("=" * 80)
    print(f"image_dir     : {image_dir}")
    print(f"save_dir      : {save_dir}")
    print(f"device        : {device}")
    print(f"model_name    : {model_name}")
    print(f"checkpoint    : {checkpoint_path}")
    print(f"target_layer  : {args.target_layer}")
    print(f"num_images    : {len(image_paths)}")
    print()

    try:
        for idx, image_path in enumerate(image_paths, start=1):
            image = Image.open(image_path).convert("RGB")
            x = transform(image).unsqueeze(0).to(device)

            cam_result = cam_explainer.generate(x, target_class=None)
            logits = cam_result["logits"]
            cam = cam_result["cam"]

            real_prob, fake_prob, pred_label = get_probs_and_pred(logits)
            pred_name = short_label_name(pred_label)

            original_rgb = denormalize_image_tensor(
                image_tensor=x[0].detach().cpu(),
                mean=mean,
                std=std,
            )
            heatmap_rgb = apply_colormap_to_cam(cam)
            overlay_rgb = overlay_cam_on_image(
                image_rgb=original_rgb,
                cam=cam,
                alpha=float(args.alpha),
            )

            text_lines = [
                f"file={image_path.name}",
                f"pred={pred_name} | real_prob={real_prob:.4f} | fake_prob={fake_prob:.4f}",
                f"model={model_name} | ckpt={checkpoint_path.name}",
            ]
            panel_rgb = make_gradcam_panel(
                original_rgb=original_rgb,
                heatmap_rgb=heatmap_rgb,
                overlay_rgb=overlay_rgb,
                text_lines=text_lines,
            )

            stem = image_path.stem
            sample_dir = ensure_dir(save_dir / stem)

            save_rgb_image(original_rgb, sample_dir / "original.png")
            save_rgb_image(heatmap_rgb, sample_dir / "heatmap.png")
            save_rgb_image(overlay_rgb, sample_dir / "overlay.png")
            save_rgb_image(panel_rgb, sample_dir / "panel.png")

            result = {
                "file_name": image_path.name,
                "file_path": str(image_path),
                "pred_label": int(pred_label),
                "pred_name": pred_name,
                "real_prob": real_prob,
                "fake_prob": fake_prob,
                "model_name": model_name,
                "checkpoint": str(checkpoint_path),
                "target_layer": args.target_layer,
                "saved_files": {
                    "original": str(sample_dir / "original.png"),
                    "heatmap": str(sample_dir / "heatmap.png"),
                    "overlay": str(sample_dir / "overlay.png"),
                    "panel": str(sample_dir / "panel.png"),
                },
            }
            save_json(result, sample_dir / "result.json")

            print(
                f"[{idx}/{len(image_paths)}] {image_path.name} -> "
                f"{pred_name} (real={real_prob:.4f}, fake={fake_prob:.4f})"
            )

            if args.show:
                plt.figure(figsize=(14, 5))
                plt.imshow(panel_rgb)
                plt.axis("off")
                plt.title(f"{image_path.name} -> {pred_name}")
                plt.tight_layout()
                plt.show()

    finally:
        cam_explainer.remove_hooks()

    print()
    print(f"Done. Results saved to: {save_dir}")


if __name__ == "__main__":
    main()