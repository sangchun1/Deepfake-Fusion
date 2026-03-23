from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Type

from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from src.deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from src.deepfake_fusion.datasets.genimage_dataset import GenImageDataset
from src.deepfake_fusion.engine.trainer import Trainer
from src.deepfake_fusion.models.build_model import build_model, get_model_summary
from src.deepfake_fusion.transforms.image_aug import build_transforms_from_config
from src.deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
)
from src.deepfake_fusion.utils.seed import (
    get_torch_generator,
    seed_everything,
    seed_worker,
)

DATASET_REGISTRY: Dict[str, Type] = {
    "cifake": CIFAKEDataset,
    "CIFAKEDataset": CIFAKEDataset,
    "face130k": FACE130KDataset,
    "FACE130KDataset": FACE130KDataset,
    "genimage": GenImageDataset,
    "GenImageDataset": GenImageDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained deepfake detector.")

    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/cifake.yaml",
        help="Path to data config YAML.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
        help="Path to model config YAML.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_cifake.yaml",
        help="Path to train config YAML.",
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
        help="Which split to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Override device from config. Example: cuda, cpu, mps",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Override dataloader batch size.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=None,
        help="Override dataloader num_workers.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save evaluation result JSON.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def format_metrics(metrics: Dict[str, Any]) -> str:
    parts = []
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


def build_loader(
    dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
):
    use_workers = num_workers > 0

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        worker_init_fn=seed_worker if use_workers else None,
        generator=get_torch_generator(seed),
        persistent_workers=use_workers,
    )


def get_split_csv_path(cfg, split: str) -> str:
    if split == "train":
        return cfg.data.paths.train_csv
    if split == "val":
        return cfg.data.paths.val_csv
    if split == "test":
        return cfg.data.paths.test_csv
    raise ValueError(f"Unsupported split: {split}")


def get_split_shuffle(cfg, split: str) -> bool:
    if split == "train":
        return bool(cfg.data.train.shuffle)
    if split == "val":
        return bool(cfg.data.val.shuffle)
    if split == "test":
        return bool(cfg.data.test.shuffle)
    raise ValueError(f"Unsupported split: {split}")


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


def build_single_dataset(dataset_cls, csv_path, root_dir, transform):
    return dataset_cls(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
    )


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )

    if args.device is not None:
        cfg.train.experiment.device = args.device
    if args.batch_size is not None:
        cfg.data.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.dataloader.num_workers = args.num_workers

    seed = int(cfg.train.experiment.seed)
    seed_everything(seed)

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    transforms = build_transforms_from_config(cfg)
    split_csv = get_split_csv_path(cfg, args.split)
    split_csv_path = resolve_path(split_csv)

    if not split_csv_path.exists():
        raise FileNotFoundError(f"{args.split} split CSV not found: {split_csv_path}")

    dataset_cls = get_dataset_class(cfg)
    dataset = build_single_dataset(
        dataset_cls=dataset_cls,
        csv_path=split_csv,
        root_dir=cfg.data.paths.root_dir,
        transform=transforms[args.split],
    )

    loader = build_loader(
        dataset=dataset,
        batch_size=int(cfg.data.dataloader.batch_size),
        num_workers=int(cfg.data.dataloader.num_workers),
        pin_memory=bool(cfg.data.dataloader.pin_memory),
        shuffle=get_split_shuffle(cfg, args.split),
        seed=seed,
    )

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"size: {len(dataset)}")
    print(f"class counts: {dataset.class_counts}")

    model = build_model(cfg.model)
    model_summary = get_model_summary(model)

    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    for key, value in model_summary.items():
        print(f"{key}: {value}")

    trainer = Trainer(
        model=model,
        train_cfg=cfg.train,
        device=cfg.train.experiment.device,
    )

    checkpoint_path = (
        resolve_path(args.checkpoint)
        if args.checkpoint is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / "best.pth")
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print("=" * 80)
    print("Load Checkpoint")
    print("=" * 80)
    print(f"checkpoint: {checkpoint_path}")

    checkpoint = trainer.load_checkpoint(checkpoint_path, strict=True)

    print("=" * 80)
    print(f"Evaluate: {args.split}")
    print("=" * 80)

    metrics = trainer.evaluate(loader, split=args.split)
    print(format_metrics(metrics))

    result = {
        "dataset_name": getattr(cfg.data, "name", "unknown"),
        "dataset_class": dataset_cls.__name__,
        "split": args.split,
        "checkpoint": checkpoint_path.as_posix(),
        "metrics": metrics,
        "dataset_size": len(dataset),
        "class_counts": dataset.class_counts,
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
    }

    output_json = (
        resolve_path(args.output_json)
        if args.output_json is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / f"eval_{args.split}.json")
    )
    save_json(result, output_json)

    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)
    print(f"Saved result to: {output_json}")


if __name__ == "__main__":
    main()