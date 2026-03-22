from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Type

import wandb
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from src.deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from src.deepfake_fusion.engine.trainer import Trainer
from src.deepfake_fusion.models.build_model import build_model, get_model_summary
from src.deepfake_fusion.transforms.image_aug import build_transforms_from_config
from src.deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
    save_yaml,
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
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run W&B sweep training.")
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
        "--device",
        type=str,
        default=None,
        help="Optional device override. Example: cuda, cpu, mps",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, allow_nan=True)


def path_exists(path_str: str) -> bool:
    try:
        return resolve_path(path_str).exists()
    except Exception:
        return False


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
    drop_last: bool,
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
        drop_last=drop_last,
        worker_init_fn=seed_worker if use_workers else None,
        generator=get_torch_generator(seed),
        persistent_workers=use_workers,
    )


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


def build_datasets_and_loaders(cfg, seed: int):
    transforms = build_transforms_from_config(cfg)

    batch_size = int(cfg.data.dataloader.batch_size)
    num_workers = int(cfg.data.dataloader.num_workers)
    pin_memory = bool(cfg.data.dataloader.pin_memory)
    drop_last = bool(cfg.data.dataloader.drop_last)

    root_dir = cfg.data.paths.root_dir
    dataset_cls = get_dataset_class(cfg)

    train_dataset = build_single_dataset(
        dataset_cls=dataset_cls,
        csv_path=cfg.data.paths.train_csv,
        root_dir=root_dir,
        transform=transforms["train"],
    )
    train_loader = build_loader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        shuffle=bool(cfg.data.train.shuffle),
        seed=seed,
    )

    val_loader = None
    if path_exists(cfg.data.paths.val_csv):
        val_dataset = build_single_dataset(
            dataset_cls=dataset_cls,
            csv_path=cfg.data.paths.val_csv,
            root_dir=root_dir,
            transform=transforms["val"],
        )
        val_loader = build_loader(
            dataset=val_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            shuffle=bool(cfg.data.val.shuffle),
            seed=seed,
        )

    test_loader = None
    if path_exists(cfg.data.paths.test_csv):
        test_dataset = build_single_dataset(
            dataset_cls=dataset_cls,
            csv_path=cfg.data.paths.test_csv,
            root_dir=root_dir,
            transform=transforms["test"],
        )
        test_loader = build_loader(
            dataset=test_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
            shuffle=bool(cfg.data.test.shuffle),
            seed=seed,
        )

    return dataset_cls, train_dataset, train_loader, val_loader, test_loader


def apply_sweep_overrides(cfg, sweep_cfg) -> Any:
    """
    wandb.config 값으로 base config override.
    default_sweep.yaml의 parameter 이름과 맞춰서 작성.
    """
    if not hasattr(cfg.train, "augmentation"):
        cfg.train.augmentation = {}

    if "lr" in sweep_cfg:
        cfg.train.optimizer.lr = float(sweep_cfg.lr)

    if "weight_decay" in sweep_cfg:
        cfg.train.optimizer.weight_decay = float(sweep_cfg.weight_decay)

    if "batch_size" in sweep_cfg:
        cfg.data.dataloader.batch_size = int(sweep_cfg.batch_size)

    if "dropout" in sweep_cfg:
        cfg.model.head.dropout = float(sweep_cfg.dropout)

    if "epochs" in sweep_cfg:
        cfg.train.train.epochs = int(sweep_cfg.epochs)
        if hasattr(cfg.train, "scheduler") and hasattr(cfg.train.scheduler, "t_max"):
            cfg.train.scheduler.t_max = int(sweep_cfg.epochs)

    if "seed" in sweep_cfg:
        cfg.train.experiment.seed = int(sweep_cfg.seed)

    if "rotation_degrees" in sweep_cfg:
        cfg.train.augmentation["rotation_degrees"] = float(sweep_cfg.rotation_degrees)

    if "color_jitter_prob" in sweep_cfg:
        cfg.train.augmentation["color_jitter_prob"] = float(sweep_cfg.color_jitter_prob)

    return cfg


def attach_unique_output_dir(cfg, run_id: str) -> Any:
    """
    sweep run끼리 checkpoint가 덮어쓰기 되지 않도록 run별 output_dir 부여.
    """
    base_output_dir = Path(cfg.train.experiment.output_dir)
    cfg.train.experiment.output_dir = str(base_output_dir / "sweeps" / run_id)
    return cfg


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )

    if args.device is not None:
        cfg.train.experiment.device = args.device

    wandb_run = wandb.init(
        project=cfg.train.wandb.project if hasattr(cfg.train, "wandb") else None,
        entity=cfg.train.wandb.entity if hasattr(cfg.train, "wandb") else None,
        tags=list(cfg.train.wandb.tags)
        if hasattr(cfg.train, "wandb") and cfg.train.wandb.tags is not None
        else None,
        mode=cfg.train.wandb.mode if hasattr(cfg.train, "wandb") else "online",
        job_type="sweep",
    )

    try:
        cfg = apply_sweep_overrides(cfg, wandb.config)
        cfg = attach_unique_output_dir(cfg, wandb_run.id)

        output_dir = ensure_dir(resolve_path(cfg.train.experiment.output_dir))
        save_yaml(cfg.to_dict(), output_dir / "config_merged.yaml")

        print("=" * 80)
        print("Merged Config (Sweep)")
        print("=" * 80)
        print(pretty_print_config(cfg))

        seed = int(cfg.train.experiment.seed)
        seed_everything(seed)

        dataset_cls, train_dataset, train_loader, val_loader, test_loader = (
            build_datasets_and_loaders(
                cfg=cfg,
                seed=seed,
            )
        )

        print("=" * 80)
        print("Dataset Summary")
        print("=" * 80)
        print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
        print(f"dataset class: {dataset_cls.__name__}")
        print(f"train size: {len(train_dataset)}")
        print(f"train class counts: {train_dataset.class_counts}")

        if val_loader is not None:
            val_dataset = val_loader.dataset
            print(f"val size: {len(val_dataset)}")
            print(f"val class counts: {val_dataset.class_counts}")

        if test_loader is not None:
            test_dataset = test_loader.dataset
            print(f"test size: {len(test_dataset)}")
            print(f"test class counts: {test_dataset.class_counts}")

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
            wandb_run=wandb_run,
        )

        print("=" * 80)
        print("Start Sweep Training")
        print("=" * 80)

        history = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
        )

        history_path = output_dir / "history.json"
        save_json({"history": history}, history_path)

        best_ckpt_path = output_dir / "best.pth"
        last_ckpt_path = output_dir / "last.pth"

        final_results: Dict[str, Any] = {
            "dataset_name": getattr(cfg.data, "name", "unknown"),
            "dataset_class": dataset_cls.__name__,
            "best_checkpoint": best_ckpt_path.as_posix(),
            "last_checkpoint": last_ckpt_path.as_posix(),
            "best_epoch": trainer.best_epoch,
            "best_monitor": cfg.train.checkpoint.monitor,
            "best_score": trainer.best_score,
            "sweep_params": dict(wandb.config),
        }

        if best_ckpt_path.exists():
            trainer.load_checkpoint(best_ckpt_path, strict=True)

            if val_loader is not None:
                val_metrics = trainer.evaluate(val_loader, split="val")
                final_results["val_metrics"] = val_metrics
                print(f"[Best Val]  {format_metrics(val_metrics)}")

                if "auc" in val_metrics:
                    wandb.log({"final_val_auc": val_metrics["auc"]})
                if "accuracy" in val_metrics:
                    wandb.log({"final_val_accuracy": val_metrics["accuracy"]})

            if test_loader is not None:
                test_metrics = trainer.evaluate(test_loader, split="test")
                final_results["test_metrics"] = test_metrics
                print(f"[Best Test] {format_metrics(test_metrics)}")

                if "auc" in test_metrics:
                    wandb.log({"final_test_auc": test_metrics["auc"]})
                if "accuracy" in test_metrics:
                    wandb.log({"final_test_accuracy": test_metrics["accuracy"]})

        results_path = output_dir / "final_results.json"
        save_json(final_results, results_path)

        wandb_run.summary["dataset_name"] = getattr(cfg.data, "name", "unknown")
        wandb_run.summary["dataset_class"] = dataset_cls.__name__
        wandb_run.summary["best_epoch"] = trainer.best_epoch
        wandb_run.summary["best_score"] = trainer.best_score
        wandb_run.summary["best_monitor"] = cfg.train.checkpoint.monitor
        wandb_run.summary["output_dir"] = output_dir.as_posix()

        print("=" * 80)
        print("Sweep Run Finished")
        print("=" * 80)
        print(f"Output dir: {output_dir}")
        print(f"Best checkpoint: {best_ckpt_path}")
        print(f"Best {cfg.train.checkpoint.monitor}: {trainer.best_score:.6f}")

    finally:
        wandb.finish()


if __name__ == "__main__":
    main()