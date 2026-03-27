from __future__ import annotations

import argparse
import csv
import json
import math
# import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Type

import numpy as np
import torch
from torch.utils.data import DataLoader

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_fusion.datasets.cifake_dataset import CIFAKEDataset
from deepfake_fusion.datasets.face130k_dataset import FACE130KDataset
from archive.genimage_dataset import GenImageDataset
from deepfake_fusion.engine.trainer import Trainer
from deepfake_fusion.metrics.classification import (
    ClassificationMeter,
    logits_to_probs,
    probs_to_preds,
)
from deepfake_fusion.models.build_model import build_model, get_model_summary
from deepfake_fusion.transforms.robustness import (
    build_clean_eval_transform,
    build_corrupted_eval_transform,
    get_benchmark_severities,
    get_corruption_params,
    get_enabled_corruptions,
)
from deepfake_fusion.utils.config import (
    load_experiment_config,
    load_yaml,
    pretty_print_config,
    resolve_path,
)
from deepfake_fusion.utils.seed import (
    get_torch_generator,
    seed_everything,
    seed_worker,
)

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DATASET_REGISTRY: Dict[str, Type] = {
    "cifake": CIFAKEDataset,
    "CIFAKEDataset": CIFAKEDataset,
    "face130k": FACE130KDataset,
    "FACE130KDataset": FACE130KDataset,
    "genimage": GenImageDataset,
    "GenImageDataset": GenImageDataset,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate robustness of a trained deepfake detector."
    )
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
        default="configs/train/spatial_resnet_cifake.yaml",
        help="Path to train config YAML.",
    )
    parser.add_argument(
        "--robustness_config",
        type=str,
        default="configs/train/robustness.yaml",
        help="Path to robustness config YAML.",
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
        choices=["val", "test"],
        help="Which split to evaluate for robustness.",
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
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Directory to save robustness results. "
            "Defaults to <experiment_output_dir>/robustness_<split>."
        ),
    )
    parser.add_argument(
        "--corruptions",
        type=str,
        default=None,
        help=(
            "Comma-separated corruption names to evaluate. "
            "Default: all enabled corruptions from robustness config."
        ),
    )
    parser.add_argument(
        "--severities",
        type=str,
        default=None,
        help=(
            "Comma-separated severity levels. "
            "Default: benchmark.severities from robustness config."
        ),
    )
    parser.add_argument(
        "--save_predictions",
        action="store_true",
        help="Save per-sample predictions for each condition.",
    )
    parser.add_argument(
        "--non_strict",
        action="store_true",
        help="Load checkpoint with strict=False.",
    )
    return parser.parse_args()


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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def parse_name_list(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [_normalize_corruption_name(part) for part in str(raw).split(",") if part.strip()]
    return items or None


def parse_int_list(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None:
        return None
    items = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    return items or None


def to_serializable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.ndim == 0:
            return obj.item()
        return obj.detach().cpu().tolist()
    return obj


def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(data), f, indent=2, ensure_ascii=False, allow_nan=True)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    ensure_parent_dir(path)

    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["condition"])
        return

    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            safe_row = {}
            for key, value in row.items():
                serializable = to_serializable(value)
                if isinstance(serializable, (dict, list)):
                    safe_row[key] = json.dumps(serializable, ensure_ascii=False)
                else:
                    safe_row[key] = serializable
            writer.writerow(safe_row)


def format_metrics(metrics: Dict[str, Any]) -> str:
    parts: List[str] = []
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
            "Could not determine dataset class. "
            "Set cfg.data.dataset_class or cfg.data.name in the data config."
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


@torch.no_grad()
def evaluate_loader_with_details(
    trainer: Trainer,
    loader,
    split: str = "test",
    save_predictions: bool = False,
) -> Dict[str, Any]:
    trainer.model.eval()
    meter = ClassificationMeter()
    prediction_rows: List[Dict[str, Any]] = []

    iterator = loader
    if trainer.use_tqdm and tqdm is not None:
        iterator = tqdm(loader, desc=f"{split.capitalize()} ", leave=False)

    for batch in iterator:
        filepaths = batch.get("filepath", None)

        batch = trainer._move_batch_to_device(batch)
        images = batch["image"]
        labels = batch["label"]

        with trainer._autocast_context():
            logits = trainer.model(images)
            loss = trainer.criterion(logits, labels)

        meter.update(
            logits=logits.detach(),
            targets=labels.detach(),
            loss=loss.detach(),
            threshold=trainer.threshold,
        )

        if trainer.use_tqdm and tqdm is not None:
            iterator.set_postfix(loss=f"{meter.loss_meter.avg:.4f}")

        if save_predictions:
            probs = logits_to_probs(logits.detach())
            preds = probs_to_preds(probs, threshold=trainer.threshold)

            probs_np = np.asarray(probs)
            labels_np = labels.detach().cpu().numpy().reshape(-1)
            batch_paths = list(filepaths) if filepaths is not None else [None] * len(labels_np)

            for idx, (filepath, target, pred) in enumerate(
                zip(batch_paths, labels_np, preds.reshape(-1))
            ):
                if probs_np.ndim == 1:
                    prob_value: Any = float(probs_np[idx])
                else:
                    prob_value = probs_np[idx].tolist()

                prediction_rows.append(
                    {
                        "filepath": filepath,
                        "label": int(target),
                        "pred": int(pred),
                        "prob": prob_value,
                    }
                )

    result = meter.compute_with_details()
    if save_predictions:
        result["predictions"] = prediction_rows
    return result


def flatten_record(record: Dict[str, Any]) -> Dict[str, Any]:
    row = {
        "condition": record["condition"],
        "corruption": record["corruption"],
        "severity": record["severity"],
        "split": record["split"],
        "dataset_size": record["dataset_size"],
        "class_counts": record.get("class_counts"),
        "params": record.get("params"),
    }
    row.update(record.get("metrics", {}))
    row.update(record.get("details", {}))
    return row


def safe_metric_value(metrics: Dict[str, Any], key: str) -> float:
    value = metrics.get(key, math.nan)
    try:
        value = float(value)
    except Exception:
        return math.nan
    return value


def nanmean(values: Iterable[float]) -> float:
    valid = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not valid:
        return math.nan
    return float(sum(valid) / len(valid))


def summarize_results(records: List[Dict[str, Any]], primary_metric: str) -> Dict[str, Any]:
    if not records:
        raise ValueError("records must not be empty")

    clean_record = next((record for record in records if record["corruption"] == "clean"), None)
    corrupted_records = [record for record in records if record["corruption"] != "clean"]

    clean_primary = math.nan
    if clean_record is not None:
        clean_primary = safe_metric_value(clean_record["metrics"], primary_metric)

    corrupted_primary_values = [
        safe_metric_value(record["metrics"], primary_metric)
        for record in corrupted_records
    ]
    mpc = nanmean(corrupted_primary_values)

    rpc = math.nan
    avg_drop = math.nan
    if not math.isnan(clean_primary) and clean_primary != 0.0 and not math.isnan(mpc):
        rpc = float(mpc / clean_primary)
        avg_drop = float(clean_primary - mpc)

    worst_record = None
    if corrupted_records:
        valid_records = [
            record
            for record in corrupted_records
            if not math.isnan(safe_metric_value(record["metrics"], primary_metric))
        ]
        if valid_records:
            worst_record = min(
                valid_records,
                key=lambda record: safe_metric_value(record["metrics"], primary_metric),
            )

    per_corruption: Dict[str, Any] = {}
    for record in corrupted_records:
        per_corruption.setdefault(record["corruption"], []).append(record)

    per_corruption_summary: Dict[str, Any] = {}
    for name, corr_records in per_corruption.items():
        corr_records = sorted(corr_records, key=lambda record: int(record["severity"]))
        values = [safe_metric_value(record["metrics"], primary_metric) for record in corr_records]
        finite_values = [v for v in values if not math.isnan(v)]

        per_corruption_summary[name] = {
            "mean_primary_metric": nanmean(values),
            "worst_primary_metric": min(finite_values) if finite_values else math.nan,
            "best_primary_metric": max(finite_values) if finite_values else math.nan,
            "num_severities": len(corr_records),
            "records": [
                {
                    "severity": record["severity"],
                    "condition": record["condition"],
                    primary_metric: safe_metric_value(record["metrics"], primary_metric),
                    "accuracy": safe_metric_value(record["metrics"], "accuracy"),
                    "f1": safe_metric_value(record["metrics"], "f1"),
                    "loss": safe_metric_value(record["metrics"], "loss"),
                }
                for record in corr_records
            ],
        }

    return {
        "primary_metric": primary_metric,
        "clean_primary_metric": clean_primary,
        "mpc": mpc,
        "rpc": rpc,
        "avg_drop": avg_drop,
        "num_conditions": len(records),
        "num_corrupted_conditions": len(corrupted_records),
        "worst_case": None
        if worst_record is None
        else {
            "condition": worst_record["condition"],
            "corruption": worst_record["corruption"],
            "severity": worst_record["severity"],
            primary_metric: safe_metric_value(worst_record["metrics"], primary_metric),
            "metrics": worst_record["metrics"],
            "params": worst_record.get("params"),
        },
        "per_corruption": per_corruption_summary,
    }


def resolve_selected_corruptions(
    robustness_cfg: Any,
    selected_raw: Optional[List[str]],
) -> List[str]:
    enabled = [_normalize_corruption_name(name) for name in get_enabled_corruptions(robustness_cfg)]

    if selected_raw is None:
        return enabled

    requested = [_normalize_corruption_name(name) for name in selected_raw]
    missing = [name for name in requested if name not in enabled]
    if missing:
        raise ValueError(
            f"Requested corruptions not enabled in robustness config: {missing}. "
            f"Enabled: {enabled}"
        )
    return requested


def resolve_selected_severities(
    robustness_cfg: Any,
    selected_raw: Optional[List[int]],
) -> List[int]:
    benchmark_severities = [int(v) for v in get_benchmark_severities(robustness_cfg)]

    if selected_raw is None:
        return benchmark_severities

    missing = [int(v) for v in selected_raw if int(v) not in benchmark_severities]
    if missing:
        raise ValueError(
            f"Requested severities not found in benchmark.severities: {missing}. "
            f"Available: {benchmark_severities}"
        )
    return [int(v) for v in selected_raw]


def build_condition_name(corruption: str, severity: int) -> str:
    if corruption == "clean":
        return "clean"
    return f"{corruption}_s{severity}"


def evaluate_condition(
    cfg: Any,
    robustness_cfg: Any,
    dataset_cls: Type,
    split_csv: str,
    trainer: Trainer,
    split: str,
    corruption: str,
    severity: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    shuffle: bool,
    seed: int,
    save_predictions: bool,
) -> Dict[str, Any]:
    if corruption == "clean":
        transform = build_clean_eval_transform(cfg.data)
        params = {"name": "clean"}
    else:
        transform = build_corrupted_eval_transform(
            data_cfg=cfg.data,
            corruption_name=corruption,
            severity=severity,
            robustness_cfg=robustness_cfg,
        )
        params = get_corruption_params(
            robustness_cfg=robustness_cfg,
            corruption_name=corruption,
            severity=severity,
        )

    dataset = build_single_dataset(
        dataset_cls=dataset_cls,
        csv_path=split_csv,
        root_dir=cfg.data.paths.root_dir,
        transform=transform,
    )
    loader = build_loader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=shuffle,
        seed=seed,
    )

    print("-" * 80)
    print(f"Condition: {build_condition_name(corruption, severity)}")
    print(f"corruption: {corruption}")
    print(f"severity: {severity}")
    print(f"params: {params}")

    result = evaluate_loader_with_details(
        trainer=trainer,
        loader=loader,
        split=split,
        save_predictions=save_predictions,
    )

    metrics = result["metrics"]
    print(format_metrics(metrics))

    return {
        "condition": build_condition_name(corruption, severity),
        "corruption": corruption,
        "severity": int(severity),
        "split": split,
        "params": params,
        "metrics": metrics,
        "details": result.get("details", {}),
        "dataset_size": len(dataset),
        "class_counts": dataset.class_counts,
        "predictions": result.get("predictions", None),
    }


def main() -> None:
    args = parse_args()

    cfg = load_experiment_config(
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
    )
    robustness_cfg = load_yaml(resolve_path(args.robustness_config))

    if args.device is not None:
        cfg.train.experiment.device = args.device
    if args.batch_size is not None:
        cfg.data.dataloader.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.data.dataloader.num_workers = args.num_workers

    seed = int(cfg.train.experiment.seed)
    seed_everything(seed)

    selected_corruptions = resolve_selected_corruptions(
        robustness_cfg=robustness_cfg,
        selected_raw=parse_name_list(args.corruptions),
    )
    selected_severities = resolve_selected_severities(
        robustness_cfg=robustness_cfg,
        selected_raw=parse_int_list(args.severities),
    )

    benchmark_cfg = getattr(robustness_cfg, "benchmark", None)
    save_predictions = bool(args.save_predictions)
    save_predictions = save_predictions or bool(
        getattr(benchmark_cfg, "save_predictions", False)
    )
    save_predictions = save_predictions or bool(
        getattr(benchmark_cfg, "save_per_sample", False)
    )

    print("=" * 80)
    print("Merged Config")
    print("=" * 80)
    print(pretty_print_config(cfg))

    print("=" * 80)
    print("Robustness Config")
    print("=" * 80)
    print(pretty_print_config(robustness_cfg))

    split_csv = get_split_csv_path(cfg, args.split)
    split_csv_path = resolve_path(split_csv)
    if not split_csv_path.exists():
        raise FileNotFoundError(f"{args.split} split CSV not found: {split_csv_path}")

    dataset_cls = get_dataset_class(cfg)

    clean_dataset = build_single_dataset(
        dataset_cls=dataset_cls,
        csv_path=split_csv,
        root_dir=cfg.data.paths.root_dir,
        transform=build_clean_eval_transform(cfg.data),
    )

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {getattr(cfg.data, 'name', 'unknown')}")
    print(f"dataset class: {dataset_cls.__name__}")
    print(f"split: {args.split}")
    print(f"size: {len(clean_dataset)}")
    print(f"class counts: {clean_dataset.class_counts}")

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
    checkpoint = trainer.load_checkpoint(checkpoint_path, strict=not args.non_strict)

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / f"robustness_{args.split}")
    )
    ensure_dir(output_dir)

    records: List[Dict[str, Any]] = []
    flattened_rows: List[Dict[str, Any]] = []

    clean_record = evaluate_condition(
        cfg=cfg,
        robustness_cfg=robustness_cfg,
        dataset_cls=dataset_cls,
        split_csv=split_csv,
        trainer=trainer,
        split=args.split,
        corruption="clean",
        severity=0,
        batch_size=int(cfg.data.dataloader.batch_size),
        num_workers=int(cfg.data.dataloader.num_workers),
        pin_memory=bool(cfg.data.dataloader.pin_memory),
        shuffle=get_split_shuffle(cfg, args.split),
        seed=seed,
        save_predictions=save_predictions,
    )
    records.append(clean_record)
    flattened_rows.append(flatten_record(clean_record))

    if save_predictions and clean_record.get("predictions") is not None:
        save_json(
            {
                "condition": clean_record["condition"],
                "predictions": clean_record["predictions"],
            },
            output_dir / "predictions" / f"{clean_record['condition']}.json",
        )
        clean_record.pop("predictions", None)

    for corruption in selected_corruptions:
        for severity in selected_severities:
            record = evaluate_condition(
                cfg=cfg,
                robustness_cfg=robustness_cfg,
                dataset_cls=dataset_cls,
                split_csv=split_csv,
                trainer=trainer,
                split=args.split,
                corruption=corruption,
                severity=severity,
                batch_size=int(cfg.data.dataloader.batch_size),
                num_workers=int(cfg.data.dataloader.num_workers),
                pin_memory=bool(cfg.data.dataloader.pin_memory),
                shuffle=get_split_shuffle(cfg, args.split),
                seed=seed,
                save_predictions=save_predictions,
            )
            records.append(record)
            flattened_rows.append(flatten_record(record))

            if save_predictions and record.get("predictions") is not None:
                save_json(
                    {
                        "condition": record["condition"],
                        "predictions": record["predictions"],
                    },
                    output_dir / "predictions" / f"{record['condition']}.json",
                )
                record.pop("predictions", None)

    primary_metric = str(getattr(benchmark_cfg, "primary_metric", "auc"))
    summary = summarize_results(records=records, primary_metric=primary_metric)

    full_result = {
        "dataset_name": getattr(cfg.data, "name", "unknown"),
        "dataset_class": dataset_cls.__name__,
        "split": args.split,
        "checkpoint": checkpoint_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "robustness_config": resolve_path(args.robustness_config).as_posix(),
        "selected_corruptions": selected_corruptions,
        "selected_severities": selected_severities,
        "dataset_size": len(clean_dataset),
        "class_counts": clean_dataset.class_counts,
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
        "records": records,
        "summary": summary,
    }

    full_json_path = output_dir / "robustness_results.json"
    summary_json_path = output_dir / "robustness_summary.json"
    csv_path = output_dir / "robustness_records.csv"

    save_json(full_result, full_json_path)
    save_json(summary, summary_json_path)
    save_csv(flattened_rows, csv_path)

    print("=" * 80)
    print("Robustness Summary")
    print("=" * 80)
    print(
        f"primary_metric: {summary['primary_metric']}"
    )
    print(
        f"clean: {summary['clean_primary_metric']:.4f}"
        if not math.isnan(summary["clean_primary_metric"])
        else "clean: nan"
    )
    print(
        f"mPC: {summary['mpc']:.4f}"
        if not math.isnan(summary["mpc"])
        else "mPC: nan"
    )
    print(
        f"rPC: {summary['rpc']:.4f}"
        if not math.isnan(summary["rpc"])
        else "rPC: nan"
    )
    print(
        f"avg_drop: {summary['avg_drop']:.4f}"
        if not math.isnan(summary["avg_drop"])
        else "avg_drop: nan"
    )

    if summary["worst_case"] is not None:
        worst_value = summary["worst_case"][primary_metric]
        worst_value_str = f"{worst_value:.4f}" if not math.isnan(float(worst_value)) else "nan"
        print(
            f"worst_case: {summary['worst_case']['condition']} "
            f"({primary_metric}={worst_value_str})"
        )

    print("=" * 80)
    print("Robustness Evaluation Finished")
    print("=" * 80)
    print(f"Saved full result to: {full_json_path}")
    print(f"Saved summary to: {summary_json_path}")
    print(f"Saved CSV to: {csv_path}")


if __name__ == "__main__":
    main()