from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from deepfake_fusion.datasets.semitruths_dataset import SemiTruthsDataset
from deepfake_fusion.engine.trainer import Trainer
from deepfake_fusion.metrics.grouped_metrics import (
    compute_group_metrics_many,
    compute_overall_metrics,
    summarize_group_tables,
)
from deepfake_fusion.models.build_model import build_model, get_model_summary
from deepfake_fusion.transforms.image_aug import build_transforms_from_config
from deepfake_fusion.utils.config import (
    load_experiment_config,
    pretty_print_config,
    resolve_path,
)
from deepfake_fusion.utils.seed import (
    get_torch_generator,
    seed_everything,
    seed_worker,
)
from deepfake_fusion.utils.semitruths_metadata import (
    add_default_analysis_bins,
    build_directional_edit_column,
    get_available_group_columns,
    standardize_semitruths_metadata,
)

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    tqdm = None


DATASET_REGISTRY: Dict[str, Type] = {
    "semitruths": SemiTruthsDataset,
    "semitruths_eval": SemiTruthsDataset,
    "SemiTruthsDataset": SemiTruthsDataset,
}


# -----------------------------------------------------------------------------
# config helpers
# -----------------------------------------------------------------------------


def cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
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


# -----------------------------------------------------------------------------
# io helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on Semi-Truths Evalset and compute grouped robustness metrics."
    )
    parser.add_argument(
        "--data_config",
        type=str,
        default="configs/data/semitruths_eval.yaml",
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
        default="configs/train/spatial_resnet_openfake.yaml",
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
        choices=["test"],
        help="Semi-Truths evaluation split. Evalset is treated as test by default.",
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
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold used to convert y_prob to y_pred.",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=10,
        help="Minimum number of samples required to keep a group in grouped metrics.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation outputs. Defaults to <train_output_dir>/semitruths_eval.",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Optional path to save a single merged result JSON.",
    )
    parser.add_argument(
        "--save_prepared_csv",
        action="store_true",
        help="Also save the standardized/binned metadata CSV used for evaluation.",
    )
    parser.add_argument(
        "--group_columns",
        type=str,
        default=None,
        help=(
            "Comma-separated metadata columns for grouped metrics. "
            "If omitted, uses available columns from data config / metadata defaults."
        ),
    )
    return parser.parse_args()



def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path



def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path



def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return obj.as_posix()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return None if np.isnan(value) else value
    if isinstance(obj, float):
        return None if np.isnan(obj) else obj
    if pd.isna(obj):
        return None
    return obj



def save_json(data: Dict[str, Any], path: Path) -> None:
    ensure_parent(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False, allow_nan=False)



def format_metrics(metrics: Mapping[str, Any], keys: Optional[Sequence[str]] = None) -> str:
    if keys is None:
        keys = (
            "loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "balanced_accuracy",
            "fake_recall",
            "real_recall",
        )
    parts: List[str] = []
    for key in keys:
        if key not in metrics:
            continue
        value = metrics[key]
        if isinstance(value, float):
            parts.append(f"{key}={value:.4f}")
        else:
            parts.append(f"{key}={value}")
    return " | ".join(parts)


# -----------------------------------------------------------------------------
# build helpers
# -----------------------------------------------------------------------------


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



def get_dataset_class(cfg: Any):
    dataset_key = None
    if cfg_get(cfg, "data", "dataset_class", default=None) is not None:
        dataset_key = str(cfg_get(cfg, "data", "dataset_class"))
    elif cfg_get(cfg, "data", "name", default=None) is not None:
        dataset_key = str(cfg_get(cfg, "data", "name"))

    if dataset_key is None:
        raise ValueError(
            "Could not determine dataset class. Set cfg.data.dataset_class or cfg.data.name."
        )
    if dataset_key not in DATASET_REGISTRY:
        supported = ", ".join(DATASET_REGISTRY.keys())
        raise ValueError(f"Unsupported dataset '{dataset_key}'. Supported values: {supported}")
    return DATASET_REGISTRY[dataset_key]



def resolve_group_columns(cfg: Any, records_df: pd.DataFrame, cli_group_columns: Optional[str]) -> List[str]:
    if cli_group_columns:
        requested = [col.strip() for col in cli_group_columns.split(",") if col.strip()]
    else:
        requested = list(cfg_get(cfg, "data", "grouping", "columns", default=[]) or [])
        if not requested:
            requested = get_available_group_columns(records_df)

    resolved: List[str] = []
    for col in requested:
        preferred = col
        if col == "area_ratio" and "area_ratio_bin" in records_df.columns:
            preferred = "area_ratio_bin"
        elif col == "semantic_mag" and "semantic_mag_bin" in records_df.columns:
            preferred = "semantic_mag_bin"
        elif col == "scene_complexity" and "scene_complexity_bin" in records_df.columns:
            preferred = "scene_complexity_bin"
        elif col == "scene_diversity" and "scene_diversity_bin" in records_df.columns:
            preferred = "scene_diversity_bin"
        if preferred in records_df.columns and records_df[preferred].notna().any():
            resolved.append(preferred)

    if "directional_edit" in records_df.columns and records_df["directional_edit"].notna().any():
        resolved.append("directional_edit")

    deduped: List[str] = []
    seen = set()
    for col in resolved:
        if col not in seen:
            deduped.append(col)
            seen.add(col)
    return deduped



def build_column_override_map(cfg: Any) -> Dict[str, str]:
    metadata_cfg = cfg_get(cfg, "data", "metadata", default=None)
    if metadata_cfg is None:
        return {}

    mapping = {
        "sample_id": cfg_get(cfg, "data", "metadata", "image_id_col", default=None),
        "filepath": cfg_get(cfg, "data", "metadata", "image_path_col", default=None),
        "label": cfg_get(cfg, "data", "metadata", "label_col", default=None),
        "dataset": cfg_get(cfg, "data", "metadata", "dataset_col", default=None),
        "entity": cfg_get(cfg, "data", "metadata", "entity_col", default=None),
        "method": cfg_get(cfg, "data", "metadata", "method_col", default=None),
        "diffusion_model": cfg_get(cfg, "data", "metadata", "diffusion_model_col", default=None),
        "area_ratio": cfg_get(cfg, "data", "metadata", "area_ratio_col", default=None),
        "semantic_mag": cfg_get(cfg, "data", "metadata", "semantic_mag_col", default=None),
        "scene_complexity": cfg_get(cfg, "data", "metadata", "scene_complexity_col", default=None),
        "scene_diversity": cfg_get(cfg, "data", "metadata", "scene_diversity_col", default=None),
        "change_type": cfg_get(cfg, "data", "metadata", "change_type_col", default=None),
        "mask_path": cfg_get(cfg, "data", "metadata", "mask_path_col", default=None),
    }
    return {k: v for k, v in mapping.items() if v}



def prepare_metadata_dataframe(cfg: Any) -> pd.DataFrame:
    metadata_csv = resolve_path(cfg.data.paths.metadata_csv)
    root_dir = cfg.data.paths.root_dir
    include_real = bool(cfg_get(cfg, "data", "metadata", "include_real", default=True))
    include_inpainting = bool(cfg_get(cfg, "data", "metadata", "include_inpainting", default=True))
    include_p2p = bool(cfg_get(cfg, "data", "metadata", "include_p2p", default=True))

    df = standardize_semitruths_metadata(
        metadata=metadata_csv,
        root_dir=root_dir,
        column_map=build_column_override_map(cfg),
        include_real=include_real,
        include_inpainting=include_inpainting,
        include_p2p=include_p2p,
        validate_paths=False,
    )

    grouping_cfg = cfg_get(cfg, "data", "grouping", default=None)
    if grouping_cfg is not None:
        df = add_default_analysis_bins(
            df,
            area_ratio_bins=cfg_get(cfg, "data", "grouping", "area_ratio_bins", default=None),
            area_ratio_labels=cfg_get(
                cfg,
                "data",
                "grouping",
                "area_ratio_bin_labels",
                default=("small", "medium", "large"),
            ),
            semantic_mag_bins=cfg_get(cfg, "data", "grouping", "semantic_mag_bins", default=None),
            semantic_mag_labels=cfg_get(
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


# -----------------------------------------------------------------------------
# model output helpers
# -----------------------------------------------------------------------------


def move_to_device(x: Any, device: torch.device) -> Any:
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return x



def extract_logits(model_output: Any) -> torch.Tensor:
    if torch.is_tensor(model_output):
        return model_output

    if isinstance(model_output, Mapping):
        for key in ("logits", "output", "pred", "prediction"):
            if key in model_output and torch.is_tensor(model_output[key]):
                return model_output[key]

    if isinstance(model_output, (tuple, list)):
        for item in model_output:
            if torch.is_tensor(item):
                return item
            if isinstance(item, Mapping):
                for key in ("logits", "output", "pred", "prediction"):
                    if key in item and torch.is_tensor(item[key]):
                        return item[key]

    raise TypeError(f"Could not extract logits from model output of type {type(model_output)}")



def logits_to_prob_fake(logits: torch.Tensor) -> torch.Tensor:
    if logits.ndim == 1:
        return torch.sigmoid(logits)
    if logits.ndim == 2 and logits.size(1) == 1:
        return torch.sigmoid(logits.squeeze(1))
    if logits.ndim == 2 and logits.size(1) >= 2:
        return torch.softmax(logits, dim=1)[:, 1]
    raise ValueError(f"Unsupported logits shape for binary classification: {tuple(logits.shape)}")



def compute_batch_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long()
    if logits.ndim == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, labels.float()
        )
    if logits.ndim == 2 and logits.size(1) == 1:
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits.squeeze(1), labels.float()
        )
    if logits.ndim == 2 and logits.size(1) >= 2:
        return torch.nn.functional.cross_entropy(logits, labels)
    raise ValueError(f"Unsupported logits shape for loss computation: {tuple(logits.shape)}")



def batch_to_records(
    batch: Mapping[str, Any],
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> List[Dict[str, Any]]:
    labels = batch["label"]
    if torch.is_tensor(labels):
        labels_np = labels.detach().cpu().numpy().astype(int)
    else:
        labels_np = np.asarray(labels, dtype=int)

    batch_size = len(labels_np)
    records: List[Dict[str, Any]] = []
    for i in range(batch_size):
        row: Dict[str, Any] = {
            "label": int(labels_np[i]),
            "y_prob": float(y_prob[i]),
            "y_pred": int(y_pred[i]),
        }
        for key, value in batch.items():
            if key == "image":
                continue
            if torch.is_tensor(value):
                if value.ndim == 0:
                    row[key] = value.detach().cpu().item()
                elif value.shape[0] == batch_size:
                    item = value[i].detach().cpu()
                    row[key] = item.item() if item.ndim == 0 else item.tolist()
            elif isinstance(value, (list, tuple)) and len(value) == batch_size:
                row[key] = value[i]
            else:
                # default collate가 처리하지 않은 스칼라/상수형
                row[key] = value
        records.append(row)
    return records


# -----------------------------------------------------------------------------
# evaluation core
# -----------------------------------------------------------------------------


def evaluate_model(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    model.eval()
    all_records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_samples = 0

    iterator: Iterable = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="Semi-Truths Eval", leave=False)

    with torch.no_grad():
        for batch in iterator:
            images = move_to_device(batch["image"], device)
            labels = move_to_device(batch["label"], device)

            outputs = model(images)
            logits = extract_logits(outputs)
            probs = logits_to_prob_fake(logits)
            preds = (probs >= float(threshold)).long()

            batch_loss = compute_batch_loss(logits, labels)
            bs = int(labels.shape[0])
            total_loss += float(batch_loss.detach().cpu().item()) * bs
            total_samples += bs

            batch_records = batch_to_records(
                batch=batch,
                y_prob=probs.detach().cpu().numpy(),
                y_pred=preds.detach().cpu().numpy(),
            )
            all_records.extend(batch_records)

    records_df = pd.DataFrame(all_records)
    overall_metrics = compute_overall_metrics(
        records_df,
        y_true_col="label",
        y_prob_col="y_prob",
        y_pred_col="y_pred",
        threshold=threshold,
    )
    overall_metrics["loss"] = (total_loss / total_samples) if total_samples > 0 else None
    overall_metrics["threshold"] = float(threshold)
    return records_df, overall_metrics


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

    output_dir = (
        resolve_path(args.output_dir)
        if args.output_dir is not None
        else resolve_path(Path(cfg.train.experiment.output_dir) / "semitruths_eval")
    )
    ensure_dir(output_dir)
    group_dir = ensure_dir(output_dir / "groups")

    prepared_df = prepare_metadata_dataframe(cfg)
    prepared_csv_path = output_dir / f"prepared_{args.split}.csv"
    prepared_df.to_csv(prepared_csv_path, index=False)

    dataset_cls = get_dataset_class(cfg)
    transforms = build_transforms_from_config(cfg)
    dataset = dataset_cls(
        csv_path=prepared_csv_path,
        root_dir=cfg.data.paths.root_dir,
        transform=transforms[args.split],
        validate_files=True,
    )
    loader = build_loader(
        dataset=dataset,
        batch_size=int(cfg.data.dataloader.batch_size),
        num_workers=int(cfg.data.dataloader.num_workers),
        pin_memory=bool(cfg.data.dataloader.pin_memory),
        shuffle=bool(cfg_get(cfg, "data", args.split, "shuffle", default=False)),
        seed=seed,
    )

    print("=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    print(f"dataset name: {cfg_get(cfg, 'data', 'name', default='unknown')}")
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
    print("Evaluate: Semi-Truths")
    print("=" * 80)
    records_df, overall_metrics = evaluate_model(
        model=trainer.model,
        loader=loader,
        device=trainer.device,
        threshold=float(args.threshold),
    )
    print(format_metrics(overall_metrics))

    group_columns = resolve_group_columns(cfg, records_df, args.group_columns)
    group_tables = compute_group_metrics_many(
        records=records_df,
        group_columns=group_columns,
        y_true_col="label",
        y_prob_col="y_prob",
        y_pred_col="y_pred",
        threshold=float(args.threshold),
        min_group_size=int(args.min_group_size),
        dropna_group_values=True,
        include_overall_row=False,
    )
    group_summary_df = summarize_group_tables(group_tables)

    records_csv_path = output_dir / f"records_{args.split}.csv"
    overall_json_path = output_dir / f"overall_{args.split}.json"
    group_summary_csv_path = output_dir / f"group_summary_{args.split}.csv"
    group_summary_json_path = output_dir / f"group_summary_{args.split}.json"

    records_df.to_csv(records_csv_path, index=False)
    group_summary_df.to_csv(group_summary_csv_path, index=False)
    save_json({"rows": group_summary_df.to_dict(orient="records")}, group_summary_json_path)

    for group_name, table in group_tables.items():
        safe_name = str(group_name).replace("/", "_")
        table.to_csv(group_dir / f"{safe_name}.csv", index=False)
        save_json({"rows": table.to_dict(orient="records")}, group_dir / f"{safe_name}.json")

    result = {
        "dataset_name": cfg_get(cfg, "data", "name", default="unknown"),
        "dataset_class": dataset_cls.__name__,
        "split": args.split,
        "checkpoint": checkpoint_path.as_posix(),
        "dataset_size": len(dataset),
        "class_counts": dataset.class_counts,
        "best_score_in_checkpoint": checkpoint.get("best_score", None),
        "best_epoch_in_checkpoint": checkpoint.get("best_epoch", None),
        "saved_epoch": checkpoint.get("epoch", None),
        "overall_metrics": overall_metrics,
        "group_columns": group_columns,
        "num_group_tables": len(group_tables),
        "files": {
            "prepared_csv": prepared_csv_path.as_posix(),
            "records_csv": records_csv_path.as_posix(),
            "group_summary_csv": group_summary_csv_path.as_posix(),
            "group_summary_json": group_summary_json_path.as_posix(),
            "group_dir": group_dir.as_posix(),
        },
    }
    save_json(result, overall_json_path)

    output_json = (
        resolve_path(args.output_json)
        if args.output_json is not None
        else output_dir / f"eval_semitruths_{args.split}.json"
    )
    save_json(result, output_json)

    print("=" * 80)
    print("Grouped Metrics")
    print("=" * 80)
    if group_columns:
        print("group columns:", ", ".join(group_columns))
        if len(group_summary_df) > 0:
            printable_cols = [
                col
                for col in [
                    "group_name",
                    "num_groups",
                    "total_samples",
                    "fake_recall_weighted_mean",
                    "auc_weighted_mean",
                    "balanced_accuracy_weighted_mean",
                    "f1_weighted_mean",
                ]
                if col in group_summary_df.columns
            ]
            if printable_cols:
                print(group_summary_df[printable_cols].to_string(index=False))
    else:
        print("No valid grouped columns found in metadata; saved overall metrics only.")

    print("=" * 80)
    print("Evaluation Finished")
    print("=" * 80)
    print(f"Saved prepared CSV: {prepared_csv_path}")
    print(f"Saved records CSV: {records_csv_path}")
    print(f"Saved overall JSON: {overall_json_path}")
    print(f"Saved merged result JSON: {output_json}")
    print(f"Saved group summary CSV: {group_summary_csv_path}")
    print(f"Saved per-group tables dir: {group_dir}")


if __name__ == "__main__":
    main()
