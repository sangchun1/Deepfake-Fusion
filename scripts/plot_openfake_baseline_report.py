from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt


DEFAULT_ROOT = Path("outputs/spatial/resnet18_openfake")
DEFAULT_OUTPUT_DIR = Path("outputs/reports/openfake_baseline")
METRIC_KEYS: Sequence[str] = ("accuracy", "precision", "recall", "f1", "auc", "loss")
MODE_ORDER = ("merged", "by_generator", "logo")


@dataclass
class EvalRecord:
    mode: str
    experiment: str
    path: Path
    dataset_name: str
    dataset_size: int
    checkpoint: str
    metrics: Dict[str, float]

    def to_row(self, include_paths: bool = False) -> Dict[str, object]:
        row: Dict[str, object] = {
            "mode": self.mode,
            "experiment": self.experiment,
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
        }
        if include_paths:
            row["eval_json"] = self.path.as_posix()
            row["checkpoint"] = self.checkpoint
        for metric in METRIC_KEYS:
            row[metric] = self.metrics.get(metric)
        return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create report plots from OpenFake baseline eval_test.json files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Root directory containing merged/, by_generator/, logo/ results.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save csv, markdown, json, and plot files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="How many best/worst experiments to include in summary text.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def discover_eval_files(root: Path) -> List[Path]:
    candidates = [
        root / "merged" / "merged" / "eval_test.json",
    ]
    for mode in ("by_generator", "logo"):
        mode_dir = root / mode
        if not mode_dir.exists():
            continue
        for child in sorted(mode_dir.iterdir()):
            eval_path = child / "eval_test.json"
            if eval_path.exists():
                candidates.append(eval_path)
    existing = [p for p in candidates if p.exists()]
    if not existing:
        raise FileNotFoundError(f"No eval_test.json files found under: {root}")
    return existing


def infer_mode_and_experiment(eval_path: Path, root: Path) -> tuple[str, str]:
    rel = eval_path.relative_to(root)
    parts = rel.parts
    if len(parts) < 3:
        raise ValueError(f"Unexpected eval path structure: {eval_path}")
    mode = parts[0]
    experiment = parts[1]
    return mode, experiment


def load_records(root: Path) -> List[EvalRecord]:
    records: List[EvalRecord] = []
    for eval_path in discover_eval_files(root):
        payload = load_json(eval_path)
        mode, experiment = infer_mode_and_experiment(eval_path, root)
        metrics_raw = payload.get("metrics", {})
        metrics = {k: float(metrics_raw[k]) for k in METRIC_KEYS if k in metrics_raw}
        records.append(
            EvalRecord(
                mode=mode,
                experiment=experiment,
                path=eval_path,
                dataset_name=str(payload.get("dataset_name", "")),
                dataset_size=int(payload.get("dataset_size", 0)),
                checkpoint=str(payload.get("checkpoint", "")),
                metrics=metrics,
            )
        )
    records.sort(key=lambda r: (MODE_ORDER.index(r.mode), r.experiment))
    return records


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_csv(records: Sequence[EvalRecord], out_path: Path) -> None:
    ensure_dir(out_path.parent)
    rows = [r.to_row(include_paths=False) for r in records]
    fieldnames = [
        "mode",
        "experiment",
        "dataset_name",
        "dataset_size",
        *METRIC_KEYS,
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_by_mode(records: Sequence[EvalRecord]) -> Dict[str, List[EvalRecord]]:
    grouped: Dict[str, List[EvalRecord]] = {mode: [] for mode in MODE_ORDER}
    for record in records:
        grouped.setdefault(record.mode, []).append(record)
    return grouped


def summarize_mode(records: Sequence[EvalRecord]) -> Dict[str, object]:
    if not records:
        return {"count": 0, "means": {}, "best_accuracy": None, "worst_accuracy": None}

    means = {
        metric: mean(r.metrics[metric] for r in records if metric in r.metrics)
        for metric in METRIC_KEYS
    }
    best_accuracy = max(records, key=lambda r: r.metrics.get("accuracy", float("-inf")))
    worst_accuracy = min(records, key=lambda r: r.metrics.get("accuracy", float("inf")))
    best_auc = max(records, key=lambda r: r.metrics.get("auc", float("-inf")))
    worst_auc = min(records, key=lambda r: r.metrics.get("auc", float("inf")))
    return {
        "count": len(records),
        "means": means,
        "best_accuracy": best_accuracy,
        "worst_accuracy": worst_accuracy,
        "best_auc": best_auc,
        "worst_auc": worst_auc,
    }


def metric_value(record: EvalRecord, metric: str) -> float:
    return float(record.metrics.get(metric, float("nan")))


def save_summary_json(records: Sequence[EvalRecord], out_path: Path) -> None:
    grouped = group_by_mode(records)
    summary = {
        "root": str(records[0].path.parents[3]) if records else "",
        "num_records": len(records),
        "modes": {
            mode: {
                "count": len(grouped.get(mode, [])),
                "records": [r.to_row(include_paths=False) for r in grouped.get(mode, [])],
                "means": summarize_mode(grouped.get(mode, [])).get("means", {}),
            }
            for mode in MODE_ORDER
        },
    }
    ensure_dir(out_path.parent)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def format_metric(x: float) -> str:
    return f"{x:.4f}"


def save_summary_markdown(records: Sequence[EvalRecord], out_path: Path, top_k: int) -> None:
    grouped = group_by_mode(records)
    merged_records = grouped.get("merged", [])
    by_generator_records = grouped.get("by_generator", [])
    logo_records = grouped.get("logo", [])

    merged_record = merged_records[0] if merged_records else None
    by_summary = summarize_mode(by_generator_records)
    logo_summary = summarize_mode(logo_records)

    def top_bottom_block(block_records: Sequence[EvalRecord], metric: str) -> str:
        if not block_records:
            return "- 없음\n"
        reverse = metric != "loss"
        ordered = sorted(block_records, key=lambda r: metric_value(r, metric), reverse=reverse)
        best = ordered[:top_k]
        worst = ordered[-top_k:]
        lines = [f"- best {metric}:"]
        for r in best:
            lines.append(f"  - {r.experiment}: {format_metric(metric_value(r, metric))}")
        lines.append(f"- worst {metric}:")
        for r in worst:
            lines.append(f"  - {r.experiment}: {format_metric(metric_value(r, metric))}")
        return "\n".join(lines) + "\n"

    lines: List[str] = []
    lines.append("# OpenFake baseline report summary\n")
    lines.append("## 1) merged\n")
    if merged_record is None:
        lines.append("- merged 결과 없음\n")
    else:
        for metric in METRIC_KEYS:
            lines.append(f"- {metric}: {format_metric(metric_value(merged_record, metric))}")
        lines.append("")

    for title, block_records, block_summary in [
        ("by_generator", by_generator_records, by_summary),
        ("logo", logo_records, logo_summary),
    ]:
        lines.append(f"## 2) {title}\n")
        lines.append(f"- num_experiments: {block_summary['count']}")
        means = block_summary.get("means", {})
        for metric in METRIC_KEYS:
            if metric in means:
                lines.append(f"- mean_{metric}: {format_metric(float(means[metric]))}")
        if block_summary.get("best_accuracy") is not None:
            best_acc: EvalRecord = block_summary["best_accuracy"]
            worst_acc: EvalRecord = block_summary["worst_accuracy"]
            best_auc: EvalRecord = block_summary["best_auc"]
            worst_auc: EvalRecord = block_summary["worst_auc"]
            lines.append(
                f"- best_accuracy: {best_acc.experiment} ({format_metric(metric_value(best_acc, 'accuracy'))})"
            )
            lines.append(
                f"- worst_accuracy: {worst_acc.experiment} ({format_metric(metric_value(worst_acc, 'accuracy'))})"
            )
            lines.append(
                f"- best_auc: {best_auc.experiment} ({format_metric(metric_value(best_auc, 'auc'))})"
            )
            lines.append(
                f"- worst_auc: {worst_auc.experiment} ({format_metric(metric_value(worst_auc, 'auc'))})"
            )
        lines.append("")
        lines.append(top_bottom_block(block_records, "accuracy"))
        lines.append(top_bottom_block(block_records, "auc"))

    ensure_dir(out_path.parent)
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")


def build_overview_rows(records: Sequence[EvalRecord]) -> Dict[str, Dict[str, float]]:
    grouped = group_by_mode(records)
    rows: Dict[str, Dict[str, float]] = {}

    if grouped.get("merged"):
        merged_record = grouped["merged"][0]
        rows["merged"] = {metric: metric_value(merged_record, metric) for metric in METRIC_KEYS}

    for mode in ("by_generator", "logo"):
        mode_records = grouped.get(mode, [])
        if not mode_records:
            continue
        rows[f"{mode}_mean"] = {
            metric: mean(metric_value(r, metric) for r in mode_records)
            for metric in METRIC_KEYS
        }

    return rows


def save_overview_comparison(records: Sequence[EvalRecord], output_dir: Path, dpi: int) -> Path:
    overview_rows = build_overview_rows(records)
    labels = list(overview_rows.keys())
    x = list(range(len(labels)))
    width = 0.12
    offsets = [
        -2.5 * width,
        -1.5 * width,
        -0.5 * width,
        0.5 * width,
        1.5 * width,
        2.5 * width,
    ]

    fig, ax = plt.subplots(figsize=(12, 6))
    for offset, metric in zip(offsets, METRIC_KEYS):
        values = [overview_rows[label][metric] for label in labels]
        ax.bar([i + offset for i in x], values, width=width, label=metric)
        for idx, value in enumerate(values):
            ax.text(
                x[idx] + offset,
                value + (0.01 if metric != "loss" else 0.015),
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    upper = max(max(overview_rows[label][metric] for metric in METRIC_KEYS) for label in labels)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.0, min(max(upper + 0.12, 1.02), 1.35))
    ax.set_ylabel("score")
    ax.set_title("OpenFake baseline overview")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3)
    fig.tight_layout()

    out_path = output_dir / "overview_comparison.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output_dir = ensure_dir(Path(args.output_dir))

    records = load_records(root)

    csv_path = output_dir / "baseline_metrics.csv"
    summary_md_path = output_dir / "summary.md"
    summary_json_path = output_dir / "summary.json"
    overview_path = output_dir / "overview_comparison.png"

    save_csv(records, csv_path)
    save_summary_markdown(records, summary_md_path, top_k=int(args.top_k))
    save_summary_json(records, summary_json_path)
    save_overview_comparison(records, output_dir, dpi=int(args.dpi))

    print("=" * 80)
    print("OpenFake baseline report files created")
    print("=" * 80)
    print(f"root: {root}")
    print(f"output_dir: {output_dir}")
    print(f"csv: {csv_path}")
    print(f"summary_md: {summary_md_path}")
    print(f"summary_json: {summary_json_path}")
    print(f"plot: {overview_path}")


if __name__ == "__main__":
    main()
