from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)


def _ensure_dir(path: str | Path) -> Path:
    path = _to_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ensure_parent_dir(path: str | Path) -> Path:
    path = _to_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path) -> Dict[str, Any]:
    path = _to_path(path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Mapping[str, Any], path: str | Path, indent: int = 2) -> None:
    path = _ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=indent)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        if value is None:
            return default
        value = float(value)
        return value
    except Exception:
        return default


def _is_finite(value: Any) -> bool:
    value = _safe_float(value)
    return not math.isnan(value) and not math.isinf(value)


def _nanmean(values: Iterable[Any]) -> float:
    values = [_safe_float(v) for v in values]
    values = [v for v in values if not math.isnan(v)]
    if len(values) == 0:
        return math.nan
    return float(sum(values) / len(values))


def _ordered_unique(values: Iterable[Any]) -> List[Any]:
    out: List[Any] = []
    seen = set()
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _sort_corruption_names(names: Sequence[str]) -> List[str]:
    def _key(name: str) -> Tuple[int, str]:
        if name == "clean":
            return (0, name)
        return (1, name)

    return sorted([str(name) for name in names], key=_key)


def _format_metric_value(value: Any, digits: int = 4) -> str:
    value = _safe_float(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def _normalize_results_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    if "records" in payload and isinstance(payload["records"], list):
        return dict(payload)

    if isinstance(payload, list):
        return {
            "records": payload,
            "summary": {},
        }

    raise ValueError(
        "Invalid robustness results payload. Expected a dict with 'records' or a list of records."
    )


def load_robustness_results(results: str | Path | Mapping[str, Any]) -> Dict[str, Any]:
    if isinstance(results, Mapping):
        payload = dict(results)
    else:
        payload = load_json(results)
    return _normalize_results_payload(payload)


def extract_metric_names(records: Sequence[Mapping[str, Any]]) -> List[str]:
    metric_names: List[str] = []
    seen = set()
    for record in records:
        metrics = record.get("metrics", {})
        if isinstance(metrics, Mapping):
            for key in metrics.keys():
                if key not in seen:
                    seen.add(key)
                    metric_names.append(str(key))
    return metric_names


def get_clean_record(records: Sequence[Mapping[str, Any]]) -> Optional[Mapping[str, Any]]:
    for record in records:
        if str(record.get("corruption", "")).lower() == "clean":
            return record
    return None


def get_corrupted_records(records: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        record for record in records
        if str(record.get("corruption", "")).lower() != "clean"
    ]


def compute_summary_from_records(
    records: Sequence[Mapping[str, Any]],
    primary_metric: str,
) -> Dict[str, Any]:
    clean_record = get_clean_record(records)
    corrupted_records = get_corrupted_records(records)

    clean_value = math.nan
    if clean_record is not None:
        clean_value = _safe_float(clean_record.get("metrics", {}).get(primary_metric))

    corrupted_values = [
        _safe_float(record.get("metrics", {}).get(primary_metric))
        for record in corrupted_records
    ]
    mpc = _nanmean(corrupted_values)

    rpc = math.nan
    avg_drop = math.nan
    if _is_finite(clean_value) and clean_value != 0.0 and _is_finite(mpc):
        rpc = float(mpc / clean_value)
        avg_drop = float(clean_value - mpc)

    worst_case = None
    valid_records = [
        record for record in corrupted_records
        if _is_finite(record.get("metrics", {}).get(primary_metric))
    ]
    if len(valid_records) > 0:
        worst_record = min(
            valid_records,
            key=lambda r: _safe_float(r.get("metrics", {}).get(primary_metric)),
        )
        worst_case = {
            "condition": worst_record.get("condition"),
            "corruption": worst_record.get("corruption"),
            "severity": int(worst_record.get("severity", -1)),
            primary_metric: _safe_float(worst_record.get("metrics", {}).get(primary_metric)),
            "metrics": dict(worst_record.get("metrics", {})),
            "params": dict(worst_record.get("params", {})),
        }

    per_corruption: Dict[str, Dict[str, Any]] = {}
    for record in corrupted_records:
        name = str(record.get("corruption"))
        per_corruption.setdefault(name, {"values": [], "records": []})
        per_corruption[name]["values"].append(
            _safe_float(record.get("metrics", {}).get(primary_metric))
        )
        per_corruption[name]["records"].append(record)

    per_corruption_summary: Dict[str, Any] = {}
    for name, bundle in per_corruption.items():
        values = [v for v in bundle["values"] if _is_finite(v)]
        per_corruption_summary[name] = {
            "mean_primary_metric": _nanmean(bundle["values"]),
            "worst_primary_metric": min(values) if len(values) > 0 else math.nan,
            "best_primary_metric": max(values) if len(values) > 0 else math.nan,
            "num_severities": len(bundle["records"]),
        }

    return {
        "primary_metric": primary_metric,
        "clean_primary_metric": clean_value,
        "mpc": mpc,
        "rpc": rpc,
        "avg_drop": avg_drop,
        "num_conditions": len(records),
        "num_corrupted_conditions": len(corrupted_records),
        "worst_case": worst_case,
        "per_corruption": per_corruption_summary,
    }


def build_metric_matrix(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> Tuple[List[str], List[int], np.ndarray]:
    corrupted_records = get_corrupted_records(records)

    corruption_names = _ordered_unique(
        str(record.get("corruption")) for record in corrupted_records
    )
    corruption_names = _sort_corruption_names(corruption_names)
    corruption_names = [name for name in corruption_names if name != "clean"]

    severities = _ordered_unique(
        int(record.get("severity")) for record in corrupted_records
    )
    severities = sorted([int(v) for v in severities])

    matrix = np.full((len(corruption_names), len(severities)), np.nan, dtype=np.float32)

    corr_to_idx = {name: idx for idx, name in enumerate(corruption_names)}
    sev_to_idx = {sev: idx for idx, sev in enumerate(severities)}

    for record in corrupted_records:
        corr = str(record.get("corruption"))
        sev = int(record.get("severity"))
        if corr not in corr_to_idx or sev not in sev_to_idx:
            continue
        value = _safe_float(record.get("metrics", {}).get(metric_name))
        matrix[corr_to_idx[corr], sev_to_idx[sev]] = value

    return corruption_names, severities, matrix


def build_per_corruption_means(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
) -> Tuple[List[str], np.ndarray]:
    corrupted_records = get_corrupted_records(records)

    grouped: Dict[str, List[float]] = {}
    for record in corrupted_records:
        corr = str(record.get("corruption"))
        grouped.setdefault(corr, [])
        grouped[corr].append(_safe_float(record.get("metrics", {}).get(metric_name)))

    corruption_names = _sort_corruption_names(grouped.keys())
    corruption_names = [name for name in corruption_names if name != "clean"]

    means = np.array([_nanmean(grouped[name]) for name in corruption_names], dtype=np.float32)
    return corruption_names, means


def plot_metric_heatmap(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    save_path: str | Path,
    title: Optional[str] = None,
    annotate: bool = True,
    digits: int = 4,
) -> Path:
    save_path = _to_path(save_path)
    _ensure_parent_dir(save_path)

    corruption_names, severities, matrix = build_metric_matrix(records, metric_name)

    if len(corruption_names) == 0 or len(severities) == 0:
        raise ValueError(f"No corrupted records available for heatmap metric '{metric_name}'.")

    height = max(4.0, 0.7 * len(corruption_names) + 2.0)
    width = max(6.0, 1.2 * len(severities) + 3.0)

    fig, ax = plt.subplots(figsize=(width, height))
    image = ax.imshow(matrix, aspect="auto")

    ax.set_xticks(np.arange(len(severities)))
    ax.set_xticklabels([str(sev) for sev in severities])
    ax.set_yticks(np.arange(len(corruption_names)))
    ax.set_yticklabels(corruption_names)
    ax.set_xlabel("Severity")
    ax.set_ylabel("Corruption")
    ax.set_title(title or f"{metric_name} heatmap")

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label(metric_name)

    if annotate:
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text = "-" if math.isnan(float(value)) else f"{float(value):.{digits}f}"
                ax.text(j, i, text, ha="center", va="center")

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_metric_lines(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    save_path: str | Path,
    title: Optional[str] = None,
    show_clean_reference: bool = True,
    digits: int = 4,
) -> Path:
    save_path = _to_path(save_path)
    _ensure_parent_dir(save_path)

    corruption_names, severities, matrix = build_metric_matrix(records, metric_name)
    clean_record = get_clean_record(records)
    clean_value = math.nan
    if clean_record is not None:
        clean_value = _safe_float(clean_record.get("metrics", {}).get(metric_name))

    if len(corruption_names) == 0 or len(severities) == 0:
        raise ValueError(f"No corrupted records available for line plot metric '{metric_name}'.")

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    x = np.array(severities, dtype=np.int32)

    for row_idx, corr_name in enumerate(corruption_names):
        y = matrix[row_idx]
        ax.plot(x, y, marker="o", linewidth=2, label=corr_name)

    if show_clean_reference and _is_finite(clean_value):
        ax.axhline(clean_value, linestyle="--", linewidth=1.5, label=f"clean={clean_value:.{digits}f}")

    ax.set_xlabel("Severity")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"{metric_name} by severity")
    ax.set_xticks(x.tolist())
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_primary_summary_bar(
    summary: Mapping[str, Any],
    save_path: str | Path,
    title: Optional[str] = None,
    digits: int = 4,
) -> Path:
    save_path = _to_path(save_path)
    _ensure_parent_dir(save_path)

    primary_metric = str(summary.get("primary_metric", "metric"))
    clean_value = _safe_float(summary.get("clean_primary_metric"))
    mpc = _safe_float(summary.get("mpc"))
    worst_case = summary.get("worst_case", None)

    worst_value = math.nan
    worst_label = "worst-case"
    if isinstance(worst_case, Mapping):
        worst_value = _safe_float(worst_case.get(primary_metric))
        corr = worst_case.get("corruption", "unknown")
        sev = worst_case.get("severity", "unknown")
        worst_label = f"worst-case\n({corr}, s={sev})"

    labels = ["clean", "mPC", worst_label]
    values = [clean_value, mpc, worst_value]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    bars = ax.bar(np.arange(len(labels)), values)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(primary_metric)
    ax.set_title(title or f"{primary_metric}: clean vs robustness summary")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, values):
        text = "-" if math.isnan(value) else f"{value:.{digits}f}"
        y = 0.0 if math.isnan(value) else value
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            text,
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def plot_per_corruption_bar(
    records: Sequence[Mapping[str, Any]],
    metric_name: str,
    save_path: str | Path,
    title: Optional[str] = None,
    digits: int = 4,
) -> Path:
    save_path = _to_path(save_path)
    _ensure_parent_dir(save_path)

    corruption_names, means = build_per_corruption_means(records, metric_name)
    if len(corruption_names) == 0:
        raise ValueError(f"No corrupted records available for bar plot metric '{metric_name}'.")

    fig_width = max(8.0, 0.8 * len(corruption_names) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.0))

    bars = ax.bar(np.arange(len(corruption_names)), means)
    ax.set_xticks(np.arange(len(corruption_names)))
    ax.set_xticklabels(corruption_names, rotation=25, ha="right")
    ax.set_ylabel(metric_name)
    ax.set_title(title or f"Mean {metric_name} by corruption")
    ax.grid(True, axis="y", alpha=0.3)

    for bar, value in zip(bars, means):
        value = _safe_float(value)
        text = "-" if math.isnan(value) else f"{value:.{digits}f}"
        y = 0.0 if math.isnan(value) else value
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            y,
            text,
            ha="center",
            va="bottom",
            rotation=0,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return save_path


def build_summary_lines(
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    digits: int = 4,
) -> List[str]:
    lines: List[str] = []

    dataset_name = payload.get("dataset_name", "unknown")
    dataset_class = payload.get("dataset_class", "unknown")
    split = payload.get("split", "unknown")
    checkpoint = payload.get("checkpoint", "unknown")

    lines.append("Robustness Summary")
    lines.append("=" * 80)
    lines.append(f"dataset_name: {dataset_name}")
    lines.append(f"dataset_class: {dataset_class}")
    lines.append(f"split: {split}")
    lines.append(f"checkpoint: {checkpoint}")
    lines.append("")

    primary_metric = str(summary.get("primary_metric", "metric"))
    lines.append(f"primary_metric: {primary_metric}")
    lines.append(f"clean_primary_metric: {_format_metric_value(summary.get('clean_primary_metric'), digits)}")
    lines.append(f"mPC: {_format_metric_value(summary.get('mpc'), digits)}")
    lines.append(f"rPC: {_format_metric_value(summary.get('rpc'), digits)}")
    lines.append(f"avg_drop: {_format_metric_value(summary.get('avg_drop'), digits)}")
    lines.append(f"num_conditions: {summary.get('num_conditions', 'unknown')}")
    lines.append(f"num_corrupted_conditions: {summary.get('num_corrupted_conditions', 'unknown')}")
    lines.append("")

    worst_case = summary.get("worst_case", None)
    if isinstance(worst_case, Mapping):
        lines.append("worst_case")
        lines.append("-" * 80)
        lines.append(f"condition: {worst_case.get('condition')}")
        lines.append(f"corruption: {worst_case.get('corruption')}")
        lines.append(f"severity: {worst_case.get('severity')}")
        lines.append(
            f"{primary_metric}: {_format_metric_value(worst_case.get(primary_metric), digits)}"
        )
        lines.append("")

    per_corruption = summary.get("per_corruption", {})
    if isinstance(per_corruption, Mapping) and len(per_corruption) > 0:
        lines.append("per_corruption")
        lines.append("-" * 80)
        for name in sorted(per_corruption.keys()):
            entry = per_corruption[name]
            lines.append(
                f"{name}: mean={_format_metric_value(entry.get('mean_primary_metric'), digits)}, "
                f"worst={_format_metric_value(entry.get('worst_primary_metric'), digits)}, "
                f"best={_format_metric_value(entry.get('best_primary_metric'), digits)}, "
                f"num_severities={entry.get('num_severities', 'unknown')}"
            )

    return lines


def save_summary_text(
    payload: Mapping[str, Any],
    summary: Mapping[str, Any],
    save_path: str | Path,
    digits: int = 4,
) -> Path:
    save_path = _to_path(save_path)
    _ensure_parent_dir(save_path)
    lines = build_summary_lines(payload=payload, summary=summary, digits=digits)
    with save_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(lines))
        f.write("\n")
    return save_path


def visualize_robustness_results(
    results: str | Path | Mapping[str, Any],
    output_dir: Optional[str | Path] = None,
    primary_metric: Optional[str] = None,
    extra_metrics: Optional[Sequence[str]] = None,
    digits: int = 4,
) -> Dict[str, Any]:
    payload = load_robustness_results(results)
    records = payload.get("records", [])
    if not isinstance(records, list) or len(records) == 0:
        raise ValueError("robustness results must contain a non-empty 'records' list.")

    summary = payload.get("summary", {})
    if not isinstance(summary, Mapping):
        summary = {}

    if primary_metric is None:
        primary_metric = str(summary.get("primary_metric", "auc"))

    if len(summary) == 0 or summary.get("primary_metric", None) is None:
        summary = compute_summary_from_records(records=records, primary_metric=primary_metric)

    if output_dir is None:
        if isinstance(results, (str, Path)):
            results_path = _to_path(results)
            output_dir = results_path.parent / "plots"
        else:
            output_dir = Path("outputs/robustness_plots")

    output_dir = _ensure_dir(output_dir)

    metric_names = extract_metric_names(records)
    if primary_metric not in metric_names:
        metric_names = [primary_metric] + metric_names

    if extra_metrics is None:
        candidates = ["accuracy", "f1", "precision", "recall", "loss"]
        extra_metrics = [name for name in candidates if name in metric_names and name != primary_metric]

    saved_paths: Dict[str, Any] = {
        "output_dir": str(output_dir),
        "primary_metric": primary_metric,
        "figures": {},
    }

    heatmap_path = output_dir / f"{primary_metric}_heatmap.png"
    line_path = output_dir / f"{primary_metric}_severity_lines.png"
    summary_bar_path = output_dir / f"{primary_metric}_summary_bar.png"
    corruption_bar_path = output_dir / f"{primary_metric}_mean_by_corruption.png"

    plot_metric_heatmap(
        records=records,
        metric_name=primary_metric,
        save_path=heatmap_path,
        title=f"{primary_metric} heatmap",
        digits=digits,
    )
    plot_metric_lines(
        records=records,
        metric_name=primary_metric,
        save_path=line_path,
        title=f"{primary_metric} by severity",
        digits=digits,
    )
    plot_primary_summary_bar(
        summary=summary,
        save_path=summary_bar_path,
        title=f"{primary_metric}: clean vs mPC vs worst-case",
        digits=digits,
    )
    plot_per_corruption_bar(
        records=records,
        metric_name=primary_metric,
        save_path=corruption_bar_path,
        title=f"Mean {primary_metric} by corruption",
        digits=digits,
    )

    saved_paths["figures"][f"{primary_metric}_heatmap"] = str(heatmap_path)
    saved_paths["figures"][f"{primary_metric}_severity_lines"] = str(line_path)
    saved_paths["figures"][f"{primary_metric}_summary_bar"] = str(summary_bar_path)
    saved_paths["figures"][f"{primary_metric}_mean_by_corruption"] = str(corruption_bar_path)

    for metric_name in extra_metrics:
        if metric_name == primary_metric:
            continue
        if metric_name not in metric_names:
            continue

        metric_heatmap_path = output_dir / f"{metric_name}_heatmap.png"
        metric_line_path = output_dir / f"{metric_name}_severity_lines.png"

        plot_metric_heatmap(
            records=records,
            metric_name=metric_name,
            save_path=metric_heatmap_path,
            title=f"{metric_name} heatmap",
            digits=digits,
        )
        plot_metric_lines(
            records=records,
            metric_name=metric_name,
            save_path=metric_line_path,
            title=f"{metric_name} by severity",
            digits=digits,
        )

        saved_paths["figures"][f"{metric_name}_heatmap"] = str(metric_heatmap_path)
        saved_paths["figures"][f"{metric_name}_severity_lines"] = str(metric_line_path)

    summary_json_path = output_dir / "robustness_summary.json"
    summary_txt_path = output_dir / "robustness_summary.txt"
    manifest_json_path = output_dir / "visualization_manifest.json"

    save_json(dict(summary), summary_json_path)
    save_summary_text(payload=payload, summary=summary, save_path=summary_txt_path, digits=digits)
    save_json(saved_paths, manifest_json_path)

    saved_paths["summary_json"] = str(summary_json_path)
    saved_paths["summary_txt"] = str(summary_txt_path)
    saved_paths["manifest_json"] = str(manifest_json_path)
    return saved_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize robustness evaluation results."
    )
    parser.add_argument(
        "--input_json",
        type=str,
        required=True,
        help="Path to robustness_results.json",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save plots. Default: <input_json_dir>/plots",
    )
    parser.add_argument(
        "--primary_metric",
        type=str,
        default=None,
        help="Primary metric to visualize. Default: summary.primary_metric or auc",
    )
    parser.add_argument(
        "--extra_metrics",
        type=str,
        default=None,
        help="Comma-separated extra metrics to plot. Example: accuracy,f1,loss",
    )
    parser.add_argument(
        "--digits",
        type=int,
        default=4,
        help="Number of digits for text annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    extra_metrics = None
    if args.extra_metrics is not None:
        extra_metrics = [x.strip() for x in args.extra_metrics.split(",") if x.strip()]

    saved = visualize_robustness_results(
        results=args.input_json,
        output_dir=args.output_dir,
        primary_metric=args.primary_metric,
        extra_metrics=extra_metrics,
        digits=int(args.digits),
    )

    print("=" * 80)
    print("Robustness Visualization Finished")
    print("=" * 80)
    print(f"output_dir: {saved['output_dir']}")
    print(f"primary_metric: {saved['primary_metric']}")
    print("saved figures:")
    for key, value in saved["figures"].items():
        print(f"  - {key}: {value}")
    print(f"summary_json: {saved['summary_json']}")
    print(f"summary_txt: {saved['summary_txt']}")
    print(f"manifest_json: {saved['manifest_json']}")


if __name__ == "__main__":
    main()