from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DEFAULT_METRICS = (
    "fake_recall",
    "balanced_accuracy",
    "auc",
    "f1",
    "precision",
    "recall",
    "accuracy",
)

SUMMARY_METRICS = (
    "fake_recall_weighted_mean",
    "balanced_accuracy_weighted_mean",
    "auc_weighted_mean",
    "f1_weighted_mean",
)


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------


def _to_path(path: str | Path) -> Path:
    return path if isinstance(path, Path) else Path(path)



def ensure_dir(path: str | Path) -> Path:
    out = _to_path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def ensure_parent(path: str | Path) -> Path:
    out = _to_path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out



def save_json(data: Mapping[str, Any], path: str | Path, indent: int = 2) -> None:
    path = ensure_parent(path)
    with _to_path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)



def safe_filename(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in str(name))



def _numeric_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")



def _drop_overall_row(df: pd.DataFrame) -> pd.DataFrame:
    if "group_value" not in df.columns:
        return df.copy()
    return df.loc[df["group_value"].astype(str) != "__overall__"].copy()



def _sort_group_df(
    df: pd.DataFrame,
    metric: str,
    descending: bool = True,
    top_k: Optional[int] = None,
) -> pd.DataFrame:
    work = _drop_overall_row(df)
    metric_s = _numeric_series(work, metric)
    work = work.loc[metric_s.notna()].copy()
    if len(work) == 0:
        return work
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    if "group_value" in work.columns:
        work["group_value"] = work["group_value"].astype(str)
    sort_cols = [metric]
    ascending = [not descending]
    if "group_value" in work.columns:
        sort_cols.append("group_value")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending, kind="stable")
    if top_k is not None and top_k > 0:
        work = work.head(int(top_k))
    return work.reset_index(drop=True)



def _resolve_metric_label(metric: str) -> str:
    return str(metric).replace("_", " ").title()



def _figsize_for_bars(n_items: int, base_w: float = 10.0, row_h: float = 0.45, min_h: float = 4.0) -> Tuple[float, float]:
    return (base_w, max(min_h, row_h * max(n_items, 1) + 1.5))



def _close_or_show(fig: plt.Figure, save_path: Optional[str | Path], dpi: int = 150) -> None:
    if save_path is not None:
        save_path = ensure_parent(save_path)
        fig.savefig(str(save_path), dpi=dpi, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


# -----------------------------------------------------------------------------
# loaders
# -----------------------------------------------------------------------------


def load_overall_metrics(result_dir: str | Path, split: str = "test") -> Dict[str, Any]:
    result_dir = _to_path(result_dir)
    json_path = result_dir / f"overall_{split}.json"
    if not json_path.exists():
        raise FileNotFoundError(f"Overall metrics JSON not found: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        return json.load(f)



def load_group_summary(result_dir: str | Path, split: str = "test") -> pd.DataFrame:
    result_dir = _to_path(result_dir)
    csv_path = result_dir / f"group_summary_{split}.csv"
    json_path = result_dir / f"group_summary_{split}.json"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data)
        return pd.DataFrame(rows)
    raise FileNotFoundError(
        f"Could not find group summary file. Expected one of: {csv_path}, {json_path}"
    )



def load_group_table(result_dir: str | Path, group_name: str) -> pd.DataFrame:
    result_dir = _to_path(result_dir)
    safe_name = safe_filename(group_name)
    csv_path = result_dir / "groups" / f"{safe_name}.csv"
    json_path = result_dir / "groups" / f"{safe_name}.json"

    if csv_path.exists():
        return pd.read_csv(csv_path)
    if json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data)
        return pd.DataFrame(rows)
    raise FileNotFoundError(
        f"Could not find group table for '{group_name}'. Expected one of: {csv_path}, {json_path}"
    )



def load_all_group_tables(result_dir: str | Path) -> Dict[str, pd.DataFrame]:
    groups_dir = _to_path(result_dir) / "groups"
    if not groups_dir.exists():
        raise FileNotFoundError(f"Groups directory not found: {groups_dir}")

    tables: Dict[str, pd.DataFrame] = {}
    for path in sorted(groups_dir.glob("*.csv")):
        tables[path.stem] = pd.read_csv(path)
    if tables:
        return tables

    for path in sorted(groups_dir.glob("*.json")):
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        rows = data.get("rows", data)
        tables[path.stem] = pd.DataFrame(rows)
    return tables


# -----------------------------------------------------------------------------
# table preparation
# -----------------------------------------------------------------------------


def prepare_group_table_for_plot(
    group_table: pd.DataFrame,
    metric: str = "fake_recall",
    top_k: Optional[int] = None,
    descending: bool = True,
    min_group_size: int = 0,
) -> pd.DataFrame:
    work = group_table.copy()
    if "n" in work.columns and min_group_size > 0:
        n_s = _numeric_series(work, "n")
        work = work.loc[n_s >= int(min_group_size)].copy()
    work = _sort_group_df(work, metric=metric, descending=descending, top_k=top_k)
    if "rank" not in work.columns:
        work.insert(0, "rank", np.arange(1, len(work) + 1))
    return work.reset_index(drop=True)



def build_group_comparison_frame(
    result_dirs: Mapping[str, str | Path],
    group_name: str,
    metric: str = "fake_recall",
    min_group_size: int = 0,
    include_counts: bool = True,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, result_dir in result_dirs.items():
        table = load_group_table(result_dir, group_name=group_name)
        table = _drop_overall_row(table)
        if len(table) == 0:
            continue

        if "n" in table.columns and min_group_size > 0:
            n_s = _numeric_series(table, "n")
            table = table.loc[n_s >= int(min_group_size)].copy()

        value_s = _numeric_series(table, metric)
        table = table.loc[value_s.notna()].copy()
        if len(table) == 0:
            continue

        for _, row in table.iterrows():
            out = {
                "model": str(model_name),
                "group_name": str(row.get("group_name", group_name)),
                "group_value": str(row.get("group_value", row.get("group_key", "unknown"))),
                metric: float(row[metric]),
            }
            if include_counts:
                for key in ("n", "n_real", "n_fake"):
                    if key in table.columns:
                        value = row.get(key, None)
                        out[key] = None if pd.isna(value) else int(value)
            rows.append(out)
    return pd.DataFrame(rows)



def build_overall_comparison_frame(
    result_dirs: Mapping[str, str | Path],
    split: str = "test",
    metrics: Sequence[str] = DEFAULT_METRICS,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for model_name, result_dir in result_dirs.items():
        metrics_dict = load_overall_metrics(result_dir=result_dir, split=split)
        row: Dict[str, Any] = {"model": str(model_name)}
        for metric in metrics:
            value = metrics_dict.get(metric, None)
            row[metric] = None if value is None else float(value)
        if "loss" in metrics_dict:
            row["loss"] = float(metrics_dict["loss"])
        if "n" in metrics_dict:
            row["n"] = int(metrics_dict["n"])
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# plotting: single-table views
# -----------------------------------------------------------------------------


def plot_group_metric_bar(
    group_table: pd.DataFrame,
    metric: str = "fake_recall",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    top_k: Optional[int] = None,
    descending: bool = True,
    min_group_size: int = 0,
    annotate: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    plot_df = prepare_group_table_for_plot(
        group_table=group_table,
        metric=metric,
        top_k=top_k,
        descending=descending,
        min_group_size=min_group_size,
    )
    if len(plot_df) == 0:
        raise ValueError(f"No valid rows to plot for metric='{metric}'.")

    labels = plot_df["group_value"].astype(str).tolist()
    values = pd.to_numeric(plot_df[metric], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_figsize_for_bars(len(plot_df)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(_resolve_metric_label(metric))
    ax.set_ylabel("Group")
    ax.set_xlim(0.0, 1.0 if np.nanmax(values) <= 1.0 else float(np.nanmax(values) * 1.05))
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    default_title = f"{plot_df.iloc[0].get('group_name', 'group')} - {_resolve_metric_label(metric)}"
    ax.set_title(title or default_title)

    if annotate:
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(
                float(bar.get_width()) + 0.01,
                bar.get_y() + bar.get_height() / 2.0,
                f"{value:.3f}",
                va="center",
                fontsize=9,
            )

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return plot_df



def plot_group_sample_count_bar(
    group_table: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    top_k: Optional[int] = None,
    descending: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    if "n" not in group_table.columns:
        raise ValueError("group_table must contain 'n' column to plot sample counts.")

    work = _drop_overall_row(group_table)
    work["n"] = pd.to_numeric(work["n"], errors="coerce")
    work = work.loc[work["n"].notna()].copy()
    work = work.sort_values(["n", "group_value"], ascending=[not descending, True], kind="stable")
    if top_k is not None and top_k > 0:
        work = work.head(int(top_k))
    if len(work) == 0:
        raise ValueError("No valid rows to plot sample counts.")

    labels = work["group_value"].astype(str).tolist()
    values = work["n"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_figsize_for_bars(len(work)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Num Samples")
    ax.set_ylabel("Group")
    ax.grid(axis="x", linestyle="--", alpha=0.35)

    default_title = f"{work.iloc[0].get('group_name', 'group')} - Sample Count"
    ax.set_title(title or default_title)

    for bar, value in zip(bars, values):
        ax.text(
            float(bar.get_width()) + max(float(np.nanmax(values)) * 0.01, 1.0),
            bar.get_y() + bar.get_height() / 2.0,
            f"{int(value)}",
            va="center",
            fontsize=9,
        )

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return work.reset_index(drop=True)



def plot_group_metric_scatter(
    group_table: pd.DataFrame,
    x: str = "n",
    y: str = "fake_recall",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    annotate: bool = False,
    dpi: int = 150,
) -> pd.DataFrame:
    work = _drop_overall_row(group_table)
    work[x] = pd.to_numeric(work[x], errors="coerce")
    work[y] = pd.to_numeric(work[y], errors="coerce")
    work = work.loc[work[x].notna() & work[y].notna()].copy()
    if len(work) == 0:
        raise ValueError(f"No valid rows to plot scatter for x='{x}', y='{y}'.")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(work[x].to_numpy(dtype=float), work[y].to_numpy(dtype=float), alpha=0.8)
    ax.set_xlabel(_resolve_metric_label(x))
    ax.set_ylabel(_resolve_metric_label(y))
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.set_title(title or f"{_resolve_metric_label(y)} vs {_resolve_metric_label(x)}")

    if annotate and "group_value" in work.columns:
        for _, row in work.iterrows():
            ax.annotate(str(row["group_value"]), (float(row[x]), float(row[y])), fontsize=8, alpha=0.85)

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return work.reset_index(drop=True)


# -----------------------------------------------------------------------------
# plotting: summaries and model comparisons
# -----------------------------------------------------------------------------


def plot_group_summary_bar(
    summary_df: pd.DataFrame,
    metric: str = "fake_recall_weighted_mean",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    descending: bool = True,
    annotate: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    work = summary_df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.loc[work[metric].notna()].copy()
    if len(work) == 0:
        raise ValueError(f"No valid rows to plot for summary metric='{metric}'.")

    sort_cols = [metric]
    ascending = [not descending]
    if "group_name" in work.columns:
        sort_cols.append("group_name")
        ascending.append(True)
    work = work.sort_values(sort_cols, ascending=ascending, kind="stable").reset_index(drop=True)

    labels = work["group_name"].astype(str).tolist() if "group_name" in work.columns else [str(i) for i in range(len(work))]
    values = work[metric].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=_figsize_for_bars(len(work)))
    y = np.arange(len(labels))
    bars = ax.barh(y, values)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(_resolve_metric_label(metric))
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.set_title(title or f"Group Summary - {_resolve_metric_label(metric)}")
    ax.set_xlim(0.0, 1.0 if np.nanmax(values) <= 1.0 else float(np.nanmax(values) * 1.05))

    if annotate:
        for bar, value in zip(bars, values):
            ax.text(float(bar.get_width()) + 0.01, bar.get_y() + bar.get_height() / 2.0, f"{value:.3f}", va="center", fontsize=9)

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return work



def plot_overall_metrics_bar(
    overall_df: pd.DataFrame,
    metrics: Sequence[str] = ("fake_recall", "auc", "f1", "balanced_accuracy"),
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> pd.DataFrame:
    if "model" not in overall_df.columns:
        raise ValueError("overall_df must contain a 'model' column.")

    work = overall_df.copy()
    valid_metrics = [metric for metric in metrics if metric in work.columns]
    if not valid_metrics:
        raise ValueError(f"None of the requested metrics exist in overall_df: {metrics}")

    melted = work[["model", *valid_metrics]].melt(id_vars="model", var_name="metric", value_name="value")
    melted["value"] = pd.to_numeric(melted["value"], errors="coerce")
    melted = melted.loc[melted["value"].notna()].copy()
    if len(melted) == 0:
        raise ValueError("No valid overall metrics to plot.")

    models = list(dict.fromkeys(melted["model"].astype(str).tolist()))
    metrics_list = list(dict.fromkeys(melted["metric"].astype(str).tolist()))
    x = np.arange(len(models), dtype=float)
    width = 0.8 / max(len(metrics_list), 1)

    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(models)), 5.5))
    for i, metric in enumerate(metrics_list):
        sub = melted.loc[melted["metric"] == metric].copy()
        sub = sub.set_index("model").reindex(models)
        values = sub["value"].to_numpy(dtype=float)
        ax.bar(x + (i - (len(metrics_list) - 1) / 2.0) * width, values, width=width, label=_resolve_metric_label(metric))

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=0)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title or "Semi-Truths Overall Metrics")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return work



def plot_model_group_heatmap(
    comparison_df: pd.DataFrame,
    metric: str = "fake_recall",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    sort_rows_by_mean: bool = True,
    sort_cols_alpha: bool = True,
    dpi: int = 150,
) -> pd.DataFrame:
    required = {"model", "group_value", metric}
    missing = required.difference(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df is missing required columns: {sorted(missing)}")

    work = comparison_df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.loc[work[metric].notna()].copy()
    if len(work) == 0:
        raise ValueError(f"No valid rows to plot heatmap for metric='{metric}'.")

    pivot = work.pivot_table(index="model", columns="group_value", values=metric, aggfunc="mean")
    if sort_rows_by_mean:
        pivot["__mean__"] = pivot.mean(axis=1)
        pivot = pivot.sort_values("__mean__", ascending=False)
        pivot = pivot.drop(columns="__mean__")
    if sort_cols_alpha:
        pivot = pivot.reindex(sorted(pivot.columns, key=lambda x: str(x)), axis=1)

    values = pivot.to_numpy(dtype=float)
    fig_w = max(8.0, 0.6 * max(1, pivot.shape[1]) + 3.0)
    fig_h = max(4.5, 0.55 * max(1, pivot.shape[0]) + 2.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(values, aspect="auto", interpolation="nearest", vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels([str(c) for c in pivot.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels([str(i) for i in pivot.index])
    ax.set_xlabel("Group Value")
    ax.set_ylabel("Model")
    ax.set_title(title or f"Model vs Group - {_resolve_metric_label(metric)}")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            val = values[i, j]
            if np.isnan(val):
                continue
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(_resolve_metric_label(metric))

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return pivot.reset_index().rename(columns={"index": "model"})



def plot_model_group_bar(
    comparison_df: pd.DataFrame,
    metric: str = "fake_recall",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
    dpi: int = 150,
) -> pd.DataFrame:
    required = {"model", "group_value", metric}
    missing = required.difference(comparison_df.columns)
    if missing:
        raise ValueError(f"comparison_df is missing required columns: {sorted(missing)}")

    work = comparison_df.copy()
    work[metric] = pd.to_numeric(work[metric], errors="coerce")
    work = work.loc[work[metric].notna()].copy()
    if len(work) == 0:
        raise ValueError(f"No valid rows to plot grouped comparison for metric='{metric}'.")

    models = list(dict.fromkeys(work["model"].astype(str).tolist()))
    groups = list(dict.fromkeys(work["group_value"].astype(str).tolist()))
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(len(models), 1)

    fig_w = max(9.0, 0.8 * max(len(groups), 1) + 2.5)
    fig, ax = plt.subplots(figsize=(fig_w, 5.8))
    for i, model in enumerate(models):
        sub = work.loc[work["model"].astype(str) == model].copy()
        sub = sub.set_index("group_value").reindex(groups)
        values = pd.to_numeric(sub[metric], errors="coerce").to_numpy(dtype=float)
        ax.bar(x + (i - (len(models) - 1) / 2.0) * width, values, width=width, label=model)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel(_resolve_metric_label(metric))
    ax.set_xlabel("Group Value")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()

    inferred_group = None
    if "group_name" in work.columns and work["group_name"].notna().any():
        inferred_group = str(work["group_name"].dropna().iloc[0])
    ax.set_title(title or f"{inferred_group or 'Group'} - {_resolve_metric_label(metric)}")

    fig.tight_layout()
    _close_or_show(fig, save_path=save_path, dpi=dpi)
    return work.reset_index(drop=True)


# -----------------------------------------------------------------------------
# convenience wrappers for result directories
# -----------------------------------------------------------------------------


def export_default_group_figures(
    result_dir: str | Path,
    output_dir: str | Path,
    split: str = "test",
    metrics: Sequence[str] = ("fake_recall", "balanced_accuracy", "auc"),
    summary_metrics: Sequence[str] = SUMMARY_METRICS,
    min_group_size: int = 0,
    top_k: Optional[int] = None,
    dpi: int = 150,
) -> Dict[str, str]:
    result_dir = _to_path(result_dir)
    output_dir = ensure_dir(output_dir)
    saved: Dict[str, str] = {}

    summary_df = load_group_summary(result_dir=result_dir, split=split)
    for metric in summary_metrics:
        if metric not in summary_df.columns:
            continue
        save_path = output_dir / f"group_summary_{safe_filename(metric)}.png"
        plot_group_summary_bar(
            summary_df=summary_df,
            metric=metric,
            save_path=save_path,
            dpi=dpi,
        )
        saved[f"summary::{metric}"] = save_path.as_posix()

    group_tables = load_all_group_tables(result_dir=result_dir)
    for group_name, group_table in group_tables.items():
        for metric in metrics:
            if metric not in group_table.columns:
                continue
            save_path = output_dir / f"{safe_filename(group_name)}__{safe_filename(metric)}.png"
            plot_group_metric_bar(
                group_table=group_table,
                metric=metric,
                save_path=save_path,
                min_group_size=min_group_size,
                top_k=top_k,
                dpi=dpi,
            )
            saved[f"{group_name}::{metric}"] = save_path.as_posix()

        if "n" in group_table.columns:
            count_path = output_dir / f"{safe_filename(group_name)}__count.png"
            plot_group_sample_count_bar(
                group_table=group_table,
                save_path=count_path,
                top_k=top_k,
                dpi=dpi,
            )
            saved[f"{group_name}::count"] = count_path.as_posix()

    manifest_path = output_dir / "figure_manifest.json"
    save_json({"saved": saved}, manifest_path)
    saved["manifest"] = manifest_path.as_posix()
    return saved



def export_model_comparison_figures(
    result_dirs: Mapping[str, str | Path],
    output_dir: str | Path,
    split: str = "test",
    overall_metrics: Sequence[str] = ("fake_recall", "auc", "f1", "balanced_accuracy"),
    group_names: Optional[Sequence[str]] = None,
    group_metric: str = "fake_recall",
    min_group_size: int = 0,
    dpi: int = 150,
) -> Dict[str, str]:
    output_dir = ensure_dir(output_dir)
    saved: Dict[str, str] = {}

    overall_df = build_overall_comparison_frame(result_dirs=result_dirs, split=split)
    if len(overall_df) > 0:
        save_path = output_dir / "overall_metrics.png"
        plot_overall_metrics_bar(
            overall_df=overall_df,
            metrics=overall_metrics,
            save_path=save_path,
            dpi=dpi,
        )
        overall_csv = output_dir / "overall_metrics.csv"
        overall_df.to_csv(overall_csv, index=False)
        saved["overall_plot"] = save_path.as_posix()
        saved["overall_csv"] = overall_csv.as_posix()

    if group_names is None:
        first_result_dir = next(iter(result_dirs.values()))
        group_names = sorted(load_all_group_tables(first_result_dir).keys())

    for group_name in group_names:
        comparison_df = build_group_comparison_frame(
            result_dirs=result_dirs,
            group_name=group_name,
            metric=group_metric,
            min_group_size=min_group_size,
        )
        if len(comparison_df) == 0:
            continue

        csv_path = output_dir / f"{safe_filename(group_name)}__comparison.csv"
        comparison_df.to_csv(csv_path, index=False)
        saved[f"{group_name}::csv"] = csv_path.as_posix()

        heatmap_path = output_dir / f"{safe_filename(group_name)}__heatmap.png"
        plot_model_group_heatmap(
            comparison_df=comparison_df,
            metric=group_metric,
            save_path=heatmap_path,
            dpi=dpi,
        )
        saved[f"{group_name}::heatmap"] = heatmap_path.as_posix()

        bar_path = output_dir / f"{safe_filename(group_name)}__bar.png"
        plot_model_group_bar(
            comparison_df=comparison_df,
            metric=group_metric,
            save_path=bar_path,
            dpi=dpi,
        )
        saved[f"{group_name}::bar"] = bar_path.as_posix()

    manifest_path = output_dir / "comparison_manifest.json"
    save_json({"saved": saved}, manifest_path)
    saved["manifest"] = manifest_path.as_posix()
    return saved


__all__ = [
    "DEFAULT_METRICS",
    "SUMMARY_METRICS",
    "build_group_comparison_frame",
    "build_overall_comparison_frame",
    "export_default_group_figures",
    "export_model_comparison_figures",
    "load_all_group_tables",
    "load_group_summary",
    "load_group_table",
    "load_overall_metrics",
    "plot_group_metric_bar",
    "plot_group_metric_scatter",
    "plot_group_sample_count_bar",
    "plot_group_summary_bar",
    "plot_model_group_bar",
    "plot_model_group_heatmap",
    "plot_overall_metrics_bar",
    "prepare_group_table_for_plot",
    "safe_filename",
    "save_json",
]
