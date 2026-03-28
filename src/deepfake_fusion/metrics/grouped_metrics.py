from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .classification import (
    compute_classification_metrics,
    compute_confusion_details,
    probs_to_preds,
)
from ..utils.semitruths_metadata import (
    add_group_key_column,
    get_available_group_columns,
    sanitize_scalar,
)

ArrayLike = Union[np.ndarray, Sequence[Any], pd.Series]
GroupBySpec = Union[str, Sequence[str]]


# -----------------------------------------------------------------------------
# internal helpers
# -----------------------------------------------------------------------------


def _to_numpy(values: Any) -> np.ndarray:
    if isinstance(values, np.ndarray):
        return values
    if isinstance(values, pd.Series):
        return values.to_numpy()
    return np.asarray(values)



def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(out):
        return None
    return out



def _safe_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None



def _as_jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return None if np.isnan(value) else value
    if isinstance(value, float) and np.isnan(value):
        return None
    return sanitize_scalar(value)



def _normalize_group_spec(group_by: GroupBySpec) -> List[str]:
    if isinstance(group_by, str):
        return [group_by]
    return [str(col) for col in group_by]



def _group_label(group_columns: Sequence[str]) -> str:
    if not group_columns:
        return "overall"
    if len(group_columns) == 1:
        return group_columns[0]
    return " + ".join(group_columns)



def _require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")



def _prepare_prediction_frame(
    records: pd.DataFrame,
    y_true_col: str = "label",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    threshold: float = 0.5,
) -> pd.DataFrame:
    """평가용 record dataframe 정리.

    필수:
    - y_true_col

    선택:
    - y_prob_col 또는 y_pred_col

    y_pred_col이 없고 y_prob_col이 있으면 threshold로 binary prediction 생성.
    """
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    _require_columns(records, [y_true_col])

    out = records.copy().reset_index(drop=True)
    out[y_true_col] = pd.to_numeric(out[y_true_col], errors="coerce")
    out = out.loc[out[y_true_col].notna()].copy()
    out[y_true_col] = out[y_true_col].astype(np.int64)

    has_prob = y_prob_col in out.columns
    has_pred = y_pred_col in out.columns
    if not has_prob and not has_pred:
        raise ValueError(
            f"At least one of '{y_prob_col}' or '{y_pred_col}' must exist in records."
        )

    if has_prob:
        out[y_prob_col] = pd.to_numeric(out[y_prob_col], errors="coerce")

    if has_pred:
        out[y_pred_col] = pd.to_numeric(out[y_pred_col], errors="coerce")

    if has_prob and not has_pred:
        valid_prob_mask = out[y_prob_col].notna()
        out = out.loc[valid_prob_mask].copy()
        out[y_pred_col] = probs_to_preds(out[y_prob_col].to_numpy(), threshold=threshold)
    elif has_pred:
        valid_pred_mask = out[y_pred_col].notna()
        out = out.loc[valid_pred_mask].copy()
        out[y_pred_col] = out[y_pred_col].astype(np.int64)
        if has_prob:
            # prob가 비어 있으면 유지하되, metrics에서 auc는 자동으로 NaN 처리된다.
            pass

    return out.reset_index(drop=True)



def _binary_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
    details = compute_confusion_details(y_true=y_true, y_pred=y_pred, num_classes=2)
    tn = int(details.get("tn", 0))
    fp = int(details.get("fp", 0))
    fn = int(details.get("fn", 0))
    tp = int(details.get("tp", 0))

    n_real = tn + fp
    n_fake = tp + fn
    n_total = n_real + n_fake

    fake_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fake_f1 = (
        2 * fake_precision * fake_recall / (fake_precision + fake_recall)
        if (fake_precision + fake_recall) > 0
        else 0.0
    )

    real_precision = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    real_recall = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    real_f1 = (
        2 * real_precision * real_recall / (real_precision + real_recall)
        if (real_precision + real_recall) > 0
        else 0.0
    )

    balanced_accuracy = 0.5 * (fake_recall + real_recall)
    prevalence_fake = n_fake / n_total if n_total > 0 else 0.0

    return {
        "n": int(n_total),
        "n_real": int(n_real),
        "n_fake": int(n_fake),
        "prevalence_fake": float(prevalence_fake),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fake_precision": float(fake_precision),
        "fake_recall": float(fake_recall),
        "fake_f1": float(fake_f1),
        "real_precision": float(real_precision),
        "real_recall": float(real_recall),
        "real_f1": float(real_f1),
        "balanced_accuracy": float(balanced_accuracy),
    }



def _compute_metric_row(
    df: pd.DataFrame,
    y_true_col: str,
    y_prob_col: str,
    y_pred_col: str,
) -> Dict[str, Any]:
    y_true = df[y_true_col].to_numpy(dtype=np.int64)
    y_pred = df[y_pred_col].to_numpy(dtype=np.int64)

    y_prob = None
    if y_prob_col in df.columns:
        valid_prob = pd.to_numeric(df[y_prob_col], errors="coerce")
        if valid_prob.notna().all():
            y_prob = valid_prob.to_numpy(dtype=np.float64)

    metrics = compute_classification_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        num_classes=2,
    )
    details = _binary_breakdown(y_true=y_true, y_pred=y_pred)

    row: Dict[str, Any] = {**details}
    for key, value in metrics.items():
        row[key] = _safe_float(value)
    return row


# -----------------------------------------------------------------------------
# public API
# -----------------------------------------------------------------------------


def compute_overall_metrics(
    records: pd.DataFrame,
    y_true_col: str = "label",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """전체 평가 지표 계산.

    반환 예:
    {
        "n": 100,
        "n_real": 40,
        "n_fake": 60,
        "accuracy": 0.91,
        "precision": 0.90,
        "recall": 0.93,
        "f1": 0.91,
        "auc": 0.96,
        "fake_recall": 0.93,
        "real_recall": 0.88,
        ...
    }
    """
    prepared = _prepare_prediction_frame(
        records=records,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        y_pred_col=y_pred_col,
        threshold=threshold,
    )
    return _compute_metric_row(
        prepared,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        y_pred_col=y_pred_col,
    )



def compute_group_metrics(
    records: pd.DataFrame,
    group_by: GroupBySpec,
    y_true_col: str = "label",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    threshold: float = 0.5,
    min_group_size: int = 1,
    dropna_group_values: bool = True,
    include_overall_row: bool = False,
    sort_by: Optional[Sequence[str]] = ("n", "group_value"),
    ascending: Optional[Sequence[bool]] = (False, True),
) -> pd.DataFrame:
    """지정한 group 기준으로 그룹별 성능 계산.

    Parameters
    ----------
    group_by:
        - str: 단일 컬럼 그룹화
        - Sequence[str]: 복수 컬럼 조합 그룹화
    """
    group_columns = _normalize_group_spec(group_by)
    _require_columns(records, group_columns + [y_true_col])

    prepared = _prepare_prediction_frame(
        records=records,
        y_true_col=y_true_col,
        y_prob_col=y_prob_col,
        y_pred_col=y_pred_col,
        threshold=threshold,
    )
    prepared = prepared.copy()

    if dropna_group_values:
        mask = prepared[group_columns].notna().all(axis=1)
        prepared = prepared.loc[mask].copy()

    if len(prepared) == 0:
        return pd.DataFrame(
            columns=[
                "group_name",
                "group_value",
                "group_key",
                "n",
                "n_real",
                "n_fake",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "auc",
                "fake_precision",
                "fake_recall",
                "fake_f1",
                "real_precision",
                "real_recall",
                "real_f1",
                "balanced_accuracy",
                "tn",
                "fp",
                "fn",
                "tp",
            ]
        )

    working = add_group_key_column(prepared, group_columns=group_columns, output_col="group_key")
    group_name = _group_label(group_columns)

    rows: List[Dict[str, Any]] = []
    grouped = working.groupby("group_key", dropna=False, sort=False)
    for group_key, group_df in grouped:
        if len(group_df) < int(min_group_size):
            continue

        metric_row = _compute_metric_row(
            group_df,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            y_pred_col=y_pred_col,
        )

        output: Dict[str, Any] = {
            "group_name": group_name,
            "group_value": group_key,
            "group_key": group_key,
        }
        for col in group_columns:
            if len(group_df) > 0:
                output[col] = _as_jsonable(group_df.iloc[0][col])
            else:
                output[col] = None
        output.update({k: _as_jsonable(v) for k, v in metric_row.items()})
        rows.append(output)

    result = pd.DataFrame(rows)

    if include_overall_row:
        overall = compute_overall_metrics(
            records=prepared,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            y_pred_col=y_pred_col,
            threshold=threshold,
        )
        overall_row: Dict[str, Any] = {
            "group_name": group_name,
            "group_value": "__overall__",
            "group_key": "__overall__",
        }
        for col in group_columns:
            overall_row[col] = "__overall__"
        overall_row.update({k: _as_jsonable(v) for k, v in overall.items()})
        result = pd.concat([pd.DataFrame([overall_row]), result], ignore_index=True)

    if sort_by is not None and len(result) > 0:
        sort_cols = [col for col in sort_by if col in result.columns]
        if sort_cols:
            ascending_flags = list(ascending or [True] * len(sort_cols))
            if len(ascending_flags) < len(sort_cols):
                ascending_flags = ascending_flags + [True] * (len(sort_cols) - len(ascending_flags))
            result = result.sort_values(
                by=sort_cols,
                ascending=ascending_flags[: len(sort_cols)],
                kind="stable",
            ).reset_index(drop=True)

    return result



def compute_group_metrics_many(
    records: pd.DataFrame,
    group_columns: Optional[Sequence[str]] = None,
    y_true_col: str = "label",
    y_prob_col: str = "y_prob",
    y_pred_col: str = "y_pred",
    threshold: float = 0.5,
    min_group_size: int = 1,
    dropna_group_values: bool = True,
    include_overall_row: bool = False,
) -> Dict[str, pd.DataFrame]:
    """여러 group column에 대해 그룹별 metric 표를 한 번에 계산.

    반환 예:
    {
        "method": DataFrame,
        "diffusion_model": DataFrame,
        ...
    }
    """
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    resolved_group_columns = list(group_columns or get_available_group_columns(records))
    tables: Dict[str, pd.DataFrame] = {}
    for group_col in resolved_group_columns:
        if group_col not in records.columns:
            continue
        table = compute_group_metrics(
            records=records,
            group_by=group_col,
            y_true_col=y_true_col,
            y_prob_col=y_prob_col,
            y_pred_col=y_pred_col,
            threshold=threshold,
            min_group_size=min_group_size,
            dropna_group_values=dropna_group_values,
            include_overall_row=include_overall_row,
        )
        tables[group_col] = table
    return tables



def summarize_group_table(
    group_table: pd.DataFrame,
    metric_cols: Sequence[str] = (
        "fake_recall",
        "auc",
        "balanced_accuracy",
        "f1",
    ),
    weight_col: str = "n",
    group_name_col: str = "group_name",
) -> Dict[str, Any]:
    """단일 그룹 metric 표를 요약.

    weighted_mean은 group size(`weight_col`)를 가중치로 사용한다.
    """
    if len(group_table) == 0:
        return {
            "group_name": None,
            "num_groups": 0,
            "total_samples": 0,
        }

    work = group_table.copy()
    if "group_value" in work.columns:
        work = work.loc[work["group_value"] != "__overall__"].copy()

    if len(work) == 0:
        return {
            "group_name": None,
            "num_groups": 0,
            "total_samples": 0,
        }

    weight_series = pd.to_numeric(work.get(weight_col), errors="coerce").fillna(0.0)
    summary: Dict[str, Any] = {
        "group_name": _as_jsonable(work.iloc[0][group_name_col]) if group_name_col in work.columns else None,
        "num_groups": int(len(work)),
        "total_samples": int(weight_series.sum()),
    }

    for metric_col in metric_cols:
        if metric_col not in work.columns:
            continue
        metric_series = pd.to_numeric(work[metric_col], errors="coerce")
        valid_mask = metric_series.notna()
        if not valid_mask.any():
            summary[f"{metric_col}_mean"] = None
            summary[f"{metric_col}_weighted_mean"] = None
            summary[f"{metric_col}_min"] = None
            summary[f"{metric_col}_max"] = None
            continue

        valid_metrics = metric_series.loc[valid_mask]
        valid_weights = weight_series.loc[valid_mask]
        weighted_mean = None
        if float(valid_weights.sum()) > 0:
            weighted_mean = float(np.average(valid_metrics, weights=valid_weights))

        summary[f"{metric_col}_mean"] = float(valid_metrics.mean())
        summary[f"{metric_col}_weighted_mean"] = weighted_mean
        summary[f"{metric_col}_min"] = float(valid_metrics.min())
        summary[f"{metric_col}_max"] = float(valid_metrics.max())

    return summary



def summarize_group_tables(
    group_tables: Mapping[str, pd.DataFrame],
    metric_cols: Sequence[str] = (
        "fake_recall",
        "auc",
        "balanced_accuracy",
        "f1",
    ),
    weight_col: str = "n",
) -> pd.DataFrame:
    """여러 group metric 표를 요약한 dataframe 반환."""
    rows: List[Dict[str, Any]] = []
    for group_name, table in group_tables.items():
        row = summarize_group_table(
            group_table=table,
            metric_cols=metric_cols,
            weight_col=weight_col,
        )
        if row.get("group_name") is None:
            row["group_name"] = group_name
        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["group_name", "num_groups", "total_samples"])
    return pd.DataFrame(rows)



def attach_group_metrics_to_records(
    records: pd.DataFrame,
    group_table: pd.DataFrame,
    group_by: GroupBySpec,
    output_suffix: str = "group_metric",
) -> pd.DataFrame:
    """record dataframe에 그룹 metric을 merge.

    예:
    - method별 fake_recall을 각 샘플 row에 붙여서 error analysis에 활용
    """
    if not isinstance(records, pd.DataFrame):
        records = pd.DataFrame(records)

    group_columns = _normalize_group_spec(group_by)
    merged = add_group_key_column(records.copy(), group_columns=group_columns, output_col="group_key")

    metric_cols = [
        col for col in group_table.columns
        if col not in {"group_name", "group_value", "group_key", *group_columns}
    ]
    rename_map = {col: f"{col}_{output_suffix}" for col in metric_cols}

    right_cols = ["group_key"] + metric_cols
    merge_table = group_table[right_cols].copy().rename(columns=rename_map)
    merged = merged.merge(merge_table, on="group_key", how="left")
    return merged



def metrics_dict_to_frame(metrics: Mapping[str, Any], index_name: str = "overall") -> pd.DataFrame:
    """dict metric 결과를 1-row dataframe으로 변환."""
    row = {k: _as_jsonable(v) for k, v in metrics.items()}
    return pd.DataFrame([row], index=[index_name]).reset_index(names="split")


__all__ = [
    "attach_group_metrics_to_records",
    "compute_group_metrics",
    "compute_group_metrics_many",
    "compute_overall_metrics",
    "metrics_dict_to_frame",
    "summarize_group_table",
    "summarize_group_tables",
]
