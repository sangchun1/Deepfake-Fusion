from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import pandas as pd

PathLike = Union[str, Path]


DEFAULT_COLUMN_CANDIDATES: Dict[str, List[str]] = {
    "sample_id": ["sample_id", "image_id", "id"],
    "filepath": ["filepath", "image_path", "path", "image", "img_path"],
    "label": ["label", "target", "class", "y", "is_fake"],
    "mask_path": ["mask_path", "mask_filepath", "mask", "mask_file"],
    "dataset": ["dataset", "benchmark", "source_dataset", "source_benchmark"],
    "entity": ["entity", "entities", "unique_entities"],
    "method": ["method", "augmentation_method", "augment_method", "technique"],
    "diffusion_model": ["diffusion_model", "model", "generator", "edit_model"],
    "language_model": ["language_model", "llm", "prompt_model", "caption_model"],
    "area_ratio": ["area_ratio", "mask_area_ratio", "area", "post_edit_ratio"],
    "semantic_mag": ["semantic_mag", "semantic_magnitude", "semantic_change", "semantic_score"],
    "scene_complexity": ["scene_complexity", "complexity"],
    "scene_diversity": ["scene_diversity", "diversity"],
    "change_type": ["change_type", "localized_or_diffused", "edit_type", "change"],
    "original_caption": ["original_caption", "source_caption", "caption_before"],
    "edited_caption": ["edited_caption", "perturbed_caption", "caption_after"],
    "original_label": ["original_label", "source_label", "label_before"],
    "edited_label": ["edited_label", "perturbed_label", "label_after"],
}

DEFAULT_GROUP_COLUMNS: List[str] = [
    "dataset",
    "method",
    "diffusion_model",
    "area_ratio_bin",
    "semantic_mag_bin",
    "scene_complexity_bin",
    "scene_diversity_bin",
    "change_type",
]


# -----------------------------------------------------------------------------
# basic helpers
# -----------------------------------------------------------------------------

def get_project_root() -> Path:
    """Repo root 반환.

    현재 파일 위치:
    src/deepfake_fusion/utils/semitruths_metadata.py
    """
    return Path(__file__).resolve().parents[3]


def normalize_slashes(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    return str(value).replace("\\", "/").strip()


def sanitize_scalar(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def find_first_existing_column(
    columns: Iterable[str],
    candidates: Sequence[str],
) -> Optional[str]:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


# -----------------------------------------------------------------------------
# label / subset / method inference
# -----------------------------------------------------------------------------

def normalize_label(label: Any) -> int:
    """Semi-Truths 라벨을 0(real), 1(fake)로 정규화."""
    if isinstance(label, bool):
        return int(label)

    if label is None or pd.isna(label):
        raise ValueError("Label is missing.")

    if isinstance(label, (int, float)):
        label_int = int(label)
        if label_int in (0, 1):
            return label_int

    label_str = str(label).strip().lower()
    if label_str in {"0", "real", "original", "pristine", "clean"}:
        return 0
    if label_str in {"1", "fake", "edited", "augmented", "inpainting", "p2p"}:
        return 1
    if label_str in {"true", "yes"}:
        return 1
    if label_str in {"false", "no"}:
        return 0

    raise ValueError(f"Unsupported label value: {label}")


def infer_subset_from_path(filepath: Any) -> Optional[str]:
    """경로에서 real / inpainting / p2p subset 추론."""
    path_str = str(normalize_slashes(filepath) or "").lower()
    if not path_str:
        return None
    if "original/" in path_str:
        return "real"
    if "inpainting/" in path_str:
        return "inpainting"
    if "p2p/" in path_str or "prompt" in path_str:
        return "p2p"
    return None


def infer_method_from_row(row: Mapping[str, Any]) -> Optional[str]:
    """row 정보에서 augmentation method 추론."""
    method = sanitize_scalar(row.get("method"))
    if method is not None:
        method_str = str(method).strip().lower()
        if method_str in {"real", "original"}:
            return "original"
        if "inpaint" in method_str:
            return "inpainting"
        if method_str in {"p2p", "prompt", "prompt-based-editing", "prompt_based_editing"}:
            return "p2p"
        return str(method)

    subset = sanitize_scalar(row.get("subset"))
    if subset is not None:
        subset_str = str(subset).strip().lower()
        if subset_str == "real":
            return "original"
        if subset_str in {"inpainting", "p2p"}:
            return subset_str

    filepath = row.get("filepath")
    subset = infer_subset_from_path(filepath)
    if subset == "real":
        return "original"
    return subset


def infer_label_from_path(filepath: Any) -> int:
    subset = infer_subset_from_path(filepath)
    if subset == "real":
        return 0
    if subset in {"inpainting", "p2p"}:
        return 1
    raise ValueError(f"Could not infer label from filepath: {filepath}")


# -----------------------------------------------------------------------------
# column map / frame normalization
# -----------------------------------------------------------------------------

def build_column_map(
    columns: Iterable[str],
    overrides: Optional[Mapping[str, str]] = None,
    candidates: Optional[Mapping[str, Sequence[str]]] = None,
) -> Dict[str, Optional[str]]:
    """표준 컬럼명 -> 실제 컬럼명 매핑 생성.

    overrides가 있으면 최우선 사용하고, 없으면 후보 컬럼명에서 자동 탐색한다.
    """
    columns = list(columns)
    candidates = candidates or DEFAULT_COLUMN_CANDIDATES
    overrides = dict(overrides or {})

    column_map: Dict[str, Optional[str]] = {}
    for standard_name, default_candidates in candidates.items():
        if standard_name in overrides:
            override_value = overrides[standard_name]
            column_map[standard_name] = override_value if override_value in columns else None
            continue
        column_map[standard_name] = find_first_existing_column(columns, default_candidates)
    return column_map


def load_metadata_csv(metadata_csv: PathLike) -> pd.DataFrame:
    metadata_csv = Path(metadata_csv)
    if not metadata_csv.exists():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_csv}")
    return pd.read_csv(metadata_csv)


def standardize_semitruths_metadata(
    metadata: Union[PathLike, pd.DataFrame],
    root_dir: Optional[PathLike] = None,
    column_map: Optional[Mapping[str, str]] = None,
    include_real: bool = True,
    include_inpainting: bool = True,
    include_p2p: bool = True,
    validate_paths: bool = False,
) -> pd.DataFrame:
    """Semi-Truths metadata를 프로젝트 공통 형식으로 정리한다.

    반환 컬럼 기본 형태:
    - sample_id
    - filepath
    - label
    - subset
    - method
    - dataset
    - diffusion_model
    - area_ratio
    - semantic_mag
    - scene_complexity
    - scene_diversity
    - change_type
    - mask_path
    - ... (기타 원본 metadata)
    """
    if isinstance(metadata, (str, Path)):
        df = load_metadata_csv(metadata)
    else:
        df = metadata.copy()

    if len(df) == 0:
        raise ValueError("Metadata dataframe is empty.")

    resolved_map = build_column_map(df.columns, overrides=column_map)
    filepath_col = resolved_map.get("filepath")
    if filepath_col is None:
        raise ValueError(
            "Could not find filepath column. "
            f"Available columns: {list(df.columns)}"
        )

    standardized = df.copy().reset_index(drop=True)
    standardized[filepath_col] = standardized[filepath_col].map(normalize_slashes)

    root_path = None
    if root_dir is not None:
        root_path = _resolve_general_path(root_dir)

    # 핵심 표준 컬럼 생성
    standardized["filepath"] = standardized[filepath_col].astype(str)

    sample_id_col = resolved_map.get("sample_id")
    if sample_id_col is not None:
        standardized["sample_id"] = standardized[sample_id_col].astype(str)
    else:
        standardized["sample_id"] = standardized.index.astype(str)

    label_col = resolved_map.get("label")
    if label_col is not None:
        standardized["label"] = standardized[label_col].map(normalize_label)
    else:
        standardized["label"] = standardized["filepath"].map(infer_label_from_path)

    standardized["subset"] = standardized["filepath"].map(infer_subset_from_path)
    standardized["method"] = standardized.apply(
        lambda row: infer_method_from_row(
            {
                "method": row[resolved_map["method"]] if resolved_map.get("method") else None,
                "subset": row["subset"],
                "filepath": row["filepath"],
            }
        ),
        axis=1,
    )

    for standard_name in [
        "dataset",
        "entity",
        "diffusion_model",
        "language_model",
        "area_ratio",
        "semantic_mag",
        "scene_complexity",
        "scene_diversity",
        "change_type",
        "original_caption",
        "edited_caption",
        "original_label",
        "edited_label",
    ]:
        original_col = resolved_map.get(standard_name)
        if original_col is not None:
            standardized[standard_name] = standardized[original_col].map(sanitize_scalar)
        else:
            standardized[standard_name] = None

    mask_col = resolved_map.get("mask_path")
    if mask_col is not None:
        standardized["mask_path"] = standardized[mask_col].map(normalize_slashes)
    else:
        standardized["mask_path"] = None

    # 숫자형으로 가능한 컬럼 변환
    for numeric_col in ["area_ratio", "semantic_mag", "scene_complexity", "scene_diversity"]:
        standardized[numeric_col] = pd.to_numeric(standardized[numeric_col], errors="coerce")

    standardized["change_type"] = standardized["change_type"].map(_normalize_change_type)

    # subset 필터링
    standardized = filter_by_subset_flags(
        standardized,
        include_real=include_real,
        include_inpainting=include_inpainting,
        include_p2p=include_p2p,
    )

    # 절대경로 컬럼은 선택적으로 추가
    if root_path is not None:
        standardized["absolute_filepath"] = standardized["filepath"].map(
            lambda x: resolve_relative_to_root(x, root_path)
        )
        standardized["absolute_mask_path"] = standardized["mask_path"].map(
            lambda x: resolve_relative_to_root(x, root_path) if x is not None else None
        )

        if validate_paths:
            missing = [
                p for p in standardized["absolute_filepath"].tolist()
                if p is not None and not Path(p).exists()
            ]
            if missing:
                raise FileNotFoundError(
                    f"Found {len(missing)} missing image files. Example: {missing[0]}"
                )

    standardized = standardized.reset_index(drop=True)
    return standardized


# -----------------------------------------------------------------------------
# filtering / binning / grouping
# -----------------------------------------------------------------------------

def filter_by_subset_flags(
    df: pd.DataFrame,
    include_real: bool = True,
    include_inpainting: bool = True,
    include_p2p: bool = True,
) -> pd.DataFrame:
    subset_to_include = {
        "real": include_real,
        "inpainting": include_inpainting,
        "p2p": include_p2p,
    }
    keep_mask = df["subset"].map(lambda x: subset_to_include.get(x, False)).fillna(False)
    return df.loc[keep_mask].reset_index(drop=True)


def apply_named_bins(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    bins: Sequence[float],
    labels: Sequence[str],
    include_lowest: bool = True,
) -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        out[target_col] = None
        return out

    numeric_series = pd.to_numeric(out[source_col], errors="coerce")
    valid_mask = numeric_series.notna()
    out[target_col] = None
    if valid_mask.any():
        out.loc[valid_mask, target_col] = pd.cut(
            numeric_series.loc[valid_mask],
            bins=bins,
            labels=labels,
            include_lowest=include_lowest,
        ).astype(object)
    return out


def apply_quantile_bins(
    df: pd.DataFrame,
    source_col: str,
    target_col: str,
    labels: Sequence[str] = ("small", "medium", "large"),
    quantiles: Sequence[float] = (0.0, 0.25, 0.75, 1.0),
) -> pd.DataFrame:
    out = df.copy()
    if source_col not in out.columns:
        out[target_col] = None
        return out

    series = pd.to_numeric(out[source_col], errors="coerce")
    valid = series.dropna()
    out[target_col] = None
    if valid.empty:
        return out

    edges = [valid.quantile(q) for q in quantiles]
    edges = _dedupe_bin_edges(edges)
    if len(edges) < 2:
        return out

    # labels 개수와 edge 구간 수 맞추기
    effective_labels = list(labels[: max(len(edges) - 1, 0)])
    if len(effective_labels) != len(edges) - 1:
        return out

    out.loc[series.notna(), target_col] = pd.cut(
        series.loc[series.notna()],
        bins=edges,
        labels=effective_labels,
        include_lowest=True,
        duplicates="drop",
    ).astype(object)
    return out


def add_default_analysis_bins(
    df: pd.DataFrame,
    area_ratio_bins: Optional[Sequence[float]] = None,
    area_ratio_labels: Sequence[str] = ("small", "medium", "large"),
    semantic_mag_bins: Optional[Sequence[float]] = None,
    semantic_mag_labels: Sequence[str] = ("small", "medium", "large"),
    scene_complexity_bins: Optional[Sequence[float]] = None,
    scene_complexity_labels: Sequence[str] = ("small", "medium", "large"),
    scene_diversity_bins: Optional[Sequence[float]] = None,
    scene_diversity_labels: Sequence[str] = ("small", "medium", "large"),
) -> pd.DataFrame:
    out = df.copy()

    if area_ratio_bins is not None:
        out = apply_named_bins(out, "area_ratio", "area_ratio_bin", area_ratio_bins, area_ratio_labels)
    else:
        out = apply_quantile_bins(out, "area_ratio", "area_ratio_bin", area_ratio_labels)

    if semantic_mag_bins is not None:
        out = apply_named_bins(out, "semantic_mag", "semantic_mag_bin", semantic_mag_bins, semantic_mag_labels)
    else:
        out = apply_quantile_bins(out, "semantic_mag", "semantic_mag_bin", semantic_mag_labels)

    if scene_complexity_bins is not None:
        out = apply_named_bins(
            out,
            "scene_complexity",
            "scene_complexity_bin",
            scene_complexity_bins,
            scene_complexity_labels,
        )
    else:
        out = apply_quantile_bins(
            out,
            "scene_complexity",
            "scene_complexity_bin",
            scene_complexity_labels,
        )

    if scene_diversity_bins is not None:
        out = apply_named_bins(
            out,
            "scene_diversity",
            "scene_diversity_bin",
            scene_diversity_bins,
            scene_diversity_labels,
        )
    else:
        out = apply_quantile_bins(
            out,
            "scene_diversity",
            "scene_diversity_bin",
            scene_diversity_labels,
        )

    return out


def get_available_group_columns(
    df: pd.DataFrame,
    preferred_columns: Optional[Sequence[str]] = None,
) -> List[str]:
    preferred_columns = list(preferred_columns or DEFAULT_GROUP_COLUMNS)
    available: List[str] = []
    for col in preferred_columns:
        if col not in df.columns:
            continue
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        available.append(col)
    return available


def build_group_key(
    row: Mapping[str, Any],
    group_columns: Sequence[str],
    sep: str = " | ",
) -> str:
    parts: List[str] = []
    for col in group_columns:
        value = sanitize_scalar(row.get(col))
        if value is None:
            parts.append(f"{col}=NA")
        else:
            parts.append(f"{col}={value}")
    return sep.join(parts)


def add_group_key_column(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    output_col: str = "group_key",
    sep: str = " | ",
) -> pd.DataFrame:
    out = df.copy()
    if not group_columns:
        out[output_col] = "all"
        return out
    out[output_col] = out.apply(
        lambda row: build_group_key(row.to_dict(), group_columns=group_columns, sep=sep),
        axis=1,
    )
    return out


def build_directional_edit_column(
    df: pd.DataFrame,
    output_col: str = "directional_edit",
) -> pd.DataFrame:
    out = df.copy()
    out[output_col] = None

    if "original_label" in out.columns and "edited_label" in out.columns:
        label_mask = out["original_label"].notna() & out["edited_label"].notna()
        out.loc[label_mask, output_col] = (
            out.loc[label_mask, "original_label"].astype(str)
            + " -> "
            + out.loc[label_mask, "edited_label"].astype(str)
        )

    if "original_caption" in out.columns and "edited_caption" in out.columns:
        empty_mask = out[output_col].isna()
        caption_mask = empty_mask & out["original_caption"].notna() & out["edited_caption"].notna()
        out.loc[caption_mask, output_col] = (
            out.loc[caption_mask, "original_caption"].astype(str)
            + " -> "
            + out.loc[caption_mask, "edited_caption"].astype(str)
        )

    return out


def group_value_counts(
    df: pd.DataFrame,
    group_columns: Sequence[str],
    sort_by_count: bool = True,
) -> pd.DataFrame:
    if not group_columns:
        return pd.DataFrame({"count": [len(df)]})

    grouped = (
        df.groupby(list(group_columns), dropna=False)
        .size()
        .reset_index(name="count")
    )
    if sort_by_count:
        grouped = grouped.sort_values("count", ascending=False).reset_index(drop=True)
    return grouped


# -----------------------------------------------------------------------------
# path helpers
# -----------------------------------------------------------------------------

def _resolve_general_path(path: PathLike) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def resolve_relative_to_root(filepath: Any, root_dir: PathLike) -> Optional[str]:
    if filepath is None or pd.isna(filepath):
        return None
    raw = Path(str(normalize_slashes(filepath)))
    root = _resolve_general_path(root_dir)
    if raw.is_absolute():
        return raw.resolve().as_posix()
    return (root / raw).resolve().as_posix()


# -----------------------------------------------------------------------------
# internal helpers
# -----------------------------------------------------------------------------

def _normalize_change_type(value: Any) -> Optional[str]:
    value = sanitize_scalar(value)
    if value is None:
        return None
    value_str = str(value).strip().lower()
    if value_str in {"localized", "localised", "local"}:
        return "localized"
    if value_str in {"diffused", "diffuse"}:
        return "diffused"
    return str(value)


def _dedupe_bin_edges(edges: Sequence[float]) -> List[float]:
    deduped: List[float] = []
    for edge in edges:
        edge = float(edge)
        if not deduped or edge > deduped[-1]:
            deduped.append(edge)
    return deduped


__all__ = [
    "DEFAULT_COLUMN_CANDIDATES",
    "DEFAULT_GROUP_COLUMNS",
    "add_default_analysis_bins",
    "add_group_key_column",
    "apply_named_bins",
    "apply_quantile_bins",
    "build_column_map",
    "build_directional_edit_column",
    "build_group_key",
    "filter_by_subset_flags",
    "find_first_existing_column",
    "get_available_group_columns",
    "get_project_root",
    "group_value_counts",
    "infer_label_from_path",
    "infer_method_from_row",
    "infer_subset_from_path",
    "load_metadata_csv",
    "normalize_label",
    "normalize_slashes",
    "resolve_relative_to_root",
    "sanitize_scalar",
    "standardize_semitruths_metadata",
]
