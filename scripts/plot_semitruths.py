from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from deepfake_fusion.visualization.semitruths_visualize import (
    DEFAULT_METRICS,
    SUMMARY_METRICS,
    export_default_group_figures,
    export_model_comparison_figures,
)


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------


def parse_csv_list(value: Optional[str]) -> Optional[List[str]]:
    if value is None:
        return None
    items = [item.strip() for item in str(value).split(",")]
    items = [item for item in items if item]
    return items or None



def parse_named_paths(items: Optional[Sequence[str]]) -> Dict[str, str]:
    """
    Parse CLI entries of the form:
      model_name=/path/to/result_dir
    into a dict mapping model_name -> path.
    """
    parsed: Dict[str, str] = {}
    if not items:
        return parsed

    for raw in items:
        item = str(raw).strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(
                f"Invalid --compare entry: {item!r}. Expected format NAME=PATH"
            )
        name, path = item.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(
                f"Invalid --compare entry: {item!r}. Expected format NAME=PATH"
            )
        parsed[name] = path
    return parsed



def ensure_dir(path: str | Path) -> Path:
    out = path if isinstance(path, Path) else Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out



def ensure_parent(path: str | Path) -> Path:
    out = path if isinstance(path, Path) else Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out



def save_json(data: Mapping[str, Any], path: str | Path) -> None:
    path = ensure_parent(path)
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)



def infer_result_name(result_dir: str | Path) -> str:
    path = Path(result_dir)
    name = path.name.strip()
    if name:
        return name
    return "model"



def default_single_output_dir(result_dir: str | Path) -> Path:
    return ensure_dir(Path(result_dir) / "figures")



def default_comparison_output_dir() -> Path:
    return ensure_dir(Path("outputs") / "semitruths_compare")


# -----------------------------------------------------------------------------
# cli
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate plots for Semi-Truths evaluation results. "
            "Supports both single-result visualization and multi-model comparison."
        )
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        default=None,
        help=(
            "Single evaluation result directory produced by scripts/evaluate_semitruths.py. "
            "Used for exporting per-group figures."
        ),
    )
    parser.add_argument(
        "--result_name",
        type=str,
        default=None,
        help=(
            "Optional display name for --result_dir when also doing comparison. "
            "Defaults to the folder name."
        ),
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="*",
        default=None,
        help=(
            "Model comparison entries in NAME=PATH format. Example: "
            "--compare resnet=runs/a spai=runs/b fusion=runs/c"
        ),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["test"],
        help="Evaluation split name used in saved filenames.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help=(
            "Base output directory. If both single and comparison are requested, "
            "files are saved under <output_dir>/single and <output_dir>/comparison."
        ),
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=",".join(DEFAULT_METRICS[:3]),
        help="Comma-separated per-group metrics for single-result plots.",
    )
    parser.add_argument(
        "--summary_metrics",
        type=str,
        default=",".join(SUMMARY_METRICS),
        help="Comma-separated summary metrics for single-result plots.",
    )
    parser.add_argument(
        "--overall_metrics",
        type=str,
        default="fake_recall,auc,f1,balanced_accuracy",
        help="Comma-separated overall metrics for multi-model comparison.",
    )
    parser.add_argument(
        "--group_metric",
        type=str,
        default="fake_recall",
        help="Metric used for comparison heatmaps/bars across models.",
    )
    parser.add_argument(
        "--group_names",
        type=str,
        default=None,
        help="Optional comma-separated list of group names to compare.",
    )
    parser.add_argument(
        "--min_group_size",
        type=int,
        default=10,
        help="Minimum group size required to keep a group in plots.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Optionally keep only top-k groups in single-result bar plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Figure DPI.",
    )
    return parser.parse_args()


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    metrics = parse_csv_list(args.metrics) or list(DEFAULT_METRICS[:3])
    summary_metrics = parse_csv_list(args.summary_metrics) or list(SUMMARY_METRICS)
    overall_metrics = parse_csv_list(args.overall_metrics) or [
        "fake_recall",
        "auc",
        "f1",
        "balanced_accuracy",
    ]
    group_names = parse_csv_list(args.group_names)

    comparison_dirs = parse_named_paths(args.compare)

    if args.result_dir is None and not comparison_dirs:
        raise SystemExit(
            "At least one of --result_dir or --compare must be provided."
        )

    single_output_dir: Optional[Path] = None
    comparison_output_dir: Optional[Path] = None

    if args.result_dir is not None:
        if args.output_dir is None:
            single_output_dir = default_single_output_dir(args.result_dir)
        elif comparison_dirs:
            single_output_dir = ensure_dir(Path(args.output_dir) / "single")
        else:
            single_output_dir = ensure_dir(Path(args.output_dir))

    if comparison_dirs:
        if args.result_dir is not None:
            result_name = args.result_name or infer_result_name(args.result_dir)
            comparison_dirs = {result_name: args.result_dir, **comparison_dirs}

        if args.output_dir is None:
            comparison_output_dir = default_comparison_output_dir()
        elif args.result_dir is not None:
            comparison_output_dir = ensure_dir(Path(args.output_dir) / "comparison")
        else:
            comparison_output_dir = ensure_dir(Path(args.output_dir))

    saved: Dict[str, Any] = {
        "single": None,
        "comparison": None,
    }

    if args.result_dir is not None:
        assert single_output_dir is not None
        print(f"[Single] result_dir={args.result_dir}")
        print(f"[Single] output_dir={single_output_dir}")
        single_saved = export_default_group_figures(
            result_dir=args.result_dir,
            output_dir=single_output_dir,
            split=args.split,
            metrics=metrics,
            summary_metrics=summary_metrics,
            min_group_size=args.min_group_size,
            top_k=args.top_k,
            dpi=args.dpi,
        )
        saved["single"] = {
            "result_dir": str(args.result_dir),
            "output_dir": single_output_dir.as_posix(),
            "saved": single_saved,
        }
        print(f"[Single] saved {len(single_saved)} artifacts")

    if comparison_dirs:
        assert comparison_output_dir is not None
        print(f"[Compare] models={list(comparison_dirs.keys())}")
        print(f"[Compare] output_dir={comparison_output_dir}")
        comparison_saved = export_model_comparison_figures(
            result_dirs=comparison_dirs,
            output_dir=comparison_output_dir,
            split=args.split,
            overall_metrics=overall_metrics,
            group_names=group_names,
            group_metric=args.group_metric,
            min_group_size=args.min_group_size,
            dpi=args.dpi,
        )
        saved["comparison"] = {
            "result_dirs": comparison_dirs,
            "output_dir": comparison_output_dir.as_posix(),
            "saved": comparison_saved,
        }
        print(f"[Compare] saved {len(comparison_saved)} artifacts")

    if args.output_dir is not None:
        summary_path = ensure_dir(Path(args.output_dir)) / "plot_run_summary.json"
    elif args.result_dir is not None and not comparison_dirs:
        summary_path = default_single_output_dir(args.result_dir) / "plot_run_summary.json"
    elif comparison_output_dir is not None:
        summary_path = comparison_output_dir / "plot_run_summary.json"
    else:
        summary_path = Path("plot_run_summary.json")

    save_json(saved, summary_path)
    print(f"Saved run summary: {summary_path.as_posix()}")


if __name__ == "__main__":
    main()
