from __future__ import annotations

import argparse
# import sys
from pathlib import Path

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))

from deepfake_fusion.visualization.robustness_visualize import (
    visualize_robustness_results,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot robustness evaluation results."
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
        help="Number of digits for plot annotations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    extra_metrics = None
    if args.extra_metrics is not None:
        extra_metrics = [
            item.strip()
            for item in args.extra_metrics.split(",")
            if item.strip()
        ]

    saved = visualize_robustness_results(
        results=args.input_json,
        output_dir=args.output_dir,
        primary_metric=args.primary_metric,
        extra_metrics=extra_metrics,
        digits=int(args.digits),
    )

    print("=" * 80)
    print("Robustness Plotting Finished")
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