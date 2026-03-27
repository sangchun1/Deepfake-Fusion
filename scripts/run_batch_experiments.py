from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import yaml


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path.resolve()
    return (get_project_root() / path).resolve()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run batch experiments for merged / by_generator / "
            "leave-one-generator-out (logo) splits, including optional evaluation "
            "and Grad-CAM explanation."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="by_generator",
        choices=["merged", "by_generator", "logo", "all"],
        help="Which experiment family to run.",
    )
    parser.add_argument(
        "--base_data_config",
        type=str,
        default="configs/data/genimage.yaml",
        help="Base data config YAML.",
    )
    parser.add_argument(
        "--model_config",
        type=str,
        default="configs/model/resnet18.yaml",
        help="Model config path.",
    )
    parser.add_argument(
        "--train_config",
        type=str,
        default="configs/train/spatial_genimage.yaml",
        help="Train config path.",
    )
    parser.add_argument(
        "--splits_root",
        type=str,
        default="data/splits/genimage",
        help="Root directory containing merged/by_generator/logo split folders.",
    )
    parser.add_argument(
        "--generated_config_dir",
        type=str,
        default="configs/_generated/genimage_batch",
        help="Where generated per-experiment data configs will be stored.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="outputs/spatial/genimage",
        help="Base output directory for train/eval artifacts.",
    )
    parser.add_argument(
        "--explain_root",
        type=str,
        default="outputs/gradcam/genimage",
        help="Base output directory for explain artifacts.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=(
            "Optional override for cfg.data.paths.root_dir. "
            "Useful when raw data is stored on an external drive."
        ),
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Optional subset of experiment folder names to run. "
            "Examples: merged flux_dev sdxl holdout_flux_dev"
        ),
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate after training.",
    )
    parser.add_argument(
        "--explain_split",
        type=str,
        default=None,
        choices=["train", "val", "test"],
        help="Which split to explain. Defaults to eval_split.",
    )
    parser.add_argument(
        "--checkpoint_name",
        type=str,
        default="best.pth",
        help="Checkpoint filename to use for evaluation and explanation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Optional device override passed to train.py / evaluate.py / explain.py.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable to use for subprocess calls.",
    )

    parser.add_argument(
        "--train_only",
        action="store_true",
        help="Only run training, skip evaluation and explanation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, skip training and explanation.",
    )
    parser.add_argument(
        "--explain_only",
        action="store_true",
        help="Only run explanation, skip training and evaluation.",
    )

    parser.add_argument(
        "--skip_existing_train",
        action="store_true",
        help="Skip training if checkpoint already exists.",
    )
    parser.add_argument(
        "--skip_existing_eval",
        action="store_true",
        help="Skip evaluation if eval json already exists.",
    )
    parser.add_argument(
        "--skip_existing_explain",
        action="store_true",
        help="Skip explanation if summary.json already exists.",
    )
    parser.add_argument(
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any experiment fails.",
    )

    parser.add_argument(
        "--run_explain",
        action="store_true",
        help="Run explain.py after evaluation.",
    )
    parser.add_argument(
        "--target_type",
        type=str,
        default="pred",
        choices=["pred", "true"],
        help="Grad-CAM target type.",
    )
    parser.add_argument(
        "--max_per_group",
        type=int,
        default=4,
        help="Max Grad-CAM examples per group.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Grad-CAM overlay alpha.",
    )
    parser.add_argument(
        "--target_layer",
        type=str,
        default=None,
        help="Optional explicit Grad-CAM target layer.",
    )

    args = parser.parse_args()

    mode_flags = [args.train_only, args.eval_only, args.explain_only]
    if sum(bool(x) for x in mode_flags) > 1:
        raise ValueError(
            "--train_only, --eval_only, --explain_only 중 하나만 사용할 수 있습니다."
        )

    if args.explain_only:
        args.run_explain = True

    if args.explain_split is None:
        args.explain_split = args.eval_split

    return args


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            allow_unicode=True,
            sort_keys=False,
        )


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def discover_experiment_dirs(
    splits_root: Path,
    mode: str,
    selected_names: Sequence[str] | None = None,
) -> List[Path]:
    mode_dir = splits_root / mode
    if not mode_dir.exists():
        raise FileNotFoundError(f"Split directory not found: {mode_dir}")

    selected = set(selected_names) if selected_names else None

    if mode == "merged":
        exp_name = mode_dir.name  # "merged"
        if selected is not None and exp_name not in selected:
            raise RuntimeError(
                f"No experiment directories found under: {mode_dir} "
                f"(selected_names={sorted(selected)})"
            )
        return [mode_dir]

    exp_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir()])

    if selected is not None:
        exp_dirs = [p for p in exp_dirs if p.name in selected]

    if not exp_dirs:
        raise RuntimeError(f"No experiment directories found under: {mode_dir}")

    return exp_dirs


def build_generated_data_config(
    base_cfg: Dict,
    exp_dir: Path,
    generated_config_path: Path,
    root_dir_override: str | None,
    mode: str,
    exp_name: str,
) -> None:
    cfg = copy.deepcopy(base_cfg)

    if "paths" not in cfg:
        raise ValueError("Base data config must contain a 'paths' section.")

    if root_dir_override is not None:
        cfg["paths"]["root_dir"] = root_dir_override

    cfg["paths"]["train_csv"] = exp_dir.joinpath("train.csv").as_posix()
    cfg["paths"]["val_csv"] = exp_dir.joinpath("val.csv").as_posix()
    cfg["paths"]["test_csv"] = exp_dir.joinpath("test.csv").as_posix()

    base_name = str(cfg.get("name", "dataset"))
    cfg["name"] = f"{base_name}_{mode}_{exp_name}"

    save_yaml(cfg, generated_config_path)


def make_train_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    output_dir: Path,
    device: str | None,
) -> List[str]:
    cmd = [
        python_executable,
        "scripts/train.py",
        "--data_config",
        data_config.as_posix(),
        "--model_config",
        model_config.as_posix(),
        "--train_config",
        train_config.as_posix(),
        "--output_dir",
        output_dir.as_posix(),
    ]
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def make_eval_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    checkpoint_path: Path,
    eval_split: str,
    output_json: Path,
    device: str | None,
) -> List[str]:
    cmd = [
        python_executable,
        "scripts/evaluate.py",
        "--data_config",
        data_config.as_posix(),
        "--model_config",
        model_config.as_posix(),
        "--train_config",
        train_config.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--split",
        eval_split,
        "--output_json",
        output_json.as_posix(),
    ]
    if device is not None:
        cmd.extend(["--device", device])
    return cmd


def make_explain_command(
    python_executable: str,
    data_config: Path,
    model_config: Path,
    train_config: Path,
    checkpoint_path: Path,
    explain_split: str,
    save_dir: Path,
    device: str | None,
    target_type: str,
    max_per_group: int,
    alpha: float,
    target_layer: str | None,
) -> List[str]:
    cmd = [
        python_executable,
        "scripts/explain.py",
        "--data_config",
        data_config.as_posix(),
        "--model_config",
        model_config.as_posix(),
        "--train_config",
        train_config.as_posix(),
        "--checkpoint",
        checkpoint_path.as_posix(),
        "--split",
        explain_split,
        "--save_dir",
        save_dir.as_posix(),
        "--target_type",
        target_type,
        "--max_per_group",
        str(max_per_group),
        "--alpha",
        str(alpha),
    ]
    if device is not None:
        cmd.extend(["--device", device])
    if target_layer is not None:
        cmd.extend(["--target_layer", target_layer])
    return cmd


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print("=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["merged", "by_generator", "logo"]
    return [mode]


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    base_data_config_path = resolve_path(args.base_data_config)
    model_config_path = resolve_path(args.model_config)
    train_config_path = resolve_path(args.train_config)
    splits_root = resolve_path(args.splits_root)
    generated_config_dir = resolve_path(args.generated_config_dir)
    output_root = resolve_path(args.output_root)
    explain_root = resolve_path(args.explain_root)

    base_data_cfg = load_yaml(base_data_config_path)

    selected_names = set(args.names) if args.names else None
    modes = resolve_modes(args.mode)

    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "mode": args.mode,
        "resolved_modes": modes,
        "base_data_config": base_data_config_path.as_posix(),
        "model_config": model_config_path.as_posix(),
        "train_config": train_config_path.as_posix(),
        "splits_root": splits_root.as_posix(),
        "generated_config_dir": generated_config_dir.as_posix(),
        "output_root": output_root.as_posix(),
        "explain_root": explain_root.as_posix(),
        "root_dir_override": args.root_dir,
        "train_only": args.train_only,
        "eval_only": args.eval_only,
        "explain_only": args.explain_only,
        "run_explain": args.run_explain,
        "eval_split": args.eval_split,
        "explain_split": args.explain_split,
        "checkpoint_name": args.checkpoint_name,
        "selected_names": sorted(selected_names) if selected_names else None,
        "results": [],
    }

    for mode in modes:
        exp_dirs = discover_experiment_dirs(
            splits_root=splits_root,
            mode=mode,
            selected_names=selected_names,
        )

        for exp_dir in exp_dirs:
            exp_name = exp_dir.name
            generated_data_config_path = generated_config_dir / mode / f"{exp_name}.yaml"
            output_dir = output_root / mode / exp_name
            checkpoint_path = output_dir / args.checkpoint_name
            eval_json_path = output_dir / f"eval_{args.eval_split}.json"

            explain_base_dir = explain_root / mode / exp_name
            explain_summary_path = (
                explain_base_dir
                / f"{base_data_cfg.get('name', 'dataset')}_{mode}_{exp_name}"
                / args.explain_split
                / Path(args.checkpoint_name).stem
                / "summary.json"
            )

            result = {
                "mode": mode,
                "experiment_name": exp_name,
                "split_dir": exp_dir.as_posix(),
                "generated_data_config": generated_data_config_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "train_status": "not_run",
                "eval_status": "not_run",
                "explain_status": "not_run",
                "checkpoint_path": checkpoint_path.as_posix(),
                "eval_json_path": eval_json_path.as_posix(),
                "explain_base_dir": explain_base_dir.as_posix(),
                "explain_summary_path": explain_summary_path.as_posix(),
                "error": None,
            }

            try:
                print("\n" + "#" * 100)
                print(f"[{mode}] {exp_name}")
                print("#" * 100)

                build_generated_data_config(
                    base_cfg=base_data_cfg,
                    exp_dir=exp_dir,
                    generated_config_path=generated_data_config_path,
                    root_dir_override=args.root_dir,
                    mode=mode,
                    exp_name=exp_name,
                )

                if not args.eval_only and not args.explain_only:
                    if args.skip_existing_train and checkpoint_path.exists():
                        print(
                            f"Skip training because checkpoint already exists: {checkpoint_path}"
                        )
                        result["train_status"] = "skipped_existing"
                    else:
                        train_cmd = make_train_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            output_dir=output_dir,
                            device=args.device,
                        )
                        run_command(train_cmd, cwd=project_root)
                        result["train_status"] = "done"

                if not args.train_only and not args.explain_only:
                    if args.skip_existing_eval and eval_json_path.exists():
                        print(
                            f"Skip evaluation because eval json already exists: {eval_json_path}"
                        )
                        result["eval_status"] = "skipped_existing"
                    else:
                        if not checkpoint_path.exists():
                            raise FileNotFoundError(
                                f"Checkpoint for evaluation not found: {checkpoint_path}"
                            )

                        eval_cmd = make_eval_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            checkpoint_path=checkpoint_path,
                            eval_split=args.eval_split,
                            output_json=eval_json_path,
                            device=args.device,
                        )
                        run_command(eval_cmd, cwd=project_root)
                        result["eval_status"] = "done"

                if args.run_explain:
                    if args.skip_existing_explain and explain_summary_path.exists():
                        print(
                            f"Skip explanation because summary already exists: {explain_summary_path}"
                        )
                        result["explain_status"] = "skipped_existing"
                    else:
                        if not checkpoint_path.exists():
                            raise FileNotFoundError(
                                f"Checkpoint for explanation not found: {checkpoint_path}"
                            )

                        explain_cmd = make_explain_command(
                            python_executable=args.python,
                            data_config=generated_data_config_path,
                            model_config=model_config_path,
                            train_config=train_config_path,
                            checkpoint_path=checkpoint_path,
                            explain_split=args.explain_split,
                            save_dir=explain_base_dir,
                            device=args.device,
                            target_type=args.target_type,
                            max_per_group=args.max_per_group,
                            alpha=args.alpha,
                            target_layer=args.target_layer,
                        )
                        run_command(explain_cmd, cwd=project_root)
                        result["explain_status"] = "done"

            except Exception as e:
                result["error"] = str(e)
                if (
                    result["train_status"] == "not_run"
                    and not args.eval_only
                    and not args.explain_only
                ):
                    result["train_status"] = "failed"
                if (
                    result["eval_status"] == "not_run"
                    and not args.train_only
                    and not args.explain_only
                ):
                    result["eval_status"] = "failed"
                if result["explain_status"] == "not_run" and args.run_explain:
                    result["explain_status"] = "failed"

                print(f"[ERROR] {mode}/{exp_name}: {e}")

                summary["results"].append(result)

                if args.stop_on_error:
                    summary_path = output_root / "batch_summary.json"
                    save_json(summary, summary_path)
                    raise
                continue

            summary["results"].append(result)

    summary_path = output_root / "batch_summary.json"
    save_json(summary, summary_path)

    print("\n" + "=" * 100)
    print("Batch experiments finished")
    print("=" * 100)
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()