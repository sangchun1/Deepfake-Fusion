from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
            "Run batch GenImage experiments for by_generator and/or "
            "leave-one-generator-out (logo) splits."
        )
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="by_generator",
        choices=["by_generator", "logo", "all"],
        help="Which experiment family to run.",
    )
    parser.add_argument(
        "--base_data_config",
        type=str,
        default="configs/data/genimage.yaml",
        help="Base GenImage data config YAML.",
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
        help="Base output directory for all experiments.",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default=None,
        help=(
            "Optional override for cfg.data.paths.root_dir. "
            "Useful when raw GenImage is stored on an external drive."
        ),
    )
    parser.add_argument(
        "--names",
        nargs="*",
        default=None,
        help=(
            "Optional subset of experiment folder names to run. "
            "Example: midjourney sd14 holdout_midjourney"
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
        "--checkpoint_name",
        type=str,
        default="best.pth",
        help="Checkpoint filename to use for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Optional device override passed to train.py / evaluate.py.",
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
        help="Only run training, skip evaluation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Only run evaluation, skip training.",
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
        "--stop_on_error",
        action="store_true",
        help="Stop immediately if any experiment fails.",
    )

    args = parser.parse_args()

    if args.train_only and args.eval_only:
        raise ValueError("--train_only and --eval_only cannot both be set.")

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

    exp_dirs = sorted([p for p in mode_dir.iterdir() if p.is_dir()])
    if selected_names:
        selected = set(selected_names)
        exp_dirs = [p for p in exp_dirs if p.name in selected]

    if not exp_dirs:
        raise RuntimeError(f"No experiment directories found under: {mode_dir}")

    return exp_dirs


def build_generated_data_config(
    base_cfg: Dict,
    exp_dir: Path,
    generated_config_path: Path,
    root_dir_override: str | None,
) -> None:
    cfg = copy.deepcopy(base_cfg)

    if "paths" not in cfg:
        raise ValueError("Base data config must contain a 'paths' section.")

    if root_dir_override is not None:
        cfg["paths"]["root_dir"] = root_dir_override

    cfg["paths"]["train_csv"] = exp_dir.joinpath("train.csv").as_posix()
    cfg["paths"]["val_csv"] = exp_dir.joinpath("val.csv").as_posix()
    cfg["paths"]["test_csv"] = exp_dir.joinpath("test.csv").as_posix()

    # dataset registry는 dataset_class를 우선 사용하므로 name은 바뀌어도 무방하지만,
    # 실험 추적을 위해 구체적인 이름을 붙여둔다.
    base_name = str(cfg.get("name", "genimage"))
    cfg["name"] = f"{base_name}_{exp_dir.parent.name}_{exp_dir.name}"

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


def run_command(cmd: Sequence[str], cwd: Path) -> None:
    print("=" * 100)
    print("Running command:")
    print(" ".join(cmd))
    print("=" * 100)
    subprocess.run(cmd, cwd=cwd, check=True)


def resolve_modes(mode: str) -> List[str]:
    if mode == "all":
        return ["by_generator", "logo"]
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
        "root_dir_override": args.root_dir,
        "train_only": args.train_only,
        "eval_only": args.eval_only,
        "eval_split": args.eval_split,
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
            generated_data_config_path = (
                generated_config_dir / mode / f"{exp_name}.yaml"
            )
            output_dir = output_root / mode / exp_name
            checkpoint_path = output_dir / args.checkpoint_name
            eval_json_path = output_dir / f"eval_{args.eval_split}.json"

            result = {
                "mode": mode,
                "experiment_name": exp_name,
                "split_dir": exp_dir.as_posix(),
                "generated_data_config": generated_data_config_path.as_posix(),
                "output_dir": output_dir.as_posix(),
                "train_status": "not_run",
                "eval_status": "not_run",
                "checkpoint_path": checkpoint_path.as_posix(),
                "eval_json_path": eval_json_path.as_posix(),
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
                )

                if not args.eval_only:
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

                if not args.train_only:
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

            except Exception as e:
                result["error"] = str(e)
                if result["train_status"] == "not_run" and not args.eval_only:
                    result["train_status"] = "failed"
                if result["eval_status"] == "not_run" and not args.train_only:
                    result["eval_status"] = "failed"

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