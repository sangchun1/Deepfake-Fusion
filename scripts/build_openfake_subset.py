from __future__ import annotations

import argparse
import io
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from PIL import Image

from datasets import load_dataset


LABEL_TO_INT = {
    "real": 0,
    "fake": 1,
    "0": 0,
    "1": 1,
}

LABEL_TO_NAME = {
    0: "real",
    1: "fake",
}


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local OpenFake test subset from Hugging Face and save "
            "images + CSV for evaluate.py / explain.py."
        )
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="ComplexDataLab/OpenFake",
        help="Hugging Face dataset name.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Official split to sample from. OpenFake provides train/test.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="data/raw/openfake",
        help="Root directory where images will be saved.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="data/splits/openfake/default_test.csv",
        help="CSV path to save metadata and relative filepaths.",
    )
    parser.add_argument(
        "--subset_name",
        type=str,
        default="default_test",
        help="Subdirectory name created under output_root.",
    )
    parser.add_argument(
        "--max_per_label",
        type=int,
        default=500,
        help="Maximum number of samples to save for each label.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for streaming shuffle.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        type=int,
        default=10_000,
        help="Buffer size for streaming shuffle.",
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Image format used when saving local subset files.",
    )
    parser.add_argument(
        "--jpeg_quality",
        type=int,
        default=95,
        help="JPEG quality used when image_format=jpg.",
    )
    parser.add_argument(
        "--allowed_models",
        type=str,
        default=None,
        help="Optional comma-separated model whitelist.",
    )
    parser.add_argument(
        "--allowed_types",
        type=str,
        default=None,
        help="Optional comma-separated type whitelist (e.g. base,lora).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing subset directory and CSV if they already exist.",
    )
    return parser.parse_args()


def parse_optional_csv_list(value: Optional[str]) -> Optional[set[str]]:
    if value is None:
        return None
    items = {item.strip().lower() for item in value.split(",") if item.strip()}
    return items or None


def normalize_label(value: Any) -> int:
    key = str(value).strip().lower()
    if key not in LABEL_TO_INT:
        raise ValueError(f"Unsupported label value: {value}")
    return LABEL_TO_INT[key]


def to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def ensure_pil_image(image_value: Any) -> Image.Image:
    if isinstance(image_value, Image.Image):
        return image_value

    if isinstance(image_value, dict):
        image_bytes = image_value.get("bytes")
        image_path = image_value.get("path")

        if image_bytes is not None:
            return Image.open(io.BytesIO(image_bytes))
        if image_path:
            return Image.open(image_path)

    raise TypeError(f"Unsupported image value type: {type(image_value)}")


def save_image(image: Image.Image, path: Path, image_format: str, jpeg_quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")

    if image_format == "jpg":
        image.save(path, format="JPEG", quality=jpeg_quality)
    else:
        image.save(path, format="PNG")


def prepare_output_paths(
    project_root: Path,
    output_root: str,
    output_csv: str,
    subset_name: str,
    overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_root_path = (project_root / output_root).resolve()
    output_csv_path = (project_root / output_csv).resolve()
    subset_dir = output_root_path / subset_name

    if overwrite:
        if subset_dir.exists():
            shutil.rmtree(subset_dir)
        if output_csv_path.exists():
            output_csv_path.unlink()
    else:
        if subset_dir.exists():
            raise FileExistsError(
                f"Subset directory already exists: {subset_dir}\n"
                "Use --overwrite to recreate it."
            )
        if output_csv_path.exists():
            raise FileExistsError(
                f"Output CSV already exists: {output_csv_path}\n"
                "Use --overwrite to recreate it."
            )

    output_root_path.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    return output_root_path, output_csv_path, subset_dir


def should_keep_example(
    example: Dict[str, Any],
    allowed_models: Optional[set[str]],
    allowed_types: Optional[set[str]],
) -> bool:
    model_name = to_text(example.get("model")).strip().lower()
    type_name = to_text(example.get("type")).strip().lower()

    if allowed_models is not None and model_name not in allowed_models:
        return False
    if allowed_types is not None and type_name not in allowed_types:
        return False
    return True


def main() -> None:
    args = parse_args()

    if args.max_per_label <= 0:
        raise ValueError("max_per_label must be a positive integer.")

    project_root = get_project_root()
    output_root_path, output_csv_path, subset_dir = prepare_output_paths(
        project_root=project_root,
        output_root=args.output_root,
        output_csv=args.output_csv,
        subset_name=args.subset_name,
        overwrite=args.overwrite,
    )

    allowed_models = parse_optional_csv_list(args.allowed_models)
    allowed_types = parse_optional_csv_list(args.allowed_types)

    print("=" * 80)
    print("Build OpenFake Subset")
    print("=" * 80)
    print(f"dataset_name: {args.dataset_name}")
    print(f"split: {args.split}")
    print(f"output_root: {output_root_path.as_posix()}")
    print(f"subset_dir: {subset_dir.as_posix()}")
    print(f"output_csv: {output_csv_path.as_posix()}")
    print(f"max_per_label: {args.max_per_label}")
    print(f"image_format: {args.image_format}")
    if allowed_models is not None:
        print(f"allowed_models: {sorted(allowed_models)}")
    if allowed_types is not None:
        print(f"allowed_types: {sorted(allowed_types)}")

    dataset = load_dataset(
        args.dataset_name,
        split=args.split,
        streaming=True,
    )
    dataset = dataset.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer_size)

    counts = {0: 0, 1: 0}
    rows: list[dict[str, Any]] = []
    target_count = {0: args.max_per_label, 1: args.max_per_label}
    suffix = ".jpg" if args.image_format == "jpg" else ".png"

    for stream_index, example in enumerate(dataset):
        if counts[0] >= target_count[0] and counts[1] >= target_count[1]:
            break

        if not should_keep_example(
            example=example,
            allowed_models=allowed_models,
            allowed_types=allowed_types,
        ):
            continue

        label = normalize_label(example.get("label"))
        if counts[label] >= target_count[label]:
            continue

        label_name = LABEL_TO_NAME[label]
        image = ensure_pil_image(example.get("image"))

        next_index = counts[label] + 1
        filename = f"{label_name}_{next_index:06d}{suffix}"
        relative_path = Path(args.subset_name) / label_name / filename
        absolute_path = output_root_path / relative_path

        save_image(
            image=image,
            path=absolute_path,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )

        counts[label] += 1
        rows.append(
            {
                "filepath": relative_path.as_posix(),
                "label": label,
                "label_name": label_name,
                "model": to_text(example.get("model")),
                "prompt": to_text(example.get("prompt")),
                "type": to_text(example.get("type")),
                "release_date": to_text(example.get("release_date")),
                "official_split": args.split,
                "stream_index": stream_index,
            }
        )

        total_saved = counts[0] + counts[1]
        if total_saved % 100 == 0:
            print(
                f"saved={total_saved} "
                f"real={counts[0]}/{target_count[0]} "
                f"fake={counts[1]}/{target_count[1]}"
            )

    if counts[0] < target_count[0] or counts[1] < target_count[1]:
        raise RuntimeError(
            "Could not collect enough samples for the requested subset size. "
            f"Saved counts: {counts}. "
            "Try lowering --max_per_label or loosening filters."
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(by=["label", "stream_index", "filepath"]).reset_index(drop=True)
    df.to_csv(output_csv_path, index=False, encoding="utf-8")

    print("=" * 80)
    print("Subset Build Finished")
    print("=" * 80)
    print(f"total_saved: {len(df)}")
    print(f"label_counts: {counts}")
    print(f"saved_csv: {output_csv_path.as_posix()}")
    print(f"saved_images_under: {subset_dir.as_posix()}")


if __name__ == "__main__":
    main()