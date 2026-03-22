from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

REAL_LABEL_NAMES = {
    "real",
    "authentic",
    "original",
    "natural",
    "pristine",
    "genuine",
}

FAKE_LABEL_NAMES = {
    "fake",
    "synthetic",
    "generated",
    "manipulated",
    "deepfake",
    "ai",
}

TRAIN_SPLIT_NAMES = {"train", "training"}
VAL_SPLIT_NAMES = {"val", "valid", "validation", "dev"}
TEST_SPLIT_NAMES = {"test", "testing"}


@dataclass
class Sample:
    filepath: str
    label: int


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create train/val/test split CSV files for deepfake image datasets."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/cifake",
        help="Root directory of the raw dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits/cifake",
        help="Directory where split CSV files will be saved.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio when validation split is not predefined.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio when test split is not predefined.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified split.",
    )
    parser.add_argument(
        "--strict_labels",
        action="store_true",
        help="Raise an error if a file path does not contain a recognizable real/fake label name.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def infer_label_from_path(path: Path) -> Optional[int]:
    """
    경로에 포함된 폴더명으로 라벨 추론.
    real 계열 -> 0
    fake 계열 -> 1
    """
    parts_lower = {part.lower() for part in path.parts}

    if parts_lower & REAL_LABEL_NAMES:
        return 0
    if parts_lower & FAKE_LABEL_NAMES:
        return 1

    return None


def infer_predefined_split(path: Path) -> Optional[str]:
    """
    경로에 train/val/test 관련 폴더가 이미 존재하면 이를 활용.
    """
    parts_lower = {part.lower() for part in path.parts}

    if parts_lower & TRAIN_SPLIT_NAMES:
        return "train"
    if parts_lower & VAL_SPLIT_NAMES:
        return "val"
    if parts_lower & TEST_SPLIT_NAMES:
        return "test"

    return None


def collect_samples(
    input_dir: Path,
    project_root: Path,
    strict_labels: bool = False,
) -> List[Sample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and is_image_file(path)
    )

    if not image_paths:
        raise RuntimeError(f"No image files found under: {input_dir}")

    samples: List[Sample] = []
    skipped_paths: List[Path] = []

    for path in image_paths:
        label = infer_label_from_path(path)

        if label is None:
            if strict_labels:
                raise ValueError(f"Could not infer label from path: {path}")
            skipped_paths.append(path)
            continue

        rel_path = path.relative_to(project_root).as_posix()
        samples.append(Sample(filepath=rel_path, label=label))

    if not samples:
        raise RuntimeError(
            "No valid samples found. Check folder names for label inference."
        )

    if skipped_paths:
        print(
            f"Skipped {len(skipped_paths)} files because label could not be inferred "
            f"(use --strict_labels to raise an error instead)."
        )

    return samples


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "filepath": [sample.filepath for sample in samples],
            "label": [sample.label for sample in samples],
        }
    )

    if df["filepath"].duplicated().any():
        dup_count = int(df["filepath"].duplicated().sum())
        print(f"Warning: found {dup_count} duplicated filepaths. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["filepath"], keep="first").reset_index(drop=True)

    return df


def stratified_split(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if df["label"].nunique() < 2:
        raise ValueError("Stratified split requires at least 2 classes.")

    train_df, split_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )
    return train_df.reset_index(drop=True), split_df.reset_index(drop=True)


def has_predefined_splits(df: pd.DataFrame, project_root: Path) -> bool:
    rel_paths = df["filepath"].apply(lambda x: Path(project_root / x))
    split_tags = rel_paths.apply(infer_predefined_split)
    return split_tags.notnull().all()


def split_with_predefined_splits(
    df: pd.DataFrame,
    project_root: Path,
    val_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    파일 경로에 train/val/test 폴더가 이미 존재하는 경우:
    - train/val/test가 모두 있으면 그대로 사용
    - train/test만 있으면 train에서 val만 다시 분리
    """
    rel_paths = df["filepath"].apply(lambda x: Path(project_root / x))
    split_tags = rel_paths.apply(infer_predefined_split)

    if split_tags.isnull().any():
        raise ValueError(
            "Some files do not belong to predefined split folders."
        )

    split_set = set(split_tags.tolist())

    train_pool_df = df[split_tags == "train"].reset_index(drop=True)
    val_df = df[split_tags == "val"].reset_index(drop=True)
    test_df = df[split_tags == "test"].reset_index(drop=True)

    if len(train_pool_df) == 0:
        raise RuntimeError("No training files found in predefined split.")

    if len(test_df) == 0:
        raise RuntimeError("No test files found in predefined split.")

    if "val" in split_set:
        if len(val_df) == 0:
            raise RuntimeError("Validation split tag detected, but no validation files found.")
        train_df = train_pool_df
    else:
        train_df, val_df = stratified_split(
            train_pool_df,
            test_size=val_ratio,
            seed=seed,
        )

    return train_df, val_df, test_df


def split_without_predefined_splits(
    df: pd.DataFrame,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    전체 데이터를 train/val/test로 직접 stratified split.
    """
    if not (0.0 < val_ratio < 1.0):
        raise ValueError("val_ratio must be between 0 and 1.")
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("test_ratio must be between 0 and 1.")
    if val_ratio + test_ratio >= 1.0:
        raise ValueError("val_ratio + test_ratio must be less than 1.")

    train_val_df, test_df = stratified_split(
        df,
        test_size=test_ratio,
        seed=seed,
    )

    adjusted_val_ratio = val_ratio / (1.0 - test_ratio)

    train_df, val_df = stratified_split(
        train_val_df,
        test_size=adjusted_val_ratio,
        seed=seed,
    )
    return train_df, val_df, test_df


def save_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def print_distribution(name: str, df: pd.DataFrame) -> None:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    print(f"[{name}] total={len(df)} label_counts={label_counts}")


def print_preview(df: pd.DataFrame, name: str, n: int = 3) -> None:
    preview = df.head(n).to_dict(orient="records")
    print(f"[{name}] preview={preview}")


def main() -> None:
    args = parse_args()
    project_root = get_project_root()

    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    samples = collect_samples(
        input_dir=input_dir,
        project_root=project_root,
        strict_labels=args.strict_labels,
    )
    df = samples_to_dataframe(samples)

    if has_predefined_splits(df, project_root):
        print("Detected predefined split folders. Reusing them when possible.")
        train_df, val_df, test_df = split_with_predefined_splits(
            df=df,
            project_root=project_root,
            val_ratio=args.val_ratio,
            seed=args.seed,
        )
    else:
        print("No predefined split folders detected. Creating train/val/test from all samples.")
        train_df, val_df, test_df = split_without_predefined_splits(
            df=df,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

    save_split_csv(train_df, output_dir / "train.csv")
    save_split_csv(val_df, output_dir / "val.csv")
    save_split_csv(test_df, output_dir / "test.csv")

    print_distribution("train", train_df)
    print_distribution("val", val_df)
    print_distribution("test", test_df)

    print_preview(train_df, "train")
    print_preview(val_df, "val")
    print_preview(test_df, "test")

    print(f"Saved train split to: {(output_dir / 'train.csv').as_posix()}")
    print(f"Saved val split to:   {(output_dir / 'val.csv').as_posix()}")
    print(f"Saved test split to:  {(output_dir / 'test.csv').as_posix()}")


if __name__ == "__main__":
    main()