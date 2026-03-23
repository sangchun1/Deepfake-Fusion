from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_NAME_TO_INT = {
    "nature": 0,
    "ai": 1,
}

OFFICIAL_SPLIT_NAMES = {"train", "val"}

GENERATOR_SLUG_MAP = {
    "Midjourney": "midjourney",
    "VQDM": "vqdm",
    "Wukong": "wukong",
    "Stable Diffusion V1.4": "sd14",
    "Stable Diffusion V1.5": "sd15",
    "GLIDE": "glide",
    "BigGAN": "biggan",
    "ADM": "adm",
}


@dataclass(frozen=True)
class Sample:
    filepath: str
    label: int
    label_name: str
    generator: str
    generator_slug: str
    official_split: str


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build GenImage CSV indices for merged / by_generator / "
            "leave-one-generator-out (logo) experiments."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/genimage",
        help="Root directory of raw GenImage dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits/genimage",
        help="Directory where CSV split files will be saved.",
    )
    parser.add_argument(
        "--official_val_test_ratio",
        type=float,
        default=0.5,
        help=(
            "When splitting official val into local val/test for merged and "
            "by_generator splits, this ratio goes to test."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting.",
    )
    parser.add_argument(
        "--save_full_index",
        action="store_true",
        help="Also save full metadata index to output_dir/full_index.csv.",
    )
    return parser.parse_args()


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def resolve_generator_slug(generator_name: str) -> str:
    if generator_name in GENERATOR_SLUG_MAP:
        return GENERATOR_SLUG_MAP[generator_name]

    slug = (
        generator_name.lower()
        .replace(".", "")
        .replace("-", " ")
        .replace("/", " ")
        .strip()
    )
    slug = "_".join(slug.split())
    return slug


def infer_sample_from_path(path: Path, input_dir: Path, project_root: Path) -> Optional[Sample]:
    """
    Expected path pattern:
    input_dir / <GENERATOR> / <train|val> / <ai|nature> / image.xxx
    """
    try:
        rel_to_input = path.relative_to(input_dir)
    except ValueError:
        return None

    parts = rel_to_input.parts
    if len(parts) < 4:
        return None

    generator_name = parts[0]
    official_split = parts[1].lower()
    label_name = parts[2].lower()

    if official_split not in OFFICIAL_SPLIT_NAMES:
        return None
    if label_name not in LABEL_NAME_TO_INT:
        return None

    generator_slug = resolve_generator_slug(generator_name)
    label = LABEL_NAME_TO_INT[label_name]
    rel_to_project = path.relative_to(project_root).as_posix()

    return Sample(
        filepath=rel_to_project,
        label=label,
        label_name=label_name,
        generator=generator_name,
        generator_slug=generator_slug,
        official_split=official_split,
    )


def collect_samples(input_dir: Path, project_root: Path) -> List[Sample]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    image_paths = sorted(
        path for path in input_dir.rglob("*") if path.is_file() and is_image_file(path)
    )
    if not image_paths:
        raise RuntimeError(f"No image files found under: {input_dir}")

    samples: List[Sample] = []
    skipped: List[Path] = []

    for path in image_paths:
        sample = infer_sample_from_path(path=path, input_dir=input_dir, project_root=project_root)
        if sample is None:
            skipped.append(path)
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No valid GenImage samples found. "
            "Expected structure: <generator>/<train|val>/<ai|nature>/image"
        )

    if skipped:
        print(
            f"Skipped {len(skipped)} files because they did not match the expected "
            "GenImage directory pattern."
        )

    return samples


def samples_to_dataframe(samples: Sequence[Sample]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "filepath": [s.filepath for s in samples],
            "label": [s.label for s in samples],
            "label_name": [s.label_name for s in samples],
            "generator": [s.generator for s in samples],
            "generator_slug": [s.generator_slug for s in samples],
            "official_split": [s.official_split for s in samples],
        }
    )

    if df["filepath"].duplicated().any():
        dup_count = int(df["filepath"].duplicated().sum())
        print(f"Warning: found {dup_count} duplicated filepaths. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["filepath"], keep="first").reset_index(drop=True)

    return df


def _build_stratify_key(df: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    if not columns:
        raise ValueError("columns must not be empty")
    return df[list(columns)].astype(str).agg("__".join, axis=1)


def _can_stratify(series: pd.Series) -> bool:
    if series.nunique() < 2:
        return False
    min_count = int(series.value_counts().min())
    return min_count >= 2


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    primary_stratify_cols: Sequence[str],
    fallback_stratify_cols: Sequence[str] = ("label",),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0 and 1.")

    stratify = None

    primary_key = _build_stratify_key(df, primary_stratify_cols)
    if _can_stratify(primary_key):
        stratify = primary_key
    else:
        fallback_key = _build_stratify_key(df, fallback_stratify_cols)
        if _can_stratify(fallback_key):
            stratify = fallback_key
        else:
            print(
                "Warning: stratified split was not possible with the provided columns. "
                "Falling back to random split."
            )

    left_df, right_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=stratify,
    )
    return left_df.reset_index(drop=True), right_df.reset_index(drop=True)


def save_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def summarize_df(name: str, df: pd.DataFrame) -> None:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    generator_counts = df["generator_slug"].value_counts().sort_index().to_dict()
    print(
        f"[{name}] total={len(df)} "
        f"label_counts={label_counts} "
        f"num_generators={df['generator_slug'].nunique()}"
    )
    print(f"[{name}] generator_counts={generator_counts}")


def build_merged_splits(
    df: pd.DataFrame,
    output_dir: Path,
    official_val_test_ratio: float,
    seed: int,
) -> None:
    train_df = df[df["official_split"] == "train"].reset_index(drop=True)
    official_val_df = df[df["official_split"] == "val"].reset_index(drop=True)

    val_df, test_df = split_dataframe(
        official_val_df,
        test_size=official_val_test_ratio,
        seed=seed,
        primary_stratify_cols=("generator_slug", "label"),
        fallback_stratify_cols=("label",),
    )

    merged_dir = output_dir / "merged"
    save_split_csv(train_df, merged_dir / "train.csv")
    save_split_csv(val_df, merged_dir / "val.csv")
    save_split_csv(test_df, merged_dir / "test.csv")

    summarize_df("merged/train", train_df)
    summarize_df("merged/val", val_df)
    summarize_df("merged/test", test_df)


def build_by_generator_splits(
    df: pd.DataFrame,
    output_dir: Path,
    official_val_test_ratio: float,
    seed: int,
) -> None:
    base_dir = output_dir / "by_generator"

    for generator_slug in sorted(df["generator_slug"].unique()):
        gen_df = df[df["generator_slug"] == generator_slug].reset_index(drop=True)
        train_df = gen_df[gen_df["official_split"] == "train"].reset_index(drop=True)
        official_val_df = gen_df[gen_df["official_split"] == "val"].reset_index(drop=True)

        val_df, test_df = split_dataframe(
            official_val_df,
            test_size=official_val_test_ratio,
            seed=seed,
            primary_stratify_cols=("label",),
            fallback_stratify_cols=("label",),
        )

        out_dir = base_dir / generator_slug
        save_split_csv(train_df, out_dir / "train.csv")
        save_split_csv(val_df, out_dir / "val.csv")
        save_split_csv(test_df, out_dir / "test.csv")

        summarize_df(f"by_generator/{generator_slug}/train", train_df)
        summarize_df(f"by_generator/{generator_slug}/val", val_df)
        summarize_df(f"by_generator/{generator_slug}/test", test_df)


def build_logo_splits(
    df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """
    leave-one-generator-out
    - train: all official train samples except holdout generator
    - val:   all official val samples except holdout generator
    - test:  all official val samples from holdout generator
    """
    base_dir = output_dir / "logo"

    for holdout_slug in sorted(df["generator_slug"].unique()):
        non_holdout_df = df[df["generator_slug"] != holdout_slug].reset_index(drop=True)
        holdout_df = df[df["generator_slug"] == holdout_slug].reset_index(drop=True)

        train_df = non_holdout_df[non_holdout_df["official_split"] == "train"].reset_index(drop=True)
        val_df = non_holdout_df[non_holdout_df["official_split"] == "val"].reset_index(drop=True)
        test_df = holdout_df[holdout_df["official_split"] == "val"].reset_index(drop=True)

        out_dir = base_dir / f"holdout_{holdout_slug}"
        save_split_csv(train_df, out_dir / "train.csv")
        save_split_csv(val_df, out_dir / "val.csv")
        save_split_csv(test_df, out_dir / "test.csv")

        summarize_df(f"logo/holdout_{holdout_slug}/train", train_df)
        summarize_df(f"logo/holdout_{holdout_slug}/val", val_df)
        summarize_df(f"logo/holdout_{holdout_slug}/test", test_df)


def main() -> None:
    args = parse_args()

    project_root = get_project_root()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    samples = collect_samples(input_dir=input_dir, project_root=project_root)
    df = samples_to_dataframe(samples)

    if args.save_full_index:
        save_split_csv(df, output_dir / "full_index.csv")
        print(f"Saved full index to: {(output_dir / 'full_index.csv').as_posix()}")

    print("\n=== Building merged splits ===")
    build_merged_splits(
        df=df,
        output_dir=output_dir,
        official_val_test_ratio=args.official_val_test_ratio,
        seed=args.seed,
    )

    print("\n=== Building by_generator splits ===")
    build_by_generator_splits(
        df=df,
        output_dir=output_dir,
        official_val_test_ratio=args.official_val_test_ratio,
        seed=args.seed,
    )

    print("\n=== Building leave-one-generator-out (logo) splits ===")
    build_logo_splits(
        df=df,
        output_dir=output_dir,
    )

    print("\nDone.")
    print(f"Saved GenImage indices under: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()