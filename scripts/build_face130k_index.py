from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL_NAME_TO_INT = {
    "real": 0,
    "fake": 1,
}
SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class Sample:
    filepath: str
    label: int
    label_name: str
    generator: str
    generator_slug: str
    source_type: str  # "real" or "fake"


def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build FACE130K CSV indices for merged / by_generator / "
            "leave-one-generator-out (logo) experiments."
        )
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/raw/face130k",
        help="Root directory of raw FACE130K dataset.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits/face130k",
        help="Directory where CSV split files will be saved.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Train ratio for merged/by_generator, and real split for logo.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.1,
        help="Validation ratio for merged/by_generator, and real split for logo.",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Test ratio for merged/by_generator, and real split for logo.",
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


def validate_ratios(train_ratio: float, val_ratio: float, test_ratio: float) -> None:
    total = train_ratio + val_ratio + test_ratio
    if any(r <= 0.0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All ratios must be > 0.")
    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            f"train_ratio + val_ratio + test_ratio must sum to 1.0, got {total:.6f}"
        )


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def slugify_generator(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "unknown"


def infer_sample_from_path(path: Path, input_dir: Path) -> Optional[Sample]:
    """
    Expected FACE130K structure:
      input_dir / real / image.xxx
      input_dir / fake / <generator_name> / image.xxx
      input_dir / fake / <generator_name> / ... / image.xxx
    """
    try:
        rel_path = path.relative_to(input_dir)
    except ValueError:
        return None

    parts = rel_path.parts
    if len(parts) < 2:
        return None

    top_level = parts[0].lower()

    if top_level == "real":
        return Sample(
            filepath=rel_path.as_posix(),
            label=LABEL_NAME_TO_INT["real"],
            label_name="real",
            generator="REAL",
            generator_slug="real",
            source_type="real",
        )

    if top_level == "fake":
        if len(parts) < 3:
            return None

        generator_name = parts[1]
        generator_slug = slugify_generator(generator_name)

        return Sample(
            filepath=rel_path.as_posix(),
            label=LABEL_NAME_TO_INT["fake"],
            label_name="fake",
            generator=generator_name,
            generator_slug=generator_slug,
            source_type="fake",
        )

    return None


def collect_samples(input_dir: Path) -> List[Sample]:
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
        sample = infer_sample_from_path(path=path, input_dir=input_dir)
        if sample is None:
            skipped.append(path)
            continue
        samples.append(sample)

    if not samples:
        raise RuntimeError(
            "No valid FACE130K samples found.\n"
            "Expected structure:\n"
            "  data/raw/face130k/real/<image>\n"
            "  data/raw/face130k/fake/<generator>/<image>"
        )

    if skipped:
        print(
            f"Skipped {len(skipped)} files because they did not match the expected "
            "FACE130K directory pattern."
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
            "source_type": [s.source_type for s in samples],
        }
    )

    if df["filepath"].duplicated().any():
        dup_count = int(df["filepath"].duplicated().sum())
        print(f"Warning: found {dup_count} duplicated filepaths. Keeping first occurrence.")
        df = df.drop_duplicates(subset=["filepath"], keep="first").reset_index(drop=True)

    return df


def _build_stratify_key(df: pd.DataFrame, columns: Sequence[str]) -> Optional[pd.Series]:
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return None
    return df[cols].astype(str).agg("__".join, axis=1)


def _can_stratify(series: Optional[pd.Series]) -> bool:
    if series is None or len(series) == 0:
        return False
    if series.nunique() < 2:
        return False
    min_count = int(series.value_counts().min())
    return min_count >= 2


def split_dataframe(
    df: pd.DataFrame,
    test_size: float,
    seed: int,
    primary_stratify_cols: Sequence[str],
    fallback_stratify_cols: Sequence[str] = (),
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


def split_three_way(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    primary_stratify_cols: Sequence[str],
    fallback_stratify_cols: Sequence[str] = (),
) -> Dict[str, pd.DataFrame]:
    validate_ratios(train_ratio, val_ratio, test_ratio)

    train_val_df, test_df = split_dataframe(
        df=df,
        test_size=test_ratio,
        seed=seed,
        primary_stratify_cols=primary_stratify_cols,
        fallback_stratify_cols=fallback_stratify_cols,
    )

    val_share_in_train_val = val_ratio / (train_ratio + val_ratio)

    train_df, val_df = split_dataframe(
        df=train_val_df,
        test_size=val_share_in_train_val,
        seed=seed,
        primary_stratify_cols=primary_stratify_cols,
        fallback_stratify_cols=fallback_stratify_cols,
    )

    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def split_two_way(
    df: pd.DataFrame,
    left_ratio: float,
    right_ratio: float,
    seed: int,
    primary_stratify_cols: Sequence[str],
    fallback_stratify_cols: Sequence[str] = (),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total = left_ratio + right_ratio
    if total <= 0:
        raise ValueError("left_ratio + right_ratio must be > 0.")

    right_share = right_ratio / total

    return split_dataframe(
        df=df,
        test_size=right_share,
        seed=seed,
        primary_stratify_cols=primary_stratify_cols,
        fallback_stratify_cols=fallback_stratify_cols,
    )


def save_split_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")


def shuffle_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if df.empty:
        return df.reset_index(drop=True)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def concat_splits(dfs: Sequence[pd.DataFrame], seed: int) -> pd.DataFrame:
    non_empty = [df for df in dfs if not df.empty]
    if not non_empty:
        return pd.DataFrame(
            columns=[
                "filepath",
                "label",
                "label_name",
                "generator",
                "generator_slug",
                "source_type",
            ]
        )
    out = pd.concat(non_empty, axis=0, ignore_index=True)
    return shuffle_df(out, seed=seed)


def summarize_df(name: str, df: pd.DataFrame) -> None:
    label_counts = df["label"].value_counts().sort_index().to_dict()
    all_generator_counts = df["generator_slug"].value_counts().sort_index().to_dict()
    fake_generator_counts = (
        df[df["label"] == LABEL_NAME_TO_INT["fake"]]["generator_slug"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print(
        f"[{name}] total={len(df)} "
        f"label_counts={label_counts} "
        f"num_fake_generators={len(fake_generator_counts)}"
    )
    print(f"[{name}] generator_counts(all)={all_generator_counts}")
    print(f"[{name}] generator_counts(fake_only)={fake_generator_counts}")


def build_base_real_and_fake_splits(
    df: pd.DataFrame,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, pd.DataFrame]]]:
    real_df = df[df["label"] == LABEL_NAME_TO_INT["real"]].reset_index(drop=True)
    fake_df = df[df["label"] == LABEL_NAME_TO_INT["fake"]].reset_index(drop=True)

    if real_df.empty:
        raise RuntimeError("No real samples found.")
    if fake_df.empty:
        raise RuntimeError("No fake samples found.")

    real_splits = split_three_way(
        df=real_df,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        primary_stratify_cols=(),
        fallback_stratify_cols=(),
    )

    fake_splits_by_generator: Dict[str, Dict[str, pd.DataFrame]] = {}

    for generator_slug in sorted(fake_df["generator_slug"].unique()):
        gen_df = fake_df[fake_df["generator_slug"] == generator_slug].reset_index(drop=True)

        gen_splits = split_three_way(
            df=gen_df,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
            primary_stratify_cols=(),
            fallback_stratify_cols=(),
        )
        fake_splits_by_generator[generator_slug] = gen_splits

    return real_splits, fake_splits_by_generator


def build_merged_splits(
    real_splits: Dict[str, pd.DataFrame],
    fake_splits_by_generator: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    seed: int,
) -> None:
    merged_dir = output_dir / "merged"

    for idx, split_name in enumerate(SPLIT_NAMES):
        df = concat_splits(
            [real_splits[split_name]]
            + [fake_splits_by_generator[g][split_name] for g in sorted(fake_splits_by_generator)],
            seed=seed + idx,
        )
        save_split_csv(df, merged_dir / f"{split_name}.csv")
        summarize_df(f"merged/{split_name}", df)


def build_by_generator_splits(
    real_splits: Dict[str, pd.DataFrame],
    fake_splits_by_generator: Dict[str, Dict[str, pd.DataFrame]],
    output_dir: Path,
    seed: int,
) -> None:
    base_dir = output_dir / "by_generator"

    for exp_idx, generator_slug in enumerate(sorted(fake_splits_by_generator)):
        out_dir = base_dir / generator_slug

        for split_idx, split_name in enumerate(SPLIT_NAMES):
            df = concat_splits(
                [
                    real_splits[split_name],
                    fake_splits_by_generator[generator_slug][split_name],
                ],
                seed=seed + exp_idx * 10 + split_idx,
            )
            save_split_csv(df, out_dir / f"{split_name}.csv")
            summarize_df(f"by_generator/{generator_slug}/{split_name}", df)


def build_logo_splits(
    df: pd.DataFrame,
    real_splits: Dict[str, pd.DataFrame],
    output_dir: Path,
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> None:
    """
    Leave-one-generator-out for FACE130K:

    - train: real_train + fake(non-holdout generators) split into train
    - val:   real_val   + fake(non-holdout generators) split into val
    - test:  real_test  + fake(holdout generator) all
    """
    base_dir = output_dir / "logo"

    fake_df = df[df["label"] == LABEL_NAME_TO_INT["fake"]].reset_index(drop=True)
    generator_slugs = sorted(fake_df["generator_slug"].unique())

    if len(generator_slugs) < 2:
        raise RuntimeError("LOGO requires at least 2 fake generators.")

    for exp_idx, holdout_slug in enumerate(generator_slugs):
        seen_fake_df = fake_df[fake_df["generator_slug"] != holdout_slug].reset_index(drop=True)
        holdout_fake_df = fake_df[fake_df["generator_slug"] == holdout_slug].reset_index(drop=True)

        seen_fake_train_df, seen_fake_val_df = split_two_way(
            df=seen_fake_df,
            left_ratio=train_ratio,
            right_ratio=val_ratio,
            seed=seed,
            primary_stratify_cols=("generator_slug",),
            fallback_stratify_cols=(),
        )

        train_df = concat_splits(
            [real_splits["train"], seen_fake_train_df],
            seed=seed + exp_idx * 10 + 0,
        )
        val_df = concat_splits(
            [real_splits["val"], seen_fake_val_df],
            seed=seed + exp_idx * 10 + 1,
        )
        test_df = concat_splits(
            [real_splits["test"], holdout_fake_df],
            seed=seed + exp_idx * 10 + 2,
        )

        out_dir = base_dir / f"holdout_{holdout_slug}"
        save_split_csv(train_df, out_dir / "train.csv")
        save_split_csv(val_df, out_dir / "val.csv")
        save_split_csv(test_df, out_dir / "test.csv")

        summarize_df(f"logo/holdout_{holdout_slug}/train", train_df)
        summarize_df(f"logo/holdout_{holdout_slug}/val", val_df)
        summarize_df(f"logo/holdout_{holdout_slug}/test", test_df)


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.val_ratio, args.test_ratio)

    project_root = get_project_root()
    input_dir = (project_root / args.input_dir).resolve()
    output_dir = (project_root / args.output_dir).resolve()

    print(f"Project root: {project_root.as_posix()}")
    print(f"Input dir: {input_dir.as_posix()}")
    print(f"Output dir: {output_dir.as_posix()}")
    print(
        f"Ratios: train={args.train_ratio:.3f}, "
        f"val={args.val_ratio:.3f}, test={args.test_ratio:.3f}"
    )
    print(f"Seed: {args.seed}")

    samples = collect_samples(input_dir=input_dir)
    df = samples_to_dataframe(samples)

    print("\n=== Full dataset summary ===")
    summarize_df("full", df)

    if args.save_full_index:
        save_split_csv(df, output_dir / "full_index.csv")
        print(f"Saved full index to: {(output_dir / 'full_index.csv').as_posix()}")

    real_splits, fake_splits_by_generator = build_base_real_and_fake_splits(
        df=df,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    print("\n=== Building merged splits ===")
    build_merged_splits(
        real_splits=real_splits,
        fake_splits_by_generator=fake_splits_by_generator,
        output_dir=output_dir,
        seed=args.seed,
    )

    print("\n=== Building by_generator splits ===")
    build_by_generator_splits(
        real_splits=real_splits,
        fake_splits_by_generator=fake_splits_by_generator,
        output_dir=output_dir,
        seed=args.seed,
    )

    print("\n=== Building leave-one-generator-out (logo) splits ===")
    build_logo_splits(
        df=df,
        real_splits=real_splits,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print("\nDone.")
    print(f"Saved FACE130K indices under: {output_dir.as_posix()}")


if __name__ == "__main__":
    main()