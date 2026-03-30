#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build OpenFake CSV indices from:
    data/raw/openfake/
      real/
      fake/<generator>/

Outputs:
    data/splits/openfake/
      merged/
        train.csv
        val.csv
        test.csv
      by_generator/
        <generator>/
          train.csv
          val.csv
          test.csv
      logo/
        <heldout_generator>/
          train.csv
          val.csv
          test.csv
      summary.json

Notes
-----
1) merged:
   - Uses all selected fake generators together.
   - Real/fake are automatically matched to the smaller available side.

2) by_generator:
   - One binary dataset per generator vs real.
   - Reuses the SAME real split size for every generator for fair comparison.
   - If some fake images were deleted/corrupted, the shared size is reduced automatically.

3) logo = Leave-One-Generator-Out:
   - For held-out generator g:
       train fake = 70% of every other generator
       val fake   = 10% of every other generator
       test fake  = 100% of held-out generator
   - Real images are matched to the fake counts for each split.
   - If real images are slightly fewer than required, train/val are reduced first
     and held-out test is preserved as much as possible.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULT_GENERATORS = [
    "sd-3.5",
    "flux.1-dev",
    "flux-1.1-pro",
    "midjourney-6",
    "dalle-3",
    "gpt-image-1",
    "ideogram-3.0",
    "hidream-i1-full",
    "grok-2-image-1212",
    "imagen-4.0",
    "sdxl-epic-realism",
    "flux-mvc5000",
]


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    files = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            files.append(p)
    return sorted(files)


def relpath_posix(path: Path, project_root: Path) -> str:
    return path.relative_to(project_root).as_posix()


def split_counts(n: int, ratios=(0.7, 0.1, 0.2)) -> tuple[int, int, int]:
    train = int(round(n * ratios[0]))
    val = int(round(n * ratios[1]))
    test = n - train - val
    return train, val, test


def shuffle_copy(items: list[str], seed: int) -> list[str]:
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    return shuffled


def sample_n(items: list[str], n: int, seed: int) -> list[str]:
    if n > len(items):
        raise ValueError(f"Requested {n} items but only {len(items)} available.")
    return shuffle_copy(items, seed)[:n]


def split_fixed(
    items: list[str],
    n_train: int,
    n_val: int,
    n_test: int,
    seed: int,
) -> dict[str, list[str]]:
    requested = n_train + n_val + n_test
    if requested > len(items):
        raise ValueError(
            f"Requested {requested} items but only {len(items)} available."
        )
    shuffled = shuffle_copy(items, seed)
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:n_train + n_val + n_test]
    return {"train": train, "val": val, "test": test}


def sample_all_and_split(items: list[str], total: int, seed: int) -> dict[str, list[str]]:
    n_train, n_val, n_test = split_counts(total, ratios=(0.7, 0.1, 0.2))
    return split_fixed(items, n_train=n_train, n_val=n_val, n_test=n_test, seed=seed)


def build_rows(
    filepaths: Iterable[str],
    label: int,
    split: str,
    generator: str,
    mode: str,
    group: str,
) -> list[dict]:
    return [
        {
            "filepath": fp,
            "label": label,
            "split": split,
            "generator": generator,
            "mode": mode,
            "group": group,
        }
        for fp in filepaths
    ]


def save_split_csvs(rows_by_split: dict[str, list[dict]], out_dir: Path) -> dict[str, int]:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for split in ["train", "val", "test"]:
        rows = rows_by_split.get(split, [])
        df = pd.DataFrame(rows)
        df.to_csv(out_dir / f"{split}.csv", index=False)
        counts[split] = len(df)
    return counts


def adjust_logo_fake_counts(
    requested_train: int,
    requested_val: int,
    requested_test: int,
    real_available: int,
) -> tuple[int, int, int]:
    """
    Preserve held-out test as much as possible.
    If real is insufficient, reduce train/val first.
    If even test cannot fit, reduce test too.
    """
    total_requested = requested_train + requested_val + requested_test
    if real_available >= total_requested:
        return requested_train, requested_val, requested_test

    keep_test = min(requested_test, real_available)
    remaining_for_seen = max(0, real_available - keep_test)

    seen_requested = requested_train + requested_val
    if remaining_for_seen >= seen_requested:
        return requested_train, requested_val, keep_test

    if seen_requested == 0:
        return 0, 0, keep_test

    # proportionally reduce train/val
    keep_train = int(round(remaining_for_seen * (requested_train / seen_requested)))
    keep_train = min(keep_train, requested_train)
    keep_val = remaining_for_seen - keep_train

    # small safety correction
    keep_val = min(keep_val, requested_val)
    if keep_train + keep_val < remaining_for_seen:
        short = remaining_for_seen - (keep_train + keep_val)
        add_to_train = min(short, requested_train - keep_train)
        keep_train += add_to_train
        short -= add_to_train
        if short > 0:
            add_to_val = min(short, requested_val - keep_val)
            keep_val += add_to_val

    return keep_train, keep_val, keep_test


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=str, default="data/raw/openfake")
    parser.add_argument("--output-root", type=str, default="data/splits/openfake")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generators",
        nargs="*",
        default=DEFAULT_GENERATORS,
        help="Selected generators. Default is the recommended 12-model set.",
    )
    parser.add_argument(
        "--by-generator-real-count",
        type=int,
        default=8000,
        help="Requested real images reused for each by_generator dataset. Default=8000.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    input_root = (project_root / args.input_root).resolve()
    output_root = (project_root / args.output_root).resolve()

    real_root = input_root / "real"
    fake_root = input_root / "fake"

    if not real_root.exists():
        raise FileNotFoundError(f"Real directory not found: {real_root}")
    if not fake_root.exists():
        raise FileNotFoundError(f"Fake directory not found: {fake_root}")

    selected_generators = args.generators
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "seed": args.seed,
        "generators": selected_generators,
        "modes": {},
    }

    # ---------------------------------------------------------------------
    # Scan real / fake
    # ---------------------------------------------------------------------
    real_files_abs = list_images(real_root)
    real_files = [relpath_posix(p, project_root) for p in real_files_abs]

    fake_files_by_gen: dict[str, list[str]] = {}
    for gen in selected_generators:
        gen_dir = fake_root / gen
        if not gen_dir.exists():
            raise FileNotFoundError(f"Generator directory not found: {gen_dir}")
        gen_files = list_images(gen_dir)
        if len(gen_files) == 0:
            raise RuntimeError(f"No images found for generator: {gen_dir}")
        fake_files_by_gen[gen] = [relpath_posix(p, project_root) for p in gen_files]

    total_fake = sum(len(v) for v in fake_files_by_gen.values())
    min_fake_across_generators = min(len(v) for v in fake_files_by_gen.values())

    summary["scan"] = {
        "total_real": len(real_files),
        "total_fake": total_fake,
        "min_fake_across_generators": min_fake_across_generators,
        "fake_counts_by_generator": {gen: len(files) for gen, files in fake_files_by_gen.items()},
    }

    # ---------------------------------------------------------------------
    # Full per-generator fake 7:1:2 split (full available)
    # Used for LOGO.
    # ---------------------------------------------------------------------
    fake_splits_by_gen: dict[str, dict[str, list[str]]] = {}
    fake_counts_by_gen = {}
    for i, gen in enumerate(selected_generators):
        files = fake_files_by_gen[gen]
        n_train, n_val, n_test = split_counts(len(files), ratios=(0.7, 0.1, 0.2))
        fake_splits_by_gen[gen] = split_fixed(
            files,
            n_train=n_train,
            n_val=n_val,
            n_test=n_test,
            seed=args.seed + i,
        )
        fake_counts_by_gen[gen] = {
            "total": len(files),
            "train": n_train,
            "val": n_val,
            "test": n_test,
        }
    summary["full_fake_counts_by_generator"] = fake_counts_by_gen

    # ---------------------------------------------------------------------
    # MERGED
    # ---------------------------------------------------------------------
    merged_fake_all = []
    for gen in selected_generators:
        merged_fake_all.extend(fake_files_by_gen[gen])

    merged_usable_total = min(len(merged_fake_all), len(real_files))
    if merged_usable_total == 0:
        raise RuntimeError("No usable data for merged split.")

    merged_fake_splits = sample_all_and_split(
        merged_fake_all,
        total=merged_usable_total,
        seed=args.seed + 1000,
    )
    merged_real_splits = sample_all_and_split(
        real_files,
        total=merged_usable_total,
        seed=args.seed + 1001,
    )

    merged_rows = {
        "train": build_rows(merged_real_splits["train"], 0, "train", "real", "merged", "merged")
        + build_rows(merged_fake_splits["train"], 1, "train", "fake", "merged", "merged"),
        "val": build_rows(merged_real_splits["val"], 0, "val", "real", "merged", "merged")
        + build_rows(merged_fake_splits["val"], 1, "val", "fake", "merged", "merged"),
        "test": build_rows(merged_real_splits["test"], 0, "test", "real", "merged", "merged")
        + build_rows(merged_fake_splits["test"], 1, "test", "fake", "merged", "merged"),
    }
    merged_counts = save_split_csvs(merged_rows, output_root / "merged")
    summary["modes"]["merged"] = {
        "usable_total_per_class": merged_usable_total,
        "real_counts": {k: len(v) for k, v in merged_real_splits.items()},
        "fake_counts": {k: len(v) for k, v in merged_fake_splits.items()},
        "csv_counts": merged_counts,
    }

    # ---------------------------------------------------------------------
    # BY_GENERATOR
    # ---------------------------------------------------------------------
    by_gen_root = output_root / "by_generator"
    by_generator_summary = {}

    bg_requested_total = int(args.by_generator_real_count)
    bg_usable_total = min(bg_requested_total, len(real_files), min_fake_across_generators)
    if bg_usable_total == 0:
        raise RuntimeError("No usable data for by_generator splits.")

    bg_rt, bg_rv, bg_rte = split_counts(bg_usable_total, ratios=(0.7, 0.1, 0.2))
    bygen_real_splits = split_fixed(
        real_files,
        n_train=bg_rt,
        n_val=bg_rv,
        n_test=bg_rte,
        seed=args.seed + 2000,
    )

    for i, gen in enumerate(selected_generators):
        fake_sampled = sample_n(fake_files_by_gen[gen], bg_usable_total, seed=args.seed + 2100 + i)
        fake_splits = split_fixed(
            fake_sampled,
            n_train=bg_rt,
            n_val=bg_rv,
            n_test=bg_rte,
            seed=args.seed + 2200 + i,
        )

        rows_by_split = {
            "train": build_rows(bygen_real_splits["train"], 0, "train", "real", "by_generator", gen)
            + build_rows(fake_splits["train"], 1, "train", gen, "by_generator", gen),
            "val": build_rows(bygen_real_splits["val"], 0, "val", "real", "by_generator", gen)
            + build_rows(fake_splits["val"], 1, "val", gen, "by_generator", gen),
            "test": build_rows(bygen_real_splits["test"], 0, "test", "real", "by_generator", gen)
            + build_rows(fake_splits["test"], 1, "test", gen, "by_generator", gen),
        }
        counts = save_split_csvs(rows_by_split, by_gen_root / gen)
        by_generator_summary[gen] = {
            "requested_total_per_class": bg_requested_total,
            "usable_total_per_class": bg_usable_total,
            "real_counts": {k: len(v) for k, v in bygen_real_splits.items()},
            "fake_counts": {k: len(v) for k, v in fake_splits.items()},
            "csv_counts": counts,
        }
    summary["modes"]["by_generator"] = by_generator_summary

    # ---------------------------------------------------------------------
    # LOGO (Leave-One-Generator-Out)
    # ---------------------------------------------------------------------
    logo_root = output_root / "logo"
    logo_summary = {}

    for j, heldout in enumerate(selected_generators):
        seen = [g for g in selected_generators if g != heldout]

        requested_train_list = []
        requested_val_list = []
        for gen in seen:
            requested_train_list.extend(fake_splits_by_gen[gen]["train"])
            requested_val_list.extend(fake_splits_by_gen[gen]["val"])

        requested_test_list = fake_files_by_gen[heldout][:]

        req_train = len(requested_train_list)
        req_val = len(requested_val_list)
        req_test = len(requested_test_list)

        keep_train, keep_val, keep_test = adjust_logo_fake_counts(
            requested_train=req_train,
            requested_val=req_val,
            requested_test=req_test,
            real_available=len(real_files),
        )

        logo_fake_train = sample_n(requested_train_list, keep_train, seed=args.seed + 3000 + j)
        logo_fake_val = sample_n(requested_val_list, keep_val, seed=args.seed + 4000 + j)
        logo_fake_test = sample_n(requested_test_list, keep_test, seed=args.seed + 5000 + j)

        logo_real_splits = split_fixed(
            real_files,
            n_train=len(logo_fake_train),
            n_val=len(logo_fake_val),
            n_test=len(logo_fake_test),
            seed=args.seed + 6000 + j,
        )

        rows_by_split = {
            "train": build_rows(logo_real_splits["train"], 0, "train", "real", "logo", heldout)
            + build_rows(logo_fake_train, 1, "train", "fake", "logo", heldout),
            "val": build_rows(logo_real_splits["val"], 0, "val", "real", "logo", heldout)
            + build_rows(logo_fake_val, 1, "val", "fake", "logo", heldout),
            "test": build_rows(logo_real_splits["test"], 0, "test", "real", "logo", heldout)
            + build_rows(logo_fake_test, 1, "test", heldout, "logo", heldout),
        }
        counts = save_split_csvs(rows_by_split, logo_root / heldout)
        logo_summary[heldout] = {
            "heldout_generator": heldout,
            "seen_generators": seen,
            "requested_fake_counts": {
                "train": req_train,
                "val": req_val,
                "test": req_test,
            },
            "used_fake_counts": {
                "train": len(logo_fake_train),
                "val": len(logo_fake_val),
                "test": len(logo_fake_test),
            },
            "real_counts": {k: len(v) for k, v in logo_real_splits.items()},
            "csv_counts": counts,
            "note": (
                "LOGO keeps the held-out generator in test as much as possible. "
                "If real images are insufficient, train/val are reduced first."
            ),
        }

    summary["modes"]["logo"] = logo_summary

    # ---------------------------------------------------------------------
    # Save summary
    # ---------------------------------------------------------------------
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print("OpenFake index build complete")
    print("=" * 80)
    print(f"input_root : {input_root}")
    print(f"output_root: {output_root}")
    print(f"generators : {len(selected_generators)}")
    print(f"total real : {len(real_files):,}")
    print(f"total fake : {total_fake:,}")
    print()
    print("[merged]")
    print(
        f"  usable_per_class={merged_usable_total:,} | "
        f"train={merged_counts['train']:,} | "
        f"val={merged_counts['val']:,} | "
        f"test={merged_counts['test']:,}"
    )
    print()
    print("[by_generator]")
    print(
        f"  requested_per_class={bg_requested_total:,} | "
        f"usable_per_class={bg_usable_total:,} | "
        f"shared real split={bg_rt:,}/{bg_rv:,}/{bg_rte:,}"
    )
    print()
    print("[logo]")
    heldout_example = selected_generators[0]
    ex = logo_summary[heldout_example]
    print(
        f"  example held-out={heldout_example}: "
        f"train={ex['csv_counts']['train']:,} | "
        f"val={ex['csv_counts']['val']:,} | "
        f"test={ex['csv_counts']['test']:,}"
    )
    print()
    print(f"Saved summary -> {output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
