#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset, load_dataset_builder
from PIL import Image
from tqdm import tqdm


DEFAULT_MODELS = [
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

EXT_MAP = {
    "PNG": "png",
    "JPEG": "jpg",
    "JPG": "jpg",
    "WEBP": "webp",
    "BMP": "bmp",
    "TIFF": "tiff",
}


class ReservoirSampler:
    def __init__(self, k: int, rng: random.Random):
        self.k = int(k)
        self.rng = rng
        self.n_seen = 0
        self.items: List[int] = []

    def consider(self, item: int) -> None:
        self.n_seen += 1
        if len(self.items) < self.k:
            self.items.append(item)
            return

        j = self.rng.randint(0, self.n_seen - 1)
        if j < self.k:
            self.items[j] = item


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a balanced OpenFake subset without split folders."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default="ComplexDataLab/OpenFake",
        help="Hugging Face dataset id.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="train",
        help="Official HF split to sample from. Default: train",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="data/raw/openfake",
        help="Root directory to save images and metadata.",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=DEFAULT_MODELS,
        help="Target fake models.",
    )
    parser.add_argument(
        "--num-per-model",
        type=int,
        default=8000,
        help="Number of fake images to sample per model.",
    )
    parser.add_argument(
        "--num-real",
        type=int,
        default=None,
        help="Number of real images to sample. Default: same as total fake selected.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-format",
        type=str,
        default="auto",
        choices=["auto", "png", "jpg", "jpeg"],
        help="How to save output images. 'auto' uses source format when available, otherwise png.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only build metadata csv/json, do not save image files.",
    )
    return parser.parse_args()


def make_output_dirs(output_root: Path, models: List[str]) -> None:
    (output_root / "real").mkdir(parents=True, exist_ok=True)
    for model in models:
        (output_root / "fake" / sanitize_model_name(model)).mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)


def get_total_rows(dataset_id: str, hf_split: str) -> int | None:
    try:
        builder = load_dataset_builder(dataset_id)
        info = builder.info
        if info.splits and hf_split in info.splits:
            return getattr(info.splits[hf_split], "num_examples", None)
    except Exception:
        return None
    return None


def load_stream(dataset_id: str, hf_split: str):
    return load_dataset(dataset_id, split=hf_split, streaming=True)


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def pass1_reservoir_sample(
    dataset_id: str,
    hf_split: str,
    target_models: List[str],
    num_per_model: int,
    total_real_needed: int,
    seed: int,
) -> tuple[Dict[str, List[int]], List[int], Dict[str, int], int]:
    rng = random.Random(seed)
    fake_samplers = {
        model: ReservoirSampler(num_per_model, random.Random(rng.randint(0, 10**9)))
        for model in target_models
    }
    real_sampler = ReservoirSampler(total_real_needed, random.Random(rng.randint(0, 10**9)))

    fake_available = Counter()
    real_available = 0

    total_rows = get_total_rows(dataset_id, hf_split)
    ds = load_stream(dataset_id, hf_split)

    for idx, row in enumerate(tqdm(ds, total=total_rows, desc="Pass 1/2: reservoir sampling", unit="row")):
        label = row.get("label")
        model = row.get("model")

        if label == "fake" and model in fake_samplers:
            fake_available[model] += 1
            fake_samplers[model].consider(idx)
        elif label == "real":
            real_available += 1
            real_sampler.consider(idx)

    missing = {m: num_per_model - fake_available[m] for m in target_models if fake_available[m] < num_per_model}
    if missing:
        lines = [f"{m}: need {num_per_model}, found {fake_available[m]}" for m in target_models if m in missing]
        raise RuntimeError(
            "Some models do not have enough fake images in the selected HF split:\n" + "\n".join(lines)
        )

    if real_available < total_real_needed:
        raise RuntimeError(f"Not enough real images: need {total_real_needed}, found {real_available}")

    sampled_fake_indices = {model: sorted(fake_samplers[model].items) for model in target_models}
    sampled_real_indices = sorted(real_sampler.items)
    return sampled_fake_indices, sampled_real_indices, dict(fake_available), real_available


def infer_extension(img: Image.Image, save_format: str) -> str:
    if save_format != "auto":
        return "jpg" if save_format == "jpeg" else save_format

    fmt = getattr(img, "format", None)
    if fmt is None:
        return "png"
    return EXT_MAP.get(fmt.upper(), "png")


def save_image(img: Image.Image, out_path: Path, ext: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if ext in {"jpg", "jpeg"}:
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(out_path, format="JPEG", quality=95)
    elif ext == "png":
        img.save(out_path, format="PNG")
    elif ext == "webp":
        img.save(out_path, format="WEBP")
    elif ext == "bmp":
        img.save(out_path, format="BMP")
    elif ext in {"tif", "tiff"}:
        img.save(out_path, format="TIFF")
    else:
        img.save(out_path, format="PNG")


def pass2_save_subset(
    dataset_id: str,
    hf_split: str,
    sampled_fake_indices: Dict[str, List[int]],
    sampled_real_indices: List[int],
    output_root: Path,
    save_format: str,
    metadata_only: bool,
) -> list[dict]:
    total_rows = get_total_rows(dataset_id, hf_split)
    ds = load_stream(dataset_id, hf_split)

    index_to_model = {}
    for model, indices in sampled_fake_indices.items():
        for idx in indices:
            index_to_model[idx] = model
    real_index_set = set(sampled_real_indices)
    target_indices = set(index_to_model.keys()) | real_index_set

    counters = Counter()
    saved_rows = []

    for idx, row in enumerate(tqdm(ds, total=total_rows, desc="Pass 2/2: save selected subset", unit="row")):
        if idx not in target_indices:
            continue

        is_real = idx in real_index_set
        label = "real" if is_real else "fake"
        model = "real" if is_real else index_to_model[idx]
        counters[(label, model)] += 1
        local_idx = counters[(label, model)]

        if is_real:
            prefix = f"real__{local_idx:06d}"
            out_dir = output_root / "real"
        else:
            safe_model = sanitize_model_name(model)
            prefix = f"{safe_model}__{local_idx:05d}"
            out_dir = output_root / "fake" / safe_model

        image_relpath = None
        if not metadata_only:
            img = row["image"]
            ext = infer_extension(img, save_format)
            out_path = out_dir / f"{prefix}.{ext}"
            save_image(img, out_path, ext)
            image_relpath = str(out_path.relative_to(output_root)).replace("\\", "/")

        saved_rows.append(
            {
                "source_hf_split": hf_split,
                "source_row_index": idx,
                "label": label,
                "binary_label": 1 if label == "fake" else 0,
                "model": model,
                "type": row.get("type"),
                "release_date": row.get("release_date"),
                "prompt": row.get("prompt"),
                "image_relpath": image_relpath,
            }
        )

        if len(saved_rows) == len(target_indices):
            break

    if len(saved_rows) != len(target_indices):
        raise RuntimeError(f"Saved {len(saved_rows)} rows, but expected {len(target_indices)} selected rows.")

    return saved_rows


def write_metadata(output_root: Path, rows: list[dict], summary: dict) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_hf_split",
        "source_row_index",
        "label",
        "binary_label",
        "model",
        "type",
        "release_date",
        "prompt",
        "image_relpath",
    ]

    with open(metadata_dir / "subset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with open(metadata_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def build_summary(
    args: argparse.Namespace,
    fake_available: Dict[str, int],
    real_available: int,
    saved_rows: list[dict],
    total_real_needed: int,
) -> dict:
    label_counter = Counter(r["label"] for r in saved_rows)
    fake_model_counter = Counter(r["model"] for r in saved_rows if r["label"] == "fake")

    return {
        "dataset_id": args.dataset_id,
        "source_hf_split": args.hf_split,
        "output_root": str(Path(args.output_root)),
        "selected_models": args.models,
        "num_models": len(args.models),
        "num_per_model_fake": args.num_per_model,
        "num_real": total_real_needed,
        "total_fake_selected": args.num_per_model * len(args.models),
        "total_real_selected": total_real_needed,
        "fake_available_in_source_split": fake_available,
        "real_available_in_source_split": real_available,
        "saved_counts_by_label": dict(label_counter),
        "saved_fake_counts_by_model": dict(fake_model_counter),
        "metadata_only": args.metadata_only,
        "save_format": args.save_format,
        "seed": args.seed,
        "directory_layout": {
            "real": "real/",
            "fake": "fake/<generator>/",
        },
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    total_real_needed = args.num_real if args.num_real is not None else args.num_per_model * len(args.models)

    make_output_dirs(output_root, args.models)

    sampled_fake_indices, sampled_real_indices, fake_available, real_available = pass1_reservoir_sample(
        dataset_id=args.dataset_id,
        hf_split=args.hf_split,
        target_models=args.models,
        num_per_model=args.num_per_model,
        total_real_needed=total_real_needed,
        seed=args.seed,
    )

    saved_rows = pass2_save_subset(
        dataset_id=args.dataset_id,
        hf_split=args.hf_split,
        sampled_fake_indices=sampled_fake_indices,
        sampled_real_indices=sampled_real_indices,
        output_root=output_root,
        save_format=args.save_format,
        metadata_only=args.metadata_only,
    )

    summary = build_summary(
        args=args,
        fake_available=fake_available,
        real_available=real_available,
        saved_rows=saved_rows,
        total_real_needed=total_real_needed,
    )
    write_metadata(output_root, saved_rows, summary)

    print("=" * 80)
    print("OpenFake raw subset download finished")
    print("=" * 80)
    print(f"output_root      : {output_root}")
    print(f"selected_models  : {len(args.models)}")
    print(f"fake per model   : {args.num_per_model}")
    print(f"total fake       : {args.num_per_model * len(args.models)}")
    print(f"total real       : {total_real_needed}")
    print(f"metadata_only    : {args.metadata_only}")
    print(f"summary json     : {output_root / 'metadata' / 'summary.json'}")
    print(f"subset csv       : {output_root / 'metadata' / 'subset.csv'}")
    print("saved structure  : real/ and fake/<generator>/")


if __name__ == "__main__":
    main()
