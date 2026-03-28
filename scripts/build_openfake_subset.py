from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Dict, List

from datasets import Image as HFImage, load_dataset, load_dataset_builder
from PIL import Image as PILImage, UnidentifiedImageError
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
        "--stage",
        type=str,
        default="all",
        choices=["all", "pass1", "pass2"],
        help="all: run pass1+pass2, pass1: sampling only, pass2: save only from selection json",
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
        "--selection-json",
        type=str,
        default=None,
        help="Path to selection json. Default: <output-root>/metadata/selection.json",
    )
    parser.add_argument(
        "--save-format",
        type=str,
        default="auto",
        choices=["auto", "png", "jpg", "jpeg"],
        help="How to save output images. 'auto' keeps source bytes/path when possible.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Only build metadata csv/json, do not save image files.",
    )
    parser.add_argument(
        "--skip-bad-images",
        action="store_true",
        help="Skip rows whose image bytes/path are unreadable.",
    )
    return parser.parse_args()


def make_output_dirs(output_root: Path, models: List[str]) -> None:
    (output_root / "real").mkdir(parents=True, exist_ok=True)
    for model in models:
        (output_root / "fake" / sanitize_model_name(model)).mkdir(parents=True, exist_ok=True)
    (output_root / "metadata").mkdir(parents=True, exist_ok=True)


def get_selection_json_path(output_root: Path, selection_json_arg: str | None) -> Path:
    if selection_json_arg:
        return Path(selection_json_arg)
    return output_root / "metadata" / "selection.json"


def get_total_rows(dataset_id: str, hf_split: str) -> int | None:
    try:
        builder = load_dataset_builder(dataset_id)
        info = builder.info
        if info.splits and hf_split in info.splits:
            return getattr(info.splits[hf_split], "num_examples", None)
    except Exception:
        return None
    return None


def load_stream(dataset_id: str, hf_split: str, metadata_only_stream: bool = False):
    if metadata_only_stream:
        return load_dataset(
            dataset_id,
            split=hf_split,
            streaming=True,
            columns=["label", "model"],
        )

    builder = load_dataset_builder(dataset_id)
    features = builder.info.features.copy()
    features["image"] = HFImage(decode=False)

    return load_dataset(
        dataset_id,
        split=hf_split,
        streaming=True,
        features=features,
        columns=["image", "label", "model", "release_date"],
    )


def sanitize_model_name(name: str) -> str:
    return name.replace("/", "-").replace(" ", "_")


def normalize_label(label) -> str:
    if isinstance(label, str):
        return label.lower()
    if isinstance(label, bool):
        return "fake" if label else "real"
    if isinstance(label, int):
        return "fake" if label == 1 else "real"
    return str(label).lower()


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
    ds = load_stream(dataset_id, hf_split, metadata_only_stream=True)

    for idx, row in enumerate(tqdm(ds, total=total_rows, desc="Pass 1: reservoir sampling", unit="row")):
        label = normalize_label(row.get("label"))
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


def build_selection_payload(
    args: argparse.Namespace,
    total_real_needed: int,
    sampled_fake_indices: Dict[str, List[int]],
    sampled_real_indices: List[int],
    fake_available: Dict[str, int],
    real_available: int,
) -> dict:
    return {
        "dataset_id": args.dataset_id,
        "hf_split": args.hf_split,
        "models": args.models,
        "num_per_model": args.num_per_model,
        "num_real": total_real_needed,
        "seed": args.seed,
        "sampled_fake_indices": sampled_fake_indices,
        "sampled_real_indices": sampled_real_indices,
        "fake_available": fake_available,
        "real_available": real_available,
    }


def save_selection(selection_path: Path, payload: dict) -> None:
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    with open(selection_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_selection(selection_path: Path) -> dict:
    if not selection_path.exists():
        raise FileNotFoundError(f"Selection file not found: {selection_path}")
    with open(selection_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    payload["models"] = [str(x) for x in payload["models"]]
    payload["num_per_model"] = int(payload["num_per_model"])
    payload["num_real"] = int(payload["num_real"])
    payload["seed"] = int(payload["seed"])
    payload["real_available"] = int(payload["real_available"])
    payload["fake_available"] = {str(k): int(v) for k, v in payload["fake_available"].items()}
    payload["sampled_fake_indices"] = {
        str(k): [int(x) for x in v]
        for k, v in payload["sampled_fake_indices"].items()
    }
    payload["sampled_real_indices"] = [int(x) for x in payload["sampled_real_indices"]]
    return payload


def _image_cell_to_bytes_and_path(image_cell) -> tuple[bytes | None, str | None]:
    if image_cell is None:
        return None, None

    if isinstance(image_cell, dict):
        return image_cell.get("bytes"), image_cell.get("path")

    if isinstance(image_cell, PILImage.Image):
        buffer = BytesIO()
        fmt = image_cell.format or "PNG"
        image_cell.save(buffer, format=fmt)
        return buffer.getvalue(), None

    return None, None


def _guess_ext_from_path(path_str: str | None) -> str | None:
    if not path_str:
        return None
    suffix = Path(path_str).suffix.lower().lstrip(".")
    if suffix in {"jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"}:
        return "jpg" if suffix == "jpeg" else suffix
    return None


def infer_extension_from_cell(image_cell, save_format: str) -> str:
    if save_format != "auto":
        return "jpg" if save_format == "jpeg" else save_format

    image_bytes, image_path = _image_cell_to_bytes_and_path(image_cell)

    ext_from_path = _guess_ext_from_path(image_path)
    if ext_from_path:
        return ext_from_path

    if image_bytes:
        try:
            with PILImage.open(BytesIO(image_bytes)) as img:
                fmt = getattr(img, "format", None)
            if fmt is not None:
                return EXT_MAP.get(fmt.upper(), "png")
        except Exception:
            pass

    return "png"


def save_image_from_cell(image_cell, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes, image_path = _image_cell_to_bytes_and_path(image_cell)

    if image_bytes is not None:
        with open(out_path, "wb") as f:
            f.write(image_bytes)
        return

    if image_path:
        shutil.copyfile(image_path, out_path)
        return

    raise ValueError("Unsupported image cell: no bytes/path found.")


def convert_and_save_image_from_cell(image_cell, out_path: Path, ext: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image_bytes, image_path = _image_cell_to_bytes_and_path(image_cell)

    if image_bytes is not None:
        img = PILImage.open(BytesIO(image_bytes))
    elif image_path:
        img = PILImage.open(image_path)
    else:
        raise ValueError("Unsupported image cell: no bytes/path found.")

    with img:
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
    skip_bad_images: bool,
) -> tuple[list[dict], int]:
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
    skipped_bad = 0

    for idx, row in enumerate(tqdm(ds, total=total_rows, desc="Pass 2: save selected subset", unit="row")):
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
            image_cell = row.get("image")
            ext = infer_extension_from_cell(image_cell, save_format)
            out_path = out_dir / f"{prefix}.{ext}"

            try:
                if save_format == "auto":
                    save_image_from_cell(image_cell, out_path)
                else:
                    convert_and_save_image_from_cell(image_cell, out_path, ext)
            except (UnidentifiedImageError, OSError, ValueError):
                if not skip_bad_images:
                    raise
                counters[(label, model)] -= 1
                skipped_bad += 1
                continue

            image_relpath = str(out_path.relative_to(output_root)).replace("\\", "/")

        saved_rows.append(
            {
                "source_hf_split": hf_split,
                "source_row_index": idx,
                "label": label,
                "binary_label": 1 if label == "fake" else 0,
                "model": model,
                "release_date": row.get("release_date"),
                "image_relpath": image_relpath,
            }
        )

        if len(saved_rows) == len(target_indices):
            break

    if not skip_bad_images and len(saved_rows) != len(target_indices):
        raise RuntimeError(f"Saved {len(saved_rows)} rows, but expected {len(target_indices)} selected rows.")

    return saved_rows, skipped_bad


def write_metadata(output_root: Path, rows: list[dict], summary: dict) -> None:
    metadata_dir = output_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "source_hf_split",
        "source_row_index",
        "label",
        "binary_label",
        "model",
        "release_date",
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
    skipped_bad_images: int,
    selection_path: Path,
) -> dict:
    label_counter = Counter(r["label"] for r in saved_rows)
    fake_model_counter = Counter(r["model"] for r in saved_rows if r["label"] == "fake")

    return {
        "stage": args.stage,
        "dataset_id": args.dataset_id,
        "source_hf_split": args.hf_split,
        "output_root": str(Path(args.output_root)),
        "selection_json": str(selection_path),
        "selected_models": args.models,
        "num_models": len(args.models),
        "num_per_model_fake": args.num_per_model,
        "num_real_requested": total_real_needed,
        "total_fake_selected_requested": args.num_per_model * len(args.models),
        "total_real_selected_requested": total_real_needed,
        "fake_available_in_source_split": fake_available,
        "real_available_in_source_split": real_available,
        "saved_counts_by_label": dict(label_counter),
        "saved_fake_counts_by_model": dict(fake_model_counter),
        "metadata_only": args.metadata_only,
        "save_format": args.save_format,
        "seed": args.seed,
        "skip_bad_images": args.skip_bad_images,
        "skipped_bad_images": skipped_bad_images,
        "directory_layout": {
            "real": "real/",
            "fake": "fake/<generator>/",
        },
    }


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    selection_path = get_selection_json_path(output_root, args.selection_json)
    total_real_needed = args.num_real if args.num_real is not None else args.num_per_model * len(args.models)

    make_output_dirs(output_root, args.models)

    if args.stage in {"all", "pass1"}:
        sampled_fake_indices, sampled_real_indices, fake_available, real_available = pass1_reservoir_sample(
            dataset_id=args.dataset_id,
            hf_split=args.hf_split,
            target_models=args.models,
            num_per_model=args.num_per_model,
            total_real_needed=total_real_needed,
            seed=args.seed,
        )

        selection_payload = build_selection_payload(
            args=args,
            total_real_needed=total_real_needed,
            sampled_fake_indices=sampled_fake_indices,
            sampled_real_indices=sampled_real_indices,
            fake_available=fake_available,
            real_available=real_available,
        )
        save_selection(selection_path, selection_payload)

        print("=" * 80)
        print("OpenFake selection finished")
        print("=" * 80)
        print(f"selection json : {selection_path}")
        print(f"selected_models: {len(args.models)}")
        print(f"fake per model : {args.num_per_model}")
        print(f"total fake req : {args.num_per_model * len(args.models)}")
        print(f"total real req : {total_real_needed}")

        if args.stage == "pass1":
            return

    if args.stage in {"all", "pass2"}:
        selection_payload = load_selection(selection_path)

        args.dataset_id = selection_payload["dataset_id"]
        args.hf_split = selection_payload["hf_split"]
        args.models = selection_payload["models"]
        args.num_per_model = selection_payload["num_per_model"]
        args.seed = selection_payload["seed"]
        total_real_needed = selection_payload["num_real"]

        make_output_dirs(output_root, args.models)

        saved_rows, skipped_bad_images = pass2_save_subset(
            dataset_id=args.dataset_id,
            hf_split=args.hf_split,
            sampled_fake_indices=selection_payload["sampled_fake_indices"],
            sampled_real_indices=selection_payload["sampled_real_indices"],
            output_root=output_root,
            save_format=args.save_format,
            metadata_only=args.metadata_only,
            skip_bad_images=args.skip_bad_images,
        )

        summary = build_summary(
            args=args,
            fake_available=selection_payload["fake_available"],
            real_available=selection_payload["real_available"],
            saved_rows=saved_rows,
            total_real_needed=total_real_needed,
            skipped_bad_images=skipped_bad_images,
            selection_path=selection_path,
        )
        write_metadata(output_root, saved_rows, summary)

        print("=" * 80)
        print("OpenFake raw subset download finished")
        print("=" * 80)
        print(f"selection json    : {selection_path}")
        print(f"output_root       : {output_root}")
        print(f"selected_models   : {len(args.models)}")
        print(f"fake per model    : {args.num_per_model}")
        print(f"total fake req    : {args.num_per_model * len(args.models)}")
        print(f"total real req    : {total_real_needed}")
        print(f"saved rows        : {len(saved_rows)}")
        print(f"skipped_bad_images: {skipped_bad_images}")
        print(f"metadata_only     : {args.metadata_only}")
        print(f"summary json      : {output_root / 'metadata' / 'summary.json'}")
        print(f"subset csv        : {output_root / 'metadata' / 'subset.csv'}")
        print("saved structure   : real/ and fake/<generator>/")


if __name__ == "__main__":
    main()
