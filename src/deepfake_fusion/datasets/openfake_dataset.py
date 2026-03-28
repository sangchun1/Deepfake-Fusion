from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

PathLike = Union[str, Path]


def get_project_root() -> Path:
    """
    repo root 반환.
    현재 파일 위치: src/deepfake_fusion/datasets/openfake_dataset.py
    """
    return Path(__file__).resolve().parents[3]


class OpenFakeDataset(Dataset):
    """
    OpenFake binary classification dataset.

    기대하는 CSV 최소 형식:
        filepath,label

    예:
        data/raw/openfake/real/real__000001.jpg,0
        data/raw/openfake/fake/sd-3.5/sd-3.5__00001.jpg,1

    또는 root_dir 기준 상대경로:
        real/real__000001.jpg,0
        fake/sd-3.5/sd-3.5__00001.jpg,1

    추가 컬럼이 있으면(예: generator, mode, group, split, type, release_date ...)
    그대로 metadata로 반환한다.

    반환 형식:
        {
            "image": Tensor[C, H, W],
            "label": LongTensor scalar,
            "filepath": str,
            # optional metadata columns...
        }
    """

    def __init__(
        self,
        csv_path: PathLike,
        root_dir: Optional[PathLike] = None,
        transform: Optional[Callable] = None,
        image_mode: str = "RGB",
        validate_files: bool = True,
    ) -> None:
        self.project_root = get_project_root()
        self.csv_path = self._resolve_general_path(csv_path)
        self.root_dir = (
            self._resolve_general_path(root_dir) if root_dir is not None else None
        )
        self.transform = transform
        self.image_mode = image_mode
        self.validate_files = validate_files

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.df = pd.read_csv(self.csv_path)

        required_columns = {"filepath", "label"}
        missing_columns = required_columns - set(self.df.columns)
        if missing_columns:
            raise ValueError(
                f"CSV must contain columns {required_columns}, but missing: {missing_columns}"
            )

        self.df = self.df.reset_index(drop=True)
        self.df["filepath"] = self.df["filepath"].astype(str)
        self.df["label"] = self.df["label"].apply(self._normalize_label).astype(int)

        self.metadata_columns = [
            col for col in self.df.columns if col not in {"filepath", "label"}
        ]

        self.samples = []
        for row in self.df.itertuples(index=False):
            image_path = self._resolve_image_path(row.filepath)

            if self.validate_files and not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            metadata = {
                col: getattr(row, col)
                for col in self.metadata_columns
                if hasattr(row, col)
            }

            self.samples.append(
                {
                    "filepath": image_path,
                    "label": int(row.label),
                    "metadata": metadata,
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self.samples[index]
        image_path: Path = sample["filepath"]
        label: int = sample["label"]
        metadata: Dict[str, Any] = sample["metadata"]

        image = Image.open(image_path)
        image = image.convert(self.image_mode)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = TF.pil_to_tensor(image)
            image = TF.convert_image_dtype(image, torch.float32)

        output = {
            "image": image,
            "label": torch.tensor(label, dtype=torch.long),
            "filepath": image_path.as_posix(),
        }
        output.update(metadata)
        return output

    @property
    def num_classes(self) -> int:
        return int(self.df["label"].nunique())

    @property
    def class_counts(self) -> Dict[int, int]:
        counts = self.df["label"].value_counts().sort_index().to_dict()
        return {int(k): int(v) for k, v in counts.items()}

    def _resolve_general_path(self, path: PathLike) -> Path:
        """
        일반 경로 해석:
        - 절대경로면 그대로
        - 상대경로면 project root 기준
        """
        path = Path(path)
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()

    def _resolve_image_path(self, filepath: str) -> Path:
        """
        CSV의 filepath를 실제 이미지 경로로 해석.

        지원 형태:
        1) 절대경로
        2) project root 기준 상대경로
           예: data/raw/openfake/real/xxx.jpg
        3) root_dir 기준 상대경로
           예: real/xxx.jpg
        """
        raw_path = Path(filepath)
        candidate_paths = []

        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append(self.project_root / raw_path)
            if self.root_dir is not None:
                candidate_paths.append(self.root_dir / raw_path)

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate.resolve()

        # validate_files=False일 때도 일관된 경로 하나는 반환
        if candidate_paths:
            return candidate_paths[0].resolve()

        raise FileNotFoundError(f"Could not resolve image path from filepath: {filepath}")

    @staticmethod
    def _normalize_label(label: Any) -> int:
        """
        라벨을 정수로 정규화.

        허용 예:
        - 0, 1
        - "0", "1"
        - "real", "fake"
        - "REAL", "FAKE"
        - False, True
        """
        if isinstance(label, bool):
            return int(label)

        if isinstance(label, (int, float)) and int(label) in (0, 1):
            return int(label)

        label_str = str(label).strip().lower()
        if label_str in {"0", "real"}:
            return 0
        if label_str in {"1", "fake"}:
            return 1

        raise ValueError(f"Unsupported label value: {label}")


def build_openfake_dataset(
    csv_path: PathLike,
    root_dir: Optional[PathLike] = None,
    transform: Optional[Callable] = None,
    image_mode: str = "RGB",
    validate_files: bool = True,
) -> OpenFakeDataset:
    """
    간단한 dataset 생성 helper.
    """
    return OpenFakeDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
        image_mode=image_mode,
        validate_files=validate_files,
    )