from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

PathLike = Union[str, Path]


def get_project_root() -> Path:
    """Repo root 반환.

    현재 파일 위치:
    src/deepfake_fusion/datasets/semitruths_dataset.py
    """
    return Path(__file__).resolve().parents[3]


class SemiTruthsDataset(Dataset):
    """Semi-Truths / Semi-Truths-Evalset binary classification dataset.

    이 Dataset은 두 가지 CSV 형식을 모두 지원한다.

    1) Semi-Truths metadata.csv 형식
       - 최소 필요 컬럼: image_path (또는 filepath)
       - label 컬럼이 있으면 그대로 사용
       - label 컬럼이 없으면 image_path를 보고 자동 추론
         * original/...   -> real (0)
         * inpainting/... -> fake (1)
         * p2p/...        -> fake (1)

    2) 프로젝트 split CSV 형식
       - 최소 필요 컬럼: filepath, label

    추가 컬럼은 모두 metadata로 반환한다.

    반환 형식 예시:
    {
        "image": Tensor[C, H, W],
        "label": LongTensor scalar,
        "filepath": str,
        "image_id": str,          # optional
        "mask_path": str | None,  # optional
        "dataset": str,           # optional
        "method": str,            # optional
        "diffusion_model": str,   # optional
        ...                         # any other metadata columns
    }
    """

    FILEPATH_CANDIDATES: List[str] = [
        "filepath",
        "image_path",
        "path",
        "image",
        "img_path",
    ]
    LABEL_CANDIDATES: List[str] = [
        "label",
        "target",
        "class",
        "y",
        "is_fake",
    ]
    MASKPATH_CANDIDATES: List[str] = [
        "mask_path",
        "mask_filepath",
        "mask",
        "mask_file",
    ]

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
        if len(self.df) == 0:
            raise ValueError(f"CSV is empty: {self.csv_path}")

        self.filepath_col = self._find_first_existing_column(
            self.df.columns, self.FILEPATH_CANDIDATES
        )
        if self.filepath_col is None:
            raise ValueError(
                "CSV must contain one of filepath columns: "
                f"{self.FILEPATH_CANDIDATES}. Found: {list(self.df.columns)}"
            )

        self.label_col = self._find_first_existing_column(
            self.df.columns, self.LABEL_CANDIDATES
        )
        self.mask_path_col = self._find_first_existing_column(
            self.df.columns, self.MASKPATH_CANDIDATES
        )

        self.df = self.df.reset_index(drop=True).copy()
        self.df[self.filepath_col] = (
            self.df[self.filepath_col].astype(str).map(self._normalize_slashes)
        )

        if self.label_col is None:
            self.df["label"] = self.df[self.filepath_col].map(self._infer_label_from_path)
            self.label_col = "label"
        else:
            self.df[self.label_col] = self.df[self.label_col].map(self._normalize_label)

        if self.mask_path_col is not None:
            self.df[self.mask_path_col] = self.df[self.mask_path_col].apply(
                lambda x: self._normalize_slashes(str(x))
                if pd.notna(x) and str(x).strip() != ""
                else None
            )

        base_columns = {self.filepath_col, self.label_col}
        if self.mask_path_col is not None:
            base_columns.add(self.mask_path_col)
        self.metadata_columns = [col for col in self.df.columns if col not in base_columns]

        self.samples: List[Dict[str, Any]] = []
        for row in self.df.itertuples(index=False):
            raw_filepath = getattr(row, self.filepath_col)
            image_path = self._resolve_image_path(raw_filepath)
            if self.validate_files and not image_path.exists():
                raise FileNotFoundError(f"Image file not found: {image_path}")

            label = int(getattr(row, self.label_col))

            raw_mask_path = None
            mask_path: Optional[Path] = None
            if self.mask_path_col is not None and hasattr(row, self.mask_path_col):
                raw_mask_path = getattr(row, self.mask_path_col)
                if raw_mask_path not in (None, "", "nan") and not pd.isna(raw_mask_path):
                    mask_path = self._resolve_mask_path(str(raw_mask_path))
                    if self.validate_files and mask_path is not None and not mask_path.exists():
                        # mask는 optional metadata로 취급하므로 strict하게 막진 않음.
                        mask_path = None

            metadata: Dict[str, Any] = {}
            for col in self.metadata_columns:
                value = getattr(row, col)
                metadata[col] = self._sanitize_metadata_value(value)

            if raw_mask_path is not None:
                metadata["mask_path"] = mask_path.as_posix() if mask_path is not None else None

            self.samples.append(
                {
                    "filepath": image_path,
                    "label": label,
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
        return int(pd.Series([s["label"] for s in self.samples]).nunique())

    @property
    def class_counts(self) -> Dict[int, int]:
        counts = pd.Series([s["label"] for s in self.samples]).value_counts().sort_index()
        return {int(k): int(v) for k, v in counts.to_dict().items()}

    def _resolve_general_path(self, path: PathLike) -> Path:
        """일반 경로 해석.

        - 절대경로면 그대로 사용
        - 상대경로면 project root 기준
        """
        path = Path(path)
        if path.is_absolute():
            return path.resolve()
        return (self.project_root / path).resolve()

    def _resolve_image_path(self, filepath: str) -> Path:
        """CSV의 filepath/image_path를 실제 이미지 경로로 해석.

        지원 형태:
        1) 절대경로
        2) project root 기준 상대경로
        3) root_dir 기준 상대경로

        Semi-Truths-Evalset 예:
        - original/images/ADE20K/xxx.png
        - inpainting/ADE20K/xxx.png
        - p2p/OpenImages/xxx.png
        """
        raw_path = Path(self._normalize_slashes(filepath))
        candidate_paths = self._build_candidate_paths(raw_path)

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate.resolve()

        if candidate_paths:
            return candidate_paths[0].resolve()
        raise FileNotFoundError(f"Could not resolve image path from filepath: {filepath}")

    def _resolve_mask_path(self, filepath: str) -> Optional[Path]:
        raw_path = Path(self._normalize_slashes(filepath))
        candidate_paths = self._build_candidate_paths(raw_path)

        for candidate in candidate_paths:
            if candidate.exists():
                return candidate.resolve()

        if candidate_paths:
            return candidate_paths[0].resolve()
        return None

    def _build_candidate_paths(self, raw_path: Path) -> List[Path]:
        candidate_paths: List[Path] = []

        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
            return candidate_paths

        candidate_paths.append(self.project_root / raw_path)

        if self.root_dir is not None:
            candidate_paths.append(self.root_dir / raw_path)

            # metadata에 dataset root 기준 상대경로가 아니라,
            # folder 내부 상대경로만 들어있는 경우를 대비한 fallback.
            raw_str = raw_path.as_posix()
            if not raw_str.startswith(("original/", "inpainting/", "p2p/", "p2p_masks/")):
                candidate_paths.extend(
                    [
                        self.root_dir / "original" / raw_path,
                        self.root_dir / "original" / "images" / raw_path,
                        self.root_dir / "inpainting" / raw_path,
                        self.root_dir / "p2p" / raw_path,
                        self.root_dir / "p2p_masks" / raw_path,
                    ]
                )

        # 중복 제거
        deduped: List[Path] = []
        seen = set()
        for path in candidate_paths:
            key = path.as_posix()
            if key not in seen:
                deduped.append(path)
                seen.add(key)
        return deduped

    @classmethod
    def _find_first_existing_column(
        cls,
        columns: Any,
        candidates: List[str],
    ) -> Optional[str]:
        column_set = set(columns)
        for candidate in candidates:
            if candidate in column_set:
                return candidate
        return None

    @staticmethod
    def _normalize_slashes(value: str) -> str:
        return str(value).replace("\\", "/").strip()

    @staticmethod
    def _sanitize_metadata_value(value: Any) -> Any:
        if pd.isna(value):
            return None
        if isinstance(value, Path):
            return value.as_posix()
        return value

    @staticmethod
    def _normalize_label(label: Any) -> int:
        """라벨을 정수로 정규화.

        허용 예:
        - 0, 1
        - "0", "1"
        - "real", "fake"
        - bool / is_fake 스타일 값
        """
        if isinstance(label, bool):
            return int(label)

        if isinstance(label, (int, float)) and not pd.isna(label):
            label_int = int(label)
            if label_int in (0, 1):
                return label_int

        label_str = str(label).strip().lower()
        if label_str in {"0", "real", "original", "true_negative", "pristine"}:
            return 0
        if label_str in {"1", "fake", "edited", "augmented", "inpainting", "p2p"}:
            return 1
        if label_str in {"true", "yes"}:
            return 1
        if label_str in {"false", "no"}:
            return 0

        raise ValueError(f"Unsupported label value: {label}")

    @staticmethod
    def _infer_label_from_path(filepath: str) -> int:
        path_str = str(filepath).replace("\\", "/").lower()
        if "original/" in path_str:
            return 0
        if "inpainting/" in path_str or "p2p/" in path_str:
            return 1
        raise ValueError(
            "Could not infer label from filepath. "
            f"Please include an explicit label column. filepath={filepath}"
        )



def build_semitruths_dataset(
    csv_path: PathLike,
    root_dir: Optional[PathLike] = None,
    transform: Optional[Callable] = None,
    image_mode: str = "RGB",
    validate_files: bool = True,
) -> SemiTruthsDataset:
    """간단한 dataset 생성 helper."""
    return SemiTruthsDataset(
        csv_path=csv_path,
        root_dir=root_dir,
        transform=transform,
        image_mode=image_mode,
        validate_files=validate_files,
    )
