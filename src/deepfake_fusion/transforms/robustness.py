from __future__ import annotations

import io
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms as T

from .image_aug import IMAGENET_MEAN, IMAGENET_STD, build_eval_transform

try:
    _PIL_RESAMPLING = Image.Resampling
except AttributeError:  # Pillow < 9
    _PIL_RESAMPLING = Image


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """
    dict 또는 attribute 방식 모두 지원하는 안전한 config getter.
    """
    current = cfg
    for key in keys:
        if current is None:
            return default
        if isinstance(current, Mapping):
            current = current.get(key, None)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return current


def _to_2tuple(size: Union[int, Sequence[int]]) -> Tuple[int, int]:
    """
    입력 크기를 (H, W) 형태로 정규화.
    """
    if isinstance(size, int):
        return (size, size)
    size = tuple(size)
    if len(size) != 2:
        raise ValueError(f"input_size must be int or sequence of length 2, got: {size}")
    return int(size[0]), int(size[1])


def _to_float_tuple(
    values: Optional[Sequence[float]],
    default: Tuple[float, ...],
) -> Tuple[float, ...]:
    if values is None:
        return default
    return tuple(float(v) for v in values)


def _normalize_corruption_name(name: Optional[str]) -> str:
    if name is None:
        return "clean"
    name = str(name).strip().lower()
    aliases = {
        "clean": "clean",
        "none": "clean",
        "jpeg": "jpeg",
        "jpg": "jpeg",
        "resize": "resize_down_up",
        "resize_down_up": "resize_down_up",
        "down_up_resize": "resize_down_up",
        "blur": "gaussian_blur",
        "gaussian_blur": "gaussian_blur",
        "noise": "gaussian_noise",
        "gaussian_noise": "gaussian_noise",
        "brightness_contrast": "brightness_contrast",
        "color": "brightness_contrast",
    }
    return aliases.get(name, name)


def _get_interpolation(interpolation: str):
    interpolation = str(interpolation).strip().lower()
    mapping = {
        "nearest": _PIL_RESAMPLING.NEAREST,
        "bilinear": _PIL_RESAMPLING.BILINEAR,
        "bicubic": _PIL_RESAMPLING.BICUBIC,
        "lanczos": _PIL_RESAMPLING.LANCZOS,
    }
    if interpolation not in mapping:
        supported = ", ".join(mapping.keys())
        raise ValueError(
            f"Unsupported interpolation '{interpolation}'. Supported: {supported}"
        )
    return mapping[interpolation]


def _image_from_uint8(array: np.ndarray, mode: str) -> Image.Image:
    if array.ndim == 2:
        return Image.fromarray(array, mode="L")
    if array.ndim == 3 and array.shape[2] == 3:
        return Image.fromarray(array, mode="RGB")
    if array.ndim == 3 and array.shape[2] == 4:
        return Image.fromarray(array, mode="RGBA")
    # fallback
    return Image.fromarray(array)


def apply_jpeg(image: Image.Image, quality: int) -> Image.Image:
    """
    JPEG 압축 artifact를 PIL 이미지에 적용.
    """
    quality = int(max(1, min(100, quality)))
    src_mode = image.mode
    work_image = image.convert("RGB") if src_mode != "RGB" else image

    buffer = io.BytesIO()
    work_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    buffer.seek(0)

    corrupted = Image.open(buffer)
    corrupted.load()
    if src_mode != "RGB":
        try:
            corrupted = corrupted.convert(src_mode)
        except ValueError:
            corrupted = corrupted.convert("RGB")
    return corrupted


def apply_resize_down_up(
    image: Image.Image,
    scale: float,
    interpolation: str = "bilinear",
) -> Image.Image:
    """
    이미지를 축소 후 원래 크기로 다시 복원.
    """
    scale = float(scale)
    if not (0.0 < scale <= 1.0):
        raise ValueError(f"scale must be in (0, 1], got: {scale}")

    orig_w, orig_h = image.size
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    resample = _get_interpolation(interpolation)

    down = image.resize((resized_w, resized_h), resample=resample)
    up = down.resize((orig_w, orig_h), resample=resample)
    return up


def apply_gaussian_blur(image: Image.Image, sigma: float) -> Image.Image:
    """
    Gaussian blur 적용.
    """
    sigma = max(0.0, float(sigma))
    return image.filter(ImageFilter.GaussianBlur(radius=sigma))


def apply_gaussian_noise(image: Image.Image, std: float) -> Image.Image:
    """
    [0, 1] 범위 기준 표준편차 std의 additive Gaussian noise 적용.
    """
    std = max(0.0, float(std))
    image_np = np.asarray(image).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=std, size=image_np.shape).astype(np.float32)
    corrupted = np.clip(image_np + noise, 0.0, 1.0)
    corrupted = (corrupted * 255.0).round().astype(np.uint8)
    return _image_from_uint8(corrupted, mode=image.mode)


def apply_brightness_contrast(
    image: Image.Image,
    brightness: float,
    contrast: float,
) -> Image.Image:
    """
    밝기/대비를 순차 적용.
    """
    brightness = float(brightness)
    contrast = float(contrast)
    out = ImageEnhance.Brightness(image).enhance(brightness)
    out = ImageEnhance.Contrast(out).enhance(contrast)
    return out


def get_enabled_corruptions(robustness_cfg: Any) -> List[str]:
    """
    config에서 enabled=true 인 corruption 이름 목록 반환.
    """
    corruptions = _cfg_get(robustness_cfg, "corruptions", default={})
    if not isinstance(corruptions, Mapping):
        raise ValueError("robustness.corruptions must be a mapping.")

    names: List[str] = []
    for name, sub_cfg in corruptions.items():
        if bool(_cfg_get(sub_cfg, "enabled", default=True)):
            names.append(_normalize_corruption_name(name))
    return names


def get_benchmark_severities(robustness_cfg: Any) -> List[int]:
    severities = _cfg_get(robustness_cfg, "benchmark", "severities", default=[1, 2, 3, 4, 5])
    severities = [int(v) for v in severities]
    if len(severities) == 0:
        raise ValueError("benchmark.severities must not be empty.")
    return severities


def _get_corruption_subcfg(robustness_cfg: Any, corruption_name: str) -> Any:
    normalized = _normalize_corruption_name(corruption_name)
    corruptions = _cfg_get(robustness_cfg, "corruptions", default={})
    if not isinstance(corruptions, Mapping):
        raise ValueError("robustness.corruptions must be a mapping.")

    # 원래 이름/정규화 이름 모두 탐색
    if normalized in corruptions:
        return corruptions[normalized]

    for key, value in corruptions.items():
        if _normalize_corruption_name(key) == normalized:
            return value

    supported = ", ".join(sorted(_normalize_corruption_name(k) for k in corruptions.keys()))
    raise ValueError(
        f"Unknown corruption '{corruption_name}'. Supported: {supported}"
    )


def _pick_severity_value(values: Sequence[Any], severity: int, name: str) -> Any:
    severity = int(severity)
    if severity < 1 or severity > len(values):
        raise ValueError(
            f"severity for '{name}' must be in [1, {len(values)}], got: {severity}"
        )
    return values[severity - 1]


def get_corruption_params(
    robustness_cfg: Any,
    corruption_name: Optional[str],
    severity: int,
) -> Dict[str, Any]:
    """
    corruption 이름과 severity(1-based)를 받아 실제 파라미터 dict 반환.
    """
    normalized = _normalize_corruption_name(corruption_name)
    if normalized == "clean":
        return {"name": "clean"}

    sub_cfg = _get_corruption_subcfg(robustness_cfg, normalized)

    if not bool(_cfg_get(sub_cfg, "enabled", default=True)):
        raise ValueError(f"Corruption '{normalized}' is disabled in config.")

    if normalized == "jpeg":
        qualities = _cfg_get(sub_cfg, "qualities", default=None)
        if qualities is None:
            raise ValueError("jpeg.qualities must be provided.")
        return {
            "name": "jpeg",
            "quality": int(_pick_severity_value(qualities, severity, normalized)),
        }

    if normalized == "resize_down_up":
        scales = _cfg_get(sub_cfg, "scales", default=None)
        interpolation = _cfg_get(sub_cfg, "interpolation", default="bilinear")
        if scales is None:
            raise ValueError("resize_down_up.scales must be provided.")
        return {
            "name": "resize_down_up",
            "scale": float(_pick_severity_value(scales, severity, normalized)),
            "interpolation": str(interpolation),
        }

    if normalized == "gaussian_blur":
        sigmas = _cfg_get(sub_cfg, "sigmas", default=None)
        if sigmas is None:
            raise ValueError("gaussian_blur.sigmas must be provided.")
        return {
            "name": "gaussian_blur",
            "sigma": float(_pick_severity_value(sigmas, severity, normalized)),
        }

    if normalized == "gaussian_noise":
        stds = _cfg_get(sub_cfg, "stds", default=None)
        if stds is None:
            raise ValueError("gaussian_noise.stds must be provided.")
        return {
            "name": "gaussian_noise",
            "std": float(_pick_severity_value(stds, severity, normalized)),
        }

    if normalized == "brightness_contrast":
        brightness = _cfg_get(sub_cfg, "brightness", default=None)
        contrast = _cfg_get(sub_cfg, "contrast", default=None)
        if brightness is None or contrast is None:
            raise ValueError(
                "brightness_contrast.brightness and brightness_contrast.contrast must be provided."
            )
        if len(brightness) != len(contrast):
            raise ValueError(
                "brightness_contrast.brightness and contrast must have the same length."
            )
        return {
            "name": "brightness_contrast",
            "brightness": float(_pick_severity_value(brightness, severity, normalized)),
            "contrast": float(_pick_severity_value(contrast, severity, normalized)),
        }

    raise ValueError(f"Unsupported corruption: {normalized}")


class CorruptionTransform:
    """
    PIL 이미지에 corruption 1개를 적용하는 callable.
    """

    def __init__(
        self,
        corruption_name: Optional[str],
        severity: int,
        robustness_cfg: Any,
    ) -> None:
        self.corruption_name = _normalize_corruption_name(corruption_name)
        self.severity = int(severity)
        self.robustness_cfg = robustness_cfg
        self.params = get_corruption_params(
            robustness_cfg=robustness_cfg,
            corruption_name=self.corruption_name,
            severity=self.severity,
        )

    def __call__(self, image: Image.Image) -> Image.Image:
        name = self.params["name"]

        if name == "clean":
            return image

        if name == "jpeg":
            return apply_jpeg(image=image, quality=self.params["quality"])

        if name == "resize_down_up":
            return apply_resize_down_up(
                image=image,
                scale=self.params["scale"],
                interpolation=self.params["interpolation"],
            )

        if name == "gaussian_blur":
            return apply_gaussian_blur(image=image, sigma=self.params["sigma"])

        if name == "gaussian_noise":
            return apply_gaussian_noise(image=image, std=self.params["std"])

        if name == "brightness_contrast":
            return apply_brightness_contrast(
                image=image,
                brightness=self.params["brightness"],
                contrast=self.params["contrast"],
            )

        raise ValueError(f"Unsupported corruption in transform call: {name}")

    def extra_repr(self) -> str:
        return f"name={self.corruption_name}, severity={self.severity}, params={self.params}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"


def build_clean_eval_transform(data_cfg: Any) -> T.Compose:
    """
    data config를 받아 clean val/test용 deterministic transform 생성.
    """
    input_size = _cfg_get(data_cfg, "image", "input_size", default=224)
    mean = _cfg_get(data_cfg, "image", "mean", default=IMAGENET_MEAN)
    std = _cfg_get(data_cfg, "image", "std", default=IMAGENET_STD)

    return build_eval_transform(
        input_size=_to_2tuple(input_size),
        mean=_to_float_tuple(mean, IMAGENET_MEAN),
        std=_to_float_tuple(std, IMAGENET_STD),
    )


def build_corrupted_eval_transform(
    data_cfg: Any,
    corruption_name: Optional[str],
    severity: int,
    robustness_cfg: Any,
) -> T.Compose:
    """
    corruption + 기존 eval preprocessing을 순차 적용하는 transform 생성.

    흐름:
    PIL Image
      -> corruption (PIL 유지)
      -> Resize / ToTensor / Normalize
      -> Tensor
    """
    corruption = CorruptionTransform(
        corruption_name=corruption_name,
        severity=severity,
        robustness_cfg=robustness_cfg,
    )
    eval_tf = build_clean_eval_transform(data_cfg)

    return T.Compose(
        [
            corruption,
            eval_tf,
        ]
    )