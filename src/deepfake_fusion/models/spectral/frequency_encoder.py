from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn


def _cfg_get(cfg: Any, *keys: str, default: Any = None) -> Any:
    """dict / attribute 접근을 모두 지원하는 config getter."""
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


def fftshift2d(x: torch.Tensor) -> torch.Tensor:
    """2D frequency map의 중심을 가운데로 이동."""
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with ndim >= 2, got {x.ndim}")
    return torch.fft.fftshift(x, dim=(-2, -1))


def ifftshift2d(x: torch.Tensor) -> torch.Tensor:
    """fftshift의 역연산."""
    if x.ndim < 2:
        raise ValueError(f"Expected tensor with ndim >= 2, got {x.ndim}")
    return torch.fft.ifftshift(x, dim=(-2, -1))


def fft2_image(x: torch.Tensor, norm: str = "ortho", shift: bool = True) -> torch.Tensor:
    """
    이미지 배치에 대해 2D FFT 수행.

    Args:
        x: [B, C, H, W]
        norm: torch.fft.fft2 의 norm 옵션
        shift: True면 zero-frequency를 중앙으로 이동

    Returns:
        complex tensor with shape [B, C, H, W]
    """
    if x.ndim != 4:
        raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")

    spectrum = torch.fft.fft2(x, dim=(-2, -1), norm=norm)
    if shift:
        spectrum = fftshift2d(spectrum)
    return spectrum


def ifft2_image(x: torch.Tensor, norm: str = "ortho", shift: bool = True) -> torch.Tensor:
    """
    2D inverse FFT 수행 후 실수부 반환.

    Args:
        x: complex tensor [B, C, H, W]
        norm: torch.fft.ifft2 의 norm 옵션
        shift: True면 inverse 전에 ifftshift 적용

    Returns:
        real tensor [B, C, H, W]
    """
    if x.ndim != 4:
        raise ValueError(
            f"Expected frequency tensor shape [B, C, H, W], got {tuple(x.shape)}"
        )

    spectrum = ifftshift2d(x) if shift else x
    recon = torch.fft.ifft2(spectrum, dim=(-2, -1), norm=norm)
    return recon.real


def magnitude_spectrum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """복소 주파수 맵 magnitude."""
    if not torch.is_complex(x):
        raise TypeError("magnitude_spectrum expects a complex tensor.")
    return x.abs().clamp_min(eps)


def log_magnitude_spectrum(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """복소 주파수 맵 log-magnitude."""
    return torch.log(magnitude_spectrum(x, eps=eps))


def build_radial_mask(
    height: int,
    width: int,
    radius_ratio: float,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    중심 기준 원형 low-frequency mask 생성.

    Args:
        height, width: spatial size
        radius_ratio: 0~1 사이 비율. min(H, W)/2 기준 반지름 비율

    Returns:
        mask: [1, 1, H, W], values in {0, 1}
    """
    if height <= 0 or width <= 0:
        raise ValueError(f"Invalid mask size: height={height}, width={width}")
    if not 0.0 < radius_ratio <= 1.0:
        raise ValueError(f"radius_ratio must be in (0, 1], got: {radius_ratio}")

    y = torch.arange(height, device=device, dtype=dtype)
    x = torch.arange(width, device=device, dtype=dtype)

    center_y = (height - 1) / 2.0
    center_x = (width - 1) / 2.0

    try:
        yy, xx = torch.meshgrid(y, x, indexing="ij")
    except TypeError:  # older PyTorch fallback
        yy, xx = torch.meshgrid(y, x)

    dist = torch.sqrt((yy - center_y) ** 2 + (xx - center_x) ** 2)

    max_radius = min(height, width) / 2.0
    radius = radius_ratio * max_radius
    mask = (dist <= radius).to(dtype=dtype)
    return mask.unsqueeze(0).unsqueeze(0)


class FrequencyEncoder(nn.Module):
    """
    FFT 기반 low/high frequency decomposition encoder.

    기본 동작:
      x -> FFT -> radial low-pass mask ->
      x_low (inverse FFT), x_high (inverse FFT 또는 residual)

    Notes:
      - forward(x)는 (x_low, x_high)를 반환
      - x_low, x_high shape은 입력 x와 동일
      - magnitude / log-magnitude / complex spectrum 유틸도 제공
    """

    def __init__(
        self,
        mask_mode: str = "radial",
        radius_ratio: float = 0.25,
        fft_norm: str = "ortho",
        high_from_residual: bool = True,
        clamp_output: bool = False,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()

        mask_mode = str(mask_mode).strip().lower()
        if mask_mode != "radial":
            raise ValueError(
                f"Unsupported mask_mode: {mask_mode}. Currently supported: ['radial']"
            )
        if not 0.0 < float(radius_ratio) <= 1.0:
            raise ValueError(f"radius_ratio must be in (0, 1], got: {radius_ratio}")
        if float(eps) <= 0.0:
            raise ValueError(f"eps must be > 0, got: {eps}")

        self.mask_mode = mask_mode
        self.radius_ratio = float(radius_ratio)
        self.fft_norm = str(fft_norm)
        self.high_from_residual = bool(high_from_residual)
        self.clamp_output = bool(clamp_output)
        self.eps = float(eps)

    def _build_low_mask(
        self,
        height: int,
        width: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mask_mode == "radial":
            return build_radial_mask(
                height=height,
                width=width,
                radius_ratio=self.radius_ratio,
                device=device,
                dtype=dtype,
            )
        raise RuntimeError("Unreachable mask_mode branch.")

    def get_spectrum(self, x: torch.Tensor, shift: bool = True) -> torch.Tensor:
        """입력 이미지의 complex spectrum 반환."""
        return fft2_image(x, norm=self.fft_norm, shift=shift)

    def get_magnitude(
        self,
        x: torch.Tensor,
        *,
        shift: bool = True,
        log_scale: bool = False,
    ) -> torch.Tensor:
        """입력 이미지의 magnitude 또는 log-magnitude spectrum 반환."""
        spectrum = self.get_spectrum(x, shift=shift)
        if log_scale:
            return log_magnitude_spectrum(spectrum, eps=self.eps)
        return magnitude_spectrum(spectrum, eps=self.eps)

    def split_spectrum(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        입력 이미지를 low/high spectrum으로 분해.

        Returns:
            {
                "spectrum": centered complex spectrum,
                "low_spectrum": centered complex low spectrum,
                "high_spectrum": centered complex high spectrum,
                "low_mask": [1, 1, H, W],
                "high_mask": [1, 1, H, W],
            }
        """
        if x.ndim != 4:
            raise ValueError(f"Expected input shape [B, C, H, W], got {tuple(x.shape)}")

        _, _, height, width = x.shape
        spectrum = self.get_spectrum(x, shift=True)

        mask_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
        low_mask = self._build_low_mask(
            height,
            width,
            device=x.device,
            dtype=mask_dtype,
        )
        high_mask = 1.0 - low_mask

        low_spectrum = spectrum * low_mask
        high_spectrum = spectrum * high_mask

        return {
            "spectrum": spectrum,
            "low_spectrum": low_spectrum,
            "high_spectrum": high_spectrum,
            "low_mask": low_mask,
            "high_mask": high_mask,
        }

    def _decompose_from_split(
        self,
        x: torch.Tensor,
        split: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x_low = ifft2_image(split["low_spectrum"], norm=self.fft_norm, shift=True)

        if self.high_from_residual:
            x_high = x - x_low
        else:
            x_high = ifft2_image(
                split["high_spectrum"],
                norm=self.fft_norm,
                shift=True,
            )

        if self.clamp_output:
            x_min = x.amin(dim=(-3, -2, -1), keepdim=True)
            x_max = x.amax(dim=(-3, -2, -1), keepdim=True)
            x_low = x_low.clamp(min=x_min, max=x_max)
            x_high = x_high.clamp(min=x_min, max=x_max)

        return x_low, x_high

    def decompose(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        입력 이미지를 spatial-domain low/high branch로 분해.

        Returns:
            x_low: [B, C, H, W]
            x_high: [B, C, H, W]
        """
        split = self.split_spectrum(x)
        return self._decompose_from_split(x, split)

    def forward(
        self,
        x: torch.Tensor,
        return_dict: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor]:
        """
        기본 반환은 (x_low, x_high).
        return_dict=True면 decomposition + spectrum 정보를 함께 반환.
        """
        split = self.split_spectrum(x)
        x_low, x_high = self._decompose_from_split(x, split)

        if not return_dict:
            return x_low, x_high

        split["x_low"] = x_low
        split["x_high"] = x_high
        split["magnitude"] = magnitude_spectrum(split["spectrum"], eps=self.eps)
        split["log_magnitude"] = log_magnitude_spectrum(
            split["spectrum"],
            eps=self.eps,
        )
        return split


def build_frequency_encoder(cfg: Any) -> FrequencyEncoder:
    """
    config 기반 FrequencyEncoder 생성 함수.

    기대하는 config 예시:
      cfg.frequency.mask_mode
      cfg.frequency.radius_ratio
      cfg.frequency.fft_norm
      cfg.frequency.high_from_residual
      cfg.frequency.clamp_output
      cfg.frequency.eps

    model_cfg 전체 또는 model_cfg.frequency 둘 다 입력 가능하도록 처리.
    """
    freq_cfg = _cfg_get(cfg, "frequency", default=cfg)

    return FrequencyEncoder(
        mask_mode=str(_cfg_get(freq_cfg, "mask_mode", default="radial")),
        radius_ratio=float(_cfg_get(freq_cfg, "radius_ratio", default=0.25)),
        fft_norm=str(_cfg_get(freq_cfg, "fft_norm", default="ortho")),
        high_from_residual=bool(
            _cfg_get(freq_cfg, "high_from_residual", default=True)
        ),
        clamp_output=bool(_cfg_get(freq_cfg, "clamp_output", default=False)),
        eps=float(_cfg_get(freq_cfg, "eps", default=1e-8)),
    )