# -*- coding: utf-8 -*-
"""Dataset transforms.

Keep legacy support:
- S-only: ImageNet-normalized RGB
- F-only: FFT log-magnitude (1-channel)
- SF dual-stream: raw RGB in [0,1], FFT done inside model

New:
- RE-only: precomputed 3-channel residual map
- SR dual-stream: paired transforms for (RGB, residual) with shared geometry

Extra:
- ConsistencyAugment (legacy, non-paired models only)
- FixedRobustnessPerturb for deterministic robustness evaluation
"""

from __future__ import annotations

import io
import random
from typing import Optional, Callable

import torch
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _maybe_strong_aug():
    if hasattr(transforms, "RandAugment"):
        return transforms.RandAugment(num_ops=2, magnitude=9)
    if hasattr(transforms, "AutoAugment") and hasattr(transforms, "AutoAugmentPolicy"):
        return transforms.AutoAugment(policy=transforms.AutoAugmentPolicy.IMAGENET)
    return transforms.Identity()


def is_freq_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("freq_")


def is_dual_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("sf_")


def is_re_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("re_")


def is_sr_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("sr_")


def is_residual_model(model_name: str) -> bool:
    return is_re_model(model_name) or is_sr_model(model_name)


def is_fire_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("fire_")


def is_sfire_model(model_name: str) -> bool:
    n = (model_name or "").lower().strip()
    return n.startswith("sfire_") or n.startswith("spatial_fire_dualstream_")


class FFTLogMagTransform:
    """Tensor(3,H,W) in [0,1] -> Tensor(1,H,W): FFT-shifted log magnitude, standardized."""
    def __init__(self, eps: float = 1e-6, clamp: float = 5.0):
        self.eps = float(eps)
        self.clamp = float(clamp)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(x):
            raise TypeError("FFTLogMagTransform expects a torch.Tensor after ToTensor()")
        if x.dim() != 3 or x.size(0) != 3:
            raise ValueError(f"Expected (3,H,W) tensor, got {tuple(x.shape)}")

        r, g, b = x[0:1], x[1:2], x[2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

        fft = torch.fft.fft2(gray, dim=(-2, -1))
        fft = torch.fft.fftshift(fft, dim=(-2, -1))
        mag = torch.abs(fft)
        feat = torch.log1p(mag)

        mean = feat.mean(dim=(-2, -1), keepdim=True)
        std = feat.std(dim=(-2, -1), keepdim=True)
        feat = (feat - mean) / (std + self.eps)

        if self.clamp > 0:
            feat = feat.clamp(-self.clamp, self.clamp)
        return feat


class FixedRobustnessPerturb:
    """Deterministic PIL-level perturbation for robustness evaluation.

    Supported perturb_type:
      - none
      - jpeg   : level is JPEG quality, e.g. 95/75/50/30
      - resize : level is round-trip scale, e.g. 0.75/0.50
      - blur   : level is Gaussian blur radius, e.g. 1/2/3
    """
    def __init__(self, perturb_type: str = "none", perturb_level: str = ""):
        self.perturb_type = str(perturb_type or "none").lower().strip()
        self.perturb_level = str(perturb_level or "").strip()
        if self.perturb_type in {"", "none", "clean"}:
            self.perturb_type = "none"
        if self.perturb_type not in {"none", "jpeg", "resize", "blur"}:
            raise ValueError(
                f"Unsupported perturb_type={perturb_type!r}; choose none/jpeg/resize/blur"
            )

    def _parse_quality(self) -> int:
        try:
            q = int(float(self.perturb_level))
        except Exception as e:
            raise ValueError(f"JPEG perturb_level should be an integer quality, got {self.perturb_level!r}") from e
        return max(1, min(100, q))

    def _parse_scale(self) -> float:
        try:
            scale = float(self.perturb_level)
        except Exception as e:
            raise ValueError(f"resize perturb_level should be a float scale, got {self.perturb_level!r}") from e
        if not (0.05 <= scale <= 1.0):
            raise ValueError(f"resize scale should be in [0.05, 1.0], got {scale}")
        return scale

    def _parse_radius(self) -> float:
        try:
            radius = float(self.perturb_level)
        except Exception as e:
            raise ValueError(f"blur perturb_level should be a float radius, got {self.perturb_level!r}") from e
        if radius < 0:
            raise ValueError(f"blur radius should be >= 0, got {radius}")
        return radius

    def _do_jpeg(self, img: Image.Image) -> Image.Image:
        q = self._parse_quality()
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert("RGB")

    def _do_resize_roundtrip(self, img: Image.Image) -> Image.Image:
        scale = self._parse_scale()
        w, h = img.size
        if w <= 1 or h <= 1:
            return img
        tw = max(2, int(round(w * scale)))
        th = max(2, int(round(h * scale)))
        tmp = img.resize((tw, th), resample=Image.BICUBIC)
        return tmp.resize((w, h), resample=Image.BICUBIC)

    def _do_blur(self, img: Image.Image) -> Image.Image:
        radius = self._parse_radius()
        if radius <= 0:
            return img
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.perturb_type == "none":
            return img
        if not isinstance(img, Image.Image):
            return img
        if self.perturb_type == "jpeg":
            return self._do_jpeg(img)
        if self.perturb_type == "resize":
            return self._do_resize_roundtrip(img)
        if self.perturb_type == "blur":
            return self._do_blur(img)
        return img


def build_spatial_transforms(image_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                _maybe_strong_aug(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )


def build_dual_raw_transforms(image_size: int = 224, is_train: bool = True):
    """Legacy SF models expect raw RGB tensor in [0,1] (no normalize)."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                _maybe_strong_aug(),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


def build_freq_transforms(image_size: int = 224, is_train: bool = True):
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                FFTLogMagTransform(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            FFTLogMagTransform(),
        ]
    )



def build_fire_transforms(image_size: int = 256, is_train: bool = True):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])


def build_residual_transforms(image_size: int = 224, is_train: bool = True):
    """Residual-only branch. Residual maps are already precomputed 3-channel images."""
    if is_train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.7, 1.0), ratio=(0.9, 1.1)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )

    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ]
    )


class PairedRGBResidualTransform:
    """Apply the same geometry to RGB and residual maps.

    RGB branch:
      - keeps raw [0,1] tensor (normalization is done inside SR model)
      - optional color jitter only on RGB

    Residual branch:
      - 3-channel residual tensor in [0,1]
      - no color jitter
    """
    def __init__(self, image_size: int = 224, is_train: bool = True,
                 scale=(0.7, 1.0), ratio=(0.9, 1.1),
                 hflip_p: float = 0.5, jitter_p: float = 0.8):
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.scale = scale
        self.ratio = ratio
        self.hflip_p = float(hflip_p)
        self.jitter_p = float(jitter_p)
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def __call__(self, img_rgb: Image.Image, img_res: Image.Image):
        if self.is_train:
            i, j, h, w = transforms.RandomResizedCrop.get_params(img_rgb, scale=self.scale, ratio=self.ratio)
            img_rgb = TF.resized_crop(img_rgb, i, j, h, w, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)
            img_res = TF.resized_crop(img_res, i, j, h, w, (self.image_size, self.image_size), interpolation=TF.InterpolationMode.BILINEAR)

            if random.random() < self.hflip_p:
                img_rgb = TF.hflip(img_rgb)
                img_res = TF.hflip(img_res)

            if random.random() < self.jitter_p:
                img_rgb = self.jitter(img_rgb)
        else:
            resize_to = int(self.image_size * 1.14)
            img_rgb = TF.resize(img_rgb, resize_to, interpolation=TF.InterpolationMode.BILINEAR)
            img_res = TF.resize(img_res, resize_to, interpolation=TF.InterpolationMode.BILINEAR)
            img_rgb = TF.center_crop(img_rgb, [self.image_size, self.image_size])
            img_res = TF.center_crop(img_res, [self.image_size, self.image_size])

        img_rgb = TF.to_tensor(img_rgb)
        img_res = TF.to_tensor(img_res)
        return img_rgb, img_res


def build_paired_transforms_for_model(model_name: str, image_size: int = 224, is_train: bool = True):
    n = (model_name or "").lower().strip()
    if n.startswith("sr_") or n.startswith("re_"):
        return PairedRGBResidualTransform(image_size=image_size, is_train=is_train)
    raise ValueError(f"build_paired_transforms_for_model only supports re_/sr_ models, got {model_name}")


def build_transforms_for_model(model_name: str, image_size: int = 224, is_train: bool = True):
    if is_fire_model(model_name):
        return build_fire_transforms(image_size=image_size, is_train=is_train)
    if is_sfire_model(model_name):
        return build_dual_raw_transforms(image_size=image_size, is_train=is_train)
    if is_freq_model(model_name):
        return build_freq_transforms(image_size=image_size, is_train=is_train)
    if is_dual_model(model_name):
        return build_dual_raw_transforms(image_size=image_size, is_train=is_train)
    if is_re_model(model_name):
        return build_residual_transforms(image_size=image_size, is_train=is_train)
    return build_spatial_transforms(image_size=image_size, is_train=is_train)


# -----------------------------
# Consistency regularization support (legacy, non-paired only)
# -----------------------------
class ConsistencyAugment:
    """PIL-level perturbations for consistency regularization."""
    def __init__(
        self,
        p: float = 1.0,
        jpeg_p: float = 0.5,
        jpeg_qmin: int = 30,
        jpeg_qmax: int = 95,
        resize_p: float = 0.5,
        resize_scale_min: float = 0.5,
        resize_scale_max: float = 1.0,
        blur_p: float = 0.2,
        blur_radius_min: float = 0.5,
        blur_radius_max: float = 1.5,
    ):
        self.p = float(p)
        self.jpeg_p = float(jpeg_p)
        self.jpeg_qmin = int(jpeg_qmin)
        self.jpeg_qmax = int(jpeg_qmax)

        self.resize_p = float(resize_p)
        self.resize_scale_min = float(resize_scale_min)
        self.resize_scale_max = float(resize_scale_max)

        self.blur_p = float(blur_p)
        self.blur_radius_min = float(blur_radius_min)
        self.blur_radius_max = float(blur_radius_max)

    def _do_jpeg(self, img: Image.Image) -> Image.Image:
        qmin = max(1, min(self.jpeg_qmin, self.jpeg_qmax))
        qmax = max(qmin, max(self.jpeg_qmin, self.jpeg_qmax))
        q = random.randint(qmin, qmax)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=q, optimize=True)
        buf.seek(0)
        out = Image.open(buf)
        out.load()
        return out.convert("RGB")

    def _do_resize_roundtrip(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        if w <= 1 or h <= 1:
            return img
        smin = max(0.05, min(self.resize_scale_min, self.resize_scale_max))
        smax = max(smin, max(self.resize_scale_min, self.resize_scale_max))
        scale = random.uniform(smin, smax)
        tw = max(2, int(round(w * scale)))
        th = max(2, int(round(h * scale)))
        tmp = img.resize((tw, th), resample=Image.BICUBIC)
        return tmp.resize((w, h), resample=Image.BICUBIC)

    def _do_blur(self, img: Image.Image) -> Image.Image:
        rmin = min(self.blur_radius_min, self.blur_radius_max)
        rmax = max(self.blur_radius_min, self.blur_radius_max)
        radius = random.uniform(rmin, rmax)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    def __call__(self, img: Image.Image) -> Image.Image:
        if not isinstance(img, Image.Image):
            return img

        if random.random() > self.p:
            return img

        out = img
        if self.resize_p > 0 and random.random() < self.resize_p:
            out = self._do_resize_roundtrip(out)
        if self.jpeg_p > 0 and random.random() < self.jpeg_p:
            out = self._do_jpeg(out)
        if self.blur_p > 0 and random.random() < self.blur_p:
            out = self._do_blur(out)
        return out


class _TransformWithPre:
    def __init__(self, base: Callable, pre: Optional[Callable] = None):
        self.base = base
        self.pre = pre

    def __call__(self, img):
        if self.pre is not None:
            img = self.pre(img)
        return self.base(img)


def wrap_transform_with_pre(base_transform: Callable, pre_pil_transform: Optional[Callable]):
    if pre_pil_transform is None:
        return base_transform
    return _TransformWithPre(base_transform, pre_pil_transform)


def build_consistency_augment(
    p: float = 1.0,
    jpeg_p: float = 0.5,
    jpeg_qmin: int = 30,
    jpeg_qmax: int = 95,
    resize_p: float = 0.5,
    resize_scale_min: float = 0.5,
    resize_scale_max: float = 1.0,
    blur_p: float = 0.2,
    blur_radius_min: float = 0.5,
    blur_radius_max: float = 1.5,
):
    return ConsistencyAugment(
        p=p,
        jpeg_p=jpeg_p,
        jpeg_qmin=jpeg_qmin,
        jpeg_qmax=jpeg_qmax,
        resize_p=resize_p,
        resize_scale_min=resize_scale_min,
        resize_scale_max=resize_scale_max,
        blur_p=blur_p,
        blur_radius_min=blur_radius_min,
        blur_radius_max=blur_radius_max,
    )


def build_fixed_robustness_perturb(perturb_type: str = "none", perturb_level: str = ""):
    perturb_type = str(perturb_type or "none").lower().strip()
    if perturb_type in {"", "none", "clean"}:
        return None
    return FixedRobustnessPerturb(perturb_type=perturb_type, perturb_level=perturb_level)
