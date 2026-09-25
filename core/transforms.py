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

Current project policy:
- For FakeImageDetect self-training, unify all model branches to a fixed input size.
- Use Resize(shorter_side=image_size) + CenterCrop(image_size) for train/val/test.
- Keep train-time color/flip augmentation, but remove RandomResizedCrop to avoid
  geometry drift across branches and experiments.
"""

from __future__ import annotations

import random

import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_INTERP = TF.InterpolationMode.BILINEAR


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


def _resize_and_crop_ops(image_size: int):
    return [
        transforms.Resize(image_size, interpolation=_INTERP),
        transforms.CenterCrop(image_size),
    ]


def build_spatial_transforms(image_size: int = 256, is_train: bool = True):
    ops = _resize_and_crop_ops(image_size)
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                _maybe_strong_aug(),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )
        return transforms.Compose(ops)

    ops.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
        ]
    )
    return transforms.Compose(ops)


def build_dual_raw_transforms(image_size: int = 256, is_train: bool = True):
    """Legacy SF models expect raw RGB tensor in [0,1] (no normalize)."""
    ops = _resize_and_crop_ops(image_size)
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
                _maybe_strong_aug(),
                transforms.ToTensor(),
            ]
        )
        return transforms.Compose(ops)

    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


def build_freq_transforms(image_size: int = 256, is_train: bool = True):
    ops = _resize_and_crop_ops(image_size)
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                FFTLogMagTransform(),
            ]
        )
        return transforms.Compose(ops)

    ops.extend(
        [
            transforms.ToTensor(),
            FFTLogMagTransform(),
        ]
    )
    return transforms.Compose(ops)


def build_residual_transforms(image_size: int = 256, is_train: bool = True):
    """Residual-only branch. Residual maps are already precomputed 3-channel images."""
    ops = _resize_and_crop_ops(image_size)
    if is_train:
        ops.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
            ]
        )
        return transforms.Compose(ops)

    ops.append(transforms.ToTensor())
    return transforms.Compose(ops)


class PairedRGBResidualTransform:
    """Apply the same geometry to RGB and residual maps.

    RGB branch:
      - keeps raw [0,1] tensor (normalization is done inside SR model)
      - optional color jitter only on RGB

    Residual branch:
      - 3-channel residual tensor in [0,1]
      - no color jitter

    Geometry policy:
      - Resize(shorter_side=image_size) + CenterCrop(image_size) for both train/val.
      - Train-time augmentation keeps size fixed via paired hflip only.
    """
    def __init__(self, image_size: int = 256, is_train: bool = True,
                 hflip_p: float = 0.5, jitter_p: float = 0.8):
        self.image_size = int(image_size)
        self.is_train = bool(is_train)
        self.hflip_p = float(hflip_p)
        self.jitter_p = float(jitter_p)
        self.jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)

    def _resize_and_crop(self, img: Image.Image) -> Image.Image:
        img = TF.resize(img, self.image_size, interpolation=_INTERP)
        img = TF.center_crop(img, [self.image_size, self.image_size])
        return img

    def __call__(self, img_rgb: Image.Image, img_res: Image.Image):
        img_rgb = self._resize_and_crop(img_rgb)
        img_res = self._resize_and_crop(img_res)

        if self.is_train:
            if random.random() < self.hflip_p:
                img_rgb = TF.hflip(img_rgb)
                img_res = TF.hflip(img_res)

            if random.random() < self.jitter_p:
                img_rgb = self.jitter(img_rgb)

        img_rgb = TF.to_tensor(img_rgb)
        img_res = TF.to_tensor(img_res)
        return img_rgb, img_res


def build_paired_transforms_for_model(model_name: str, image_size: int = 256, is_train: bool = True):
    n = (model_name or "").lower().strip()
    if n.startswith("sr_") or n.startswith("re_"):
        return PairedRGBResidualTransform(image_size=image_size, is_train=is_train)
    raise ValueError(f"build_paired_transforms_for_model only supports re_/sr_ models, got {model_name}")


def build_transforms_for_model(model_name: str, image_size: int = 256, is_train: bool = True):
    if is_freq_model(model_name):
        return build_freq_transforms(image_size=image_size, is_train=is_train)
    if is_dual_model(model_name):
        return build_dual_raw_transforms(image_size=image_size, is_train=is_train)
    if is_re_model(model_name):
        return build_residual_transforms(image_size=image_size, is_train=is_train)
    return build_spatial_transforms(image_size=image_size, is_train=is_train)
