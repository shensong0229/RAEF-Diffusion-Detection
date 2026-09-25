# -*- coding: utf-8 -*-
"""Official FIRE single-branch model integrated into FakeImageDetect.

Cloud-oriented updates in this version
--------------------------------------
1) create_vae() now prefers a local SD1.5 VAE directory instead of always
   downloading from Hugging Face.
2) The local VAE path is auto-resolved from common project locations and can
   also be overridden by FIRE_VAE_DIR.
3) If no local VAE is found, it falls back to the original Hugging Face repo.
4) low_cpu_mem_usage is explicitly disabled so loading does not depend on
   accelerate being installed.
5) VAE slicing/tiling are enabled when available to reduce GPU memory pressure
   on 1x A30 training.
6) Added extract_features() for the new Spatial+FIRE dual-stream model while
   keeping the original single-branch forward path unchanged.
"""
from __future__ import annotations

import contextlib
import math
import os
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
import torchvision
from diffusers import AutoencoderKL
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import retrieve_latents


# ----------------------------- helpers -----------------------------

def _resolve_norm(norm_layer: Literal["batch", "instance"]):
    if norm_layer == "batch":
        return nn.BatchNorm2d
    if norm_layer == "instance":
        return nn.InstanceNorm2d
    raise AssertionError(f"Unknown norm layer: {norm_layer}")


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _default_cache_dir() -> Optional[str]:
    for key in ["HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE"]:
        v = os.environ.get(key, "").strip()
        if v:
            return v

    hf_home = os.environ.get("HF_HOME", "").strip()
    if hf_home:
        return os.path.join(hf_home, "hub")

    return os.path.join(_project_root(), ".cache", "huggingface", "hub")


def _looks_like_vae_dir(path: str) -> bool:
    if not path:
        return False
    if not os.path.isdir(path):
        return False
    if not os.path.isfile(os.path.join(path, "config.json")):
        return False
    has_weights = (
        os.path.isfile(os.path.join(path, "diffusion_pytorch_model.safetensors"))
        or os.path.isfile(os.path.join(path, "diffusion_pytorch_model.bin"))
    )
    return has_weights


def _candidate_local_vae_dirs() -> list[str]:
    prj = _project_root()
    cwd = os.getcwd()
    candidates = [
        os.environ.get("FIRE_VAE_DIR", "").strip(),
        os.path.join(prj, "pretrained", "sd15_vae"),
        os.path.join(cwd, "pretrained", "sd15_vae"),
        r"F:\FakeImageDetect\pretrained\sd15_vae",
    ]

    seen = set()
    out = []
    for p in candidates:
        if not p:
            continue
        rp = os.path.abspath(p)
        if rp in seen:
            continue
        seen.add(rp)
        out.append(rp)
    return out


# ----------------------------- backbone -----------------------------

def get_frq_resnet_model(
    mode: Literal["rgb", "ours", "frq"],
    norm_layer: Literal["batch", "instance"] = "instance",
    pretrained: bool = True,
) -> nn.Module:
    norm = _resolve_norm(norm_layer)
    if norm == nn.InstanceNorm2d:
        model = torchvision.models.resnet50(num_classes=1000, pretrained=False, norm_layer=norm)
        if pretrained:
            weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
            model.load_state_dict(weights.get_state_dict(progress=True, check_hash=True), strict=False)
    else:
        weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        model = torchvision.models.resnet50(num_classes=1000, weights=weights, norm_layer=norm)

    if mode == "frq":
        model.conv1.weight = nn.Parameter(torch.cat([model.conv1.weight * 0.25] * 2, dim=1))
        model.conv1.in_channels = 6

    model.fc = nn.Linear(2048, 1)
    torch.nn.init.normal_(model.fc.weight.data, 0.0, 0.02)
    if model.fc.bias is not None:
        torch.nn.init.zeros_(model.fc.bias.data)
    return model


class ESPCN(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, channels: int, upscale_factor: int) -> None:
        super().__init__()
        hidden_channels = channels // 2
        out_channels = int(out_channels * (upscale_factor ** 2))
        self.bn = nn.BatchNorm2d(in_channels)
        self.feature_maps = nn.Sequential(
            nn.Conv2d(in_channels, channels, (5, 5), (1, 1), (2, 2)),
            nn.Tanh(),
            nn.Conv2d(channels, hidden_channels, (3, 3), (1, 1), (1, 1)),
            nn.Tanh(),
        )
        self.sub_pixel_0 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )
        self.sub_pixel_1 = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, (3, 3), (1, 1), (1, 1)),
            nn.PixelShuffle(upscale_factor),
            nn.Sigmoid(),
        )
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                if module.in_channels == 32:
                    nn.init.normal_(module.weight.data, 0.0, 0.001)
                    nn.init.zeros_(module.bias.data)
                else:
                    nn.init.normal_(
                        module.weight.data,
                        0.0,
                        math.sqrt(2 / (module.out_channels * module.weight.data[0][0].numel())),
                    )
                    nn.init.zeros_(module.bias.data)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.bn(x)
        x = self.feature_maps(x)
        return self.sub_pixel_0(x), self.sub_pixel_1(x)


class FFTFilter(nn.Module):
    def __init__(self, radiuslow: int = 40, radiushigh: int = 120, rows: int = 256, cols: int = 256):
        super().__init__()
        self.radiuslow = int(radiuslow)
        self.radiushigh = int(radiushigh)
        self.rows = int(rows)
        self.cols = int(cols)
        i_mask, r_i_mask = self.init_mask()
        self.register_buffer("i_mask", i_mask, persistent=True)
        self.register_buffer("r_i_mask", r_i_mask, persistent=True)
        self.mask_autoencoder = ESPCN(in_channels=3, out_channels=1, channels=64, upscale_factor=1)

    def init_mask(self):
        mask = torch.ones((1, self.rows, self.cols), dtype=torch.float32, requires_grad=False)
        crow, ccol = self.rows // 2, self.cols // 2
        x, y = torch.meshgrid(torch.arange(self.rows), torch.arange(self.cols), indexing="ij")
        area = (x - crow) ** 2 + (y - ccol) ** 2 < self.radiuslow * self.radiuslow
        mask[:, area] = 0
        area = (x - crow) ** 2 + (y - ccol) ** 2 >= self.radiushigh * self.radiushigh
        mask[:, area] = 0
        return mask, 1 - mask

    def middle_pass_filter(self, image: Tensor):
        freq_image = torch.fft.fftn(image * 255, dim=(-2, -1))
        freq_image = torch.fft.fftshift(freq_image, dim=(-2, -1))
        mask_input = (20 * torch.log(torch.abs(freq_image) + 1e-7)) / 255
        mask_mid_frq_real, mask_mid_filterd_real = self.mask_autoencoder(mask_input.float())

        middle_freq = torch.fft.ifftshift(freq_image * mask_mid_frq_real.to(freq_image.dtype), dim=(-2, -1))
        masked_image_array = torch.fft.ifftn(middle_freq, dim=(-2, -1))
        z = torch.abs(masked_image_array)
        middle_freq_image = z / (torch.max(z) - torch.min(z) + 1e-8)

        middle_filtered = torch.fft.ifftshift(freq_image * mask_mid_filterd_real.to(freq_image.dtype), dim=(-2, -1))
        middle_filtered_array = torch.fft.ifftn(middle_filtered, dim=(-2, -1))
        z = torch.abs(middle_filtered_array)
        middle_filtered_image = z / (torch.max(z) - torch.min(z) + 1e-8)
        return middle_freq_image, middle_filtered_image, mask_mid_frq_real.float(), mask_mid_filterd_real.float()

    def forward(self, image: Tensor):
        return self.middle_pass_filter(image)


# ----------------------------- VAE loading -----------------------------

def create_vae() -> nn.Module:
    cache_dir = _default_cache_dir()
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    for local_dir in _candidate_local_vae_dirs():
        if _looks_like_vae_dir(local_dir):
            print(f"[INFO] loading local FIRE VAE from: {local_dir}")
            return AutoencoderKL.from_pretrained(
                local_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=False,
                local_files_only=True,
            )

    repo_id = os.environ.get("FIRE_VAE_REPO", "runwayml/stable-diffusion-v1-5").strip() or "runwayml/stable-diffusion-v1-5"
    print(f"[INFO] local FIRE VAE not found, fallback to HF repo: {repo_id}")
    return AutoencoderKL.from_pretrained(
        repo_id,
        subfolder="vae",
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
        low_cpu_mem_usage=False,
    )


class FIREOfficialModel(nn.Module):
    def __init__(
        self,
        mode: str = "frq",
        norm_layer: str = "instance",
        pretrained: bool = True,
        radiuslow: int = 40,
        radiushigh: int = 120,
        rows: int = 256,
        cols: int = 256,
    ):
        super().__init__()
        self.vae = create_vae()
        if hasattr(self.vae, "enable_slicing"):
            self.vae.enable_slicing()
        if hasattr(self.vae, "enable_tiling"):
            self.vae.enable_tiling()

        self.decode_dtype = next(iter(self.vae.post_quant_conv.parameters())).dtype
        for param in self.vae.parameters():
            param.requires_grad = False

        self.resnet = get_frq_resnet_model(mode=mode, norm_layer=norm_layer, pretrained=pretrained)
        self.fft_filter_module = FFTFilter(radiuslow=radiuslow, radiushigh=radiushigh, rows=rows, cols=cols)

    def _reconstruct(self, x: Tensor) -> Tensor:
        if torch.cuda.is_available():
            autocast_ctx = torch.amp.autocast(device_type="cuda", enabled=False)
        else:
            autocast_ctx = contextlib.nullcontext()

        with autocast_ctx:
            xin = x.to(device=x.device, dtype=self.decode_dtype)
            latents = retrieve_latents(self.vae.encode(xin))
            rec = self.vae.decode(latents.to(self.decode_dtype), return_dict=False)[0]
        return rec.float()

    def _forward_resnet_backbone(self, x6: Tensor):
        m = self.resnet
        x = m.conv1(x6)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        feat_map = x
        pooled = m.avgpool(x)
        pooled = torch.flatten(pooled, 1)
        logits = m.fc(pooled).view(-1)
        return feat_map, pooled, logits

    def extract_features(self, x: Tensor, return_aux: bool = False):
        middle_freq_image, middle_filtered_image, mask_mid_frq, mask_mid_filterd = self.fft_filter_module(x)
        reconstructions_x = self._reconstruct(x)
        reconstructions_middle_filtered = self._reconstruct(middle_filtered_image)
        raw_reconstructions_delta = torch.abs(reconstructions_x - x)
        filtered_reconstructions_delta = torch.abs(reconstructions_middle_filtered - x)
        anomaly_prior = torch.mean(torch.abs(raw_reconstructions_delta - filtered_reconstructions_delta), dim=1, keepdim=True)
        feature_map, global_feature, logits = self._forward_resnet_backbone(
            torch.cat([raw_reconstructions_delta, filtered_reconstructions_delta], dim=1)
        )
        if not return_aux:
            return {
                "logits": logits,
                "feature_map": feature_map,
                "global_feature": global_feature,
                "anomaly_prior": anomaly_prior,
            }
        return {
            "logits": logits,
            "feature_map": feature_map,
            "global_feature": global_feature,
            "anomaly_prior": anomaly_prior,
            "middle_freq_image": middle_freq_image,
            "middle_filtered_image": middle_filtered_image,
            "raw_reconstructions_delta": raw_reconstructions_delta,
            "filtered_reconstructions_delta": filtered_reconstructions_delta,
            "mask_mid_frq": mask_mid_frq,
            "mask_mid_filterd": mask_mid_filterd,
            "ideal_mid_mask": self.fft_filter_module.i_mask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).detach(),
            "ideal_comp_mask": self.fft_filter_module.r_i_mask.unsqueeze(0).repeat(x.shape[0], 1, 1, 1).detach(),
        }

    def forward(self, x: Tensor, return_aux: bool = False):
        aux = self.extract_features(x, return_aux=True)
        logits = aux["logits"]
        if not return_aux:
            return logits
        return aux
