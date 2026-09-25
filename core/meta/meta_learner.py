# E:\FakeImageDetect\core\meta\meta_learner.py
from typing import Dict, Tuple, Literal
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

# ===== AMP 兼容层：优先用 torch.amp，旧版本回落到 torch.cuda.amp =====
def create_grad_scaler(use_amp: bool):
    try:
        from torch import amp
        return amp.GradScaler('cuda', enabled=use_amp)
    except Exception:
        from torch.cuda.amp import GradScaler as CudaGradScaler
        return CudaGradScaler(enabled=use_amp)

def autocast_ctx(use_amp: bool):
    try:
        from torch import amp
        return amp.autocast('cuda', enabled=use_amp)
    except Exception:
        from torch.cuda.amp import autocast as cuda_autocast
        return cuda_autocast(enabled=use_amp)
# =====================================================================

AlgoName = Literal["reptile", "fomaml"]

def gpu_fft_mag_log1p_norm(x: torch.Tensor) -> torch.Tensor:
    """
    x: [B,1,H,W] float on CUDA
    return: [B,1,H,W] z-score normalized log1p(|fft|) with fftshift
    """
    X = torch.fft.fft2(x)
    mag = torch.abs(X)
    y = torch.log1p(mag)
    y = torch.fft.fftshift(y, dim=(-2, -1))
    mu = y.mean(dim=(-2, -1), keepdim=True)
    std = y.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    y = (y - mu) / std
    return y

class MetaLearner:
    def __init__(self,
                 model: nn.Module,
                 algo: AlgoName = "reptile",
                 inner_steps: int = 5,
                 inner_lr: float = 1e-3,
                 meta_lr: float = 1e-3,
                 weight_decay: float = 0.0,
                 use_amp: bool = True):
        self.model = model
        self.algo = algo
        self.inner_steps = inner_steps
        self.inner_lr = inner_lr
        self.meta_lr = meta_lr
        self.weight_decay = weight_decay
        self.use_amp = use_amp
        self.scaler = create_grad_scaler(use_amp)

    @torch.no_grad()
    def _reptile_merge(self, theta_before: Dict[str, torch.Tensor], theta_inner: Dict[str, torch.Tensor], meta_lr: float):
        for n, p in self.model.state_dict().items():
            if n in theta_inner and theta_inner[n].dtype.is_floating_point:
                p.copy_(p + meta_lr * (theta_inner[n] - theta_before[n]))

    def _inner_loop_clone(self) -> nn.Module:
        return copy.deepcopy(self.model)

    def _inner_opt(self, fast: nn.Module):
        return torch.optim.SGD(fast.parameters(), lr=self.inner_lr, momentum=0.9, weight_decay=self.weight_decay)

    def _loss(self, logits: torch.Tensor, y: torch.Tensor):
        return F.cross_entropy(logits, y)

    def _forward_logits(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        return net(x)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        # x on CUDA, shape [B,1,H,W]
        return gpu_fft_mag_log1p_norm(x)

    def step_episode(self, x_sup: torch.Tensor, y_sup: torch.Tensor, x_que: torch.Tensor, y_que: torch.Tensor) -> Tuple[float,float]:
        """
        执行一个 episode 的外层更新。
        返回: (support_loss, query_loss)
        """
        device = next(self.model.parameters()).device

        x_sup = x_sup.to(device, non_blocking=True)
        y_sup = y_sup.to(device, non_blocking=True)
        x_que = x_que.to(device, non_blocking=True)
        y_que = y_que.to(device, non_blocking=True)

        x_sup = self._preprocess(x_sup)
        x_que = self._preprocess(x_que)

        if self.algo == "reptile":
            theta0 = {k: v.detach().clone() for k, v in self.model.state_dict().items()}

            fast = self._inner_loop_clone().train()
            opt = self._inner_opt(fast)

            sup_loss_val = 0.0
            for _ in range(self.inner_steps):
                opt.zero_grad(set_to_none=True)
                with autocast_ctx(self.use_amp):
                    logits_sup = self._forward_logits(fast, x_sup)
                    loss_sup = self._loss(logits_sup, y_sup)
                self.scaler.scale(loss_sup).backward()
                self.scaler.step(opt)
                self.scaler.update()
                sup_loss_val = float(loss_sup.detach().item())

            theta_inner = {k: v.detach().clone() for k, v in fast.state_dict().items()}
            self._reptile_merge(theta0, theta_inner, self.meta_lr)

            with torch.no_grad(), autocast_ctx(self.use_amp):
                q_logits = self.model(x_que)
                q_loss = self._loss(q_logits, y_que)
            return sup_loss_val, float(q_loss.item())

        elif self.algo == "fomaml":
            self.model.train()
            opt = torch.optim.SGD(self.model.parameters(), lr=self.inner_lr, momentum=0.9, weight_decay=self.weight_decay)

            opt.zero_grad(set_to_none=True)
            with autocast_ctx(self.use_amp):
                logits_sup = self._forward_logits(self.model, x_sup)
                loss_sup = self._loss(logits_sup, y_sup)
            self.scaler.scale(loss_sup).backward()
            self.scaler.step(opt)
            self.scaler.update()

            opt.zero_grad(set_to_none=True)
            with autocast_ctx(self.use_amp):
                logits_que = self._forward_logits(self.model, x_que)
                loss_que = self._loss(logits_que, y_que)

            # 一阶近似外层更新
            for g in self.model.parameters():
                if g.grad is not None:
                    g.grad = None
            loss_que.backward()
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.add_( - self.meta_lr * p.grad )

            return float(loss_sup.detach().item()), float(loss_que.detach().item())

        else:
            raise ValueError(f"Unknown algo={self.algo}")
