# core/models/sf_gate.py
import torch
import torch.nn as nn
import timm


class SFConcatClassifier(nn.Module):
    """
    Baseline: dual-branch (spatial + frequency) feature concat -> MLP -> classifier.
    - Spatial branch uses ImageNet-normalized RGB.
    - Freq branch uses FFT-magnitude "image" (3-ch) produced by transforms.
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained_spatial: bool = True,
        pretrained_freq: bool = False,
        dropout: float = 0.2,
        mlp_ratio: float = 0.5,
    ):
        super().__init__()
        self.s_net = timm.create_model(backbone, pretrained=pretrained_spatial, num_classes=0, global_pool="avg")
        self.f_net = timm.create_model(backbone, pretrained=pretrained_freq, num_classes=0, global_pool="avg")

        # infer feat dim
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            d = self.s_net(dummy).shape[-1]

        hidden = max(128, int((2 * d) * mlp_ratio))
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(2 * d, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, x_s: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        fs = self.s_net(x_s)  # [B, D]
        ff = self.f_net(x_f)  # [B, D]
        z = torch.cat([fs, ff], dim=1)
        return self.head(z)


class SFGatedClassifier(nn.Module):
    """
    Innovation #1: Adaptive gated fusion.
    alpha = sigmoid(g([fs; ff])) in [0,1]
    f = alpha * fs + (1-alpha) * ff
    logits = cls(f)
    """
    def __init__(
        self,
        backbone: str = "resnet50",
        num_classes: int = 2,
        pretrained_spatial: bool = True,
        pretrained_freq: bool = False,
        dropout: float = 0.2,
        gate_hidden_ratio: float = 0.25,
    ):
        super().__init__()
        self.s_net = timm.create_model(backbone, pretrained=pretrained_spatial, num_classes=0, global_pool="avg")
        self.f_net = timm.create_model(backbone, pretrained=pretrained_freq, num_classes=0, global_pool="avg")

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            d = self.s_net(dummy).shape[-1]

        gate_h = max(64, int(2 * d * gate_hidden_ratio))

        self.gate = nn.Sequential(
            nn.Linear(2 * d, gate_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_h, 1),
        )
        self.cls = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d, num_classes),
        )

    def forward(self, x_s: torch.Tensor, x_f: torch.Tensor) -> torch.Tensor:
        fs = self.s_net(x_s)  # [B, D]
        ff = self.f_net(x_f)  # [B, D]
        z = torch.cat([fs, ff], dim=1)
        alpha = torch.sigmoid(self.gate(z))  # [B, 1]
        fused = alpha * fs + (1.0 - alpha) * ff
        return self.cls(fused)
