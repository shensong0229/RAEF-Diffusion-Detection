import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import timm
except Exception:
    timm = None

try:
    from torchvision import models as tvm
except Exception:
    tvm = None


class SpatialClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet50", num_classes: int = 2, pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone_name = backbone
        self.num_classes = int(num_classes)
        self.dropout_p = float(dropout)
        self.backend = None

        # 优先 timm（可换更强骨干），没有 timm 再用 torchvision
        if timm is not None:
            self.net = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes, drop_rate=dropout)
            self.backend = "timm"
        else:
            if tvm is None:
                raise ImportError("Neither timm nor torchvision is available.")
            backbone = backbone.lower()
            if backbone == "resnet50":
                net = tvm.resnet50(pretrained=pretrained)
            elif backbone == "resnet18":
                net = tvm.resnet18(pretrained=pretrained)
            else:
                raise ValueError(f"Unsupported backbone without timm: {backbone}")
            in_features = net.fc.in_features
            net.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
            self.net = net
            self.backend = "torchvision"

        self.feature_dim = int(self._infer_feature_dim())

    def _infer_feature_dim(self) -> int:
        if hasattr(self.net, "num_features"):
            nf = getattr(self.net, "num_features")
            if isinstance(nf, int) and nf > 0:
                return int(nf)

        if self.backend == "torchvision":
            if hasattr(self.net, "fc") and isinstance(self.net.fc, nn.Sequential):
                for m in self.net.fc:
                    if isinstance(m, nn.Linear):
                        return int(m.in_features)
            if hasattr(self.net, "fc") and isinstance(self.net.fc, nn.Linear):
                return int(self.net.fc.in_features)

        raise AttributeError("Cannot infer feature_dim for SpatialClassifier.")

    def _extract_features_torchvision(self, x: torch.Tensor):
        # torchvision ResNet feature pipeline
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)
        x = self.net.layer1(x)
        x = self.net.layer2(x)
        x = self.net.layer3(x)
        fmap = self.net.layer4(x)
        pooled = self.net.avgpool(fmap)
        pooled = torch.flatten(pooled, 1)
        logits = self.net.fc(pooled)
        return fmap, pooled, logits

    def _extract_features_timm(self, x: torch.Tensor):
        if not hasattr(self.net, "forward_features"):
            raise AttributeError("Current timm backbone does not expose forward_features().")

        fmap = self.net.forward_features(x)
        if isinstance(fmap, (tuple, list)):
            fmap = fmap[-1]

        # Most timm CNN backbones (e.g. resnet) return [B,C,H,W].
        # If some backbone returns [B,N,C], we still produce a pooled vector,
        # but dual-stream currently expects CNN-style 4D feature maps.
        if fmap.ndim == 4:
            if hasattr(self.net, "forward_head"):
                pooled = self.net.forward_head(fmap, pre_logits=True)
                logits = self.net.forward_head(fmap, pre_logits=False)
            else:
                pooled = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
                logits = self.net(x)
            return fmap, pooled, logits

        if fmap.ndim == 3:
            # token-style fallback: pool tokens, but keep an explicit error message for dual-stream
            pooled = fmap.mean(dim=1)
            logits = self.net(x)
            return fmap, pooled, logits

        raise ValueError(f"Unsupported timm feature shape: {tuple(fmap.shape)}")

    def extract_features(self, x: torch.Tensor, return_logits: bool = True):
        if self.backend == "torchvision":
            fmap, pooled, logits = self._extract_features_torchvision(x)
        elif self.backend == "timm":
            fmap, pooled, logits = self._extract_features_timm(x)
        else:
            raise ValueError(f"Unsupported backend: {self.backend}")

        out = {
            "feature_map": fmap,
            "global_feature": pooled,
        }
        if return_logits:
            out["logits"] = logits
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
