import torch

def save_ckpt(path: str, model, optimizer=None, epoch: int = 0, meta=None):
    payload = {"model": model.state_dict(), "epoch": epoch, "meta": meta or {}}
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)

def load_ckpt(path: str, model, optimizer=None, map_location="cpu"):
    payload = torch.load(path, map_location=map_location)
    state = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    model.load_state_dict(state, strict=True)
    if optimizer is not None and isinstance(payload, dict) and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
