"""Small helpers shared by all training scripts."""
import json
import random
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy and PyTorch so runs are repeatable."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Return the GPU. We refuse to silently fall back to CPU (CLAUDE.md rule)."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Use the .venv Python with the CUDA build of PyTorch.")
    return torch.device("cuda")


def save_json(obj, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Compact separators keep manifest files small.
    path.write_text(json.dumps(obj, separators=(",", ":")), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_tensor(img: np.ndarray) -> torch.Tensor:
    """uint8 HxWxC (or HxW) -> float32 CxHxW in [0, 1]."""
    if img.ndim == 2:
        img = img[:, :, None]
    return torch.from_numpy(np.ascontiguousarray(img.transpose(2, 0, 1))).float().div_(255.0)


def to_uint8_image(t: torch.Tensor) -> np.ndarray:
    """float CxHxW in [0, 1] -> uint8 HxWxC (for saving / plotting)."""
    arr = t.detach().clamp(0, 1).mul(255).round().byte().cpu().numpy()
    return arr.transpose(1, 2, 0)
