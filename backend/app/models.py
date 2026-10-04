"""Loads the ONNX models once at start-up and runs them with ONNX Runtime (CPU only, no PyTorch).

Missing or broken files don't crash the app: /api/health reports them and only the endpoints that
need them return 503. A common fresh-clone problem is a Git LFS "pointer" file (a ~130-byte text
file starting with 'version https://git-lfs') instead of the real model, so we detect that explicitly.
"""
import time

import numpy as np
import onnxruntime as ort

from .config import MODEL_DOWNLOADS, MODEL_FILES, MODELS_DIR

LFS_MAGIC = b"version https://git-lfs"


class ModelStore:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.sessions: dict[str, ort.InferenceSession] = {}
        self.status: dict[str, dict] = {}

    def load_all(self) -> None:
        for name, description in MODEL_FILES.items():
            path = self.models_dir / f"{name}.onnx"
            info = {"description": description, "file": path.name, "loaded": False, "lfs_pointer": False}
            if not path.exists():
                info["error"] = "file not found"
                if name in MODEL_DOWNLOADS:   # large model: tell the user where to get it
                    info["error"] += f" - download it into models/onnx/ from {MODEL_DOWNLOADS[name]}"
                    info["download_url"] = MODEL_DOWNLOADS[name]
            else:
                info["size_mb"] = round(path.stat().st_size / 1e6, 2)
                with open(path, "rb") as f:
                    info["lfs_pointer"] = f.read(len(LFS_MAGIC)) == LFS_MAGIC
                if info["lfs_pointer"]:
                    info["error"] = "Git LFS pointer, not the real model (run 'git lfs pull')"
                else:
                    try:
                        self.sessions[name] = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
                        info["loaded"] = True
                    except Exception as e:  # corrupted file etc.: report it, keep the app running
                        info["error"] = f"could not load: {e}"
            self.status[name] = info

    def available(self, name: str) -> bool:
        return name in self.sessions

    def run(self, name: str, feed: dict) -> tuple[list[np.ndarray], float]:
        """Run one model; returns (outputs, inference time in ms)."""
        t = time.perf_counter()
        outputs = self.sessions[name].run(None, feed)
        return outputs, (time.perf_counter() - t) * 1000.0


store = ModelStore()
