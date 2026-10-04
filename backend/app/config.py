"""Backend settings (overridable with environment variables, e.g. in docker-compose.yml)."""
import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.environ.get("MODELS_DIR", BACKEND_DIR.parent / "models" / "onnx"))
SAMPLES_DIR = Path(os.environ.get("SAMPLES_DIR", BACKEND_DIR / "samples"))
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "CORS_ORIGINS", "http://localhost:5173,http://localhost:8080,http://127.0.0.1:8080").split(",") if o.strip()]

MAX_UPLOAD_BYTES = 10 * 1024 * 1024                     # 10 MB
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}
IMG_SIZE = 128                                           # every model works on 128x128 inputs

# ONNX files the app needs (file name -> what it is). Names match scripts/export_onnx.py.
MODEL_FILES = {
    "task1_universal_ae": "Task 1 universal denoising autoencoder",
    "task2_classifier": "Task 2 corruption classifier",
    "task2_expert_salt": "Task 2 salt-and-pepper specialist",
    "task2_expert_blur": "Task 2 blur specialist",
    "task2_expert_occlusion": "Task 2 occlusion specialist",
    "task3_soft_moe": "Task 3 soft mixture-of-experts (whole pipeline)",
    "task4_generator": "Task 4 face-to-sketch generator",
}

# Models too large for the git repository (> 50 MB) are published as GitHub Release assets.
MODEL_DOWNLOADS = {
    "task4_generator": "https://github.com/FaatehHaneef/Image-Processing-Engine/releases/download/models-v1/task4_generator.onnx",
}
