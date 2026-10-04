"""Upload validation, preprocessing, runtime corruption and PNG encoding.

Preprocessing is identical to training (src/data/pet.py, src/data/fs2k.py): convert to RGB, PIL
bilinear resize to 128x128, scale to [0, 1], layout [N, C, H, W] float32.
Corruptions come from the SAME module the training/evaluation code used (src/corruptions.py),
with the same fixed low / medium / high settings as the test manifest.
"""
import base64
import io

import numpy as np
from fastapi import HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from src import config as C
from src.corruptions import fixed_test_params, apply_corruption, severity_value

from .config import ALLOWED_TYPES, IMG_SIZE, MAX_UPLOAD_BYTES

CORRUPTIONS = {"salt": C.SALT, "blur": C.BLUR, "occlusion": C.OCCLUSION}


async def read_upload(file: UploadFile) -> Image.Image:
    """Validate type, size and that the bytes really decode as an image. Returns an RGB PIL image."""
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(415, f"Unsupported file type '{file.content_type}'. Use JPEG, PNG, WebP or BMP.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).")
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()                                   # forces a full decode (catches truncated files)
    except (UnidentifiedImageError, OSError) as e:
        raise HTTPException(400, f"Could not decode the image: {e}") from e
    img = ImageOps.exif_transpose(img)               # phone/webcam photos: respect the rotation flag
    return img.convert("RGB")


def to_model_array(img: Image.Image, fit: str = "stretch") -> np.ndarray:
    """uint8 HxWx3 at 128x128. fit='stretch' = direct resize (as in training);
    fit='crop' = centre square crop first (better for webcam / portrait photos in Task 4)."""
    if fit == "crop":
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        img = img.crop((left, top, left + side, top + side))
    return np.asarray(img.resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR), dtype=np.uint8)


def corrupt(arr: np.ndarray, corruption: str, level: str, seed: int | None) -> tuple[np.ndarray, dict | None]:
    """Apply one of the assignment's corruptions at a fixed test severity. 'none' = leave as is
    (e.g. the user uploaded an image that is already corrupted)."""
    if corruption == "none":
        return arr, None
    if corruption not in CORRUPTIONS:
        raise HTTPException(422, f"corruption must be one of none, {', '.join(CORRUPTIONS)}")
    if level not in C.LEVELS:
        raise HTTPException(422, f"level must be one of {', '.join(C.LEVELS)}")
    seed = int(np.random.default_rng().integers(2 ** 31)) if seed is None else seed
    params = fixed_test_params(CORRUPTIONS[corruption], level, np.random.default_rng(seed))
    info = {"type": corruption, "level": level, "seed": seed, "severity": round(severity_value(params), 4),
            "params": params}
    return apply_corruption(arr, params), info


def to_batch(arr: np.ndarray) -> np.ndarray:
    """uint8 HxWxC -> float32 [1, C, H, W] in [0, 1] (the ONNX input format)."""
    return (arr.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]


def to_png_data_url(x: np.ndarray) -> str:
    """Model output [C, H, W] or uint8 HxWxC -> 'data:image/png;base64,...' for the browser."""
    if x.dtype != np.uint8:
        x = np.clip(np.rint(x * 255.0), 0, 255).astype(np.uint8)
        x = x[0] if x.shape[0] == 1 else x.transpose(1, 2, 0)    # 1-channel sketch or RGB image
    buf = io.BytesIO()
    Image.fromarray(x).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# Error map colours: the UI theme's ramp (near-black -> steel blue -> white), so the map and its
# LOW -> HIGH legend in the frontend match. Same fixed scale as the report figures (0 .. 0.5).
ERROR_STOPS = np.array([[14, 15, 17], [122, 150, 186], [236, 240, 245]], dtype=np.float32)
ERROR_VMAX = 0.5


def error_map(output: np.ndarray, clean: np.ndarray) -> tuple[str, float]:
    """|output - clean| averaged over RGB -> colour PNG + mean absolute error.
    output: model output [3, H, W] in [0, 1]; clean: uint8 HxWx3 reference."""
    err = np.abs(np.clip(output, 0, 1).transpose(1, 2, 0) - clean.astype(np.float32) / 255.0).mean(axis=2)
    t = np.clip(err / ERROR_VMAX, 0, 1) * (len(ERROR_STOPS) - 1)        # position along the ramp
    i = np.minimum(t.astype(int), len(ERROR_STOPS) - 2)
    f = (t - i)[..., None]
    rgb = ERROR_STOPS[i] * (1 - f) + ERROR_STOPS[i + 1] * f
    return to_png_data_url(rgb.round().astype(np.uint8)), round(float(err.mean()), 5)
