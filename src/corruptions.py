"""The three corruptions (salt-and-pepper, Gaussian blur, occlusion), NumPy only.

Design:
- Every corruption is fully described by a small JSON-friendly dict ("params").
  `apply_corruption(img, params)` is deterministic: same image + same params -> same output.
  The val/test manifests simply store these dicts, so evaluation is exactly repeatable.
- Randomness lives only in the *samplers* (`sample_training_params`, `fixed_test_params`),
  which turn a NumPy random generator into params.
- No PyTorch here, so the FastAPI backend can import this file unchanged (no train/serve skew).

Images are uint8 arrays of shape (H, W, 3).
"""
import numpy as np

from src import config as C


# ---------------------------------------------------------------- apply (deterministic)

def salt_and_pepper(img: np.ndarray, p: float, seed: int) -> np.ndarray:
    """Each pixel is hit with probability p; a hit pixel becomes black or white (50/50).
    A "pixel" means all 3 channels together, so hit pixels are pure black/white, not coloured."""
    rng = np.random.default_rng(seed)
    h, w = img.shape[:2]
    hit = rng.random((h, w)) < p
    white = rng.random((h, w)) < 0.5
    out = img.copy()
    out[hit & white] = 255
    out[hit & ~white] = 0
    return out


def gaussian_kernel_1d(kernel: int, sigma: float) -> np.ndarray:
    x = np.arange(kernel) - kernel // 2
    k = np.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(img: np.ndarray, kernel: int, sigma: float) -> np.ndarray:
    """Separable Gaussian blur (rows, then columns) with reflect padding at the borders."""
    k = gaussian_kernel_1d(kernel, sigma).astype(np.float32)
    r = kernel // 2
    x = img.astype(np.float32)
    # Blur along height: pad top/bottom, then weighted sum of shifted copies.
    xp = np.pad(x, ((r, r), (0, 0), (0, 0)), mode="reflect")
    x = sum(k[i] * xp[i:i + img.shape[0]] for i in range(kernel))
    # Blur along width.
    xp = np.pad(x, ((0, 0), (r, r), (0, 0)), mode="reflect")
    x = sum(k[i] * xp[:, i:i + img.shape[1]] for i in range(kernel))
    return np.clip(np.rint(x), 0, 255).astype(np.uint8)


def occlusion(img: np.ndarray, rects) -> np.ndarray:
    """Paint black rectangles. rects = [[y, x, h, w], ...] in pixel coordinates."""
    out = img.copy()
    for y, x, h, w in rects:
        out[y:y + h, x:x + w] = 0
    return out


def apply_corruption(img: np.ndarray, params: dict) -> np.ndarray:
    kind = params["type"]
    if kind == "clean":
        return img.copy()
    if kind == "salt":
        return salt_and_pepper(img, params["p"], params["seed"])
    if kind == "blur":
        return gaussian_blur(img, params["kernel"], params["sigma"])
    if kind == "occlusion":
        return occlusion(img, params["rects"])
    raise ValueError(f"unknown corruption type: {kind}")


# ---------------------------------------------------------------- occlusion geometry

def _overlaps(a, b) -> bool:
    ay, ax, ah, aw = a
    by, bx, bh, bw = b
    return ay < by + bh and by < ay + ah and ax < bx + bw and bx < ax + aw


def make_rects(n: int, area_frac: float, rng: np.random.Generator, size: int = C.IMG_SIZE):
    """n non-overlapping rectangles that together cover ~area_frac of a size x size image.

    Non-overlapping, so the covered area is exactly the sum of the rectangle areas (overlaps
    would make the real coverage smaller than requested). The total area is split randomly
    between rectangles; each rectangle gets a random aspect ratio between 1:2 and 2:1.
    """
    total = area_frac * size * size
    for _ in range(200):  # restart with new shapes if placement gets stuck
        shares = rng.dirichlet(np.full(n, 3.0)) if n > 1 else np.array([1.0])
        rects = []
        for share in shares:
            area = share * total
            aspect = np.exp(rng.uniform(np.log(0.5), np.log(2.0)))   # h / w
            h = int(np.clip(round(np.sqrt(area * aspect)), 2, size))
            w = int(np.clip(round(area / h), 2, size))
            for _ in range(100):
                y = int(rng.integers(0, size - h + 1))
                x = int(rng.integers(0, size - w + 1))
                if not any(_overlaps((y, x, h, w), r) for r in rects):
                    rects.append([y, x, h, w])
                    break
            else:
                break  # this rectangle did not fit -> restart
        if len(rects) == n:
            return rects
    raise RuntimeError(f"could not place {n} rectangles covering {area_frac:.0%}")


def rect_coverage(rects, size: int = C.IMG_SIZE) -> float:
    """Fraction of the image covered (computed on a mask, so it is exact)."""
    mask = np.zeros((size, size), bool)
    for y, x, h, w in rects:
        mask[y:y + h, x:x + w] = True
    return float(mask.mean())


# ---------------------------------------------------------------- samplers (random -> params)

def sample_training_params(condition: int, rng: np.random.Generator) -> dict:
    """Random params from the training ranges in the assignment table."""
    name = C.CONDITIONS[condition]
    if name == "clean":
        return {"type": "clean"}
    if name == "salt":
        return {"type": "salt", "p": float(rng.uniform(*C.SALT_P_RANGE)),
                "seed": int(rng.integers(2 ** 31))}
    if name == "blur":
        return {"type": "blur", "kernel": int(rng.choice(C.BLUR_KERNELS)),
                "sigma": float(rng.uniform(*C.BLUR_SIGMA_RANGE))}
    n = int(rng.integers(C.OCC_RECTS_RANGE[0], C.OCC_RECTS_RANGE[1] + 1))
    area = float(rng.uniform(*C.OCC_AREA_RANGE))
    return {"type": "occlusion", "rects": make_rects(n, area, rng), "target_area": area}


def fixed_test_params(condition: int, level: str, rng: np.random.Generator) -> dict:
    """Params for the fixed test severities (low / medium / high)."""
    name = C.CONDITIONS[condition]
    if name == "clean":
        return {"type": "clean"}
    if name == "salt":
        return {"type": "salt", "p": C.TEST_SALT_P[level], "seed": int(rng.integers(2 ** 31))}
    if name == "blur":
        k, s = C.TEST_BLUR[level]
        return {"type": "blur", "kernel": k, "sigma": s}
    n, area = C.TEST_OCCLUSION[level]
    return {"type": "occlusion", "rects": make_rects(n, area, rng), "target_area": area}


def severity_value(params: dict) -> float:
    """One number describing how strong a corruption is (for plots / tables)."""
    kind = params["type"]
    if kind == "salt":
        return params["p"]
    if kind == "blur":
        return params["sigma"]
    if kind == "occlusion":
        return rect_coverage(params["rects"])
    return 0.0
