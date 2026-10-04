"""Corruptions follow the assignment definitions and are deterministic given their params."""
import numpy as np
import pytest

from src import config as C
from src.corruptions import (apply_corruption, fixed_test_params, gaussian_blur, make_rects,
                             rect_coverage, sample_training_params, salt_and_pepper)

RNG = np.random.default_rng(0)
IMG = RNG.integers(30, 220, size=(128, 128, 3), dtype=np.uint8)  # mid-grey noise, no pure 0/255


def test_salt_fraction_and_values():
    out = salt_and_pepper(IMG, p=0.10, seed=1)
    changed = np.any(out != IMG, axis=2)
    assert abs(changed.mean() - 0.10) < 0.01                    # about p of the pixels are hit
    hit = out[changed]
    assert np.all((hit == 0).all(1) | (hit == 255).all(1))      # whole pixel black or white
    black = (hit == 0).all(1).mean()
    assert abs(black - 0.5) < 0.05                              # black/white 50/50


def test_blur_matches_reference_and_keeps_shape():
    from scipy.ndimage import gaussian_filter1d
    out = gaussian_blur(IMG, kernel=5, sigma=1.5)
    assert out.shape == IMG.shape and out.dtype == np.uint8
    # Reference: SciPy 1-D Gaussian with the same 5-tap support (truncate=2/1.5 -> radius 2) and
    # 'mirror' borders (= NumPy 'reflect').
    ref = IMG.astype(np.float64)
    for axis in (0, 1):
        ref = gaussian_filter1d(ref, 1.5, axis=axis, mode="mirror", truncate=2 / 1.5)
    assert np.abs(out.astype(float) - ref).max() <= 1.0         # only rounding differences


@pytest.mark.parametrize("n,area", [(1, 0.10), (2, 0.20), (3, 0.35), (3, 0.10), (1, 0.35)])
def test_occlusion_area_and_no_overlap(n, area):
    rng = np.random.default_rng(n * 100)
    for _ in range(50):
        rects = make_rects(n, area, rng)
        assert len(rects) == n
        cov = rect_coverage(rects)
        assert abs(cov - area) < 0.01                           # within 1 percentage point
        assert abs(cov * 128 * 128 - sum(h * w for _, _, h, w in rects)) < 1e-6  # no overlap
        for y, x, h, w in rects:                                # inside the image
            assert 0 <= y and y + h <= 128 and 0 <= x and x + w <= 128


def test_training_ranges():
    rng = np.random.default_rng(3)
    for _ in range(300):
        s = sample_training_params(C.SALT, rng)
        assert 0.02 <= s["p"] <= 0.15
        b = sample_training_params(C.BLUR, rng)
        assert b["kernel"] in (3, 5, 7) and 0.5 <= b["sigma"] <= 2.5
        o = sample_training_params(C.OCCLUSION, rng)
        assert 1 <= len(o["rects"]) <= 3
        assert 0.09 <= rect_coverage(o["rects"]) <= 0.36


def test_fixed_test_severities():
    rng = np.random.default_rng(4)
    assert [fixed_test_params(C.SALT, lv, rng)["p"] for lv in C.LEVELS] == [0.03, 0.08, 0.15]
    assert [(p["kernel"], p["sigma"]) for p in (fixed_test_params(C.BLUR, lv, rng) for lv in C.LEVELS)] \
        == [(3, 0.7), (5, 1.5), (7, 2.5)]
    occ = [fixed_test_params(C.OCCLUSION, lv, rng) for lv in C.LEVELS]
    assert [len(p["rects"]) for p in occ] == [1, 2, 3]


def test_apply_is_deterministic():
    rng = np.random.default_rng(5)
    for cond in range(4):
        params = sample_training_params(cond, rng)
        assert np.array_equal(apply_corruption(IMG, params), apply_corruption(IMG, params))
    assert np.array_equal(apply_corruption(IMG, {"type": "clean"}), IMG)
