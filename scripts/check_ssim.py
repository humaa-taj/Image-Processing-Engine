"""Compare our SSIM (src/losses.py) with scikit-image on REAL validation images.

Uses the first 40 entries of each condition from the validation manifest (corrupted vs clean),
so the comparison covers clean (SSIM = 1), salt, blur and occlusion inputs.
Run:  .venv\\Scripts\\python scripts/check_ssim.py   -> artifacts/results/ssim_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from skimage.metrics import structural_similarity

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import pet  # noqa: E402
from src.losses import ssim  # noqa: E402


def compare(per_condition: int = 40) -> dict:
    ds = pet.load_manifest_dataset("val")
    rows = []
    for cond in range(4):
        idx = [i for i, e in enumerate(ds.entries) if e["condition"] == cond][:per_condition]
        for i in idx:
            x, y, _, _ = ds[i]
            ours = ssim(x[None], y[None]).item()
            ref = structural_similarity(x.numpy().transpose(1, 2, 0), y.numpy().transpose(1, 2, 0),
                                        channel_axis=2, data_range=1.0, gaussian_weights=True,
                                        sigma=1.5, use_sample_covariance=False)
            rows.append((cond, ours, ref))
    diff = np.array([abs(o - r) for _, o, r in rows])
    out = {"n_images": len(rows), "max_abs_diff": float(diff.max()), "mean_abs_diff": float(diff.mean()),
           "per_condition_mean_ssim_ours": {C.CONDITIONS[c]: float(np.mean([o for cc, o, _ in rows if cc == c]))
                                            for c in range(4)},
           "per_condition_mean_ssim_skimage": {C.CONDITIONS[c]: float(np.mean([r for cc, _, r in rows if cc == c]))
                                               for c in range(4)},
           "skimage_settings": "gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=1"}
    return out


if __name__ == "__main__":
    torch.set_grad_enabled(False)
    result = compare()
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "ssim_check.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
