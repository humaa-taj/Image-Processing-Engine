"""Evaluation metrics (per image) and the fixed Optuna objective."""
import pandas as pd
import torch

from src.losses import ssim


@torch.no_grad()
def per_image_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """PSNR (dB), SSIM and L1 for each image in a batch. Images in [0, 1]."""
    pred, target = pred.float().clamp(0, 1), target.float()
    mse = (pred - target).pow(2).flatten(1).mean(1).clamp_min(1e-10)
    return {
        "psnr": (10 * torch.log10(1.0 / mse)).cpu(),
        "ssim": ssim(pred, target, per_image=True).cpu(),
        "l1": (pred - target).abs().flatten(1).mean(1).cpu(),
    }


def objective_score(l1: float, ssim_value: float) -> float:
    """Fixed validation score for Optuna in Tasks 1-3 (lower is better).

    It does NOT contain the tuned loss weight alpha. If we used the training loss itself, Optuna
    could improve the score just by moving alpha towards whichever term is numerically smaller,
    without the images getting better. A fixed yardstick keeps all trials comparable.
    """
    return 0.5 * l1 + 0.5 * (1.0 - ssim_value)


def summarize(df: pd.DataFrame, by=("condition", "level")) -> pd.DataFrame:
    """Mean PSNR / SSIM / L1 grouped by corruption and severity (rows = one per test input)."""
    return df.groupby(list(by), sort=False)[["psnr", "ssim", "l1"]].mean().reset_index()
