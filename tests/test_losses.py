"""Our SSIM agrees with scikit-image; loss and metrics behave as expected."""
import numpy as np
import optuna
import pytest
import torch
from skimage.metrics import structural_similarity

from src.losses import restoration_loss, ssim
from src.metrics import objective_score, per_image_metrics
from src.tracking import TrainingDiverged, safe_objective


def test_ssim_identical_is_one():
    x = torch.rand(2, 3, 64, 64)
    assert torch.allclose(ssim(x, x), torch.tensor(1.0), atol=1e-5)


def test_ssim_matches_skimage():
    rng = np.random.default_rng(0)
    a = rng.random((64, 64, 3)).astype(np.float32)
    b = np.clip(a + rng.normal(0, 0.1, a.shape), 0, 1).astype(np.float32)
    ref = structural_similarity(a, b, channel_axis=2, data_range=1.0, gaussian_weights=True,
                                sigma=1.5, use_sample_covariance=False)
    t = lambda z: torch.from_numpy(z.transpose(2, 0, 1))[None]  # noqa: E731
    ours = ssim(t(a), t(b)).item()
    # skimage averages over a slightly different border region, so allow a small gap.
    assert abs(ours - ref) < 0.01


def test_loss_and_metrics():
    x = torch.rand(4, 3, 32, 32)
    assert restoration_loss(x, x, alpha=0.8).item() == pytest.approx(0.0, abs=1e-5)
    y = (x + 0.1).clamp(0, 1)
    m = per_image_metrics(y, x)
    assert m["psnr"].shape == (4,) and (m["psnr"] > 15).all()
    assert objective_score(0.0, 1.0) == 0.0


def test_safe_objective_turns_errors_into_pruned():
    def oom(trial):
        raise torch.cuda.OutOfMemoryError("fake")

    def diverge(trial):
        raise TrainingDiverged("nan")

    study = optuna.create_study()
    study.optimize(safe_objective(oom), n_trials=1)
    study.optimize(safe_objective(diverge), n_trials=1)
    assert [t.state for t in study.trials] == [optuna.trial.TrialState.PRUNED] * 2
    assert study.trials[0].user_attrs["pruned_reason"] == "cuda_oom"


def test_ssim_matches_skimage_on_real_images():
    """Real validation images (all 4 conditions): our SSIM equals scikit-image up to float32 rounding."""
    from scripts.check_ssim import compare
    result = compare(per_condition=10)
    assert result["max_abs_diff"] < 1e-4
