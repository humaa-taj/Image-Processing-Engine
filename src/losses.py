"""SSIM and the combined L1 + SSIM restoration loss.

SSIM (Wang et al., 2004) compares two images in small Gaussian-weighted windows (11x11, sigma 1.5):
    SSIM = ((2*mu_x*mu_y + C1) * (2*cov_xy + C2)) / ((mu_x^2 + mu_y^2 + C1) * (var_x + var_y + C2))
It is 1 for identical images. Images are in [0, 1], so C1 = 0.01^2 and C2 = 0.03^2.
Always computed in float32, even under AMP, because the small variance terms are unstable in fp16.
"""
import torch
import torch.nn.functional as F

C1, C2 = 0.01 ** 2, 0.03 ** 2


def _gaussian_window(channels: int, size: int = 11, sigma: float = 1.5, device=None) -> torch.Tensor:
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = g[:, None] * g[None, :]                       # 2D Gaussian = outer product
    return window.expand(channels, 1, size, size).contiguous()  # one window per channel


def ssim_map(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Per-pixel SSIM map, shape (N, C, H-10, W-10) ('valid' windows only, no padding)."""
    x, y = x.float(), y.float()
    ch = x.shape[1]
    w = _gaussian_window(ch, device=x.device)
    blur = lambda t: F.conv2d(t, w, groups=ch)  # noqa: E731  (local Gaussian average per channel)
    mu_x, mu_y = blur(x), blur(y)
    var_x = blur(x * x) - mu_x ** 2
    var_y = blur(y * y) - mu_y ** 2
    cov = blur(x * y) - mu_x * mu_y
    return ((2 * mu_x * mu_y + C1) * (2 * cov + C2)) / ((mu_x ** 2 + mu_y ** 2 + C1) * (var_x + var_y + C2))


def ssim(x: torch.Tensor, y: torch.Tensor, per_image: bool = False) -> torch.Tensor:
    """Mean SSIM over the batch (scalar), or one value per image if per_image=True."""
    m = ssim_map(x, y)
    return m.flatten(1).mean(1) if per_image else m.mean()


def restoration_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
    """L = alpha * L1 + (1 - alpha) * (1 - SSIM)   (Task 1 formula; also used by the specialists)."""
    pred, target = pred.float(), target.float()
    return alpha * F.l1_loss(pred, target) + (1 - alpha) * (1 - ssim(pred, target))
