"""Task 4 final training: retrain the best Optuna config for the full schedule (configs/task4.yaml).

- Logs D real / D fake / G adversarial / G L1 losses and validation metrics separately (MLflow).
- Every `sample_every` epochs: the SAME 6 validation photos (2 per style) + one photo in all 3 styles.
- Every `checkpoint_every` epochs: G, D and both optimizers -> models/checkpoints/task4_last.pt
  (GANs can diverge, so we can always go back). The generator with the best validation score is
  saved separately -> models/checkpoints/task4_generator.pt (this is what gets exported to ONNX).
Run:  .venv\\Scripts\\python scripts/train_task4.py
"""
import argparse
import sys
from pathlib import Path

import matplotlib
import mlflow
import pandas as pd
import torch
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.gan_training import fs2k_loaders, generate, train_gan  # noqa: E402
from src.models.autoencoder import count_parameters  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

CKPT = C.CHECKPOINTS / "task4_generator.pt"
LAST = C.CHECKPOINTS / "task4_last.pt"


def make_sample_logger(device, every: int):
    _, (photos, sketches, styles) = fs2k_loaders(4, num_workers=0)
    idx = torch.cat([torch.nonzero(styles == k)[:2, 0] for k in range(3)])     # fixed: 2 photos per style
    photos, sketches, styles = photos[idx], sketches[idx], styles[idx]

    def log(g, epoch):
        if epoch != 1 and epoch % every:
            return
        fake = generate(g, photos, styles, device)
        all_styles = generate(g, photos[:1].repeat(3, 1, 1, 1), torch.arange(3), device)
        n = len(idx)
        fig, axes = plt.subplots(3, n + 1, figsize=(2 * (n + 1), 6.4))
        for j in range(n):
            axes[0, j].imshow(photos[j].permute(1, 2, 0).numpy())
            axes[0, j].set_title(f"photo (style {int(styles[j]) + 1})", fontsize=8)
            axes[1, j].imshow(fake[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[1, j].set_title("generated", fontsize=8)
            axes[2, j].imshow(sketches[j, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[2, j].set_title("real sketch", fontsize=8)
        for k in range(3):
            axes[k, n].imshow(all_styles[k, 0].numpy(), cmap="gray", vmin=0, vmax=1)
            axes[k, n].set_title(f"photo 1 as Style {k + 1}", fontsize=8)
        for a in axes.flat:
            a.axis("off")
        fig.suptitle(f"Task 4 fixed validation samples, epoch {epoch}")
        fig.tight_layout()
        mlflow.log_figure(fig, f"samples/epoch_{epoch:03d}.png")
        plt.close(fig)
    return log


def make_checkpointer(cfg: dict, every: int):
    def save(g, d, opt_g, opt_d, epoch, best):
        if epoch % every:
            return
        torch.save({"epoch": epoch, "config": cfg, "generator": g.state_dict(), "discriminator": d.state_dict(),
                    "opt_g": opt_g.state_dict(), "opt_d": opt_d.state_dict(), "best_epoch": best["epoch"]}, LAST)
    return save


def plot_curves(h: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(h.epoch, h.d_real, label="D real loss")
    axes[0].plot(h.epoch, h.d_fake, label="D fake loss")
    axes[0].plot(h.epoch, h.g_adv, label="G adversarial loss")
    axes[0].set_title("Adversarial losses")
    axes[1].plot(h.epoch, h.g_l1, label="G L1 loss (train)")
    axes[1].plot(h.epoch, h.val_l1, label="val L1")
    axes[1].set_title("Reconstruction (L1)")
    axes[2].plot(h.epoch, h.val_ssim, label="val SSIM (all)", color="black")
    for k in range(3):
        axes[2].plot(h.epoch, h[f"val_ssim_style{k + 1}"], label=f"val SSIM style {k + 1}", alpha=0.8)
    axes[2].set_title("Validation SSIM")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "task4.yaml"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if CKPT.exists() and not args.overwrite:
        sys.exit(f"{CKPT} already exists. Use --overwrite to replace it.")
    cfg = yaml.safe_load(Path(args.config).read_text())
    tc, mc = cfg["train"], cfg["model"]
    run_cfg = {**mc, **tc}
    device = get_device()
    seed_everything(C.SEED)
    C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    setup_mlflow("task4-face-to-sketch")
    with mlflow.start_run(run_name="final-train"):
        mlflow.log_params({**run_cfg, "config_source": cfg.get("source", "")})
        result = train_gan(run_cfg, tc["epochs"], device, sample_fn=make_sample_logger(device, tc["sample_every"]),
                           checkpoint_fn=make_checkpointer(run_cfg, tc["checkpoint_every"]))
        best, history = result["best"], pd.DataFrame(result["history"])
        g = result["generator"]
        torch.save({"model_config": g.config, "train_config": tc, "state_dict": best["state"],
                    "best_epoch": best["epoch"], "best_val": {k: v for k, v in best.items() if k != "state"},
                    "history": result["history"]}, CKPT)
        history.to_csv(C.RESULTS / "task4_history.csv", index=False)
        plot_curves(history, C.FIGURES / "task4_curves.png")
        mlflow.log_metrics({"best_epoch": best["epoch"], "best_val_score": best["val_score"],
                            "best_val_ssim": best["val_ssim"], "generator_params_M": count_parameters(g) / 1e6})
        for f in (CKPT, C.FIGURES / "task4_curves.png", C.RESULTS / "task4_history.csv"):
            mlflow.log_artifact(str(f))
    print(f"best epoch {best['epoch']}: val score {best['val_score']:.4f}, SSIM {best['val_ssim']:.4f}, "
          f"L1 {best['val_l1']:.4f} -> {CKPT}")


if __name__ == "__main__":
    main()
