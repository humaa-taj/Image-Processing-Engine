"""Task 1 final training with the best Optuna config (configs/task1.yaml).

Model selection uses the VALIDATION manifest only (best val score + early stopping). Test is not touched.
Run:  .venv\\Scripts\\python scripts/train_task1.py
Outputs: models/checkpoints/task1_universal_ae.pt (gitignored), artifacts/results/task1_history.csv,
         docs/figures/task1_curves.png, MLflow run 'final-train' (metrics, samples, checkpoint).
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
from src.models.autoencoder import ConvAutoencoder, count_parameters  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.training import load_val_tensors, predict, train_restoration  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402



def make_sample_logger(device, every: int = 10):
    """Log the same 8 validation images (2 per condition) every `every` epochs -> see progress over time."""
    x, y, labels = load_val_tensors()
    idx = torch.cat([torch.nonzero(labels == c)[:2, 0] for c in range(4)])
    x, y, labels = x[idx], y[idx], labels[idx]

    def log(model, epoch):
        if epoch != 1 and epoch % every:
            return
        out = predict(model, x, device)
        fig, axes = plt.subplots(3, len(idx), figsize=(2 * len(idx), 6.3))
        for j in range(len(idx)):
            for r, (img, name) in enumerate([(x[j], "input"), (out[j], "output"), (y[j], "target")]):
                axes[r, j].imshow(img.permute(1, 2, 0).clamp(0, 1).numpy())
                axes[r, j].axis("off")
                if j == 0:
                    axes[r, j].set_title(name, loc="left", fontsize=9)
            axes[0, j].set_title(C.CONDITIONS[labels[j]], fontsize=9)
        fig.suptitle(f"Task 1 validation samples, epoch {epoch}")
        fig.tight_layout()
        mlflow.log_figure(fig, f"samples/epoch_{epoch:03d}.png")
        plt.close(fig)
    return log


def plot_curves(history: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(history.epoch, history.train_loss, label="train loss (alpha-weighted)")
    axes[0].plot(history.epoch, history.val_score, label="val score 0.5*L1+0.5*(1-SSIM)")
    axes[0].set_title("Loss")
    axes[1].plot(history.epoch, history.val_ssim, label="all", color="black")
    for c in C.CONDITIONS:
        axes[1].plot(history.epoch, history[f"val_ssim_{c}"], label=c, alpha=0.8)
    axes[1].set_title("Validation SSIM")
    axes[2].plot(history.epoch, history.val_psnr)
    axes[2].set_title("Validation PSNR (dB)")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
    axes[0].legend()
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="v2 = upgrade pass: own config, checkpoint and outputs")
    ap.add_argument("--overwrite", action="store_true", help="replace an existing checkpoint")
    args = ap.parse_args()
    tag = "" if args.version == "v1" else f"_{args.version}"
    ckpt = C.CHECKPOINTS / f"task1_universal_ae{tag}.pt"
    if ckpt.exists() and not args.overwrite:
        sys.exit(f"{ckpt} already exists. Use --overwrite to replace it.")

    cfg = yaml.safe_load((C.ROOT / "configs" / f"task1{tag}.yaml").read_text())
    device = get_device()
    seed_everything(C.SEED)
    model = ConvAutoencoder(**cfg["model"]).to(device)
    train_cfg = cfg["train"]

    setup_mlflow("task1-universal-ae")
    with mlflow.start_run(run_name=f"final-train{tag}"):
        mlflow.log_params({**cfg["model"], **train_cfg, "params_M": count_parameters(model) / 1e6,
                           "config_source": cfg.get("source", "")})
        result = train_restoration(model, train_cfg, train_cfg["epochs"], device,
                                   patience=train_cfg["patience"], sample_fn=make_sample_logger(device))
        best = result["best"]
        history = pd.DataFrame(result["history"])

        C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"model_config": cfg["model"], "train_config": train_cfg, "state_dict": best["state"],
                    "best_epoch": best["epoch"], "best_val": {k: v for k, v in best.items() if k != "state"},
                    "history": result["history"]}, ckpt)
        C.RESULTS.mkdir(parents=True, exist_ok=True)
        history.to_csv(C.RESULTS / f"task1{tag}_history.csv", index=False)
        plot_curves(history, C.FIGURES / f"task1{tag}_curves.png")

        mlflow.log_metrics({"best_epoch": best["epoch"], "best_val_score": best["val_score"],
                            "best_val_ssim": best["val_ssim"], "best_val_psnr": best["val_psnr"]})
        mlflow.log_artifact(str(ckpt), "checkpoint")
        mlflow.log_artifact(str(C.FIGURES / f"task1{tag}_curves.png"))
        mlflow.log_artifact(str(C.RESULTS / f"task1{tag}_history.csv"))
    print(f"best epoch {best['epoch']}: val score {best['val_score']:.4f}, SSIM {best['val_ssim']:.4f}, "
          f"PSNR {best['val_psnr']:.2f}  -> {ckpt}")


if __name__ == "__main__":
    main()
