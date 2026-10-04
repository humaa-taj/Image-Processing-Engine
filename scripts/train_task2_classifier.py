"""Task 2 final classifier training with the best Optuna config (configs/task2_classifier.yaml).

Model selection: lowest validation cross-entropy (+ early stopping). Test is not touched.
Run:  .venv\\Scripts\\python scripts/train_task2_classifier.py
Outputs: models/checkpoints/task2_classifier.pt, artifacts/results/task2_classifier_history.csv,
         docs/figures/task2_classifier_curves.png, MLflow run 'classifier-final-train'.
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
from src.models.autoencoder import count_parameters  # noqa: E402
from src.models.classifier import CorruptionClassifier  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.training import train_classifier  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

CKPT = C.CHECKPOINTS / "task2_classifier.pt"


def plot_curves(h: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(h.epoch, h.train_loss, label="train")
    axes[0].plot(h.epoch, h.val_loss, label="validation")
    axes[0].set_title("Cross-entropy loss")
    axes[1].plot(h.epoch, h.train_acc, label="train accuracy")
    axes[1].plot(h.epoch, h.val_acc, label="val accuracy")
    axes[1].plot(h.epoch, h.val_macro_f1, label="val macro-F1", linestyle="--")
    axes[1].set_title("Accuracy")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(C.ROOT / "configs" / "task2_classifier.yaml"))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    if CKPT.exists() and not args.overwrite:
        sys.exit(f"{CKPT} already exists. Use --overwrite to replace it.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    device = get_device()
    seed_everything(C.SEED)
    model = CorruptionClassifier(**cfg["model"]).to(device)
    tc = cfg["train"]
    setup_mlflow("task2-hard-routing")
    with mlflow.start_run(run_name="classifier-final-train"):
        mlflow.log_params({**cfg["model"], **tc, "params_M": count_parameters(model) / 1e6,
                           "config_source": cfg.get("source", "")})
        result = train_classifier(model, tc, tc["epochs"], device, patience=tc["patience"])
        best, history = result["best"], pd.DataFrame(result["history"])
        C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"model_config": cfg["model"], "train_config": tc, "state_dict": best["state"],
                    "best_epoch": best["epoch"], "best_val": {k: v for k, v in best.items() if k != "state"},
                    "history": result["history"]}, CKPT)
        history.to_csv(C.RESULTS / "task2_classifier_history.csv", index=False)
        plot_curves(history, C.FIGURES / "task2_classifier_curves.png")
        mlflow.log_metrics({"best_epoch": best["epoch"], "best_val_loss": best["val_loss"],
                            "best_val_acc": best["val_acc"], "best_val_macro_f1": best["val_macro_f1"]})
        for f in (CKPT, C.FIGURES / "task2_classifier_curves.png", C.RESULTS / "task2_classifier_history.csv"):
            mlflow.log_artifact(str(f))
    print(f"best epoch {best['epoch']}: val loss {best['val_loss']:.4f}, acc {best['val_acc']:.4f}, "
          f"macro-F1 {best['val_macro_f1']:.4f} -> {CKPT}")


if __name__ == "__main__":
    main()
