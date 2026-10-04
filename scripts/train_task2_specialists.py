"""Task 2: train the three specialist autoencoders with the shared Optuna config.

Each specialist has its OWN weights and sees ONLY its own corruption (training and validation).
Model selection: best validation score on that corruption's 184 val-manifest images.
Run:  .venv\\Scripts\\python scripts/train_task2_specialists.py              (all three)
      .venv\\Scripts\\python scripts/train_task2_specialists.py --only blur  (one of them)
Outputs: models/checkpoints/task2_expert_{salt,blur,occlusion}.pt,
         artifacts/results/task2_expert_<name>_history.csv, docs/figures/task2_specialists_curves.png,
         one MLflow run per specialist.
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
from src.training import train_restoration  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

EXPERTS = {"salt": C.SALT, "blur": C.BLUR, "occlusion": C.OCCLUSION}


TAG = ""   # "" for v1, "_v2" for the upgrade pass (set from --version in main())


def ckpt_path(name: str) -> Path:
    return C.CHECKPOINTS / f"task2_expert_{name}{TAG}.pt"


def plot_all_curves() -> None:
    """One figure with the validation SSIM / PSNR curves of every specialist trained so far."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for name in EXPERTS:
        f = C.RESULTS / f"task2_expert_{name}{TAG}_history.csv"
        if not f.exists():
            continue
        h = pd.read_csv(f)
        axes[0].plot(h.epoch, h.train_loss, label=f"{name} train")
        axes[0].plot(h.epoch, h.val_score, "--", label=f"{name} val score")
        axes[1].plot(h.epoch, h.val_ssim, label=name)
        axes[2].plot(h.epoch, h.val_psnr, label=name)
    for ax, title in zip(axes, ["Loss / val score", "Validation SSIM (own corruption)", "Validation PSNR (dB)"]):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGURES / f"task2_specialists{TAG}_curves.png", dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="v2 = upgrade pass: own config, checkpoints, outputs")
    ap.add_argument("--only", choices=list(EXPERTS), nargs="*")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    global TAG
    TAG = "" if args.version == "v1" else f"_{args.version}"
    args.config = str(C.ROOT / "configs" / f"task2_specialists{TAG}.yaml")
    names = args.only or list(EXPERTS)
    existing = [n for n in names if ckpt_path(n).exists()]
    if existing and not args.overwrite:
        sys.exit(f"checkpoints already exist for {existing}. Use --overwrite or --only <others>.")

    cfg = yaml.safe_load(Path(args.config).read_text())
    tc = cfg["train"]
    device = get_device()
    setup_mlflow("task2-hard-routing")
    for name in names:
        print(f"===== specialist: {name} =====", flush=True)
        seed_everything(C.SEED)
        model = ConvAutoencoder(**cfg["model"]).to(device)
        with mlflow.start_run(run_name=f"expert-{name}-final-train{TAG}"):
            mlflow.log_params({**cfg["model"], **tc, "expert": name, "params_M": count_parameters(model) / 1e6,
                               "config_source": cfg.get("source", "")})
            result = train_restoration(model, tc, tc["epochs"], device, conditions=(EXPERTS[name],),
                                       patience=tc["patience"])
            best = result["best"]
            C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            torch.save({"model_config": cfg["model"], "train_config": tc, "expert": name,
                        "condition": EXPERTS[name], "state_dict": best["state"], "best_epoch": best["epoch"],
                        "best_val": {k: v for k, v in best.items() if k != "state"},
                        "history": result["history"]}, ckpt_path(name))
            pd.DataFrame(result["history"]).to_csv(C.RESULTS / f"task2_expert_{name}{TAG}_history.csv", index=False)
            mlflow.log_metrics({"best_epoch": best["epoch"], "best_val_score": best["val_score"],
                                "best_val_ssim": best["val_ssim"], "best_val_psnr": best["val_psnr"]})
            mlflow.log_artifact(str(ckpt_path(name)), "checkpoint")
        print(f"{name}: best epoch {best['epoch']}, val SSIM {best['val_ssim']:.4f}, PSNR {best['val_psnr']:.2f}")
    plot_all_curves()


if __name__ == "__main__":
    main()
