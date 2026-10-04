"""Task 3 final training: soft mixture-of-experts with the best Optuna config (configs/task3.yaml).

Starts from the Task 2 classifier (gate) and specialists (experts); warm-up (gate only), then joint
fine-tuning. Model selection on the validation manifest only.
Run:  .venv\\Scripts\\python scripts/train_task3.py
Outputs: models/checkpoints/task3_soft_moe.pt, artifacts/results/task3_history.csv,
         docs/figures/task3_curves.png, MLflow run 'final-train' in experiment 'task3-soft-moe'.
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
from src.onnx_utils import build_moe_from_task2  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.training import train_moe  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402


def plot_curves(h: pd.DataFrame, warmup: int, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    t = h[h.epoch > 0]
    axes[0].plot(t.epoch, t.train_total, label="train total loss")
    axes[0].plot(h.epoch, h.val_score, label="val score 0.5*L1+0.5*(1-SSIM)")
    axes[0].set_title("Loss")
    axes[1].plot(h.epoch, h.val_ssim, label="val SSIM")
    axes[1].plot(h.epoch, h.gate_acc, label="gate top-1 accuracy")
    axes[1].set_title("Validation SSIM / gate accuracy")
    for name in C.CONDITIONS:
        axes[2].plot(h.epoch, h[f"own_w_{name}"], label=f"{name} inputs -> own branch")
    axes[2].set_title("Mean weight on the correct branch (val)")
    for ax in axes:
        ax.axvline(warmup + 0.5, color="grey", linestyle=":", label="end of warm-up")
        ax.set_xlabel("epoch (0 = Task 2 init)")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1", help="v2 = built on the v2 specialists (own config/outputs)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    tag = "" if args.version == "v1" else f"_{args.version}"
    ckpt = C.CHECKPOINTS / f"task3_soft_moe{tag}.pt"
    if ckpt.exists() and not args.overwrite:
        sys.exit(f"{ckpt} already exists. Use --overwrite to replace it.")

    cfg = yaml.safe_load((C.ROOT / "configs" / f"task3{tag}.yaml").read_text())
    tc = cfg["train"]
    device = get_device()
    seed_everything(C.SEED)
    model = build_moe_from_task2(cfg["moe"]["temperature"], expert_tag=tag).to(device)
    setup_mlflow("task3-soft-moe")
    with mlflow.start_run(run_name=f"final-train{tag}"):
        mlflow.log_params({**tc, **cfg["moe"], "config_source": cfg.get("source", "")})
        result = train_moe(model, tc, device, patience=tc["patience"])
        best, history = result["best"], pd.DataFrame(result["history"])
        C.CHECKPOINTS.mkdir(parents=True, exist_ok=True)
        torch.save({"gate_config": model.gate.config, "expert_config": model.experts[0].config,
                    "temperature": cfg["moe"]["temperature"], "train_config": tc, "state_dict": best["state"],
                    "best_epoch": best["epoch"], "best_val": {k: v for k, v in best.items() if k != "state"},
                    "init_val": result["init"], "history": result["history"]}, ckpt)
        history.to_csv(C.RESULTS / f"task3{tag}_history.csv", index=False)
        plot_curves(history, tc["warmup_epochs"], C.FIGURES / f"task3{tag}_curves.png")
        mlflow.log_metrics({"best_epoch": best["epoch"], "best_val_score": best["val_score"],
                            "best_val_ssim": best["val_ssim"], "init_val_score": result["init"]["val_score"]})
        for f in (ckpt, C.FIGURES / f"task3{tag}_curves.png", C.RESULTS / f"task3{tag}_history.csv"):
            mlflow.log_artifact(str(f))
    print(f"init val score {result['init']['val_score']:.4f} -> best epoch {best['epoch']} ({best['stage']}): "
          f"val score {best['val_score']:.4f}, SSIM {best['val_ssim']:.4f}, gate acc {best['gate_acc']:.4f} -> {ckpt}")


if __name__ == "__main__":
    main()
