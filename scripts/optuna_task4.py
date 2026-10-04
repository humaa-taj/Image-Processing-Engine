"""Task 4 Optuna study: style-conditioned face-to-sketch cGAN.

Search space (PDF: G lr, D lr, batch size, base channels, dropout, style-embedding dim, L1 weight):
    lr_g           log-uniform [5e-5, 5e-4]   (pix2pix default 2e-4)
    lr_d           log-uniform [5e-5, 5e-4]
    batch_size     {4, 8, 16}                 (small for the GAN; batch 8 measured at 2.7 GB with base 64)
    base_channels  {32, 48, 64}               generator width (64 = standard pix2pix, 42 M params,
                                              ~168 MB ONNX). No file-size cap: model quality comes first,
                                              files > 50 MB stay local (CLAUDE.md section 7). Configs that
                                              don't fit in 4 GB are pruned by the OOM handler. D fixed at 64
    dropout        uniform [0.0, 0.5]         (pix2pix: 0.5 in the 3 innermost decoder layers)
    emb_dim        {4, 8, 16, 32}             style-embedding size
    lambda_l1      log-uniform [10, 200]      (pix2pix default 100)
Objective (minimize): best val 0.5*L1 + 0.5*(1-SSIM) of generated vs real sketches (159 val pairs).
Short trials (default 20 epochs, PDF allows fewer epochs), MedianPruner, NaN / divergence -> pruned.
Run:  .venv\\Scripts\\python scripts/optuna_task4.py --n-trials 20     (--smoke for a dry run)
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import optuna
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.gan_training import D_BASE, train_gan  # noqa: E402
from src.tracking import (create_study, safe_objective, save_study_report, setup_mlflow,  # noqa: E402
                          study_summary, trial_run)
from src.utils import get_device, seed_everything  # noqa: E402

STUDY = "task4"
SEARCH_SPACE = {"lr_g": "log-uniform [5e-5, 5e-4]", "lr_d": "log-uniform [5e-5, 5e-4]", "batch_size": [4, 8, 16],
                "base_channels": [32, 48, 64], "dropout": "uniform [0.0, 0.5]", "emb_dim": [4, 8, 16, 32],
                "lambda_l1": "log-uniform [10, 200]", "discriminator_base": f"fixed {D_BASE}"}


def suggest(trial: optuna.Trial) -> dict:
    return {"lr_g": trial.suggest_float("lr_g", 5e-5, 5e-4, log=True),
            "lr_d": trial.suggest_float("lr_d", 5e-5, 5e-4, log=True),
            "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
            "base_channels": trial.suggest_categorical("base_channels", SEARCH_SPACE["base_channels"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "emb_dim": trial.suggest_categorical("emb_dim", SEARCH_SPACE["emb_dim"]),
            "lambda_l1": trial.suggest_float("lambda_l1", 10.0, 200.0, log=True)}


def make_objective(epochs: int, device):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial)
        seed_everything(C.SEED)
        with trial_run(trial, {**cfg, "epochs": epochs}):
            best = train_gan(cfg, epochs, device, trial=trial)["best"]
            mlflow.log_metrics({"best_val_score": best["val_score"], "best_val_ssim": best["val_ssim"],
                                "best_val_l1": best["val_l1"], "best_epoch": best["epoch"]})
            for k in ("val_ssim", "val_l1", "epoch"):
                trial.set_user_attr(f"best_{k}", best[k])
            return best["val_score"]
    return objective


def write_config(study: optuna.Study) -> None:
    p = study.best_params
    config = {"model": {"base_channels": p["base_channels"], "emb_dim": p["emb_dim"], "dropout": round(p["dropout"], 4)},
              "train": {"lr_g": float(f"{p['lr_g']:.3g}"), "lr_d": float(f"{p['lr_d']:.3g}"),
                        "batch_size": p["batch_size"], "lambda_l1": round(p["lambda_l1"], 2), "epochs": 150,
                        "checkpoint_every": 10, "sample_every": 10},
              "source": f"Optuna study '{STUDY}', best trial #{study.best_trial.number} "
                        f"(val score {study.best_value:.4f})"}
    (C.ROOT / "configs" / "task4.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--smoke", action="store_true", help="2 trials x 2 epochs, in memory, nothing saved")
    args = ap.parse_args()
    device = get_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow("task4-face-to-sketch")
    log = lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} params={t.params}", flush=True)  # noqa: E731

    if args.smoke:
        study = optuna.create_study(direction="minimize")
        with mlflow.start_run(run_name="smoke-task4"):
            study.optimize(safe_objective(make_objective(2, device)), n_trials=2, callbacks=[log])
        print("smoke OK:", study_summary(study))
        return

    study = create_study(STUDY, n_startup_trials=5, n_warmup_steps=5)
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{STUDY}': {len(study.trials)} trials so far, running {max(remaining, 0)} more")
    with mlflow.start_run(run_name=f"optuna-{STUDY}"):
        mlflow.log_params({"n_trials_target": args.n_trials, "epochs_per_trial": args.epochs,
                           "search_space": json.dumps(SEARCH_SPACE)})
        if remaining > 0:
            study.optimize(safe_objective(make_objective(args.epochs, device)), n_trials=remaining, callbacks=[log])
        summary = save_study_report(study, STUDY, {
            "study": STUDY, "direction": "minimize", "epochs_per_trial": args.epochs,
            "objective": "best-epoch 0.5*val_L1 + 0.5*(1 - val_SSIM), generated vs real sketch (159 val pairs)",
            "search_space": SEARCH_SPACE})
        mlflow.log_metrics({k: summary[k] for k in ("completed", "pruned", "failed")})
    write_config(study)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
