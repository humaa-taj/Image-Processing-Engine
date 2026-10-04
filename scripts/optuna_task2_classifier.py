"""Task 2 Optuna study: corruption classifier.

Search space (PDF: learning rate, batch size, channel configuration, dropout, weight decay):
    lr             log-uniform [1e-4, 3e-3]
    batch_size     {32, 64, 128}          (multiples of 4 for exactly balanced batches; 128 peaks ~2.1 GB)
    base_channels  {16, 24, 32, 48}       (stage widths b, 2b, 4b, 8b)
    dropout        uniform [0.0, 0.5]     (before the final linear layer)
    weight_decay   log-uniform [1e-6, 1e-2]
Objective (minimize): best validation cross-entropy over the trial's epochs (calibrated probabilities
matter because this network becomes the Task 3 gate). Accuracy / macro-F1 are logged too.
Short trials (default 12 epochs), MedianPruner, OOM/NaN -> pruned. Resumable.

Run:  .venv\\Scripts\\python scripts/optuna_task2_classifier.py --n-trials 25     (--smoke for a dry run)
Outputs: artifacts/optuna/task2_classifier.db, artifacts/results/task2_classifier_optuna_*,
         docs/figures/task2_classifier_optuna_*.png, configs/task2_classifier.yaml
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
from src.models.autoencoder import count_parameters  # noqa: E402
from src.models.classifier import CorruptionClassifier  # noqa: E402
from src.tracking import (create_study, safe_objective, save_study_report, setup_mlflow,  # noqa: E402
                          study_summary, trial_run)
from src.training import train_classifier  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

STUDY = "task2_classifier"
SEARCH_SPACE = {"lr": "log-uniform [1e-4, 3e-3]", "batch_size": [32, 64, 128], "base_channels": [16, 24, 32, 48],
                "dropout": "uniform [0.0, 0.5]", "weight_decay": "log-uniform [1e-6, 1e-2]"}


def suggest(trial: optuna.Trial) -> dict:
    return {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
            "base_channels": trial.suggest_categorical("base_channels", SEARCH_SPACE["base_channels"]),
            "dropout": trial.suggest_float("dropout", 0.0, 0.5),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)}


def make_objective(epochs: int, device):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial)
        seed_everything(C.SEED)
        model = CorruptionClassifier(cfg["base_channels"], cfg["dropout"]).to(device)
        with trial_run(trial, {**cfg, "epochs": epochs, "params_M": count_parameters(model) / 1e6}):
            best = train_classifier(model, cfg, epochs, device, trial=trial)["best"]
            mlflow.log_metrics({"best_val_loss": best["val_loss"], "best_val_acc": best["val_acc"],
                                "best_val_macro_f1": best["val_macro_f1"], "best_epoch": best["epoch"]})
            for k in ("val_acc", "val_macro_f1", "epoch"):
                trial.set_user_attr(f"best_{k}", best[k])
            return best["val_loss"]
    return objective


def write_config(study: optuna.Study) -> None:
    p = study.best_params
    config = {"model": {"base_channels": p["base_channels"], "dropout": round(p["dropout"], 4)},
              "train": {"lr": float(f"{p['lr']:.3g}"), "batch_size": p["batch_size"],
                        "weight_decay": float(f"{p['weight_decay']:.3g}"), "epochs": 40, "patience": 10},
              "source": f"Optuna study '{STUDY}', best trial #{study.best_trial.number} "
                        f"(val cross-entropy {study.best_value:.4f})"}
    (C.ROOT / "configs" / "task2_classifier.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=25)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--smoke", action="store_true", help="2 trials x 2 epochs, in memory, nothing saved")
    args = ap.parse_args()
    device = get_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow("task2-hard-routing")
    log = lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} params={t.params}", flush=True)  # noqa: E731

    if args.smoke:
        study = optuna.create_study(direction="minimize")
        with mlflow.start_run(run_name="smoke-task2-classifier"):
            study.optimize(safe_objective(make_objective(2, device)), n_trials=2, callbacks=[log])
        print("smoke OK:", study_summary(study))
        return

    study = create_study(STUDY, n_startup_trials=5, n_warmup_steps=3)
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{STUDY}': {len(study.trials)} trials so far, running {max(remaining, 0)} more")
    with mlflow.start_run(run_name=f"optuna-{STUDY}"):
        mlflow.log_params({"n_trials_target": args.n_trials, "epochs_per_trial": args.epochs,
                           "search_space": json.dumps(SEARCH_SPACE)})
        if remaining > 0:
            study.optimize(safe_objective(make_objective(args.epochs, device)), n_trials=remaining, callbacks=[log])
        summary = save_study_report(study, STUDY, {
            "study": STUDY, "direction": "minimize", "epochs_per_trial": args.epochs,
            "objective": "best-epoch validation cross-entropy (val manifest, 736 images)",
            "search_space": SEARCH_SPACE})
        mlflow.log_metrics({k: summary[k] for k in ("completed", "pruned", "failed")})
    write_config(study)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
