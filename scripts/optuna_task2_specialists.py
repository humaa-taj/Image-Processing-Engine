"""Task 2 Optuna study: ONE shared architecture search for the three specialist autoencoders.

The PDF allows a shared search to keep the cost feasible. Each trial trains three independent
models with the SAME hyperparameters: salt -> blur -> occlusion, each only on its own corruption,
each validated only on the val-manifest images of its own corruption (184 each).

Search space (PDF: learning rate, bottleneck size, channel configuration, batch size, L1/SSIM weight):
    lr             log-uniform [1e-4, 3e-3]
    batch_size     {16, 32, 64}
    bottleneck_dim {512, 1024, 2048}
    base_channels  {24, 32, 40}     capped so that Task 3's single ONNX file with all three experts
                                    stays under 50 MB (3 x 11.2 MB at base 40)
    alpha          uniform [0.5, 0.95]
    dropout        fixed 0          (not required by the PDF here; Task 1 found ~0 best)
Objective (minimize): mean over the three specialists of their best 0.5*L1 + 0.5*(1-SSIM) score.
Pruning: the three models are reported as consecutive steps (salt = steps 1-10, blur = 11-20,
occlusion = 21-30), so the MedianPruner always compares trials at the same specialist and epoch.

Run:  .venv\\Scripts\\python scripts/optuna_task2_specialists.py --n-trials 20     (--smoke for a dry run)
Outputs: artifacts/optuna/task2_specialists.db, artifacts/results/task2_specialists_optuna_*,
         docs/figures/task2_specialists_optuna_*.png, configs/task2_specialists.yaml
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
from src.models.autoencoder import ConvAutoencoder, count_parameters  # noqa: E402
from src.tracking import (create_study, safe_objective, save_study_report, setup_mlflow,  # noqa: E402
                          study_summary, trial_run)
from src.training import train_restoration  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

EXPERTS = {"salt": C.SALT, "blur": C.BLUR, "occlusion": C.OCCLUSION}
# v1 capped base channels at 40 for the old 50 MB file rule; its best trial hit that cap.
# v2 (upgrade pass, 2026-10-04) removes the cap: no file-size limit on models any more (CLAUDE.md s.7).
SPACES = {
    "v1": {"lr": "log-uniform [1e-4, 3e-3]", "batch_size": [16, 32, 64], "bottleneck_dim": [512, 1024, 2048],
           "base_channels": [24, 32, 40], "alpha": "uniform [0.5, 0.95]", "dropout": "fixed 0.0"},
    "v2": {"lr": "log-uniform [1e-4, 3e-3]", "batch_size": [16, 32, 64], "bottleneck_dim": [1024, 2048, 4096],
           "base_channels": [40, 64, 96], "alpha": "uniform [0.5, 0.95]", "dropout": "fixed 0.0"},
}
STUDY, SEARCH_SPACE = "task2_specialists", SPACES["v1"]    # set from --version in main()


def suggest(trial: optuna.Trial) -> dict:
    return {"lr": trial.suggest_float("lr", 1e-4, 3e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", SEARCH_SPACE["batch_size"]),
            "bottleneck_dim": trial.suggest_categorical("bottleneck_dim", SEARCH_SPACE["bottleneck_dim"]),
            "base_channels": trial.suggest_categorical("base_channels", SEARCH_SPACE["base_channels"]),
            "alpha": trial.suggest_float("alpha", 0.5, 0.95)}


def make_objective(epochs: int, device):
    def objective(trial: optuna.Trial) -> float:
        cfg = suggest(trial)
        with trial_run(trial, {**cfg, "epochs_per_expert": epochs}):
            scores = {}
            for k, (name, cond) in enumerate(EXPERTS.items()):
                seed_everything(C.SEED)
                model = ConvAutoencoder(cfg["base_channels"], cfg["bottleneck_dim"], 0.0).to(device)
                best = train_restoration(model, cfg, epochs, device, conditions=(cond,), trial=trial,
                                         step_offset=k * epochs, metric_prefix=f"{name}/")["best"]
                scores[name] = best["val_score"]
                trial.set_user_attr(f"{name}_val_score", best["val_score"])
                trial.set_user_attr(f"{name}_val_ssim", best["val_ssim"])
                mlflow.log_metrics({f"{name}/best_val_score": best["val_score"],
                                    f"{name}/best_val_ssim": best["val_ssim"]})
            mean = sum(scores.values()) / len(scores)
            mlflow.log_metrics({"mean_best_val_score": mean, "params_M": count_parameters(model) / 1e6})
            return mean
    return objective


def write_config(study: optuna.Study) -> None:
    p = study.best_params
    config = {"model": {"base_channels": p["base_channels"], "bottleneck_dim": p["bottleneck_dim"], "dropout": 0.0},
              "train": {"lr": float(f"{p['lr']:.3g}"), "batch_size": p["batch_size"], "alpha": round(p["alpha"], 4),
                        "weight_decay": 1e-5, "epochs": 100, "patience": 20},
              "source": f"Optuna study '{STUDY}', best trial #{study.best_trial.number} "
                        f"(mean val score {study.best_value:.4f})"}
    (C.ROOT / "configs" / f"{STUDY}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=10, help="epochs per specialist per trial")
    ap.add_argument("--smoke", action="store_true", help="2 trials x 1 epoch per expert, in memory, nothing saved")
    ap.add_argument("--version", choices=list(SPACES), default="v1", help="v2 = upgrade pass (own study/outputs)")
    args = ap.parse_args()
    global STUDY, SEARCH_SPACE
    STUDY = "task2_specialists" if args.version == "v1" else f"task2_specialists_{args.version}"
    SEARCH_SPACE = SPACES[args.version]
    device = get_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow("task2-hard-routing")
    log = lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} params={t.params} "  # noqa: E731
                             f"per-expert={ {k: round(v, 4) for k, v in t.user_attrs.items() if k.endswith('score')} }",
                             flush=True)
    if args.smoke:
        study = optuna.create_study(direction="minimize")
        with mlflow.start_run(run_name="smoke-task2-specialists"):
            study.optimize(safe_objective(make_objective(1, device)), n_trials=2, callbacks=[log])
        print("smoke OK:", study_summary(study))
        return

    study = create_study(STUDY, n_startup_trials=5, n_warmup_steps=3)
    if args.version != "v1" and not study.trials:
        # Start v2 by re-running v1's best settings: v2 can then only match or beat v1 (on validation).
        v1 = optuna.load_study(study_name="task2_specialists",
                               storage=f"sqlite:///{(C.OPTUNA_DIR / 'task2_specialists.db').as_posix()}")
        study.enqueue_trial(v1.best_params)
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{STUDY}': {len(study.trials)} trials so far, running {max(remaining, 0)} more")
    with mlflow.start_run(run_name=f"optuna-{STUDY}"):
        mlflow.log_params({"n_trials_target": args.n_trials, "epochs_per_expert": args.epochs,
                           "search_space": json.dumps(SEARCH_SPACE)})
        if remaining > 0:
            study.optimize(safe_objective(make_objective(args.epochs, device)), n_trials=remaining, callbacks=[log])
        summary = save_study_report(study, STUDY, {
            "study": STUDY, "direction": "minimize", "epochs_per_expert": args.epochs,
            "objective": "mean over salt/blur/occlusion specialists of best 0.5*val_L1 + 0.5*(1 - val_SSIM)",
            "search_space": SEARCH_SPACE})
        mlflow.log_metrics({k: summary[k] for k in ("completed", "pruned", "failed")})
    write_config(study)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
