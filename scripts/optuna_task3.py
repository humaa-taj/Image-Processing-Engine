"""Task 3 Optuna study: soft mixture-of-experts fine-tuning.

Every trial starts from the SAME Task 2 weights (gate <- classifier, experts <- specialists).
Search space (PDF: fine-tune learning rate, temperature, classification weight, balance weight,
reconstruction weighting):
    lr            log-uniform [1e-5, 3e-4]   joint fine-tuning lr (smaller than the specialists' 2e-3)
    temperature   uniform [0.5, 3.0]         T in softmax(logits / T)
    cls_w         log-uniform [0.01, 1.0]    weight of the cross-entropy term (start 0.1)
    bal_w         log-uniform [0.001, 0.1]   weight of the balance term (start 0.01)
    l1_share      uniform [0.5, 0.95]        reconstruction weighting: l1_w = l1_share, ssim_w = 1 - l1_share
                                             (start 0.8 / 0.2)
Fixed: batch 32 (balanced), warm-up 2 epochs (gate only, lr 5e-4), then 6 joint epochs.
Objective (minimize): best val 0.5*L1 + 0.5*(1-SSIM) -- the same fixed yardstick as Tasks 1-2.
Pruning: MedianPruner, plus routing collapse (a branch's mean weight on the balanced val set < 0.05
or > 0.5), OOM, NaN.
Run:  .venv\\Scripts\\python scripts/optuna_task3.py --n-trials 20     (--smoke for a dry run)
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
from src.onnx_utils import build_moe_from_task2  # noqa: E402
from src.tracking import (create_study, safe_objective, save_study_report, setup_mlflow,  # noqa: E402
                          study_summary, trial_run)
from src.training import train_moe  # noqa: E402
from src.utils import get_device, seed_everything  # noqa: E402

STUDY, TAG = "task3", ""   # set from --version in main(); v2 = built on the v2 specialists
SEARCH_SPACE = {"lr": "log-uniform [1e-5, 3e-4]", "temperature": "uniform [0.5, 3.0]",
                "cls_w": "log-uniform [0.01, 1.0]", "bal_w": "log-uniform [0.001, 0.1]",
                "l1_share": "uniform [0.5, 0.95] (l1_w = l1_share, ssim_w = 1 - l1_share)"}
FIXED = {"batch_size": 32, "warmup_epochs": 2, "warmup_lr": 5e-4}


def suggest(trial: optuna.Trial) -> dict:
    l1_share = trial.suggest_float("l1_share", 0.5, 0.95)
    return {"lr": trial.suggest_float("lr", 1e-5, 3e-4, log=True),
            "temperature": trial.suggest_float("temperature", 0.5, 3.0),
            "cls_w": trial.suggest_float("cls_w", 0.01, 1.0, log=True),
            "bal_w": trial.suggest_float("bal_w", 0.001, 0.1, log=True),
            "l1_w": l1_share, "ssim_w": 1 - l1_share}


def make_objective(joint_epochs: int, device):
    def objective(trial: optuna.Trial) -> float:
        cfg = {**suggest(trial), **FIXED, "joint_epochs": joint_epochs}
        seed_everything(C.SEED)
        model = build_moe_from_task2(cfg["temperature"], expert_tag=TAG).to(device)
        with trial_run(trial, cfg):
            result = train_moe(model, cfg, device, trial=trial)
            best = result["best"]
            mlflow.log_metrics({"best_val_score": best["val_score"], "best_val_ssim": best["val_ssim"],
                                "best_val_psnr": best["val_psnr"], "best_epoch": best["epoch"],
                                "init_val_score": result["init"]["val_score"]})
            for k in ("val_ssim", "val_psnr", "gate_acc", "epoch", "min_branch_w", "max_branch_w"):
                trial.set_user_attr(f"best_{k}", best[k])
            return best["val_score"]
    return objective


def write_config(study: optuna.Study) -> None:
    p = study.best_params
    config = {"moe": {"temperature": round(p["temperature"], 4)},
              "train": {"lr": float(f"{p['lr']:.3g}"), "l1_w": round(p["l1_share"], 4),
                        "ssim_w": round(1 - p["l1_share"], 4), "cls_w": float(f"{p['cls_w']:.3g}"),
                        "bal_w": float(f"{p['bal_w']:.3g}"), **FIXED, "warmup_epochs": 3, "joint_epochs": 30,
                        "patience": 8},
              "source": f"Optuna study '{STUDY}', best trial #{study.best_trial.number} "
                        f"(val score {study.best_value:.4f})"}
    (C.ROOT / "configs" / f"{STUDY}.yaml").write_text(yaml.safe_dump(config, sort_keys=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--joint-epochs", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="2 trials, 1 joint epoch, in memory, nothing saved")
    ap.add_argument("--version", default="v1", help="v2 = start from the v2 specialists (own study/outputs)")
    args = ap.parse_args()
    global STUDY, TAG
    TAG = "" if args.version == "v1" else f"_{args.version}"
    STUDY = f"task3{TAG}"
    device = get_device()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    setup_mlflow("task3-soft-moe")
    log = lambda s, t: print(f"trial {t.number}: {t.state.name} value={t.value} params={t.params}", flush=True)  # noqa: E731

    if args.smoke:
        study = optuna.create_study(direction="minimize")
        with mlflow.start_run(run_name="smoke-task3"):
            study.optimize(safe_objective(make_objective(1, device)), n_trials=2, callbacks=[log])
        print("smoke OK:", study_summary(study))
        return

    study = create_study(STUDY, n_startup_trials=5, n_warmup_steps=3)
    remaining = args.n_trials - len([t for t in study.trials if t.state.is_finished()])
    print(f"study '{STUDY}': {len(study.trials)} trials so far, running {max(remaining, 0)} more")
    with mlflow.start_run(run_name=f"optuna-{STUDY}"):
        mlflow.log_params({"n_trials_target": args.n_trials, "joint_epochs_per_trial": args.joint_epochs,
                           "search_space": json.dumps(SEARCH_SPACE), "fixed": json.dumps(FIXED)})
        if remaining > 0:
            study.optimize(safe_objective(make_objective(args.joint_epochs, device)), n_trials=remaining,
                           callbacks=[log])
        summary = save_study_report(study, STUDY, {
            "study": STUDY, "direction": "minimize", "joint_epochs_per_trial": args.joint_epochs, "fixed": FIXED,
            "objective": "best-epoch 0.5*val_L1 + 0.5*(1 - val_SSIM) on the balanced val manifest",
            "collapse_rule": "prune if a branch's mean val weight < 0.05 or > 0.5", "search_space": SEARCH_SPACE})
        mlflow.log_metrics({k: summary[k] for k in ("completed", "pruned", "failed")})
    write_config(study)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
