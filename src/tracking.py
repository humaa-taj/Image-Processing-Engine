"""MLflow and Optuna helpers shared by every task.

MLflow: MLflow 3.x refuses the old plain-folder store, so we use a local SQLite database for run
metadata (mlruns/mlflow.db) and a local folder for files (mlruns/artifacts/). Both are gitignored.
View the runs with:
    .venv\\Scripts\\mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db
One MLflow experiment per task; each Optuna trial is a nested run under one parent run per study.

Optuna: one SQLite file per study in artifacts/optuna/, so a study can be stopped and resumed
(load_if_exists=True). The TPE sampler is seeded so the search is repeatable.
"""
import contextlib
import functools
import gc

import mlflow
import optuna
import torch

from src import config as C


def setup_mlflow(experiment: str) -> None:
    """Point MLflow at the local store and select (or create) the experiment for this task."""
    C.MLRUNS.mkdir(exist_ok=True)
    mlflow.set_tracking_uri("sqlite:///" + (C.MLRUNS / "mlflow.db").as_posix())
    if mlflow.get_experiment_by_name(experiment) is None:
        mlflow.create_experiment(experiment, artifact_location=(C.MLRUNS / "artifacts").as_uri())
    mlflow.set_experiment(experiment)


def create_study(name: str, direction: str = "minimize", n_startup_trials: int = 5,
                 n_warmup_steps: int = 2) -> optuna.Study:
    """Persistent study with a seeded TPE sampler and a MedianPruner.

    MedianPruner stops a trial when its intermediate validation score is worse than the median of
    earlier trials at the same epoch. It waits for n_startup_trials finished trials and ignores the
    first n_warmup_steps epochs of each trial (early epochs are too noisy to judge).
    """
    C.OPTUNA_DIR.mkdir(parents=True, exist_ok=True)
    return optuna.create_study(
        study_name=name,
        storage="sqlite:///" + (C.OPTUNA_DIR / f"{name}.db").as_posix(),
        load_if_exists=True,
        direction=direction,
        sampler=optuna.samplers.TPESampler(seed=C.SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, n_warmup_steps=n_warmup_steps),
    )


class TrainingDiverged(Exception):
    """Raise inside a trial when the loss becomes NaN/inf or explodes (mainly for the GAN)."""


def safe_objective(objective):
    """Wrap an Optuna objective so one bad trial doesn't kill the whole study.

    - CUDA out-of-memory (config too big for 4 GB) -> free GPU memory, mark the trial as pruned.
    - TrainingDiverged                             -> mark the trial as pruned.
    """
    @functools.wraps(objective)
    def wrapper(trial: optuna.Trial):
        try:
            return objective(trial)
        except torch.cuda.OutOfMemoryError:
            trial.set_user_attr("pruned_reason", "cuda_oom")
            reason = "CUDA out of memory"
        except TrainingDiverged as e:
            trial.set_user_attr("pruned_reason", f"diverged: {e}")
            reason = f"diverged: {e}"
        # Outside the except block, so the exception (and the tensors it references) are released.
        gc.collect()
        torch.cuda.empty_cache()
        raise optuna.TrialPruned(reason)
    return wrapper


def study_summary(study: optuna.Study) -> dict:
    """Counts of completed / pruned / failed trials and the best trial (for the report)."""
    states = optuna.trial.TrialState
    count = lambda s: sum(t.state == s for t in study.trials)  # noqa: E731
    out = {"completed": count(states.COMPLETE), "pruned": count(states.PRUNED),
           "failed": count(states.FAIL), "total": len(study.trials)}
    if out["completed"]:
        out.update(best_value=study.best_value, best_params=study.best_params,
                   best_trial=study.best_trial.number)
    return out


@contextlib.contextmanager
def trial_run(trial: optuna.Trial, params: dict):
    """Nested MLflow run for one Optuna trial (call inside the study's parent run).

    Pruned / OOM / diverged trials end with MLflow status KILLED and a tag saying why, instead of
    looking like crashes (FAILED).
    """
    run = mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True)
    mlflow.log_params(params)
    mlflow.set_tag("optuna_trial", trial.number)
    try:
        yield run
    except optuna.TrialPruned as e:
        mlflow.set_tag("optuna_state", f"PRUNED: {e}")
        mlflow.end_run(status="KILLED")
        raise
    except (torch.cuda.OutOfMemoryError, TrainingDiverged) as e:
        mlflow.set_tag("optuna_state", f"PRUNED: {type(e).__name__}")
        mlflow.end_run(status="KILLED")
        raise
    except Exception:
        mlflow.set_tag("optuna_state", "FAIL")
        mlflow.end_run(status="FAILED")
        raise
    else:
        mlflow.set_tag("optuna_state", "COMPLETE")
        mlflow.end_run()


def save_study_report(study: optuna.Study, prefix: str, extra: dict) -> dict:
    """Write the study summary (JSON), all trials (CSV) and two plots for the report.

    -> artifacts/results/<prefix>_optuna_summary.json, <prefix>_optuna_trials.csv
    -> docs/figures/<prefix>_optuna_history.png, <prefix>_optuna_importance.png
    """
    import json

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary = {**extra, **study_summary(study)}
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / f"{prefix}_optuna_summary.json").write_text(json.dumps(summary, indent=2))
    study.trials_dataframe().to_csv(C.RESULTS / f"{prefix}_optuna_trials.csv", index=False)
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    for name, plot in [("history", optuna.visualization.matplotlib.plot_optimization_history),
                       ("importance", optuna.visualization.matplotlib.plot_param_importances)]:
        try:
            ax = plot(study)
            ax.figure.set_size_inches(8, 5)
            ax.figure.tight_layout()
            ax.figure.savefig(C.FIGURES / f"{prefix}_optuna_{name}.png", dpi=120)
        except Exception as e:  # e.g. importance needs >1 completed trial; never fail a study over a plot
            print(f"could not draw {name} plot: {e}")
        plt.close("all")
    return summary
