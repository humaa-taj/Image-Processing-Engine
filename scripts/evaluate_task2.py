"""Task 2 final evaluation on the OFFICIAL TEST manifest (run once, at the end).

1. Classifier: accuracy, macro precision/recall/F1, per-class metrics, normalized 4x4 confusion
   matrix, and accuracy per corruption x severity (where do mistakes happen?).
2. Hard-routed restoration in two modes, per corruption x severity:
     oracle    : route = true label from the manifest  (how good are the specialists?)
     predicted : route = classifier argmax              (how good is the whole system?)
   Clean routes always use the identity bypass. Compared with Task 1 on the same inputs.
3. Misrouting analysis: every test input whose predicted route differs from the true label, how
   much PSNR that cost, and a figure of the worst cases.
Outputs (artifacts/results/ and docs/figures/), all prefixed task2_. Logged to MLflow.
Run:  .venv\\Scripts\\python scripts/evaluate_task2.py          (--dry-run: 40 VALIDATION images)
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_task1 import ERROR_VMAX, get_dataset, latex, pick_examples, summarize  # noqa: E402
from src import config as C  # noqa: E402
from src.corruptions import severity_value  # noqa: E402
from src.metrics import per_image_metrics  # noqa: E402
from src.onnx_utils import load_autoencoder, load_classifier  # noqa: E402
from src.routing import BRANCH_NAMES, hard_route  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device  # noqa: E402

PREFIX = "task2"
EXPERT_CONDS = {"salt": C.SALT, "blur": C.BLUR, "occlusion": C.OCCLUSION}


def load_models(device, ckpt_dir: Path = C.CHECKPOINTS, tag: str = ""):
    classifier = load_classifier(ckpt_dir / "task2_classifier.pt").to(device)    # same classifier in v1 and v2
    experts = {cond: load_autoencoder(ckpt_dir / f"task2_expert_{name}{tag}.pt").to(device)
               for name, cond in EXPERT_CONDS.items()}
    return classifier, experts


@torch.no_grad()
def run_test(classifier, experts, device, ds) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=128, num_workers=2)
    rows = []
    for x, y, cond, idx in loader:
        xg = x.to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            probs = torch.softmax(classifier(xg).float(), 1)
            pred = probs.argmax(1)
            oracle_out = hard_route(xg, cond.to(device), experts).float().cpu()
            pred_out = hard_route(xg, pred, experts).float().cpu()
        m_in, m_or, m_pr = per_image_metrics(x, y), per_image_metrics(oracle_out, y), per_image_metrics(pred_out, y)
        probs, pred = probs.cpu(), pred.cpu()
        for j in range(len(idx)):
            e = ds.entries[int(idx[j])]
            row = {"entry": int(idx[j]), "id": e["id"], "condition": C.CONDITIONS[e["condition"]],
                   "level": e["level"], "severity": severity_value(e["params"]), "true": e["condition"],
                   "pred": int(pred[j]), **{f"p_{c}": float(probs[j, k]) for k, c in enumerate(C.CONDITIONS)}}
            for tag, m in (("input", m_in), ("oracle", m_or), ("pred", m_pr)):
                row.update({f"{tag}_{k}": m[k][j].item() for k in ("psnr", "ssim", "l1")})
            rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[df.condition == "clean", ["input_psnr", "oracle_psnr"]] = np.inf  # identity on clean = perfect
    df.loc[(df.condition == "clean") & (df.pred == C.CLEAN), "pred_psnr"] = np.inf
    return df


# ------------------------------------------------------------------ classifier metrics

def classifier_report(df: pd.DataFrame, fig_path: Path) -> dict:
    y, p = df.true.values, df.pred.values
    rep = classification_report(y, p, labels=range(4), target_names=C.CONDITIONS, output_dict=True, zero_division=0)
    cm = confusion_matrix(y, p, labels=range(4), normalize="true")  # rows = true class, sum to 1
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]:.3f}", ha="center", va="center", color="white" if cm[i, j] > 0.5 else "black")
    ax.set_xticks(range(4), C.CONDITIONS)
    ax.set_yticks(range(4), C.CONDITIONS)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Task 2 classifier, normalized confusion matrix (test)")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    per_level = (df.assign(correct=df.true == df.pred).groupby(["condition", "level"]).correct.mean()
                 .rename("accuracy").reset_index())
    return {"accuracy": rep["accuracy"],
            "macro_precision": rep["macro avg"]["precision"], "macro_recall": rep["macro avg"]["recall"],
            "macro_f1": rep["macro avg"]["f1-score"],
            "per_class": {c: {k: rep[c][k] for k in ("precision", "recall", "f1-score", "support")}
                          for c in C.CONDITIONS},
            "confusion_matrix_normalized": cm.round(4).tolist(),
            "accuracy_per_condition_level": per_level.to_dict("records")}


# ------------------------------------------------------------------ restoration tables

def mode_frame(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Rename <mode>_psnr -> psnr etc. so the Task 1 summarize() can be reused."""
    out = df[["entry", "condition", "level", "input_psnr", "input_ssim", "input_l1"]].copy()
    for k in ("psnr", "ssim", "l1"):
        out[k] = df[f"{mode}_{k}"]
    return out


def comparison_table(oracle: pd.DataFrame, pred: pd.DataFrame) -> pd.DataFrame:
    keep = ["condition", "level", "n", "input_psnr", "input_ssim"]
    t = oracle[keep].copy()
    t["oracle_psnr"], t["oracle_ssim"] = oracle.psnr, oracle.ssim
    t["pred_psnr"], t["pred_ssim"] = pred.psnr, pred.ssim
    task1 = C.RESULTS / "task1_test_summary.csv"
    if task1.exists():  # same test inputs -> direct comparison with the universal autoencoder
        t1 = pd.read_csv(task1)[["condition", "level", "psnr", "ssim"]].rename(
            columns={"psnr": "task1_psnr", "ssim": "task1_ssim"})
        t = t.merge(t1, on=["condition", "level"], how="left")
    return t


def finite_mean(s: pd.Series) -> float:
    return float(s.replace(np.inf, np.nan).mean())


def comparison_latex(t: pd.DataFrame) -> str:
    fmt = lambda v, d: "$\\infty$" if np.isinf(v) else f"{v:.{d}f}"  # noqa: E731
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\hline",
             r"Cond. & Level & \multicolumn{2}{c}{Input} & \multicolumn{2}{c}{Task 1} & "
             r"\multicolumn{2}{c}{Oracle} & \multicolumn{2}{c}{Predicted} \\",
             r" & & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\", r"\hline"]
    for r in t.itertuples():
        lines.append(f"{r.condition} & {r.level} & {fmt(r.input_psnr, 2)} & {r.input_ssim:.3f} & "
                     f"{fmt(r.task1_psnr, 2)} & {r.task1_ssim:.3f} & {fmt(r.oracle_psnr, 2)} & {r.oracle_ssim:.3f} & "
                     f"{fmt(r.pred_psnr, 2)} & {r.pred_ssim:.3f} \\\\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------ misrouting

def misrouting(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    wrong = df[df.true != df.pred].copy()
    # PSNR cost of the wrong route. If a clean image is wrongly sent to an expert, the oracle (identity)
    # PSNR is infinite, so we measure that cost in SSIM and L1 as well.
    wrong["ssim_drop"] = wrong.oracle_ssim - wrong.pred_ssim
    wrong["l1_increase"] = wrong.pred_l1 - wrong.oracle_l1
    wrong["psnr_drop"] = (wrong.oracle_psnr - wrong.pred_psnr).replace([np.inf, -np.inf], np.nan)
    pairs = (wrong.groupby(["condition", "pred"]).agg(n=("entry", "size"), mean_ssim_drop=("ssim_drop", "mean"),
                                                     mean_l1_increase=("l1_increase", "mean"))
             .reset_index())
    pairs["pred"] = pairs.pred.map(lambda k: C.CONDITIONS[k])
    summary = {"n_test_inputs": len(df), "n_misrouted": len(wrong), "misrouted_fraction": len(wrong) / len(df),
               "mean_ssim_drop_when_misrouted": float(wrong.ssim_drop.mean()) if len(wrong) else 0.0,
               "mean_psnr_drop_when_misrouted_finite": float(wrong.psnr_drop.mean()) if len(wrong) else 0.0,
               "overall_ssim_cost_vs_oracle": float(df.oracle_ssim.mean() - df.pred_ssim.mean()),
               "by_true_and_predicted": pairs.to_dict("records")}
    return summary, wrong.sort_values("ssim_drop", ascending=False)


# ------------------------------------------------------------------ figures

def pick_examples_task2(df: pd.DataFrame) -> list[int]:
    """9 corrupted examples = median predicted-route SSIM of each corruption x severity group (as in Task 1).
    3 clean examples = 25th / 50th / 75th percentile of the classifier's p_clean, because almost every
    clean input scores SSIM 1.0 under the identity route (so SSIM percentiles would all pick one image)."""
    pdf = mode_frame(df, "pred")
    picks = pick_examples(pdf)[:9]
    clean = df[df.condition == "clean"].sort_values("p_clean").reset_index(drop=True)
    picks += [int(clean.loc[int(q * (len(clean) - 1)), "entry"]) for q in (0.25, 0.5, 0.75)]
    return picks


@torch.no_grad()
def draw_routed(entries, df, classifier, experts, device, ds, title, path, per_row=2, show_oracle=False):
    """Rows: clean target | input (+ probabilities) | routed output | [oracle output] | |error|."""
    cols = 5 if show_oracle else 4
    rows = int(np.ceil(len(entries) / per_row))
    fig, axes = plt.subplots(rows, cols * per_row, figsize=(2.3 * cols * per_row, 2.5 * rows + 0.6), squeeze=False)
    d = df.set_index("entry")
    for k, entry in enumerate(entries):
        x, y, cond, _ = ds[entry]
        r = d.loc[entry]
        xg = x[None].to(device)
        pred_out = hard_route(xg, torch.tensor([int(r.pred)], device=device), experts)[0].float().cpu()
        oracle_out = hard_route(xg, torch.tensor([cond], device=device), experts)[0].float().cpu()
        probs = " ".join(f"{c[:4]} {r[f'p_{c}']:.2f}" for c in C.CONDITIONS)
        panels = [(y, "clean target"), (x, f"{r.condition} {r.level}\n{probs}"),
                  (pred_out, f"route: {BRANCH_NAMES[int(r.pred)]}\n{r.pred_ssim:.3f} SSIM")]
        if show_oracle:
            panels.append((oracle_out, f"oracle: {BRANCH_NAMES[int(r.true)]}\n{r.oracle_ssim:.3f} SSIM"))
        panels.append((None, "|error| (routed)"))
        r0, c0 = divmod(k, per_row)
        for j, (img, label) in enumerate(panels):
            ax = axes[r0, cols * c0 + j]
            if img is None:
                im = ax.imshow((pred_out - y).abs().mean(0).numpy(), cmap="magma", vmin=0, vmax=ERROR_VMAX)
            else:
                ax.imshow(img.clamp(0, 1).permute(1, 2, 0).numpy())
            ax.set_title(label, fontsize=7)
            ax.axis("off")
    for ax in axes.flat[len(entries) * cols:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.01, label="mean abs error")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check the script on 40 VALIDATION images")
    ap.add_argument("--checkpoint-dir", default=str(C.CHECKPOINTS))
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the example figures from the saved per-image results (no re-evaluation)")
    ap.add_argument("--version", default="v1", help="v2 = evaluate the upgrade-pass models (own output files)")
    args = ap.parse_args()
    global PREFIX
    tag = "" if args.version == "v1" else f"_{args.version}"
    PREFIX = f"task2{tag}"
    out_dir = C.ARTIFACTS / "logs" / "dryrun" if args.dry_run else C.RESULTS
    fig_dir = out_dir if args.dry_run else C.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image = out_dir / f"{PREFIX}_test_per_image.csv"
    if args.figures_only:
        device = get_device()
        classifier, experts = load_models(device, Path(args.checkpoint_dir), tag)
        df, ds = pd.read_csv(per_image), get_dataset(args.dry_run)
        draw_routed(pick_examples_task2(df), df, classifier, experts, device, ds,
                    "Task 2 (predicted routing): representative test examples", fig_dir / f"{PREFIX}_examples.png")
        print("redrew", fig_dir / f"{PREFIX}_examples.png")
        return
    if per_image.exists() and not (args.force or args.dry_run):
        sys.exit(f"{per_image} exists: the test set was already evaluated. Use --force to re-run.")

    device = get_device()
    classifier, experts = load_models(device, Path(args.checkpoint_dir), tag)
    ds = get_dataset(args.dry_run)
    df = run_test(classifier, experts, device, ds)
    df.to_csv(per_image, index=False)

    cls = classifier_report(df, fig_dir / f"{PREFIX}_confusion_matrix.png")
    (out_dir / f"{PREFIX}_classifier_metrics.json").write_text(json.dumps(cls, indent=2))

    oracle, pred = summarize(mode_frame(df, "oracle")), summarize(mode_frame(df, "pred"))
    oracle.to_csv(out_dir / f"{PREFIX}_oracle_summary.csv", index=False)
    pred.to_csv(out_dir / f"{PREFIX}_predicted_summary.csv", index=False)
    (out_dir / f"{PREFIX}_oracle_table.tex").write_text(latex(oracle))
    (out_dir / f"{PREFIX}_predicted_table.tex").write_text(latex(pred))
    comp = comparison_table(oracle, pred)
    comp.to_csv(out_dir / f"{PREFIX}_comparison.csv", index=False)
    if "task1_psnr" in comp:
        (out_dir / f"{PREFIX}_comparison_table.tex").write_text(comparison_latex(comp))

    mis, wrong = misrouting(df)
    (out_dir / f"{PREFIX}_misrouting.json").write_text(json.dumps(mis, indent=2))
    wrong.to_csv(out_dir / f"{PREFIX}_misrouted_inputs.csv", index=False)

    draw_routed(pick_examples_task2(df), df, classifier, experts, device, ds,
                "Task 2 (predicted routing): representative test examples", fig_dir / f"{PREFIX}_examples.png")
    if len(wrong):
        draw_routed(wrong.entry.head(6).tolist(), df, classifier, experts, device, ds,
                    "Task 2: worst misrouted inputs (classifier error -> wrong branch)",
                    fig_dir / f"{PREFIX}_misrouted.png", show_oracle=True)

    pd.set_option("display.width", 220)
    print(json.dumps({k: v for k, v in cls.items() if k in ("accuracy", "macro_precision", "macro_recall",
                                                             "macro_f1")}, indent=2))
    print(comp.round(4).to_string(index=False))
    print(json.dumps({k: v for k, v in mis.items() if k != "by_true_and_predicted"}, indent=2))
    if args.dry_run:
        return

    setup_mlflow("task2-hard-routing")
    with mlflow.start_run(run_name=f"test-evaluation-{PREFIX}"):
        mlflow.log_metrics({f"test_cls_{k}": cls[k] for k in ("accuracy", "macro_precision", "macro_recall", "macro_f1")})
        mlflow.log_metrics({"test_misrouted_fraction": mis["misrouted_fraction"],
                            "test_oracle_ssim": float(df.oracle_ssim.mean()), "test_pred_ssim": float(df.pred_ssim.mean()),
                            "test_oracle_psnr_corrupted": finite_mean(df[df.condition != "clean"].oracle_psnr),
                            "test_pred_psnr_corrupted": finite_mean(df[df.condition != "clean"].pred_psnr)})
        for f in sorted(out_dir.glob(f"{PREFIX}_*")) + sorted(fig_dir.glob(f"{PREFIX}_*.png")):
            if f.name != per_image.name:
                mlflow.log_artifact(str(f))


if __name__ == "__main__":
    main()
