"""Task 3 final evaluation on the OFFICIAL TEST manifest (run once, at the end).

1. Restoration per corruption x severity, compared with Task 1 and Task 2 (oracle / predicted).
2. Gate behaviour (PDF requirements):
   - average weight of every branch per true corruption type AND severity -> table + routing heatmap
   - weight distribution (how sharp the routing is) per corruption
   - examples where one expert dominates vs where weights are spread over several branches
   - inactive experts / one expert dominating unrelated inputs
3. Re-check of the 63 test inputs that Task 2's hard routing sent to the wrong branch.
Outputs: artifacts/results/task3_*, docs/figures/task3_*.png, MLflow run 'test-evaluation'.
Run:  .venv\\Scripts\\python scripts/evaluate_task3.py        (--dry-run: 40 VALIDATION images)
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
from torch.utils.data import DataLoader

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_task1 import ERROR_VMAX, get_dataset, latex, pick_examples, summarize  # noqa: E402
from src import config as C  # noqa: E402
from src.corruptions import severity_value  # noqa: E402
from src.metrics import per_image_metrics  # noqa: E402
from src.onnx_utils import load_moe  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device  # noqa: E402

PREFIX = "task3"
BRANCHES = ["identity", "salt", "blur", "occlusion"]
LEVEL_ORDER = ["none", "low", "medium", "high"]


@torch.no_grad()
def run_test(model, device, ds) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=64, num_workers=2)
    rows = []
    for x, y, cond, idx in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            out, w, _ = model(x.to(device))
        out, w = out.float().cpu(), w.float().cpu()
        m_out, m_in = per_image_metrics(out, y), per_image_metrics(x, y)
        entropy = -(w * w.clamp_min(1e-12).log()).sum(1)   # 0 = one branch, log(4)=1.386 = all equal
        for j in range(len(idx)):
            e = ds.entries[int(idx[j])]
            row = {"entry": int(idx[j]), "id": e["id"], "condition": C.CONDITIONS[e["condition"]],
                   "level": e["level"], "severity": severity_value(e["params"]), "true": e["condition"],
                   **{f"w_{b}": float(w[j, k]) for k, b in enumerate(BRANCHES)},
                   "w_max": float(w[j].max()), "top_branch": int(w[j].argmax()), "entropy": float(entropy[j]),
                   "psnr": m_out["psnr"][j].item(), "ssim": m_out["ssim"][j].item(), "l1": m_out["l1"][j].item(),
                   "input_psnr": m_in["psnr"][j].item(), "input_ssim": m_in["ssim"][j].item(),
                   "input_l1": m_in["l1"][j].item()}
            rows.append(row)
    df = pd.DataFrame(rows)
    df.loc[df.condition == "clean", "input_psnr"] = np.inf
    return df


# ------------------------------------------------------------------ comparison with Tasks 1 and 2

def comparison_table(summary: pd.DataFrame, tag: str = "") -> pd.DataFrame:
    t = summary[["condition", "level", "n", "input_psnr", "input_ssim", "psnr", "ssim"]].rename(
        columns={"psnr": "task3_psnr", "ssim": "task3_ssim"})
    t2 = C.RESULTS / f"task2{tag}_comparison.csv"
    if t2.exists():
        prev = pd.read_csv(t2)[["condition", "level", "task1_psnr", "task1_ssim", "oracle_psnr", "oracle_ssim",
                                "pred_psnr", "pred_ssim"]].rename(
            columns={"oracle_psnr": "task2_oracle_psnr", "oracle_ssim": "task2_oracle_ssim",
                     "pred_psnr": "task2_pred_psnr", "pred_ssim": "task2_pred_ssim"})
        t = t.merge(prev, on=["condition", "level"], how="left")
    return t


def comparison_latex(t: pd.DataFrame) -> str:
    fmt = lambda v: "$\\infty$" if np.isinf(v) else f"{v:.2f}"  # noqa: E731
    lines = [r"\begin{tabular}{llrrrrrrrr}", r"\hline",
             r"Cond. & Level & \multicolumn{2}{c}{Task 1} & \multicolumn{2}{c}{Task 2 (pred.)} & "
             r"\multicolumn{2}{c}{Task 3} \\", r" & & PSNR & SSIM & PSNR & SSIM & PSNR & SSIM \\", r"\hline"]
    for r in t.itertuples():
        lines.append(f"{r.condition} & {r.level} & {fmt(r.task1_psnr)} & {r.task1_ssim:.3f} & "
                     f"{fmt(r.task2_pred_psnr)} & {r.task2_pred_ssim:.3f} & {fmt(r.task3_psnr)} & {r.task3_ssim:.3f} \\\\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}"]) + "\n"


# ------------------------------------------------------------------ routing analysis

def routing_tables(df: pd.DataFrame) -> pd.DataFrame:
    """Mean weight of every branch per true condition and severity (+ per condition overall)."""
    wcols = [f"w_{b}" for b in BRANCHES]
    by_level = df.groupby(["condition", "level"])[wcols + ["entropy", "w_max"]].mean().reset_index()
    by_cond = df.groupby("condition")[wcols + ["entropy", "w_max"]].mean().reset_index().assign(level="all")
    t = pd.concat([by_level, by_cond[by_cond.condition != "clean"]])
    t["_c"] = t.condition.map({c: i for i, c in enumerate(C.CONDITIONS)})
    t["_l"] = t.level.map({**{lv: i for i, lv in enumerate(LEVEL_ORDER)}, "all": 9})
    return t.sort_values(["_c", "_l"]).drop(columns=["_c", "_l"]).reset_index(drop=True)


def routing_heatmap(table: pd.DataFrame, df: pd.DataFrame, path: Path) -> None:
    t = table[table.level != "all"]
    mat = t[[f"w_{b}" for b in BRANCHES]].values
    labels = [f"{r.condition} / {r.level}" for r in t.itertuples()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [1.1, 1]})
    im = axes[0].imshow(mat, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(4):
            axes[0].text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                         color="black" if mat[i, j] > 0.6 else "white", fontsize=9)
    axes[0].set_xticks(range(4), BRANCHES)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_xlabel("branch")
    axes[0].set_title("Mean routing weight per true corruption / severity (test)")
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    # Weight distribution: weight given to the CORRECT branch, per condition (1.0 = fully routed).
    data = [df[df.true == k][f"w_{BRANCHES[k]}"].values for k in range(4)]
    axes[1].boxplot(data, tick_labels=C.CONDITIONS, showfliers=True, flierprops={"markersize": 2, "alpha": 0.3})
    axes[1].set_ylim(-0.02, 1.02)
    axes[1].set_ylabel("weight on the correct branch")
    axes[1].set_title("Distribution of the correct-branch weight (test)")
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def expert_health(df: pd.DataFrame) -> dict:
    """Inactive experts (hardly ever used) and experts that take over unrelated inputs."""
    out = {}
    for k, b in enumerate(BRANCHES):
        w = df[f"w_{b}"]
        unrelated = df[df.true != k][f"w_{b}"]
        out[b] = {"mean_weight_all_inputs": float(w.mean()),
                  "share_of_inputs_where_top": float((df.top_branch == k).mean()),
                  "mean_weight_on_own_inputs": float(df[df.true == k][f"w_{b}"].mean()),
                  "mean_weight_on_unrelated_inputs": float(unrelated.mean()),
                  "share_unrelated_inputs_with_weight_over_0.5": float((unrelated > 0.5).mean())}
        out[b]["inactive"] = out[b]["mean_weight_all_inputs"] < 0.05
        out[b]["dominates_unrelated"] = out[b]["mean_weight_on_unrelated_inputs"] > 0.2
    return out


def task2_misrouted_recheck(df: pd.DataFrame, tag: str = "") -> dict:
    f = C.RESULTS / f"task2{tag}_misrouted_inputs.csv"
    if not f.exists():
        return {}
    t2 = pd.read_csv(f)[["entry", "condition", "level", "pred", "pred_ssim", "oracle_ssim"]]
    m = t2.merge(df[["entry", "ssim", "w_identity", "w_salt", "w_blur", "w_occlusion"]], on="entry")
    m["task3_better_than_task2"] = m.ssim > m.pred_ssim
    groups = m.assign(t2_route=m.pred.map(lambda k: BRANCHES[k])).groupby(["condition", "t2_route"]).agg(
        n=("entry", "size"), task2_pred_ssim=("pred_ssim", "mean"), task2_oracle_ssim=("oracle_ssim", "mean"),
        task3_ssim=("ssim", "mean"), task3_better=("task3_better_than_task2", "mean"),
        mean_w_identity=("w_identity", "mean")).reset_index()
    return {"n": len(m), "task2_pred_ssim": float(m.pred_ssim.mean()), "task2_oracle_ssim": float(m.oracle_ssim.mean()),
            "task3_ssim": float(m.ssim.mean()), "share_task3_better": float(m.task3_better_than_task2.mean()),
            "by_case": groups.round(4).to_dict("records")}


# ------------------------------------------------------------------ figures

@torch.no_grad()
def draw(entries, df, model, device, ds, title, path, per_row=2):
    """Rows: clean target | input | output | |error| | weight bars."""
    rows = int(np.ceil(len(entries) / per_row))
    fig, axes = plt.subplots(rows, 5 * per_row, figsize=(2.4 * 5 * per_row, 2.5 * rows + 0.6), squeeze=False)
    d = df.set_index("entry")
    colors = ["tab:gray", "tab:orange", "tab:blue", "tab:green"]
    for k, entry in enumerate(entries):
        x, y, _, _ = ds[entry]
        out, w, _ = model(x[None].to(device))
        out, w = out[0].float().cpu().clamp(0, 1), w[0].float().cpu()
        r = d.loc[entry]
        r0, c0 = divmod(k, per_row)
        ax = axes[r0, 5 * c0:5 * c0 + 5]
        for a, img, label in [(ax[0], y, "clean target"), (ax[1], x, f"{r.condition} {r.level}"),
                              (ax[2], out, f"output {r.ssim:.3f} SSIM")]:
            a.imshow(img.permute(1, 2, 0).numpy())
            a.set_title(label, fontsize=8)
            a.axis("off")
        im = ax[3].imshow((out - y).abs().mean(0).numpy(), cmap="magma", vmin=0, vmax=ERROR_VMAX)
        ax[3].set_title("|error|", fontsize=8)
        ax[3].axis("off")
        ax[4].barh(range(4), w.numpy(), color=colors)
        ax[4].set_yticks(range(4), ["id", "salt", "blur", "occl"], fontsize=7)
        ax[4].set_xlim(0, 1)
        ax[4].invert_yaxis()
        ax[4].set_title(f"weights (H={r.entropy:.2f})", fontsize=8)
        ax[4].tick_params(axis="x", labelsize=6)
    for a in axes.flat[len(entries) * 5:]:
        a.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.01, label="mean abs error")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def pick_dominant_and_distributed(df: pd.DataFrame) -> tuple[list[int], list[int]]:
    """Dominant: per condition, the input whose weights are most concentrated (lowest entropy).
    Distributed: the 4 inputs with the most spread-out weights (highest entropy)."""
    dominant = [int(df[df.true == k].sort_values("entropy").iloc[0].entry) for k in range(4)]
    distributed = df.sort_values("entropy", ascending=False).entry.head(4).astype(int).tolist()
    return dominant, distributed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="check the script on 40 VALIDATION images")
    ap.add_argument("--checkpoint", default=None, help="default: the checkpoint of --version")
    ap.add_argument("--version", default="v1", help="v2 = evaluate the upgrade-pass models (own output files)")
    args = ap.parse_args()
    global PREFIX
    tag = "" if args.version == "v1" else f"_{args.version}"
    PREFIX = f"task3{tag}"
    args.checkpoint = args.checkpoint or str(C.CHECKPOINTS / f"task3_soft_moe{tag}.pt")
    out_dir = C.ARTIFACTS / "logs" / "dryrun" if args.dry_run else C.RESULTS
    fig_dir = out_dir if args.dry_run else C.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image = out_dir / f"{PREFIX}_test_per_image.csv"
    if per_image.exists() and not (args.force or args.dry_run):
        sys.exit(f"{per_image} exists: the test set was already evaluated. Use --force to re-run.")

    device = get_device()
    model = load_moe(args.checkpoint).to(device)
    ds = get_dataset(args.dry_run)
    df = run_test(model, device, ds)
    df.to_csv(per_image, index=False)

    summary = summarize(df)
    summary.to_csv(out_dir / f"{PREFIX}_test_summary.csv", index=False)
    (out_dir / f"{PREFIX}_test_table.tex").write_text(latex(summary))
    comp = comparison_table(summary, tag)
    comp.to_csv(out_dir / f"{PREFIX}_comparison.csv", index=False)
    if "task1_psnr" in comp:
        (out_dir / f"{PREFIX}_comparison_table.tex").write_text(comparison_latex(comp))

    rt = routing_tables(df)
    rt.to_csv(out_dir / f"{PREFIX}_routing_weights.csv", index=False)
    routing_heatmap(rt, df, fig_dir / f"{PREFIX}_routing_heatmap.png")
    health = expert_health(df)
    (out_dir / f"{PREFIX}_expert_health.json").write_text(json.dumps(health, indent=2))
    recheck = task2_misrouted_recheck(df, tag)
    (out_dir / f"{PREFIX}_task2_misrouted_recheck.json").write_text(json.dumps(recheck, indent=2))

    dominant, distributed = pick_dominant_and_distributed(df)
    draw(dominant + distributed, df, model, device, ds,
         "Task 3: one dominant expert (top 2 rows) vs distributed weights (bottom 2 rows)",
         fig_dir / f"{PREFIX}_dominant_vs_distributed.png")
    draw(pick_examples(df), df, model, device, ds, "Task 3: representative test examples (median SSIM per group)",
         fig_dir / f"{PREFIX}_examples.png")
    failures = [int(df[df.condition == c].sort_values("ssim").iloc[0].entry) for c in C.CONDITIONS]
    draw(failures, df, model, device, ds, "Task 3: failure cases (lowest output SSIM per condition)",
         fig_dir / f"{PREFIX}_failures.png")

    pd.set_option("display.width", 250)
    print(comp.round(4).to_string(index=False))
    print(rt.round(3).to_string(index=False))
    print(json.dumps({b: {k: round(v, 4) if isinstance(v, float) else v for k, v in h.items()}
                      for b, h in health.items()}, indent=1))
    print(json.dumps({k: v for k, v in recheck.items() if k != "by_case"}, indent=1))
    if args.dry_run:
        return

    setup_mlflow("task3-soft-moe")
    with mlflow.start_run(run_name=f"test-evaluation-{PREFIX}"):
        corrupted = df[df.condition != "clean"]
        mlflow.log_metrics({"test_ssim": float(df.ssim.mean()), "test_l1": float(df.l1.mean()),
                            "test_psnr_corrupted": float(corrupted.psnr.mean()),
                            "test_ssim_corrupted": float(corrupted.ssim.mean())})
        for f in sorted(out_dir.glob(f"{PREFIX}_*")) + sorted(fig_dir.glob(f"{PREFIX}_*.png")):
            if f.name != per_image.name:
                mlflow.log_artifact(str(f))


if __name__ == "__main__":
    main()
