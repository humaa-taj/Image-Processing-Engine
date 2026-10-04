"""Task 1 final evaluation on the OFFICIAL TEST manifest (run once, at the end).

For every one of the 36,690 test inputs (3,669 images x [clean + 3 corruptions x 3 severities]):
  PSNR / SSIM / L1 of the OUTPUT vs the clean target, and of the INPUT vs the clean target (baseline,
  i.e. "what if we did nothing"). The difference shows what the autoencoder actually adds.
Outputs:
  artifacts/results/task1_test_per_image.csv   one row per test input
  artifacts/results/task1_test_summary.csv     per corruption x severity (+ per corruption overall)
  artifacts/results/task1_test_table.tex       same, LaTeX (for the report)
  docs/figures/task1_examples.png              12 representative examples (median SSIM of their group)
  docs/figures/task1_failures.png              4 failure cases (lowest output PSNR per condition)
Each figure row: clean target | corrupted input | output | absolute error map.
Run:  .venv\\Scripts\\python scripts/evaluate_task1.py
"""
import argparse
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

from src import config as C  # noqa: E402
from src.corruptions import severity_value  # noqa: E402
from src.data import pet  # noqa: E402
from src.metrics import per_image_metrics  # noqa: E402
from src.onnx_utils import load_autoencoder  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device  # noqa: E402

CKPT = C.CHECKPOINTS / "task1_universal_ae.pt"
PREFIX = "task1"
ERROR_VMAX = 0.5  # same colour scale on every error map so they can be compared


def get_dataset(dry_run: bool) -> pet.PetManifestDataset:
    """The official test manifest, or (dry run) a test-style manifest built in memory for 40 VALIDATION
    images, so the script can be checked end to end without touching the test set."""
    if not dry_run:
        return pet.load_manifest_dataset("test")
    from scripts.prepare_data import make_test_manifest
    ids = pet.load_split()["val"][:40]
    return pet.PetManifestDataset(dict(zip(ids, pet.load_cached("trainval", ids))),
                                  make_test_manifest(ids)["entries"])


def run_test(model, device, ds) -> pd.DataFrame:
    loader = DataLoader(ds, batch_size=128, num_workers=2)
    rows = []
    with torch.no_grad():
        for x, y, cond, idx in loader:
            with torch.autocast("cuda", dtype=torch.float16):
                out = model(x.to(device)).float().cpu()
            m_out, m_in = per_image_metrics(out, y), per_image_metrics(x, y)
            for j in range(len(idx)):
                e = ds.entries[int(idx[j])]
                rows.append({"entry": int(idx[j]), "id": e["id"], "condition": C.CONDITIONS[e["condition"]],
                             "level": e["level"], "severity": severity_value(e["params"]),
                             "psnr": m_out["psnr"][j].item(), "ssim": m_out["ssim"][j].item(),
                             "l1": m_out["l1"][j].item(), "input_psnr": m_in["psnr"][j].item(),
                             "input_ssim": m_in["ssim"][j].item(), "input_l1": m_in["l1"][j].item()})
    df = pd.DataFrame(rows)
    df.loc[df.condition == "clean", "input_psnr"] = np.inf  # input == target: PSNR is infinite
    return df


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["psnr", "ssim", "l1", "input_psnr", "input_ssim", "input_l1"]
    order = {c: i for i, c in enumerate(C.CONDITIONS)}
    lv = {"none": 0, "low": 1, "medium": 2, "high": 3, "all": 4}
    by_level = df.groupby(["condition", "level"])[cols].mean().reset_index()
    by_cond = df.groupby("condition")[cols].mean().reset_index().assign(level="all")
    corrupted = df[df.condition != "clean"]
    overall_corrupted = corrupted[cols].mean().to_frame().T.assign(condition="ALL corrupted", level="all")
    overall = df[cols].mean().to_frame().T.assign(condition="ALL", level="all")
    table = pd.concat([by_level, by_cond[by_cond.condition != "clean"], overall_corrupted, overall],
                      ignore_index=True)
    table["n"] = [len(df) if r.condition == "ALL" else len(corrupted) if r.condition == "ALL corrupted" else
                  len(df[(df.condition == r.condition) & ((df.level == r.level) | (r.level == "all"))])
                  for r in table.itertuples()]
    table["psnr_gain"] = table.psnr - table.input_psnr
    table["ssim_gain"] = table.ssim - table.input_ssim
    table["_c"] = table.condition.map({**order, "ALL corrupted": 8, "ALL": 9})
    table["_l"] = table.level.map(lv)
    return table.sort_values(["_c", "_l"]).drop(columns=["_c", "_l"]).reset_index(drop=True)


def latex(table: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{llrrrrrr}", r"\hline",
             r"Condition & Level & PSNR in & PSNR out & SSIM in & SSIM out & L1 in & L1 out \\", r"\hline"]
    for r in table.itertuples():
        pin = "$\\infty$" if np.isinf(r.input_psnr) else f"{r.input_psnr:.2f}"
        lines.append(f"{r.condition} & {r.level} & {pin} & {r.psnr:.2f} & {r.input_ssim:.3f} & {r.ssim:.3f} & "
                     f"{r.input_l1:.4f} & {r.l1:.4f} \\\\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}"]) + "\n"


def pick_examples(df: pd.DataFrame) -> list[int]:
    """12 representative examples: the median-SSIM entry of each corruption x severity group (9),
    plus clean images at the 25th / 50th / 75th SSIM percentile (3). Median = typical, not cherry-picked."""
    picks = []
    for cond in ("salt", "blur", "occlusion"):
        for level in C.LEVELS:
            g = df[(df.condition == cond) & (df.level == level)]
            picks.append(int(g.iloc[(g.ssim - g.ssim.median()).abs().argsort().iloc[0]].entry))
    clean = df[df.condition == "clean"]
    for q in (0.25, 0.5, 0.75):
        picks.append(int(clean.iloc[(clean.ssim - clean.ssim.quantile(q)).abs().argsort().iloc[0]].entry))
    return picks


def pick_failures(df: pd.DataFrame) -> list[int]:
    """One failure per condition: the test input with the lowest output PSNR."""
    return [int(df[df.condition == c].sort_values("psnr").iloc[0].entry) for c in C.CONDITIONS]


def draw(entries: list[int], df: pd.DataFrame, model, device, ds, title: str, path: Path, per_row: int) -> None:
    rows = int(np.ceil(len(entries) / per_row))
    fig, axes = plt.subplots(rows, 4 * per_row, figsize=(3.2 * per_row * 4 / 2, 2.0 * rows + 0.6), squeeze=False)
    for k, entry in enumerate(entries):
        x, y, _, _ = ds[entry]
        with torch.no_grad():
            out = model(x[None].to(device)).float().cpu()[0].clamp(0, 1)
        err = (out - y).abs().mean(0).numpy()
        r = df.set_index("entry").loc[entry]
        r0, c0 = divmod(k, per_row)
        panels = [(y, "clean target"), (x, f"{r.condition} {r.level}"),
                  (out, f"output {r.psnr:.1f} dB / {r.ssim:.3f}"), (None, "|error|")]
        for j, (img, label) in enumerate(panels):
            ax = axes[r0, 4 * c0 + j]
            if img is None:
                im = ax.imshow(err, cmap="magma", vmin=0, vmax=ERROR_VMAX)
            else:
                ax.imshow(img.permute(1, 2, 0).numpy())
            ax.set_title(label, fontsize=8)
            ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.01, pad=0.01, label="mean abs error")
    fig.suptitle(title)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="re-run even though test results exist")
    ap.add_argument("--dry-run", action="store_true", help="check the script on 40 VALIDATION images")
    ap.add_argument("--checkpoint", default=None, help="default: the checkpoint of --version")
    ap.add_argument("--version", default="v1", help="v2 = evaluate the upgrade-pass model (own output files)")
    args = ap.parse_args()
    global PREFIX
    tag = "" if args.version == "v1" else f"_{args.version}"
    PREFIX = f"task1{tag}"
    args.checkpoint = args.checkpoint or str(C.CHECKPOINTS / f"task1_universal_ae{tag}.pt")
    results_dir = C.ARTIFACTS / "logs" / "dryrun" if args.dry_run else C.RESULTS
    fig_dir = results_dir if args.dry_run else C.FIGURES
    results_dir.mkdir(parents=True, exist_ok=True)
    per_image = results_dir / f"{PREFIX}_test_per_image.csv"
    if per_image.exists() and not (args.force or args.dry_run):
        sys.exit(f"{per_image} exists: the test set was already evaluated. Use --force to re-run.")

    device = get_device()
    model = load_autoencoder(args.checkpoint).to(device)
    ds = get_dataset(args.dry_run)
    df = run_test(model, device, ds)
    table = summarize(df)
    df.to_csv(per_image, index=False)
    table.to_csv(results_dir / f"{PREFIX}_test_summary.csv", index=False)
    (results_dir / f"{PREFIX}_test_table.tex").write_text(latex(table))

    draw(pick_examples(df), df, model, device, ds, "Task 1: representative test examples (median SSIM of each group)",
         fig_dir / f"{PREFIX}_examples.png", per_row=2)
    draw(pick_failures(df), df, model, device, ds, "Task 1: failure cases (lowest output PSNR per condition)",
         fig_dir / f"{PREFIX}_failures.png", per_row=2)
    if args.dry_run:
        print(table.round(4).to_string(index=False))
        return

    setup_mlflow("task1-universal-ae")
    with mlflow.start_run(run_name=f"test-evaluation-{PREFIX}"):
        overall = table[table.condition == "ALL"].iloc[0]
        mlflow.log_metrics({"test_psnr": overall.psnr, "test_ssim": overall.ssim, "test_l1": overall.l1})
        for r in table.itertuples():
            if r.condition != "ALL":
                mlflow.log_metrics({f"test_ssim_{r.condition}_{r.level}": r.ssim,
                                    f"test_psnr_{r.condition}_{r.level}": r.psnr})
        for f in (C.RESULTS / f"{PREFIX}_test_summary.csv", C.RESULTS / f"{PREFIX}_test_table.tex",
                  C.FIGURES / f"{PREFIX}_examples.png", C.FIGURES / f"{PREFIX}_failures.png"):
            mlflow.log_artifact(str(f))
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
