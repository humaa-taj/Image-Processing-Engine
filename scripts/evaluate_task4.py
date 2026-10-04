"""Task 4 final evaluation on the OFFICIAL FS2K TEST set (1,046 pairs; run once, at the end).

Per test pair (generated with its TRUE style): L1, SSIM, PSNR vs the real sketch.
Reported per style and per photo source (photo1 / photo3). Style is confounded with photo source
in FS2K (see docs/dataset_findings.md), so per-style numbers must be read together with the source.
Style effect: for every test photo we also generate the other two styles and measure how much the
output changes (mean |difference|); a value near 0 would mean the style embedding is ignored.
Figures: photo | real sketch | generated (true style) | all 3 styles, for 8 test photos; 4 failure cases.
Run:  .venv\\Scripts\\python scripts/evaluate_task4.py        (--dry-run: uses the VALIDATION pairs)
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data.fs2k import FS2KDataset  # noqa: E402
from src.gan_training import generate  # noqa: E402
from src.metrics import per_image_metrics  # noqa: E402
from src.onnx_utils import load_generator  # noqa: E402
from src.tracking import setup_mlflow  # noqa: E402
from src.utils import get_device, load_json  # noqa: E402

PREFIX = "task4"


def load_split_tensors(part: str):
    records = load_json(C.SPLITS / "fs2k_split.json")[part]
    ds = FS2KDataset(records, augment=False)
    photos, sketches, styles = zip(*[ds[i] for i in range(len(ds))])
    return records, torch.stack(photos), torch.stack(sketches), torch.tensor(styles)


def evaluate(g, device, records, photos, sketches, styles) -> pd.DataFrame:
    fake = generate(g, photos, styles, device)
    m = per_image_metrics(fake, sketches)
    others = [generate(g, photos, torch.full_like(styles, k), device) for k in range(3)]
    rows = []
    for i, r in enumerate(records):
        k = int(styles[i])
        style_effect = np.mean([(others[j][i] - others[k][i]).abs().mean().item() for j in range(3) if j != k])
        rows.append({"index": i, "name": r["name"], "source": r["name"].split("/")[0], "style": k + 1,
                     "l1": m["l1"][i].item(), "ssim": m["ssim"][i].item(), "psnr": m["psnr"][i].item(),
                     "style_effect": float(style_effect)})
    return pd.DataFrame(rows)


def summaries(df: pd.DataFrame) -> pd.DataFrame:
    cols = ["l1", "ssim", "psnr", "style_effect"]
    parts = [df[cols].mean().to_frame().T.assign(group="all", n=len(df))]
    for s, g in df.groupby("style"):
        parts.append(g[cols].mean().to_frame().T.assign(group=f"style {s}", n=len(g)))
    for (src, s), g in df.groupby(["source", "style"]):
        parts.append(g[cols].mean().to_frame().T.assign(group=f"{src} / style {s}", n=len(g)))
    return pd.concat(parts)[["group", "n"] + cols].reset_index(drop=True)


def latex(t: pd.DataFrame) -> str:
    lines = [r"\begin{tabular}{lrrrrr}", r"\hline", r"Group & n & L1 & SSIM & PSNR & Style effect \\", r"\hline"]
    for r in t.itertuples():
        lines.append(f"{r.group} & {r.n} & {r.l1:.4f} & {r.ssim:.3f} & {r.psnr:.2f} & {r.style_effect:.4f} \\\\")
    return "\n".join(lines + [r"\hline", r"\end{tabular}"]) + "\n"


def draw(indices, df, g, device, photos, sketches, styles, title, path):
    fig, axes = plt.subplots(len(indices), 6, figsize=(12, 2.1 * len(indices) + 0.5), squeeze=False)
    for row, i in enumerate(indices):
        r = df.iloc[i]
        allk = generate(g, photos[i:i + 1].repeat(3, 1, 1, 1), torch.arange(3), device)
        panels = [(photos[i].permute(1, 2, 0).numpy(), f"photo ({r.source})", None),
                  (sketches[i, 0].numpy(), f"real sketch (Style {r['style']})", "gray"),
                  (allk[r["style"] - 1, 0].numpy(), f"generated, SSIM {r.ssim:.3f}", "gray")]
        panels += [(allk[k, 0].numpy(), f"as Style {k + 1}", "gray") for k in range(3)]
        for j, (img, label, cmap) in enumerate(panels):
            axes[row, j].imshow(img, cmap=cmap, vmin=0, vmax=1)
            axes[row, j].set_title(label, fontsize=8)
            axes[row, j].axis("off")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))   # leave room for the title above the first row
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="evaluate on the VALIDATION pairs instead of test")
    ap.add_argument("--checkpoint", default=str(C.CHECKPOINTS / "task4_generator.pt"))
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw the figures from the saved per-image results (no re-evaluation)")
    args = ap.parse_args()
    out_dir = C.ARTIFACTS / "logs" / "dryrun" if args.dry_run else C.RESULTS
    fig_dir = out_dir if args.dry_run else C.FIGURES
    out_dir.mkdir(parents=True, exist_ok=True)
    per_image = out_dir / f"{PREFIX}_test_per_image.csv"
    if per_image.exists() and not (args.force or args.dry_run or args.figures_only):
        sys.exit(f"{per_image} exists: the test set was already evaluated. Use --force to re-run.")

    device = get_device()
    g = load_generator(args.checkpoint).to(device)
    records, photos, sketches, styles = load_split_tensors("val" if args.dry_run else "test")
    if args.figures_only:
        df = pd.read_csv(per_image)
    else:
        df = evaluate(g, device, records, photos, sketches, styles)
        df.to_csv(per_image, index=False)
    table = summaries(df)
    table.to_csv(out_dir / f"{PREFIX}_test_summary.csv", index=False)
    (out_dir / f"{PREFIX}_test_table.tex").write_text(latex(table))

    # 8 examples: per style, the photos at the 25th / 75th SSIM percentile (+2 medians from the largest style)
    examples = []
    for s in (1, 2, 3):
        gdf = df[df["style"] == s].sort_values("ssim").reset_index(drop=True)
        qs = (0.25, 0.5, 0.75) if s == 1 else (0.25, 0.75) if s == 2 else (0.25, 0.5, 0.75)
        examples += [int(gdf.loc[int(q * (len(gdf) - 1)), "index"]) for q in qs]
    draw(examples, df, g, device, photos, sketches, styles,
         "Task 4: test examples (25/50/75th SSIM percentile per style) and the same photo in all 3 styles",
         fig_dir / f"{PREFIX}_examples.png")
    failures = [int(df[df["style"] == s].sort_values("ssim").iloc[0]["index"]) for s in (1, 2, 3)]
    failures.append(int(df.sort_values("l1", ascending=False).iloc[0]["index"]))
    draw(failures, df, g, device, photos, sketches, styles,
         "Task 4: failure cases (lowest SSIM per style + highest L1 overall)", fig_dir / f"{PREFIX}_failures.png")

    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))
    if args.dry_run or args.figures_only:
        return
    setup_mlflow("task4-face-to-sketch")
    with mlflow.start_run(run_name="test-evaluation"):
        overall = table.iloc[0]
        mlflow.log_metrics({"test_l1": overall.l1, "test_ssim": overall.ssim, "test_psnr": overall.psnr,
                            "test_style_effect": overall.style_effect})
        for f in sorted(out_dir.glob(f"{PREFIX}_*")) + sorted(fig_dir.glob(f"{PREFIX}_*.png")):
            if f.name != per_image.name:
                mlflow.log_artifact(str(f))


if __name__ == "__main__":
    main()
