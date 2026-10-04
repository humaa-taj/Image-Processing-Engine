"""Figure of all corruptions for the report and for a visual sanity check.

Top rows: test images exactly as the saved test manifest corrupts them (clean + 3 severities each).
Bottom row: random TRAINING samples from the runtime pipeline (fresh random condition each time).
Run:  .venv\\Scripts\\python scripts/preview_corruptions.py   -> docs/figures/corruption_grid.png
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.corruptions import apply_corruption, severity_value  # noqa: E402
from src.data import pet  # noqa: E402
from src.utils import load_json, to_uint8_image  # noqa: E402


def caption(entry) -> str:
    p = entry["params"]
    if p["type"] == "clean":
        return "clean"
    if p["type"] == "salt":
        return f"salt {entry['level']}\np={p['p']:.2f}"
    if p["type"] == "blur":
        return f"blur {entry['level']}\nk={p['kernel']}, s={p['sigma']:.1f}"
    return f"occl. {entry['level']}\n{len(p['rects'])} rect, {severity_value(p):.0%}"


def main():
    entries = load_json(C.MANIFESTS / "test_manifest.json")["entries"]
    test_ids = sorted({e["id"] for e in entries})
    show_ids = [test_ids[0], test_ids[1500], test_ids[3000]]
    images = dict(zip(show_ids, pet.load_cached("test", show_ids)))

    train_ids = pet.load_split()["train"][:10]
    train_ds = pet.PetTrainDataset(pet.load_cached("trainval", train_ids))

    fig, axes = plt.subplots(len(show_ids) + 1, 10, figsize=(20, 2.4 * (len(show_ids) + 1)))
    for row, image_id in enumerate(show_ids):
        for col, e in enumerate([e for e in entries if e["id"] == image_id]):
            axes[row, col].imshow(apply_corruption(images[image_id], e["params"]))
            axes[row, col].set_title(caption(e), fontsize=9)
    for col in range(10):
        corrupted, _, label = train_ds[col]
        axes[-1, col].imshow(to_uint8_image(corrupted))
        axes[-1, col].set_title(f"train: {C.CONDITIONS[label]}", fontsize=9, color="tab:blue")
    for ax in axes.flat:
        ax.axis("off")
    fig.suptitle("Test manifest corruptions (rows 1-3) and random runtime training samples (row 4)")
    fig.tight_layout()
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    out = C.FIGURES / "corruption_grid.png"
    fig.savefig(out, dpi=110)
    print(f"saved {out.relative_to(C.ROOT)} ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    np.random.seed(0)
    main()
