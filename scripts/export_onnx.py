"""Export inference models to ONNX (models/onnx/<name>.onnx) and record file sizes.

Run:  .venv\\Scripts\\python scripts/export_onnx.py task1_universal_ae   (or --all)
- model.eval() first (dropout off, BatchNorm frozen), exported on CPU in float32.
- Fixed 3x128x128 input, dynamic batch axis.
- Uses PyTorch's torch.export-based exporter (dynamo=True), then onnx.checker validates the file.
Sizes -> artifacts/results/onnx_sizes.csv.
Size policy (CLAUDE.md section 7): no cap on model size. Files over 50 MB stay LOCAL: the script adds
them to models/onnx/.gitignore so they are never pushed to GitHub (they get a download link instead).
"""
import argparse
import sys
from pathlib import Path

import onnx
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.onnx_utils import REGISTRY  # noqa: E402

SIZE_LIMIT_MB = 50


def keep_local(filename: str) -> None:
    """List a large ONNX file in models/onnx/.gitignore so git never commits it."""
    ignore = C.ONNX_DIR / ".gitignore"
    lines = ignore.read_text().splitlines() if ignore.exists() else [
        "# ONNX files over 50 MB: kept local, not pushed to GitHub (download link in README)"]
    if filename not in lines:
        ignore.write_text("\n".join(lines + [filename]) + "\n")


def export(name: str) -> dict:
    entry = REGISTRY[name]
    model = entry["load"](entry["checkpoint"])
    # Most models take one image batch; the Task 4 generator also takes a style id per image.
    example = entry["example"]() if "example" in entry else (torch.rand(2, 3, C.IMG_SIZE, C.IMG_SIZE),)
    batch = torch.export.Dim("batch")   # every input shares the same dynamic batch axis
    path = C.ONNX_DIR / f"{name}.onnx"
    C.ONNX_DIR.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, example, str(path), input_names=entry["inputs"], output_names=entry["outputs"],
                      dynamic_shapes=tuple({0: batch} for _ in example), dynamo=True, external_data=False,
                      verbose=False)
    onnx.checker.check_model(onnx.load(str(path)))
    size_mb = path.stat().st_size / 1e6
    flag = ""
    if size_mb > SIZE_LIMIT_MB:
        keep_local(path.name)
        flag = "  <-- over 50 MB: kept LOCAL (added to models/onnx/.gitignore), needs a download link"
    print(f"{name}: {path.relative_to(C.ROOT)}  {size_mb:.2f} MB{flag}")
    return {"model": name, "file": path.name, "size_mb": round(size_mb, 2),
            "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(REGISTRY) if args.all else args.names
    rows = [export(n) for n in names]

    out = C.RESULTS / "onnx_sizes.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=rows[0].keys())
    table = pd.concat([old[~old.model.isin(names)], pd.DataFrame(rows)]).sort_values("model")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
