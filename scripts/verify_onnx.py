"""Check that ONNX Runtime gives the same outputs as PyTorch, on REAL images.

Inputs: 64 images from the Pet VALIDATION manifest (16 per condition: clean, salt, blur, occlusion);
for the Task 4 generator, 48 FS2K validation photos (16 per style) with their style ids.
Both run on CPU in float32. Checked at batch size 64 and at batch size 1 (dynamic batch axis).
Results -> artifacts/results/onnx_verification.csv (max / mean absolute difference per model).
Run:  .venv\\Scripts\\python scripts/verify_onnx.py task1_universal_ae   (or --all)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.data import pet  # noqa: E402
from src.onnx_utils import REGISTRY  # noqa: E402

TOLERANCE = 1e-4  # images are in [0, 1]; 1e-4 is far below one grey level (1/255 = 0.0039)


def real_inputs(per_condition: int = 16) -> torch.Tensor:
    ds = pet.load_manifest_dataset("val")
    idx = [i for c in range(4) for i in [j for j, e in enumerate(ds.entries) if e["condition"] == c][:per_condition]]
    return torch.stack([ds[i][0] for i in idx])


def verify(name: str, x: torch.Tensor) -> dict:
    entry = REGISTRY[name]
    model = entry["load"](entry["checkpoint"])
    session = ort.InferenceSession(str(C.ONNX_DIR / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    inputs = entry["real_inputs"]() if "real_inputs" in entry else (x,)
    with torch.no_grad():
        ref = model(*inputs)
    refs = [r.numpy() for r in (ref if isinstance(ref, (tuple, list)) else (ref,))]  # models may return several outputs
    feed = lambda n: {k: v[:n].numpy() for k, v in zip(entry["inputs"], inputs)}  # noqa: E731
    got = session.run(None, feed(len(inputs[0])))
    single = session.run(None, feed(1))
    diffs = [np.abs(g - r) for g, r in zip(got, refs)]
    max_diff = max(float(d.max()) for d in diffs)
    row = {"model": name, "n_images": len(inputs[0]), "outputs": "+".join(entry["outputs"]), "max_abs_diff": max_diff,
           "mean_abs_diff": float(np.mean([d.mean() for d in diffs])),
           "batch1_max_abs_diff": max(float(np.abs(s - r[:1]).max()) for s, r in zip(single, refs)),
           "passed": bool(max_diff < TOLERANCE)}
    print(f"{name}: max {row['max_abs_diff']:.2e}, mean {row['mean_abs_diff']:.2e}, "
          f"batch-1 max {row['batch1_max_abs_diff']:.2e} -> {'PASS' if row['passed'] else 'FAIL'}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    names = list(REGISTRY) if args.all else args.names
    torch.set_num_threads(4)
    x = real_inputs()
    rows = [verify(n, x) for n in names]

    out = C.RESULTS / "onnx_verification.csv"
    old = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=rows[0].keys())
    table = pd.concat([old[~old.model.isin(names)], pd.DataFrame(rows)]).sort_values("model")
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    if not all(r["passed"] for r in rows):
        sys.exit("ONNX verification FAILED for at least one model")


if __name__ == "__main__":
    main()
