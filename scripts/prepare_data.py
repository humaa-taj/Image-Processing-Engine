"""Phase 1 data preparation. Run once:  .venv\\Scripts\\python scripts/prepare_data.py

Creates (never touches data/):
  artifacts/splits/pet_split.json        80/20 split of official trainval, seed 42
  artifacts/cache/pet_trainval.npy       clean 128x128 images (gitignored, ~180 MB each)
  artifacts/cache/pet_test.npy
  artifacts/manifests/val_manifest.json  one condition per val image, exactly 184 per class
  artifacts/manifests/test_manifest.json clean + 3 corruptions x 3 fixed severities per test image
  artifacts/splits/fs2k_split.json       FS2K train/val (15%, stratified by style, seed 42) + test

Safe to re-run: if a file already exists it is regenerated in memory and compared. Identical ->
"unchanged"; different -> the script stops (so a saved split can never change by accident).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # make `src` importable

from src import config as C  # noqa: E402
from src.corruptions import fixed_test_params, sample_training_params  # noqa: E402
from src.data import fs2k, pet  # noqa: E402
from src.utils import load_json, save_json  # noqa: E402


def write_or_verify(obj, path: Path, force: bool) -> None:
    roundtrip = json.loads(json.dumps(obj))  # what the file would contain
    if path.exists() and not force:
        if load_json(path) != roundtrip:
            sys.exit(f"ERROR: {path} exists and differs from a fresh build. Not overwriting. "
                     "Use --force only if you really want to replace it.")
        print(f"  unchanged (verified identical): {path.relative_to(C.ROOT)}")
        return
    save_json(obj, path)
    print(f"  wrote {path.relative_to(C.ROOT)}  ({path.stat().st_size / 1e6:.2f} MB)")


def make_val_manifest(val_ids: list[str]) -> dict:
    """One condition per val image, exactly balanced, severity from the TRAINING ranges.

    A master generator (seed 42) shuffles the condition list and draws one seed per image;
    that per-image seed produces the corruption params, and it is stored for traceability.
    """
    master = np.random.default_rng(C.SEED)
    n = len(val_ids)
    assert n % 4 == 0, "val set size must be divisible by 4 for an exact balance"
    conditions = np.repeat(np.arange(4), n // 4)
    master.shuffle(conditions)
    entries = []
    for image_id, cond in zip(val_ids, conditions.tolist()):
        seed = int(master.integers(2 ** 31))
        params = sample_training_params(cond, np.random.default_rng(seed))
        entries.append({"id": image_id, "condition": cond, "level": "random", "seed": seed, "params": params})
    return {"split": "val", "master_seed": C.SEED, "entries": entries}


def make_test_manifest(test_ids: list[str]) -> dict:
    """Every test image: 1 clean entry + (salt, blur, occlusion) x (low, medium, high) = 10 entries."""
    master = np.random.default_rng(C.SEED + 1)  # separate stream from the val manifest
    entries = []
    for image_id in test_ids:
        entries.append({"id": image_id, "condition": C.CLEAN, "level": "none", "seed": None,
                        "params": {"type": "clean"}})
        for cond in (C.SALT, C.BLUR, C.OCCLUSION):
            for level in C.LEVELS:
                seed = int(master.integers(2 ** 31))
                params = fixed_test_params(cond, level, np.random.default_rng(seed))
                entries.append({"id": image_id, "condition": cond, "level": level, "seed": seed, "params": params})
    return {"split": "test", "master_seed": C.SEED + 1, "entries": entries}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="overwrite existing split/manifest files")
    args = ap.parse_args()

    print("1) Pet split")
    split = pet.make_split()
    write_or_verify(split, C.SPLITS / "pet_split.json", args.force)
    print(f"   train={len(split['train'])} val={len(split['val'])}")

    print("2) Pet clean 128x128 cache")
    for name, ids in [("trainval", sorted(pet.read_ids("trainval"))), ("test", sorted(pet.read_ids("test")))]:
        if pet.cache_path(name).exists() and not args.force:
            print(f"   exists: {pet.cache_path(name).relative_to(C.ROOT)}")
        else:
            print(f"   building {name} ({len(ids)} images)...")
            pet.build_cache(name, ids)
        print(f"   {name}: {np.load(pet.cache_path(name), mmap_mode='r').shape}, "
              f"{pet.cache_path(name).stat().st_size / 1e6:.0f} MB")

    print("3) Validation manifest")
    val = make_val_manifest(split["val"])
    write_or_verify(val, C.MANIFESTS / "val_manifest.json", args.force)
    counts = np.bincount([e["condition"] for e in val["entries"]], minlength=4)
    print(f"   entries={len(val['entries'])} per class={dict(zip(C.CONDITIONS, counts.tolist()))}")

    print("4) Test manifest")
    test = make_test_manifest(sorted(pet.read_ids("test")))
    write_or_verify(test, C.MANIFESTS / "test_manifest.json", args.force)
    print(f"   entries={len(test['entries'])}")

    print("5) FS2K split")
    fsplit = fs2k.make_split()
    write_or_verify(fsplit, C.SPLITS / "fs2k_split.json", args.force)
    for part in ("train", "val", "test"):
        styles = np.bincount([r["style"] for r in fsplit[part]], minlength=3).tolist()
        print(f"   {part}: {len(fsplit[part])}  styles(0/1/2)={styles}")


if __name__ == "__main__":
    main()
