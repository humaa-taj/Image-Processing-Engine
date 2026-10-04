"""Saved splits/manifests are correct and reproducible; datasets and sampler behave.

Needs `scripts/prepare_data.py` to have been run (uses the saved artifacts and the image cache).
"""
import json
from collections import Counter

import numpy as np
import torch

from src import config as C
from src.corruptions import severity_value
from src.data import fs2k, pet
from src.data.samplers import BalancedBatchSampler
from src.utils import load_json

import scripts.prepare_data as prep  # noqa: E402  (to rebuild manifests in memory)


def test_pet_split_saved_and_reproducible():
    split = pet.load_split()
    assert len(split["train"]) == 2944 and len(split["val"]) == 736
    assert not set(split["train"]) & set(split["val"])
    assert set(split["train"]) | set(split["val"]) == set(pet.read_ids("trainval"))
    assert not (set(split["train"]) | set(split["val"])) & set(pet.read_ids("test"))  # test untouched
    assert pet.make_split() == split


def test_val_manifest():
    m = load_json(C.MANIFESTS / "val_manifest.json")
    assert [e["id"] for e in m["entries"]] == pet.load_split()["val"]
    assert Counter(e["condition"] for e in m["entries"]) == {0: 184, 1: 184, 2: 184, 3: 184}
    assert m == json.loads(json.dumps(prep.make_val_manifest(pet.load_split()["val"])))


def test_test_manifest():
    m = load_json(C.MANIFESTS / "test_manifest.json")
    entries = m["entries"]
    assert len(entries) == 3669 * 10
    per_image = Counter(e["id"] for e in entries)
    assert set(per_image.values()) == {10}
    groups = Counter((e["condition"], e["level"]) for e in entries)
    assert groups[(C.CLEAN, "none")] == 3669
    for cond in (C.SALT, C.BLUR, C.OCCLUSION):
        for level in C.LEVELS:
            assert groups[(cond, level)] == 3669
    occ_high = [severity_value(e["params"]) for e in entries if e["condition"] == C.OCCLUSION and e["level"] == "high"]
    assert abs(np.mean(occ_high) - 0.35) < 0.005


def test_cache_matches_fresh_load():
    ids = pet.load_split()["val"][:3]
    cached = pet.load_cached("trainval", ids)
    for i, image_id in enumerate(ids):
        assert np.array_equal(cached[i], pet.load_image(image_id))


def test_datasets_shapes_and_labels():
    ids = pet.load_split()["train"][:8]
    ds = pet.PetTrainDataset(pet.load_cached("trainval", ids))
    corrupted, clean, label = ds[(0, C.BLUR)]
    assert corrupted.shape == clean.shape == (3, 128, 128) and label == C.BLUR
    assert 0 <= corrupted.min() and corrupted.max() <= 1
    val = pet.load_manifest_dataset("val")
    assert len(val) == 736
    x, y, label, idx = val[5]
    assert x.shape == (3, 128, 128) and label == val.entries[5]["condition"]
    assert torch.equal(val[5][0], x)  # deterministic


def test_balanced_sampler():
    s = BalancedBatchSampler(n_images=2944, batch_size=32, seed=1)
    batches = list(s)
    assert len(batches) == 2944 // 32
    for b in batches:
        assert Counter(c for _, c in b) == {0: 8, 1: 8, 2: 8, 3: 8}
    all_idx = [i for b in batches for i, _ in b]
    assert len(set(all_idx)) == len(all_idx)            # no image twice in an epoch
    s.set_epoch(1)
    assert [i for i, _ in next(iter(s))] != [i for i, _ in batches[0]]  # new shuffle each epoch


def test_fs2k_split_and_pairs():
    split = load_json(C.SPLITS / "fs2k_split.json")
    assert (len(split["train"]), len(split["val"]), len(split["test"])) == (899, 159, 1046)
    names = lambda part: {r["name"] for r in split[part]}  # noqa: E731
    assert not names("train") & names("val") and not (names("train") | names("val")) & names("test")
    assert Counter(r["style"] for r in split["val"]) == {0: 54, 1: 52, 2: 53}
    assert split == load_json(C.SPLITS / "fs2k_split.json") == json.loads(
        json.dumps(fs2k.make_split()))
    photo, sketch = fs2k.pair_paths("photo2/image0015")      # .jpg photo with .png RGBA sketch
    assert photo.stem == "image0015" and sketch.name == "sketch0015.png"


def test_fs2k_paired_augmentation_keeps_alignment():
    """Use the sketch as both members: if photo and sketch got the same crop/flip, they stay equal."""
    rec = [{"name": "photo1/image0110", "style": 0}]
    ds = fs2k.FS2KDataset(rec, augment=True)
    torch.manual_seed(0)
    photo, sketch, style = ds[0]
    assert photo.shape == (3, 128, 128) and sketch.shape == (1, 128, 128) and style == 0
    # Re-run with a patched loader that returns the same image as "photo" and "sketch".
    orig = fs2k.load_pair
    fs2k.load_pair = lambda name, size: (np.repeat(orig(name, size)[1][:, :, None], 3, 2), orig(name, size)[1])
    try:
        for seed in range(10):
            torch.manual_seed(seed)
            p, s, _ = ds[0]
            assert torch.equal(p[0:1], s)
    finally:
        fs2k.load_pair = orig
