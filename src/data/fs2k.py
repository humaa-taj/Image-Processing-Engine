"""FS2K photo->sketch pairs for Task 4.

Pairing rule (same as the official tools/split_train_test.py):
    photo/photo1/image0110.jpg  <->  sketch/sketch1/sketch0110.jpg
Extensions are mixed (.jpg / .JPG / .png), so we look for whichever exists.
Style label: JSON field "style" in {0, 1, 2} (= "Style 1/2/3" in the app).
"""
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from src import config as C
from src.utils import load_json, to_tensor

EXTENSIONS = (".jpg", ".JPG", ".png", ".PNG", ".jpeg")


def _find(stem: Path) -> Path:
    for ext in EXTENSIONS:
        p = stem.with_suffix(ext)
        if p.exists():
            return p
    raise FileNotFoundError(f"no image file for {stem}")


def pair_paths(image_name: str) -> tuple[Path, Path]:
    """'photo1/image0110' -> (photo path, sketch path)."""
    folder, stem = image_name.split("/")
    photo = _find(C.FS2K_DIR / "photo" / folder / stem)
    sketch = _find(C.FS2K_DIR / "sketch" / folder.replace("photo", "sketch") / stem.replace("image", "sketch"))
    return photo, sketch


def read_records(split: str) -> list[dict]:
    """Official records, reduced to what we need: name and style. split = 'train' or 'test'."""
    data = load_json(C.FS2K_DIR / f"anno_{split}.json")
    return [{"name": d["image_name"], "style": int(d["style"])} for d in data]


def make_split(seed: int = C.SEED) -> dict:
    """Hold out 15% of the official TRAIN set as validation, stratified by style."""
    records = sorted(read_records("train"), key=lambda r: r["name"])
    train, val = train_test_split(records, test_size=C.FS2K_VAL_FRACTION, random_state=seed,
                                  stratify=[r["style"] for r in records])
    test = sorted(read_records("test"), key=lambda r: r["name"])
    by_name = lambda rs: sorted(rs, key=lambda r: r["name"])  # noqa: E731
    return {"seed": seed, "train": by_name(train), "val": by_name(val), "test": test}


def load_pair(image_name: str, size: int) -> tuple[np.ndarray, np.ndarray]:
    """Photo as RGB, sketch as 1-channel grayscale (sketches are grayscale: R=G=B), both size x size.
    RGBA sketches have white under the transparent pixels, so a plain convert is safe."""
    photo_path, sketch_path = pair_paths(image_name)
    with Image.open(photo_path) as p, Image.open(sketch_path) as s:
        photo = p.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        sketch = s.convert("L").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(photo), np.asarray(sketch)


class FS2KDataset(Dataset):
    """Returns (photo [3,128,128], sketch [1,128,128], style) with values in [0, 1].

    augment=True applies pix2pix-style "jitter": load at 143x143, take the SAME random 128x128 crop
    from photo and sketch, and flip BOTH horizontally with probability 0.5. Using one set of random
    numbers for both images keeps them pixel-aligned (different transforms would break the pairing).
    """

    def __init__(self, records: list[dict], augment: bool = False, jitter_size: int = 143):
        self.records = records
        self.augment = augment
        self.jitter_size = jitter_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        if not self.augment:
            photo, sketch = load_pair(r["name"], C.IMG_SIZE)
        else:
            photo, sketch = load_pair(r["name"], self.jitter_size)
            rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,))))  # reproducible, per-worker
            top = int(rng.integers(0, self.jitter_size - C.IMG_SIZE + 1))
            left = int(rng.integers(0, self.jitter_size - C.IMG_SIZE + 1))
            photo = photo[top:top + C.IMG_SIZE, left:left + C.IMG_SIZE]
            sketch = sketch[top:top + C.IMG_SIZE, left:left + C.IMG_SIZE]
            if rng.random() < 0.5:
                photo, sketch = photo[:, ::-1], sketch[:, ::-1]
        return to_tensor(photo), to_tensor(sketch), r["style"]
