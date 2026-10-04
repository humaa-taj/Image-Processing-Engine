"""Oxford-IIIT Pet: split, 128x128 cache, and PyTorch datasets for Tasks 1-3.

Flow:
  prepare_data.py -> make_split()  -> artifacts/splits/pet_split.json   (seed 42, saved once)
                  -> build_cache() -> artifacts/cache/pet_{trainval,test}.npy  (clean uint8 images)
  training        -> PetTrainDataset   (corruption sampled at runtime, every time an image is loaded)
  val / test      -> PetManifestDataset (corruption read from the fixed manifest)

The cache holds only CLEAN resized images (no corrupted copies), so decoding big JPEGs is done
once instead of every epoch. Resize is a direct 128x128 resize (no crop/pad), as the assignment
says; non-square images get squashed (documented decision).
"""
import random

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from src import config as C
from src.corruptions import apply_corruption, sample_training_params
from src.utils import load_json, save_json, to_tensor


def read_ids(split: str) -> list[str]:
    """Image ids from the official trainval.txt / test.txt (first column)."""
    lines = (C.PET_DIR / "annotations" / f"{split}.txt").read_text().splitlines()
    return [ln.split()[0] for ln in lines if ln.strip() and not ln.startswith("#")]


def load_image(image_id: str) -> np.ndarray:
    """Open one image, force RGB (3 files are RGBA), resize to 128x128. Returns uint8 HxWx3."""
    with Image.open(C.PET_DIR / "images" / f"{image_id}.jpg") as im:
        im = im.convert("RGB").resize((C.IMG_SIZE, C.IMG_SIZE), Image.Resampling.BILINEAR)
        return np.asarray(im, dtype=np.uint8)


def make_split(seed: int = C.SEED, train_fraction: float = 0.8) -> dict:
    """80/20 split of the official trainval ids. Sorted first so the result never depends on file order."""
    ids = sorted(read_ids("trainval"))
    random.Random(seed).shuffle(ids)
    n_train = int(train_fraction * len(ids))
    return {"seed": seed, "train": sorted(ids[:n_train]), "val": sorted(ids[n_train:])}


def load_split() -> dict:
    return load_json(C.SPLITS / "pet_split.json")


def cache_path(name: str):
    return C.CACHE / f"pet_{name}.npy"


def build_cache(name: str, ids: list[str]) -> None:
    """Decode + resize every image once and store them in one uint8 array (N, 128, 128, 3)."""
    arr = np.stack([load_image(i) for i in ids])
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.save(cache_path(name), arr)
    save_json(ids, C.CACHE / f"pet_{name}_ids.json")


def load_cached(name: str, ids: list[str]) -> np.ndarray:
    """Return the cached images for `ids` (in that order). name = 'trainval' or 'test'."""
    arr = np.load(cache_path(name), mmap_mode="r")
    index = {i: n for n, i in enumerate(load_json(C.CACHE / f"pet_{name}_ids.json"))}
    return np.ascontiguousarray(arr[[index[i] for i in ids]])


class CachedImages:
    """Lazy, memory-mapped view of some images in the cache (behaves like an array of images).

    On Windows each DataLoader worker is a fresh process and receives a pickled copy of the
    dataset. Passing a 145 MB array makes every worker start slowly; this object only pickles the
    file name and the row numbers, and each worker memory-maps the .npy file itself.
    """

    def __init__(self, name: str, ids: list[str]):
        index = {i: n for n, i in enumerate(load_json(C.CACHE / f"pet_{name}_ids.json"))}
        self.name, self.rows, self._arr = name, [index[i] for i in ids], None

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        if self._arr is None:
            self._arr = np.load(cache_path(self.name), mmap_mode="r")
        return np.asarray(self._arr[self.rows[i]])

    def __getstate__(self):  # don't pickle the open memory map
        return {**self.__dict__, "_arr": None}


class PetTrainDataset(Dataset):
    """Training data with runtime corruption.

    dataset[i]                -> condition picked uniformly at random (1/4 each), fresh severity
    dataset[(i, condition)]   -> fixed condition, fresh severity (used by the balanced sampler
                                 and by the Task 2 specialists that only see one corruption)
    Returns (corrupted, clean, condition_label) as float tensors in [0, 1] and an int label.
    """

    def __init__(self, images: np.ndarray, conditions=(0, 1, 2, 3)):
        self.images = images
        self.conditions = list(conditions)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, key):
        # Seed NumPy from torch's RNG: torch seeds each DataLoader worker differently,
        # so workers never repeat the same corruption, and runs stay reproducible.
        rng = np.random.default_rng(int(torch.randint(0, 2 ** 31, (1,))))
        if isinstance(key, tuple):
            idx, cond = key
        else:
            idx, cond = key, self.conditions[rng.integers(len(self.conditions))]
        clean = self.images[idx]
        corrupted = apply_corruption(clean, sample_training_params(cond, rng))
        return to_tensor(corrupted), to_tensor(clean), cond


class PetManifestDataset(Dataset):
    """Validation / test data with deterministic corruption from a manifest.

    Each manifest entry: {"id", "condition", "level", "params"}.
    Returns (corrupted, clean, condition_label, entry_index).
    """

    def __init__(self, images_by_id: dict, entries: list[dict]):
        self.images_by_id = images_by_id
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        clean = self.images_by_id[e["id"]]
        corrupted = apply_corruption(clean, e["params"])
        return to_tensor(corrupted), to_tensor(clean), e["condition"], i


def load_manifest_dataset(split: str) -> PetManifestDataset:
    """split = 'val' or 'test'. Loads the cached clean images and the saved manifest."""
    manifest = load_json(C.MANIFESTS / f"{split}_manifest.json")
    ids = sorted({e["id"] for e in manifest["entries"]})
    images = load_cached("trainval" if split == "val" else "test", ids)
    return PetManifestDataset(dict(zip(ids, images)), manifest["entries"])
