"""Read-only dataset inspection for Oxford-IIIT Pet and FS2K.

Prints facts used in docs/dataset_findings.md. Never writes anything under data/.
Run:  python scripts/inspect_datasets.py
"""
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
PET = ROOT / "data" / "oxford-iiit-pet"
FS2K = ROOT / "data" / "fs2k" / "FS2K"


def read_ids(txt):
    """Each line: '<image_id> <class> <species> <breed>'; '#' lines are comments."""
    lines = (txt).read_text().splitlines()
    return [ln.split()[0] for ln in lines if ln.strip() and not ln.startswith("#")]


def size_summary(sizes):
    ws = [w for w, _ in sizes]
    hs = [h for _, h in sizes]
    return f"w {min(ws)}-{max(ws)}, h {min(hs)}-{max(hs)}"


def inspect_pet():
    print("=== Oxford-IIIT Pet ===")
    img_dir = PET / "images"
    for split in ["trainval", "test"]:
        ids = read_ids(PET / "annotations" / f"{split}.txt")
        missing, unreadable, sizes, modes = [], [], [], Counter()
        for i in ids:
            p = img_dir / f"{i}.jpg"
            if not p.exists():
                missing.append(i)
                continue
            try:
                with Image.open(p) as im:
                    im.load()  # force full decode to catch truncated files
                    sizes.append(im.size)
                    modes[im.mode] += 1
            except Exception as e:  # noqa: BLE001 - we want to report any failure
                unreadable.append(f"{i}: {e}")
        print(f"{split}: listed={len(ids)} unique={len(set(ids))} missing={len(missing)} "
              f"unreadable={len(unreadable)}")
        print(f"  modes={dict(modes)}  sizes: {size_summary(sizes)}")
        if missing:
            print("  missing:", missing[:10])
        if unreadable:
            print("  unreadable:", unreadable[:10])
        non_rgb = [i for i in ids if (img_dir / f"{i}.jpg").exists()
                   and Image.open(img_dir / f"{i}.jpg").mode != "RGB"]
        print(f"  non-RGB ids: {non_rgb}")

    tv, te = set(read_ids(PET / "annotations" / "trainval.txt")), set(read_ids(PET / "annotations" / "test.txt"))
    print(f"trainval/test overlap: {len(tv & te)}")
    all_jpgs = {p.stem for p in img_dir.glob("*.jpg")}
    print(f"jpgs in images/: {len(all_jpgs)}; not listed in trainval/test: {len(all_jpgs - tv - te)}")
    print(f"other files in images/: {sorted(p.name for p in img_dir.iterdir() if p.suffix != '.jpg')}")

    # Preview of the 80/20 split with seed 42 (nothing saved here; Phase 1 saves it)
    ids = sorted(tv)
    rng = random.Random(42)
    rng.shuffle(ids)
    n_train = int(0.8 * len(ids))
    print(f"80/20 split preview (seed 42): train={n_train} val={len(ids) - n_train}")


def find_with_ext(stem_path):
    """FS2K mixes .jpg, .JPG and .png; return the existing file or None."""
    for ext in [".jpg", ".JPG", ".png", ".PNG", ".jpeg"]:
        p = stem_path.with_suffix(ext)
        if p.exists():
            return p
    return None


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def inspect_fs2k():
    print("\n=== FS2K ===")
    all_names = []
    for split in ["train", "test"]:
        data = json.loads((FS2K / f"anno_{split}.json").read_text())
        keys = Counter(tuple(sorted(d.keys())) for d in data)
        styles = Counter(d["style"] for d in data)
        by_folder = Counter(d["image_name"].split("/")[0] for d in data)
        print(f"{split}: records={len(data)} unique names={len({d['image_name'] for d in data})}")
        print(f"  key sets: {list(keys.items())}")
        print(f"  style counts (0/1/2): {dict(sorted(styles.items()))}")
        print(f"  photo folder counts: {dict(sorted(by_folder.items()))}")
        style_by_folder = Counter((d["image_name"].split("/")[0], d["style"]) for d in data)
        print(f"  (folder, style): {dict(sorted(style_by_folder.items()))}")

        missing, p_sizes, s_sizes, mismatch, p_modes, s_modes, exts = [], [], [], [], Counter(), Counter(), Counter()
        for d in data:
            name = d["image_name"]  # e.g. "photo1/image0110"
            folder, stem = name.split("/")
            photo = find_with_ext(FS2K / "photo" / folder / stem)
            sketch = find_with_ext(FS2K / "sketch" / folder.replace("photo", "sketch")
                                   / stem.replace("image", "sketch"))
            if photo is None or sketch is None:
                missing.append((name, photo is not None, sketch is not None))
                continue
            exts[(photo.suffix, sketch.suffix)] += 1
            with Image.open(photo) as a, Image.open(sketch) as b:
                p_sizes.append(a.size)
                s_sizes.append(b.size)
                p_modes[a.mode] += 1
                s_modes[b.mode] += 1
                if a.size != b.size:
                    mismatch.append((name, a.size, b.size))
        print(f"  missing photo/sketch: {len(missing)} {missing[:10]}")
        print(f"  ext pairs: {dict(exts)}")
        print(f"  photo modes {dict(p_modes)}; sketch modes {dict(s_modes)}")
        print(f"  photo sizes: {size_summary(p_sizes)}; sketch sizes: {size_summary(s_sizes)}")
        print(f"  photo/sketch size mismatches: {len(mismatch)} {mismatch[:5]}")
        print(f"  distinct photo sizes (top 5): {Counter(p_sizes).most_common(5)}")
        all_names += [d["image_name"] for d in data]

    print(f"train/test name overlap: {len(all_names) - len(set(all_names))}")

    # Files on disk not referenced by any JSON record
    referenced = set(all_names)
    on_disk = {f"{p.parent.name}/{p.stem}" for p in (FS2K / "photo").glob("*/*") if p.is_file()}
    print(f"photos on disk={len(on_disk)}, unreferenced={sorted(on_disk - referenced)[:10]}")
    sk_disk = [p for p in (FS2K / "sketch").glob("*/*") if p.is_file()]
    print(f"sketches on disk={len(sk_disk)}")

    # Exact duplicate photos (identical bytes) across the whole set
    hashes = Counter(md5(p) for p in (FS2K / "photo").glob("*/*") if p.is_file())
    print(f"exact duplicate photo files: {sum(c - 1 for c in hashes.values() if c > 1)}")


if __name__ == "__main__":
    inspect_pet()
    inspect_fs2k()
