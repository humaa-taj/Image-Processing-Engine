"""Landing-page card images made by OUR models (no stock photos, no invented numbers).

Runs the exported ONNX models on CPU with ONNX Runtime (no GPU) on bundled sample images:
  card1_universal.png  salt-and-pepper (medium) input | Task 1 output       (split view)
  card2_hard.png       blur (medium) input | Task 2 hard-routed output
  card3_soft.png       occlusion (medium) input | Task 3 soft-MoE output
  card3_soft.json      the REAL gate weights for that card-3 image (shown as the card's bars)
  card4_sketch.png     face photo | Task 4 sketch (only if the generator ONNX exists; else photo only)
Output: frontend/src/assets/landing/. Fixed seeds, so re-running gives the same images.
Run:  .venv\\Scripts\\python scripts/make_landing_assets.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as C  # noqa: E402
from src.corruptions import apply_corruption, fixed_test_params  # noqa: E402

OUT = C.ROOT / "frontend" / "src" / "assets" / "landing"
SAMPLES = C.ROOT / "backend" / "samples"
W, H = 480, 360   # card image size (4:3, as in the Stitch design)


def load(path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB").resize((128, 128), Image.Resampling.BILINEAR))


def run(name, x: np.ndarray, extra=None):
    sess = ort.InferenceSession(str(C.ONNX_DIR / f"{name}.onnx"), providers=["CPUExecutionProvider"])
    feed = {sess.get_inputs()[0].name: (x.astype(np.float32) / 255).transpose(2, 0, 1)[None]}
    if extra:
        feed.update(extra)
    return sess.run(None, feed)


def to_img(t: np.ndarray) -> Image.Image:
    a = np.clip(np.rint(t * 255), 0, 255).astype(np.uint8)
    a = a[0] if a.shape[0] == 1 else a.transpose(1, 2, 0)
    return Image.fromarray(a).convert("RGB")


def split(left: Image.Image, right: Image.Image) -> Image.Image:
    """Left half of the 'before' image + right half of the 'after' image, with a thin divider line."""
    big_l, big_r = (im.resize((W, W), Image.Resampling.BICUBIC) for im in (left, right))
    top = (W - H) // 2
    big_l, big_r = big_l.crop((0, top, W, top + H)), big_r.crop((0, top, W, top + H))
    out = big_l.copy()
    out.paste(big_r.crop((W // 2, 0, W, H)), (W // 2, 0))
    px = out.load()
    for y in range(H):
        px[W // 2, y] = (210, 214, 220)
    return out


def corrupted(clean, cond, level, seed):
    return apply_corruption(clean, fixed_test_params(cond, level, np.random.default_rng(seed)))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    dog, cat = load(SAMPLES / "pets/great_pyrenees_54.jpg"), load(SAMPLES / "pets/Bombay_56.jpg")
    dog2 = load(SAMPLES / "pets/keeshond_53.jpg")

    x = corrupted(dog, C.SALT, "medium", 1)
    split(Image.fromarray(x), to_img(run("task1_universal_ae", x)[0][0])).save(OUT / "card1_universal.png")

    x = corrupted(cat, C.BLUR, "medium", 2)
    _, probs = run("task2_classifier", x)
    route = int(np.argmax(probs[0]))
    out = run(["", "task2_expert_salt", "task2_expert_blur", "task2_expert_occlusion"][route], x)[0][0] if route else x / 255
    split(Image.fromarray(x), to_img(out if route else x.transpose(2, 0, 1) / 255)).save(OUT / "card2_hard.png")

    x = corrupted(dog2, C.OCCLUSION, "medium", 5)
    out, weights, _ = run("task3_soft_moe", x)
    split(Image.fromarray(x), to_img(out[0])).save(OUT / "card3_soft.png")
    w = {k: round(float(v), 2) for k, v in zip(["Identity", "Salt-and-pepper", "Blur", "Occlusion"], weights[0])}
    (OUT / "card3_soft.json").write_text(json.dumps(
        {"caption": "Gate weights for this image (occlusion, medium)", "weights": w}, indent=2))

    face_path = sorted((SAMPLES / "faces").glob("*.jpg"))[0]
    face = load(face_path)
    if (C.ONNX_DIR / "task4_generator.onnx").exists():
        sketch = run("task4_generator", face, {"style": np.array([0], dtype=np.int64)})[0][0]
        split(Image.fromarray(face), to_img(sketch)).save(OUT / "card4_sketch.png")
    else:   # generator not trained yet: show only the real photo, no invented sketch
        img = Image.fromarray(face).resize((W, W), Image.Resampling.BICUBIC)
        img.crop((0, (W - H) // 2, W, (W - H) // 2 + H)).save(OUT / "card4_sketch.png")
    print("written:", sorted(p.name for p in OUT.iterdir()), "| card 3 weights:", w, "| card 2 route:", route)


if __name__ == "__main__":
    main()
