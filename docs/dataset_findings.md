# Dataset findings (read-only inspection)

Produced by `python scripts/inspect_datasets.py` on 2026-10-03, plus a one-off check of RGBA alpha channels and sketch colour channels. Nothing under `data/` was modified.

## Oxford-IIIT Pet (Tasks 1-3)

| Item | Result |
|---|---|
| `trainval.txt` | 3,680 ids, all unique, 0 missing, 0 unreadable |
| `test.txt` | 3,669 ids, all unique, 0 missing, 0 unreadable |
| trainval / test overlap | 0 |
| Files in `images/` | 7,390 `.jpg` + 3 `.mat` (`Abyssinian_100/101/102.mat`, junk) = 7,393 |
| `.jpg` not listed in trainval/test | 41 (ignored; we only load listed ids) |
| Image sizes (trainval) | width 114-3264, height 108-2606 (varied aspect ratios) |
| Image sizes (test) | width 137-1646, height 103-2160 |
| Non-RGB images | 3 RGBA: `Egyptian_Mau_14`, `Egyptian_Mau_186` (trainval), `Abyssinian_5` (test; actually a PNG file with a `.jpg` name). Alpha is 255 everywhere, so `.convert("RGB")` is safe. No grayscale or CMYK among listed ids. |
| macOS junk | 7,391 `._*` files under `annotations/` (e.g. `._trimaps`, `trimaps/._*.png`). Not used. |
| 80/20 split, seed 42 | Straightforward: 2,944 train / 736 val (preview only; Phase 1 saves the real split file) |

Note: resizing straight to 128x128 squashes non-square images. The assignment says "resize to 128 x 128", so a direct resize follows the text literally. This is an open decision (see `docs/explanation.md`, risks).

## FS2K (Task 4)

### JSON structure
Both `anno_train.json` and `anno_test.json` are a list of records with these keys (all records have the same key set):

`image_name, skin_color, lip_color, eye_color, hair, hair_color, gender, earring, smile, frontal_face, style`

- **Style label field: `style`**, integer in {0, 1, 2}. In the UI, Style 1/2/3 = `style` 0/1/2 (the official `vis.py` does `style += 1` for display).
- The README calls the point field `skin_patch`, but the JSON key is actually `skin_color`. We don't need it.
- `image_name` looks like `"photo1/image0110"`, with no file extension.

### Photo-sketch pairing
`photo/<photoN>/image<XXXX>.<ext>` pairs with `sketch/<sketchN>/sketch<XXXX>.<ext>`: replace `photo` with `sketch` and `image` with `sketch`. This is the same rule the official `split_train_test.py` and `vis.py` use. Extensions are mixed, so the loader must try `.jpg`, `.JPG` and `.png`:
- train: 960 pairs jpg/jpg, 98 pairs jpg (photo2) / png (sketch2)
- `photo3/image0449.JPG` has an uppercase extension (Windows doesn't care, Linux does)

All 2,104 records resolve to an existing photo and sketch. Every photo has exactly the same pixel size as its sketch, which is a good sign the pairs are aligned.

### Counts
| Split | Records | style 0 | style 1 | style 2 |
|---|---|---|---|---|
| Official train | 1,058 | 357 | 350 | 351 |
| Official test | 1,046 | 619 | 381 | 46 |

- Photos on disk: 2,104; sketches on disk: 2,104; no unreferenced files; no train/test name overlap; 0 byte-identical duplicate photos.
- The README attribute table lists train S2=351, S3=350. The JSON gives style1=350, style2=351. That's a 1-image swap in the README; we trust the JSON.
- **Stratified 15% validation is feasible:** about 54 / 52 / 53 val images per style (~159 val, ~899 train).

### Image properties
| Source | Size | Count (train / test) |
|---|---|---|
| photo1 (CASIA-WebFace) | 250x250 | 707 / 822 |
| photo2 (actors) | 223x318 (one 205x292) | 98 / 0 |
| photo3 (stock) | 475x340 | 253 / 224 |

- Photos are all RGB. Sketches are RGB, plus 8 RGBA PNGs in `sketch2/`. Their transparent pixels are already white underneath, so `.convert("RGB")` is safe.
- Sketches are pure grayscale (R=G=B in all 408 sketches checked), so the generator could output 1 channel. To be decided in Task 4.

### Things to watch (FS2K)
1. **Style is confounded with photo source.** In train, styles 0 and 1 come only from `photo1` (250x250 faces), and style 2 comes only from `photo2` + `photo3` (non-square, different backgrounds). In test, 179 of the style-0 images come from `photo3`, and style 2 has only 46 images. So the model may learn "photo type -> style" instead of a real style. Per-style test results must be read with this in mind, and style-2 test numbers rest on 46 images.
2. **Aspect ratio:** photo2/photo3 are not square, so a direct resize to 128x128 distorts faces. Photo and sketch are distorted identically, so pairing stays correct.
3. Train/test style distributions differ a lot (balanced vs 59% style 0).
