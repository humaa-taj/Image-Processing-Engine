# API contract (frontend <-> FastAPI backend)

Implemented in `backend/app/main.py` and tested in `backend/tests/test_api.py`. All paths start with `/api`.
In Docker Compose the frontend's nginx proxies `/api` to the backend service; in development Vite proxies `/api` to `http://localhost:8000`.

## Conventions

- **Requests** that send images are `multipart/form-data`. An image is given EITHER as an uploaded `file` (JPEG / PNG / WebP / BMP, max 10 MB) OR as `sample` = a bundled sample name from `GET /api/samples` (e.g. `pets/beagle_54.jpg`).
- **Images in responses** are PNG data URLs (`"data:image/png;base64,..."`), 128 x 128 pixels (the models' resolution). The browser can show them directly in `<img src>`.
- **Numbers** are plain JSON numbers. Times are milliseconds measured on the server (`timing_ms`).
- **Errors** use HTTP status codes with `{"detail": "<human readable message>"}`:
  `400` image could not be decoded, `404` unknown sample, `413` file too large, `415` unsupported file type, `422` invalid field value, `503` model not loaded (see `/api/health`).

### Corruption fields (restoration endpoints and `/api/corrupt`)

| Field | Values | Default |
|---|---|---|
| `corruption` | `none`, `salt`, `blur`, `occlusion` | `none` |
| `level` | `low`, `medium`, `high` | `medium` |
| `seed` | integer, optional | random (returned in the response) |
| `input_is_clean` | `true` / `false` | `false` |

The severity of each level is fixed and identical to the official test manifest (`src/config.py`, applied by `src/corruptions.py`):

| Corruption | low | medium | high |
|---|---|---|---|
| `salt` (probability p) | 0.03 | 0.08 | 0.15 |
| `blur` (kernel, sigma) | (3, 0.7) | (5, 1.5) | (7, 2.5) |
| `occlusion` (rectangles, area) | 1, ~10% | 2, ~20% | 3, ~35% |

`corruption = none` means "use the image as it is" (e.g. an upload that is already corrupted).
`seed` makes salt-and-pepper pixels and occlusion rectangles reproducible: the seed returned by `/api/corrupt` can be sent to a restore endpoint to restore exactly the previewed image.

### The `corruption` object in responses

`null` when no corruption was applied, otherwise:
```json
{"type": "salt", "level": "medium", "seed": 123456, "severity": 0.08,
 "params": {"type": "salt", "p": 0.08, "seed": 123456}}
```
`params` is exactly what `src/corruptions.py` applied: `{"type":"blur","kernel":5,"sigma":1.5}` or `{"type":"occlusion","rects":[[y,x,h,w],...],"target_area":0.2}`. `severity` = p (salt), sigma (blur) or covered area fraction (occlusion).

### Clean reference and error map (restoration endpoints)

An error map needs the clean original. The backend returns one only when the clean image is really known:
- a corruption was applied by the backend (`corruption != none`): the clean image is the image before corruption;
- or the client says the image itself is clean (`input_is_clean = true`, e.g. a clean sample restored as is).
Otherwise (e.g. an uploaded image that is already corrupted) these fields are `null` and `reference_available` is `false`. The UI must not fake an error map.

```json
"reference_available": true,
"clean_image": "data:image/png;base64,...",
"error_map_image": "data:image/png;base64,...",   // |output - clean| averaged over RGB, colour ramp 0 .. 0.5
"mean_abs_error": 0.0312                           // mean of |output - clean|, images in [0, 1]
```

## Endpoints

### `GET /api/health`
```json
{"status": "ok",                        // "ok" if all 7 models loaded, else "degraded"
 "models_loaded": 7, "models_expected": 7,
 "lfs_pointer_files": [],               // models that are Git LFS pointer files instead of real files
 "models": {"task1_universal_ae": {"description": "...", "file": "task1_universal_ae.onnx",
                                    "loaded": true, "lfs_pointer": false, "size_mb": 28.55}, ...},
 "onnxruntime": "1.30.0", "providers": ["CPUExecutionProvider"]}
```
A model that failed has `"loaded": false` and an `"error"` string.

### `GET /api/samples`
```json
{"pets":  [{"name": "pets/beagle_54.jpg", "url": "/api/samples/files/pets/beagle_54.jpg"}, ...],
 "faces": [{"name": "faces/fs2k_photo3_image0113.jpg", "url": "/api/samples/files/faces/..."}, ...]}
```
`pets` = 16 clean Oxford-IIIT Pet test images (Tasks 1-3); `faces` = FS2K stock photos (Task 4).

### `POST /api/corrupt` (Apply corruption)
Form: `file` | `sample`, `corruption` (required, not `none`), `level`, `seed`.
```json
{"corruption": {...}, "input_image": "data:...", "clean_image": "data:...", "timing_ms": {"total": 12.3}}
```

### `POST /api/restore/universal` (Task 1, Universal Restoration)
Form: `file` | `sample`, corruption fields.
```json
{"task": "Universal Restoration", "corruption": {...} | null,
 "input_image": "data:...", "output_image": "data:...",
 "reference_available": true, "clean_image": "...", "error_map_image": "...", "mean_abs_error": 0.03,
 "timing_ms": {"inference": 31.5, "total": 58.2}}
```

### `POST /api/restore/hard` (Task 2, Hard-Routed Restoration)
Form: as universal.
```json
{"task": "Hard-Routed Restoration", "corruption": {...} | null,
 "probabilities": {"clean": 0.01, "salt": 0.0, "blur": 0.98, "occlusion": 0.01},   // softmax, sums to 1
 "predicted_class": "blur",                       // argmax of the probabilities
 "selected_expert": "blur expert",               // "identity" when predicted_class == "clean" (bypass, no expert runs)
 "input_image": "...", "output_image": "...", <reference fields>,
 "timing_ms": {"classifier": 4.1, "expert": 27.9, "inference": 32.0, "total": 60.4}}   // expert = 0 for identity
```

### `POST /api/restore/soft` (Task 3, Soft Mixture-of-Experts Restoration)
Form: as universal.
```json
{"task": "Soft Mixture-of-Experts Restoration", "corruption": {...} | null,
 "weights": {"identity": 0.55, "salt expert": 0.0, "blur expert": 0.44, "occlusion expert": 0.01},  // sum to 1
 "ranking": ["identity", "blur expert", "occlusion expert", "salt expert"],   // by weight, largest first
 "dominant_branch": "identity",
 "top_contributors": ["identity", "blur expert"],   // branches with weight >= top_contributor_threshold (at least 1)
 "top_contributor_threshold": 0.1,
 "input_image": "...", "output_image": "...", <reference fields>,
 "timing_ms": {"inference": 95.0, "total": 120.3}}
```

### `POST /api/sketch` (Task 4, Face-to-Sketch Generator)
Form: `file` | `sample`, `style` (1, 2 or 3; UI default 1), `fit` (`crop` = centre square crop, default; `stretch` = direct resize as in training, for photos that are already a cropped face). A webcam capture is sent as an uploaded PNG `file`.
```json
{"task": "Face-to-Sketch Generator", "style": 2,
 "photo_image": "data:...",      // the 128x128 photo the generator saw
 "sketch_image": "data:...",     // grayscale 128x128 sketch
 "timing_ms": {"inference": 40.2, "total": 61.0}}
```

## Mock mode (frontend only)

`VITE_MOCK=1` makes `frontend/src/api.js` return placeholder responses with the same shapes, for UI work without a backend. It is OFF by default; when ON the UI shows a "MOCK DATA" badge in the top bar. Mock output is never real and must never be used in the report.
