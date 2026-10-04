"""FastAPI backend: one API for the four workspaces.

  GET  /api/health              which models are loaded (and Git LFS pointer detection)
  GET  /api/samples             bundled sample images (pets for Tasks 1-3, faces for Task 4)
  POST /api/corrupt             preview of a corruption (same code + severities as the test set)
  POST /api/restore/universal   Task 1  Universal Restoration
  POST /api/restore/hard        Task 2  Hard-Routed Restoration
  POST /api/restore/soft        Task 3  Soft Mixture-of-Experts Restoration
  POST /api/sketch              Task 4  Face-to-Sketch Generator

Restoration endpoints take a multipart form: an uploaded `file` OR a `sample` name, plus optional
`corruption` (none | salt | blur | occlusion), `level` (low | medium | high) and `seed`.
The corruption is applied on the server with the same code and severities as the test set, so
"pick a clean sample + apply a corruption" and "upload an already corrupted image" both work.
Error maps need the clean original: it is known when a corruption is applied here, or when the
client says the input itself is clean (`input_is_clean=true`). Otherwise no error map is returned.
The full request/response contract is in docs/api_contract.md.
"""
import time
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

from src import config as C

from .config import CORS_ORIGINS, MODEL_FILES, SAMPLES_DIR
from .images import corrupt, error_map, read_upload, to_batch, to_model_array, to_png_data_url
from .models import store

BRANCHES = ["identity", "salt expert", "blur expert", "occlusion expert"]   # gate / classifier order
EXPERT_MODELS = {1: "task2_expert_salt", 2: "task2_expert_blur", 3: "task2_expert_occlusion"}
TOP_CONTRIBUTOR_MIN = 0.10   # soft-MoE: a branch "contributes" if its weight is at least 10%


@asynccontextmanager
async def lifespan(app: FastAPI):
    store.load_all()                 # load every ONNX model once, at start-up
    yield


app = FastAPI(title="Image Restoration & Face-to-Sketch API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
if SAMPLES_DIR.exists():
    app.mount("/api/samples/files", StaticFiles(directory=SAMPLES_DIR), name="samples")


def require(*names: str) -> None:
    missing = [n for n in names if not store.available(n)]
    if missing:
        raise HTTPException(503, f"Model(s) not loaded: {', '.join(missing)}. See /api/health.")


async def load_input(file: UploadFile | None, sample: str | None, fit: str = "stretch") -> np.ndarray:
    """The uploaded file or a bundled sample -> uint8 128x128x3."""
    if sample:
        path = (SAMPLES_DIR / sample).resolve()
        if SAMPLES_DIR.resolve() not in path.parents or not path.is_file():   # no '../' tricks
            raise HTTPException(404, f"Unknown sample '{sample}'.")
        with Image.open(path) as im:
            return to_model_array(im.convert("RGB"), fit)
    if file is None:
        raise HTTPException(422, "Send an image file or choose a sample.")
    return to_model_array(await read_upload(file), fit)


async def prepare(file, sample, corruption, level, seed, input_is_clean=False):
    """Returns (start time, model input uint8, corruption info, clean reference or None)."""
    t0 = time.perf_counter()
    original = await load_input(file, sample)
    arr, info = corrupt(original, corruption, level, seed)
    clean = original if (info is not None or input_is_clean) else None   # only when really known
    return t0, arr, info, clean


def reference_fields(output: np.ndarray, clean) -> dict:
    """Error map + clean image, only when a clean reference exists (never faked)."""
    if clean is None:
        return {"reference_available": False, "clean_image": None, "error_map_image": None, "mean_abs_error": None}
    img, mae = error_map(output, clean)
    return {"reference_available": True, "clean_image": to_png_data_url(clean), "error_map_image": img,
            "mean_abs_error": mae}


def ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 1)


@app.get("/api/health")
def health():
    loaded = sum(s["loaded"] for s in store.status.values())
    return {"status": "ok" if loaded == len(MODEL_FILES) else "degraded",
            "models_loaded": loaded, "models_expected": len(MODEL_FILES),
            "lfs_pointer_files": [n for n, s in store.status.items() if s["lfs_pointer"]],
            "models": store.status, "onnxruntime": ort.__version__, "providers": ort.get_available_providers()}


@app.get("/api/samples")
def samples():
    def listing(sub: str):
        folder = SAMPLES_DIR / sub
        files = sorted(p.name for p in folder.glob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png")) \
            if folder.exists() else []
        return [{"name": f"{sub}/{f}", "url": f"/api/samples/files/{sub}/{f}"} for f in files]
    return {"pets": listing("pets"), "faces": listing("faces")}


@app.post("/api/corrupt")
async def corrupt_preview(file: UploadFile | None = File(None), sample: str | None = Form(None),
                          corruption: str = Form(...), level: str = Form("medium"), seed: int | None = Form(None)):
    """'Apply corruption' in the UI. Returns the corrupted 128x128 image and the seed; sending the same
    seed to a restore endpoint reproduces exactly this corruption."""
    t0, arr, info, clean = await prepare(file, sample, corruption, level, seed)
    return {"corruption": info, "input_image": to_png_data_url(arr), "clean_image": to_png_data_url(clean),
            "timing_ms": {"total": ms(t0)}}


@app.post("/api/restore/universal")
async def restore_universal(file: UploadFile | None = File(None), sample: str | None = Form(None),
            input_is_clean: bool = Form(False),
                            corruption: str = Form("none"), level: str = Form("medium"), seed: int | None = Form(None)):
    require("task1_universal_ae")
    t0, arr, info, clean = await prepare(file, sample, corruption, level, seed, input_is_clean)
    (out,), t_inf = store.run("task1_universal_ae", {"input": to_batch(arr)})
    return {"task": "Universal Restoration", "corruption": info, "input_image": to_png_data_url(arr),
            "output_image": to_png_data_url(out[0]), **reference_fields(out[0], clean),
            "timing_ms": {"inference": round(t_inf, 1), "total": ms(t0)}}


@app.post("/api/restore/hard")
async def restore_hard(file: UploadFile | None = File(None), sample: str | None = Form(None),
            input_is_clean: bool = Form(False),
                       corruption: str = Form("none"), level: str = Form("medium"), seed: int | None = Form(None)):
    require("task2_classifier", *EXPERT_MODELS.values())
    t0, arr, info, clean = await prepare(file, sample, corruption, level, seed, input_is_clean)
    x = to_batch(arr)
    (_, probs), t_cls = store.run("task2_classifier", {"input": x})
    route = int(np.argmax(probs[0]))                       # r = argmax_k p_k
    if route == C.CLEAN:                                   # identity bypass: no expert runs
        out, t_exp = x[0], 0.0
    else:
        (o,), t_exp = store.run(EXPERT_MODELS[route], {"input": x})
        out = o[0]
    return {"task": "Hard-Routed Restoration", "corruption": info,
            "probabilities": {c: round(float(p), 4) for c, p in zip(C.CONDITIONS, probs[0])},
            "predicted_class": C.CONDITIONS[route], "selected_expert": BRANCHES[route],
            "input_image": to_png_data_url(arr), "output_image": to_png_data_url(out), **reference_fields(out, clean),
            "timing_ms": {"classifier": round(t_cls, 1), "expert": round(t_exp, 1),
                          "inference": round(t_cls + t_exp, 1), "total": ms(t0)}}


@app.post("/api/restore/soft")
async def restore_soft(file: UploadFile | None = File(None), sample: str | None = Form(None),
            input_is_clean: bool = Form(False),
                       corruption: str = Form("none"), level: str = Form("medium"), seed: int | None = Form(None)):
    require("task3_soft_moe")
    t0, arr, info, clean = await prepare(file, sample, corruption, level, seed, input_is_clean)
    (out, weights, _), t_inf = store.run("task3_soft_moe", {"input": to_batch(arr)})
    w = {b: round(float(v), 4) for b, v in zip(BRANCHES, weights[0])}
    ranking = sorted(w, key=w.get, reverse=True)
    # "Top contributors" = branches with at least 10% of the mixture (always at least the largest one).
    top = [b for b in ranking if w[b] >= TOP_CONTRIBUTOR_MIN] or ranking[:1]
    return {"task": "Soft Mixture-of-Experts Restoration", "corruption": info, "weights": w,
            "ranking": ranking, "dominant_branch": ranking[0], "top_contributors": top,
            "top_contributor_threshold": TOP_CONTRIBUTOR_MIN,
            "input_image": to_png_data_url(arr), "output_image": to_png_data_url(out[0]),
            **reference_fields(out[0], clean), "timing_ms": {"inference": round(t_inf, 1), "total": ms(t0)}}


@app.post("/api/sketch")
async def sketch(file: UploadFile | None = File(None), sample: str | None = Form(None),
                 style: int = Form(1), fit: str = Form("crop")):
    require("task4_generator")
    if style not in (1, 2, 3):
        raise HTTPException(422, "style must be 1, 2 or 3")
    if fit not in ("crop", "stretch"):
        raise HTTPException(422, "fit must be 'crop' or 'stretch'")
    t0 = time.perf_counter()
    arr = await load_input(file, sample, fit)
    (out,), t_inf = store.run("task4_generator", {"photo": to_batch(arr),
                                                  "style": np.array([style - 1], dtype=np.int64)})
    return {"task": "Face-to-Sketch Generator", "style": style, "photo_image": to_png_data_url(arr),
            "sketch_image": to_png_data_url(out[0]), "timing_ms": {"inference": round(t_inf, 1), "total": ms(t0)}}
