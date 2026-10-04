"""API tests against the real ONNX files in models/onnx (run from the repo root):
    .venv\\Scripts\\python -m pytest backend/tests
Endpoints whose model is not exported yet must answer 503 (not crash)."""
import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from backend.app.main import app
from backend.app.models import store


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:    # 'with' runs the start-up hook that loads the models
        yield c


def png_bytes(size=(200, 150)) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(np.random.default_rng(0).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def needs(name):
    return pytest.mark.skipif(not (store.models_dir / f"{name}.onnx").exists(), reason=f"{name}.onnx not exported yet")


def test_health(client):
    r = client.get("/api/health").json()
    assert r["models_expected"] == 7 and r["lfs_pointer_files"] == []
    assert r["models"]["task1_universal_ae"]["loaded"]


def test_samples(client):
    s = client.get("/api/samples").json()
    assert len(s["pets"]) >= 10 and len(s["faces"]) >= 5
    assert client.get(s["pets"][0]["url"]).status_code == 200


@needs("task1_universal_ae")
def test_universal_with_sample_and_corruption(client):
    r = client.post("/api/restore/universal", data={"sample": "pets/beagle_54.jpg", "corruption": "salt",
                                                     "level": "high", "seed": "7"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["corruption"]["params"]["p"] == 0.15 and j["output_image"].startswith("data:image/png;base64,")
    # same seed -> same corruption (deterministic, like the test manifest)
    j2 = client.post("/api/restore/universal", data={"sample": "pets/beagle_54.jpg", "corruption": "salt",
                                                      "level": "high", "seed": "7"}).json()
    assert j2["input_image"] == j["input_image"]


@needs("task2_classifier")
def test_hard_routing_upload(client):
    r = client.post("/api/restore/hard", files={"file": ("x.png", png_bytes(), "image/png")},
                    data={"corruption": "blur", "level": "high"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-3
    assert j["predicted_class"] in ("clean", "salt", "blur", "occlusion") and "expert" in j["timing_ms"]


def test_soft_and_sketch_endpoints(client):
    for path, data in (("/api/restore/soft", {"sample": "pets/beagle_54.jpg"}),
                       ("/api/sketch", {"sample": "faces/fs2k_photo3_image0113.jpg", "style": "2"})):
        name = "task3_soft_moe" if "soft" in path else "task4_generator"
        r = client.post(path, data=data)
        assert r.status_code == (200 if store.available(name) else 503), r.text


def test_upload_validation(client):
    bad_type = client.post("/api/restore/universal", files={"file": ("a.txt", b"hello", "text/plain")})
    assert bad_type.status_code == 415
    not_image = client.post("/api/restore/universal", files={"file": ("a.png", b"not really a png", "image/png")})
    assert not_image.status_code == 400
    traversal = client.post("/api/restore/universal", data={"sample": "../app/main.py"})
    assert traversal.status_code == 404
    bad_level = client.post("/api/restore/universal", data={"sample": "pets/beagle_54.jpg", "corruption": "salt",
                                                             "level": "extreme"})
    assert bad_level.status_code == 422


def test_corrupt_preview_matches_restore_input(client):
    """'Apply corruption' preview and the later restore use the same seed -> identical corrupted input."""
    form = {"sample": "pets/beagle_54.jpg", "corruption": "occlusion", "level": "high", "seed": "3"}
    prev = client.post("/api/corrupt", data=form).json()
    assert len(prev["corruption"]["params"]["rects"]) == 3
    rest = client.post("/api/restore/universal", data=form).json()
    assert rest["input_image"] == prev["input_image"]


def test_error_map_only_with_a_real_clean_reference(client):
    sample = "pets/beagle_54.jpg"
    applied = client.post("/api/restore/universal", data={"sample": sample, "corruption": "blur", "level": "low"}).json()
    assert applied["reference_available"] and applied["error_map_image"].startswith("data:image/png")
    clean_in = client.post("/api/restore/universal", data={"sample": sample, "input_is_clean": "true"}).json()
    assert clean_in["reference_available"]
    already = client.post("/api/restore/universal", files={"file": ("x.png", png_bytes(), "image/png")}).json()
    assert not already["reference_available"] and already["error_map_image"] is None   # never faked


@needs("task3_soft_moe")
def test_soft_weights_and_top_contributors(client):
    j = client.post("/api/restore/soft", data={"sample": "pets/beagle_54.jpg", "corruption": "blur", "level": "medium"}).json()
    w = j["weights"]
    assert abs(sum(w.values()) - 1) < 1e-3 and j["ranking"][0] == j["dominant_branch"]
    assert j["top_contributors"] and all(w[b] >= 0.1 for b in j["top_contributors"][1:])
