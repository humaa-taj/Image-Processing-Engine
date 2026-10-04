# Implementation plan

Seven phases. Each phase ends with a summary, a git commit + push, an updated `PROGRESS.md`, and a STOP for review. Re-read `docs/explanation.md` at the start of every phase. Every phase also adds a short entry to `docs/ai_use_log.md` (what AI produced, how it was checked).

**Reminder (user): confirm the real submission deadline with your instructor.** The PDF date (March 16, 2024) is stale.

**Time estimates are based on a real benchmark** (`scripts/benchmark_epoch.py` -> `artifacts/results/benchmark_epoch.json`, run 2026-10-03 on the RTX 3050 Ti 4 GB, torch 2.14.1+cu126):

| Measured | Result |
|---|---|
| Data pipeline (runtime corruption, CPU) | ~1,400 img/s (0 workers), ~1,900 img/s (2 workers) -> not a bottleneck |
| Stand-in autoencoder (9.8 M params, batch 32, AMP) | **2.9 s / train epoch** (2,944 images) + 0.5 s val pass (736), peak 0.38 GB |
| Stand-in pix2pix GAN (G 41.8 M + D 2.8 M, batch 8, fp32) | **7.9 s / epoch** (899 pairs), peak **2.71 GB** |

The first epoch is ~5x slower (worker start-up, cuDNN autotune), so short trials pay that once. Estimates below add a 1.5-2x margin for larger configs, validation and MLflow logging. Each phase re-estimates before its long runs, and you're told before anything over ~30 min starts.

CLAUDE.md section 0 was updated to these 7 phases (approved 2026-10-03). Approved design decisions are listed in `docs/explanation.md` section 13.

## Dependency overview

```
P1 Foundation ──► P2 Task 1 ──► P3 Task 2 ──► P4 Task 3 ──┐
       │                                                  ├──► P6 App (backend + frontend) ──► P7 Docker, fresh clone, README, report material
       └────────────► P5 Task 4 (independent) ────────────┘
                         ▲ its long GPU runs can go overnight / on Colab while you review other phases
You: Google Stitch design must be ready before P6 frontend work starts.
```

Highest risk / time cost: **P4 (soft-MoE: joint training, collapse, single-graph ONNX)** and **P5 (GAN: instability, longest GPU time)**. Medium: **P7 (Docker on a fresh machine)**.

---

## Phase 1: Foundation (environment, data pipeline, corruptions, manifests, shared tools)

**Goal:** everything the four tasks share, tested once, so later phases only add models.

**Deliverables**
- `.venv` with CUDA PyTorch (after the driver decision), `requirements.txt` pinned; `torch.cuda.is_available()` verified.
- Folder skeleton: `src/`, `scripts/`, `configs/`, `tests/`, `artifacts/` (ignored), `models/onnx/`.
- `src/data/pet.py`: read trainval/test ids, 80/20 seed-42 split saved to `artifacts/splits/pet_split.json`, clean 128x128 cache in `artifacts/cache/` (uint8 `.npy`), Dataset classes for train (runtime corruption) and val/test (manifest corruption).
- `src/corruptions.py`: salt-and-pepper, Gaussian blur, occlusion (non-overlapping rects with exact area control). Every function is pure: given an image + parameters it returns the corrupted image, so the backend can reuse the same code.
- `scripts/make_manifests.py`: `artifacts/manifests/val_manifest.json` (one condition per val image, exactly balanced, training ranges, per-image seed) and `test_manifest.json` (clean + 3 fixed severities x 3 corruptions per test image).
- Balanced batch sampler (exactly B/4 per condition), used by the classifier and MoE.
- `src/data/fs2k.py`: pairing, stratified 15% val split (seed 42) saved to `artifacts/splits/fs2k_split.json`, paired transforms (same random flip/crop for photo + sketch).
- `src/losses.py` (L1 + SSIM combo), `src/metrics.py` (PSNR, SSIM, L1, per-corruption/severity aggregation).
- `src/tracking.py`: small helpers for MLflow (parent/nested runs) + Optuna (SQLite storage in `artifacts/optuna/`, OOM -> `TrialPruned` wrapper).
- `scripts/preview_corruptions.py` -> `docs/figures/corruption_grid.png` (all corruptions x severities, for the report).
- `scripts/benchmark_epoch.py`: times one epoch of a small AE to calibrate all estimates.
- `tests/`: corruption parameter ranges, occlusion area within tolerance, manifest determinism (regenerate -> identical), split reuse, FS2K pairing.
- `.gitignore` exception so `artifacts/splits/`, `artifacts/manifests/` and later `artifacts/optuna/*.db` + final result tables are committed (sizes checked first; each under ~20 MB, otherwise tell the user).

**You:** review the corruption grid and the split/manifest counts. (PyTorch + driver were already done on 2026-10-03.)
**Verify:** `pytest` passes; corruption grid looks right to you; counts: 2,944/736 split, 736 val manifest entries (184 per class), 36,690 test manifest inputs; FS2K ~899/159 split with per-style counts printed.
**Time:** ~2-3 h of coding/review; GPU ~2 min (benchmark only). DONE 2026-10-03.
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 1: environment, data pipeline, corruptions, manifests, shared utilities`

---

## Phase 2: Task 1, Universal denoising autoencoder (+ ONNX tooling)

**Deliverables**
- `src/models/autoencoder.py`: conv encoder -> flattened dense bottleneck (size = `bottleneck_dim`) -> conv decoder. Configurable base channels and dropout. Reused by the Task 2 specialists.
- `scripts/train_task1.py` (single config, MLflow logging, early stopping on val only, checkpoint to `models/checkpoints/`).
- `scripts/optuna_task1.py`: LR, batch size, bottleneck dim, encoder channels, dropout, alpha. **Objective is fixed and independent of alpha** (e.g. `0.5 * val_L1 + 0.5 * (1 - val_SSIM)`), MedianPruner, OOM handling, nested MLflow runs.
- Small ablation: no skips vs one limited skip (only if we decide to use one).
- `scripts/evaluate_task1.py`: test-manifest PSNR/SSIM/L1 per corruption x severity -> `artifacts/results/task1_*.csv` + LaTeX-ready table; 12-example grid + 4 failure cases (target / input / output / error map); training curves.
- `scripts/export_onnx.py` and `scripts/verify_onnx.py` (generic, extended in later phases) -> `models/onnx/task1_universal_ae.onnx`, `artifacts/results/onnx_verification.csv`.

**Verify:** best trial reproduced by the final run; ONNX vs PyTorch max abs diff ~1e-5 or better; figures look sensible.
**Time (estimate):** Optuna ~30 trials x up to 15 epochs ~ 30-45 min (pruning cuts this); final training ~100 epochs ~ 10 min; skip ablation ~10 min; test evaluation (36,690 inputs) ~2 min. **Total GPU ~1 h.**
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 2: Task 1 universal autoencoder, Optuna study, evaluation, ONNX`

---

## Phase 3: Task 2, Classifier + hard-routed specialists

**Deliverables**
- `src/models/classifier.py` (small CNN, 4 logits). The same class later serves as the Task 3 gate.
- `scripts/optuna_task2_classifier.py` (LR, batch, channels, dropout, weight decay; balanced batches) + final training.
- `scripts/optuna_task2_specialists.py`: one shared search for a common architecture (LR, bottleneck, channels, batch, L1/SSIM weight), objective averaged over the three corruptions; then `scripts/train_task2_specialists.py` trains salt, blur and occlusion experts independently, each only on its corruption.
- `src/routing.py`: hard routing with identity bypass.
- `scripts/evaluate_task2.py`: classifier accuracy, macro P/R/F1, per-class metrics, normalized confusion matrix plot; restoration in **oracle** and **predicted** modes per corruption x severity; list and visualize failures caused by misclassification.
- ONNX: classifier + 3 specialists, added to verification.

**Verify:** confusion matrix sane (clean/blur confusion expected and discussed); oracle >= predicted; ONNX diffs small.
**Time (estimate):** classifier Optuna ~20-30 min + final ~5 min; shared specialist Optuna ~30-45 min; 3 final specialists ~10 min each; evaluation (two routing modes) ~5 min. **Total GPU ~1.5-2 h.**
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 3: Task 2 classifier, specialists, hard routing evaluation, ONNX`

---

## Phase 4: Task 3, Soft mixture-of-experts (HIGH RISK)

**Deliverables**
- `src/models/soft_moe.py`: gate (from Task 2 classifier weights) + identity + 3 experts (from Task 2 specialists), `softmax(logits / T)`, weighted sum; returns image and weights.
- `scripts/train_task3.py`: stage 1 gate-only warm-up (experts frozen), stage 2 joint fine-tune with a smaller LR; loss `l1_w*L1 + ssim_w*(1-SSIM) + cls_w*CE + bal_w*balance` on balanced batches.
- `scripts/optuna_task3.py`: fine-tune LR, T, cls weight, balance weight, reconstruction weighting; fixed objective; prune on routing collapse (e.g. any branch mean weight < 0.02 or > 0.9 on balanced val).
- `scripts/evaluate_task3.py`: restoration per corruption x severity (compare to Tasks 1 and 2); mean weights per true type x severity; routing heatmap; dominant-vs-distributed examples; inactive/over-dominant expert check.
- ONNX: whole MoE as **one graph** (outputs: image + 4 weights), verified.

**Risks:** 4 GB VRAM with 3 AEs + gate training at once (mitigation: smaller batch, AMP, gradient accumulation); collapse; the gain over hard routing may be small (that's a valid finding, reported honestly).
**Time (estimate):** one MoE step runs the gate + all 3 experts (~3-4x an AE step, ~10-15 s/epoch); Optuna ~25 trials x ~10 epochs ~ 45-60 min; final ~15-20 min; evaluation ~5 min. **Total GPU ~1-1.5 h.**
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 4: Task 3 soft mixture-of-experts, routing analysis, single-graph ONNX`

---

## Phase 5: Task 4, FS2K style-conditioned cGAN (HIGH TIME COST; independent)

Can be started any time after Phase 1. Its long runs can run overnight, or on Colab/Kaggle for the full retrain.

**Deliverables**
- `src/models/cgan.py`: U-Net generator with style embedding (the embedding is broadcast as extra input channels and/or injected at the bottleneck); PatchGAN discriminator taking photo + sketch + style embedding map.
- `scripts/train_task4.py`: separate logging of D real, D fake, G adv, G L1; val L1/SSIM (+ optional FID/LPIPS if time allows); fixed val photos sampled every N epochs; frequent checkpoints; NaN/divergence guard.
- `scripts/optuna_task4.py`: G LR, D LR, batch size (small), base channels, dropout, embedding dim, lambda_L1; short trials; divergent trial -> pruned. Objective fixed (e.g. val L1 + (1 - SSIM)), not the GAN loss.
- Full retrain of the best config; `scripts/evaluate_task4.py`: test metrics per style, sample grids (same photo in all 3 styles), failure cases.
- Optional (Colab): `notebooks/task4_colab.ipynb` that only calls the same scripts.
- ONNX: generator only (inputs: photo + style id), verified.

**Verify:** losses stay bounded; sample grids improve over time; style changes the output visibly for the same photo.
**Time (estimate):** Optuna ~20 trials x ~20 epochs ~ 1-1.5 h; full retrain ~200 epochs ~ 30-45 min. **Total GPU ~2 h, fine locally (Colab not needed).** Memory is the constraint: the stand-in GAN peaked at 2.71 GB at batch 8, so the search caps batch size and base channels (OOM trials are pruned automatically).
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 5: Task 4 conditional GAN, Optuna, full retrain, ONNX`

---

## Phase 6: Application (FastAPI backend + React/Tailwind frontend)

**You first:** Google Stitch design (screenshots/exports into `docs/stitch/`). Frontend work waits for it.

**Deliverables**
- `backend/app/`: FastAPI with `GET /health` (models loaded, LFS-pointer detection), `GET /samples`, `POST /corrupt`, `POST /restore/universal`, `POST /restore/hard`, `POST /restore/soft`, `POST /sketch`. Upload validation (type, size limit, decodable), same preprocessing as training, onnxruntime CPU, timing in responses, CORS. `backend/requirements.txt` (CPU only, no torch). `backend/samples/` with 15-20 small images (pets + faces; licence-checked).
- Backend tests with FastAPI TestClient.
- `frontend/`: React + Tailwind (Vite), four workspaces with the exact names, following the Stitch design: upload, sample picker, corruption type + severity, probabilities/weights bars, expert highlight, webcam capture, style selector, side-by-side view, download, timing.

**Verify:** backend tests pass; all four workspaces work locally (`uvicorn` + `npm run dev`) on unseen images; results match the ONNX verification.
**Time:** ~1-2 days of coding/review, no GPU.
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 6: FastAPI backend and React/Tailwind frontend`

---

## Phase 7: Docker Compose, fresh-clone test, README, report material

**Deliverables**
- `backend/Dockerfile` (`python:3.11-slim`, pinned), `frontend/Dockerfile` (multi-stage: Node build -> nginx serving the static files and proxying `/api` to `backend`), `.dockerignore` files, `docker-compose.yml` (models mounted `./models/onnx:/app/models:ro`, healthchecks, log rotation, no restart loops).
- Strict Docker safety procedure (CLAUDE.md section 9) at every build.
- Fresh-clone test: clone into a new folder, `docker compose up --build`, open the browser, run all four workspaces.
- `README.md`: overview, requirements, clone, models (already in repo), one command to start, how to reproduce training/evaluation, repo map.
- Report material in `docs/figures/` and `artifacts/results/`: all tables (CSV + LaTeX), figures, Optuna summaries (search space, completed/pruned/failed counts, best trial), ONNX sizes and verification table, architecture diagrams.
- `docs/ai_use_log.md` complete.

**You:** YouTube demo (5-7 min, checklist provided), IEEE LaTeX report, AI-use appendix. Re-confirm the deadline with your instructor.
**Verify:** fresh clone works with one command; `/health` all green; `docker system df` before/after reported.
**Time:** ~0.5-1 day; Docker builds ~5-10 min each.
**AI-use log:** add this phase's entry to `docs/ai_use_log.md`.
**Commit:** `Phase 7: Docker Compose, README, report material`

---

## Proposed repository structure

```
src/
  data/        pet.py, fs2k.py, samplers.py
  models/      autoencoder.py, classifier.py, soft_moe.py, cgan.py
  corruptions.py  losses.py  metrics.py  tracking.py  routing.py
scripts/       inspect_datasets.py, make_manifests.py, preview_corruptions.py, benchmark_epoch.py,
               train_task*.py, optuna_task*.py, evaluate_task*.py, export_onnx.py, verify_onnx.py
configs/       task1.yaml ... task4.yaml (final chosen hyperparameters, written from Optuna results)
tests/         pytest unit tests
artifacts/     splits/, manifests/, cache/, optuna/*.db, results/  (gitignored)
models/
  onnx/        committed .onnx files
  checkpoints/ (gitignored)
backend/       app/, samples/, tests/, requirements.txt, Dockerfile, .dockerignore
frontend/      React + Tailwind (Vite), Dockerfile, nginx.conf, .dockerignore
docs/          assignment.pdf, ieee_format.pdf, explanation.md, implementation_plan.md,
               dataset_findings.md, ai_use_log.md, figures/, stitch/
report/        IEEE LaTeX source (you write it; we provide figures/tables)
docker-compose.yml  README.md  PROGRESS.md  requirements.txt
```

Committed artifacts (approved): `artifacts/splits/`, `artifacts/manifests/`, `artifacts/optuna/*.db`, and final result tables, each checked to be under ~20 MB. `artifacts/cache/` and everything else in `artifacts/` stay ignored.
