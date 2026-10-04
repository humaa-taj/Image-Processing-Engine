# Progress

Resume guide: read `CLAUDE.md`, then `docs/explanation.md`, then `docs/implementation_plan.md`, then this file.

## Done
- **Phase 0, setup and planning (2026-10-03)**
  - Read `docs/assignment.pdf` (PDFs were already in `docs/`).
  - Read-only dataset inspection: `scripts/inspect_datasets.py`, results in `docs/dataset_findings.md`.
  - Environment and git checks (see Known issues).
  - Wrote `docs/explanation.md` (knowledge base) and `docs/implementation_plan.md` (7 phases), plus `docs/ai_use_log.md`.

- **Plan approved + environment (2026-10-03)**
  - NVIDIA driver updated by user to 616.92 (CUDA 13.4).
  - `.venv` (Python 3.11.7) with torch 2.14.1+cu126, torchvision 0.29.1+cu126. Verified: `torch.cuda.is_available()` = True, CUDA 12.6, cuDNN 9.10, RTX 3050 Ti 4 GiB, GPU matmul + fp16 autocast conv OK.
  - Approved decisions recorded in `docs/explanation.md` section 13; CLAUDE.md section 0 updated to 7 phases; plan updated (AI-log entry per phase, deadline reminder, committed artifacts).

- **Phase 1, foundation (2026-10-03)**
  - `requirements.txt` pinned; packages installed in `.venv`.
  - `src/`: config, corruptions (NumPy-only), Pet data + cache, FS2K data + paired augmentation, balanced sampler, SSIM/L1 loss, metrics + fixed Optuna objective, MLflow (SQLite) + Optuna helpers (OOM/divergence -> pruned).
  - `scripts/prepare_data.py` built: Pet split 2,944/736; cache (181 + 180 MB, gitignored); val manifest 736 (184/class); test manifest 36,690; FS2K split 899/159/1,046 (val styles 54/52/53). Re-run verified byte-identical.
  - `docs/figures/corruption_grid.png`; 22 pytest tests pass.
  - Benchmark (logged to MLflow `phase1-benchmark`): AE 2.8-2.9 s/epoch, GAN 7.9 s/epoch (2.71 GB peak). Plan estimates updated.
  - Fixed `.gitignore`: `data/` was also hiding `src/data/` -> now `/data/`.

- **Phase 2, Task 1 universal autoencoder (2026-10-03)**
  - SSIM checked vs scikit-image on 160 real images: max diff 1.9e-5 (`artifacts/results/ssim_check.json`).
  - Model: conv encoder 128->8 px, 8x8xc latent (no skips), upsample+conv decoder. Best: base 64, bottleneck 2048 (24x compression), 7.12 M params.
  - Optuna `task1` (artifacts/optuna/task1.db): 30 trials = 16 completed, 14 pruned, 0 failed (0 OOM); best trial #24, val score 0.1495; ~70 min. Best params in `configs/task1.yaml` (lr 5.8e-4, batch 16, dropout 0.013, alpha 0.558).
  - Final training: 100 epochs (~25 min), best epoch 96: val score 0.1067, SSIM 0.8215, PSNR 26.11.
  - Test (36,690 inputs, run once): all corrupted inputs PSNR 19.71 -> 25.55 dB, SSIM 0.639 -> 0.806. Salt +10.8 dB; occlusion +8.9 dB; blur -2.1 dB overall (low/medium blur made worse); clean 27.29 dB / 0.849 (quality lost on clean input). Full table: `artifacts/results/task1_test_summary.csv` / `.tex`.
  - Figures: `docs/figures/task1_{curves,examples,failures,optuna_history,optuna_importance}.png`.
  - ONNX `models/onnx/task1_universal_ae.onnx` 28.55 MB; vs PyTorch max diff 4.05e-6, mean 5.7e-8 (PASS).

- **Phase 3, Task 2 classifier + hard-routed specialists (2026-10-04)**
  - Classifier Optuna `task2_classifier`: 25 trials = 7 completed, 18 pruned, 0 failed (~20 min); best #23 (val CE 0.0135). Final: val acc 99.73%. `configs/task2_classifier.yaml`.
  - Specialist Optuna `task2_specialists` (shared search, each trial trains all 3): 20 trials = 10 completed, 10 pruned, 0 failed (~48 min); best #17 (mean val score 0.1735): base 40, bottleneck 2048, lr 2.1e-3, batch 16, alpha 0.53. `configs/task2_specialists.yaml`.
  - Final specialists (100 epochs each, ~15 min each): val SSIM salt 0.844, blur 0.848, occlusion 0.739.
  - Test (once): classifier acc 99.83%, macro-F1 0.997; 63/36,690 misrouted (41 clean->blur, 3 clean->occlusion, 19 low occlusion->clean). Oracle vs predicted overall SSIM 0.8269 vs 0.8268. vs Task 1: overall SSIM 0.810 -> 0.827 (identity on clean); per-corruption SSIM equal, PSNR 0.3-0.8 dB lower (smaller experts).
  - Files: `artifacts/results/task2_*` (comparison table .csv/.tex, classifier metrics, misrouting), `docs/figures/task2_{confusion_matrix,examples,misrouted,classifier_curves,specialists_curves,*_optuna_*}.png`.
  - ONNX: classifier 10.61 MB (outputs logits + probs), 3 experts 11.23 MB each; all verified (max diff <= 1.1e-5). Task 3 single file estimate ~44 MB.

- **Phase 4, Task 3 soft mixture-of-experts (2026-10-04)**
  - Optuna `task3`: 20 trials = 16 completed, 4 pruned, 0 failed; best #17 (val 0.0720): T 2.63, lr 2.7e-4, cls_w 0.010, bal_w 0.0011, l1/ssim 0.59/0.41. `configs/task3.yaml`.
  - Final: val score 0.0860 (Task 2 init) -> 0.0702; val SSIM 0.859 -> 0.885; gate top-1 acc 99.7% -> 77.5% (blends instead of classifying).
  - Test: corrupted SSIM 0.806 / 0.808 / 0.845 and PSNR 25.55 / 24.93 / 26.18 dB (Task 1 / Task 2 / Task 3). Low blur 33.1 dB (was 26.3). 63 Task-2 misrouted inputs: SSIM 0.891 -> 0.974, better on 95%.
  - Routing: identity weight grows as damage gets milder (blur 79/55/46%, occlusion 52/38/27%); salt ~96% salt expert. No inactive experts.
  - ONNX `task3_soft_moe.onnx` 44.36 MB (one graph), verified max diff 2.3e-5.
  - Files: `artifacts/results/task3_*`, `docs/figures/task3_*` (routing heatmap, dominant vs distributed, examples, failures, curves).
- **Prepared in parallel (not yet run/finished):** Task 4 cGAN code; FastAPI backend (6 tests pass, samples, Dockerfile, allow-list .dockerignore); `--version v2` upgrade options for Tasks 1-3 (bigger models, own studies/checkpoints/outputs).
- **Policy change (2026-10-04):** no model-size cap; ONNX files > 50 MB stay local (auto-listed in `models/onnx/.gitignore`), download link later. CLAUDE.md section 7 updated.

- **Frontend, first 7 screens (2026-10-04, separate from phase work)**
  - `frontend/` React 19 + Vite 8 + Tailwind 4 (fonts bundled locally: Newsreader, Inter, IBM Plex Mono). Theme tokens in `src/index.css` (colours sampled from the Stitch landing PNG).
  - Screens: Landing; Universal Restoration and Hard-Routed Restoration (input / processing / result). Soft-MoE and Face-to-Sketch tabs open a "Design in progress" page; System page shows /api/health; Experiments links to MLflow (`VITE_MLFLOW_URL`).
  - `docs/api_contract.md` = the backend API. New backend bits: `POST /api/corrupt`, error map only when a clean reference exists (`input_is_clean`), 16 pet samples. Mock mode `VITE_MOCK=1` (off by default, MOCK DATA badge).
  - Run: backend `.venv\Scripts\python -m uvicorn backend.app.main:app --port 8000`; frontend `cd frontend; npm install; npm run dev` -> http://localhost:5173.

- **Frontend complete (2026-10-04):** all 4 workspaces + landing + System page. Soft-MoE: weight bars, top contributors (from the backend), gate routing diagram, restored/error-map switch. Face-to-Sketch: upload or webcam (permission errors handled), Style 1/2/3 (default 1), "photo is already a cropped face" option, Download sketch. Landing: Stitch illustrations (decorative), one screen without scrolling. 36/36 navigation checks pass. Face-to-Sketch shows the backend's 503 until the Task 4 generator ONNX exists.

- **Phase 5, Task 4 face-to-sketch cGAN (2026-10-04)**
  - Optuna `task4`: 20 trials = 19 completed, 1 pruned, 0 failed; best #11 (val 0.3022): generator base 64, batch 8, lr_G/lr_D 3.4e-4, dropout 0.33, emb 16, lambda_L1 152.6. `configs/task4.yaml`.
  - Final: 150 epochs, best val at epoch 55 (score 0.2884, SSIM 0.513).
  - Test (1,046 pairs, once): L1 0.0986, SSIM 0.498, PSNR 16.4 dB; style 1/2/3 SSIM 0.542/0.411/0.626; style effect 0.11 (style clearly changes the sketch).
  - ONNX `task4_generator.onnx` 167.9 MB, verified (7.2e-7), LOCAL ONLY (needs a download link before submission).
  - All 7 models load in the backend (health "ok"); Face-to-Sketch workspace works for real.

- **Phase 7, Docker + README (2026-10-04)**
  - `docker-compose.yml`: backend (python:3.11.13-slim-bookworm, CPU onnxruntime, non-root) + frontend (node:22.19.0-alpine build -> nginx:1.29.1-alpine, /api proxied to backend:8000). Models mounted read-only. App: http://localhost:8080.
  - Images: backend 448 MB, frontend 82 MB. Docker usage: build cache 842 MB, images 2.77 GB (incl. the user's own n8n image).
  - Fresh clone from GitHub + README steps -> healthy, 7/7 models, 36/36 navigation checks.
  - Large model `task4_generator.onnx` is a GitHub Release asset (models-v1); README has the download command; /api/health points to it if missing.
  - `docs/report_material.md` indexes every figure/table per task.
- **Architecture figure polish (2026-10-04)**
  - Regenerated `docs/figures/diagram_app.png`, `diagram_task1_autoencoder.png`, `diagram_task3_soft_moe.png`, and `diagram_task4_cgan.png` from `scripts/make_diagrams.py`.
  - Improved title hierarchy, spacing, annotation placement, formula wrapping, arrow-label contrast, and export padding; verified the source with `py_compile` and visually checked all four renders.
  - Follow-up correction moved the Task 4 bottleneck/style annotation and generator notes away from the U-Net blocks; the cGAN figure was regenerated and rechecked.

## In progress
- Nothing running. The app runs in Docker at http://localhost:8080 (`docker compose down` to stop).

## Next
- Student: report (IEEE LaTeX), demo video (5-7 min), AI-use appendix (from docs/ai_use_log.md), app/MLflow screenshots, confirm the deadline.
- `Stitch Screens/` folder still uncommitted (move to docs/stitch/?).
- Planned upgrades: see `docs/planned_upgrades.md`.

## Known issues / decisions pending
- Global Python 3.11.7 has `torch 2.11.0+cpu`. Always use `.venv`.
- `gh` CLI not installed (push works via Git Credential Manager).
- MLflow UI: `.venv\Scripts\mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`
- After a fresh clone, run `scripts/prepare_data.py` once to rebuild the (gitignored) image cache.
- Reminder for user: confirm the real deadline with the instructor.
- Optional polish: example/failure figures have extra vertical whitespace (cosmetic; can be redrawn from the saved per-image CSV without re-running the test).

## Environment snapshot (2026-10-03)
- Windows 11, Python 3.11.7, Node 22.19.0, npm 10.9.3, git 2.50.1
- GPU: RTX 3050 Ti Laptop 4 GB, driver 616.92 (CUDA 13.4); `.venv` torch 2.14.1+cu126
- Docker 28.3.2 engine running, Compose v2.38.2, 7.61 GiB RAM; C: 115 GB free
- Repo: https://github.com/humaa-taj/Image-Processing-Engine (main)
